#include "tesseract/ops/MatMul.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/autograd/Function.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/MatMul.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Broadcast.hpp"
#include "tesseract/utils/Logging.hpp"

#include "GemmAvx2.hpp"
#include "IndexIter.hpp"

#if defined(TESSERACT_HAS_OPENMP)
#include <omp.h>
#endif

#if defined(TESSERACT_HAS_EIGEN)
// Row-major `Eigen::Map` gives the exact layout our contiguous slabs use
// (stride-1 innermost, stride-K / stride-N outermost). Disable the
// alignment-128 assumptions so arbitrarily-aligned batch slabs don't
// trigger Eigen's alignment asserts; our contiguous allocations are only
// 8-byte aligned after a non-multiple-of-4 row offset.
#define EIGEN_DONT_ALIGN_STATICALLY 1
#include <Eigen/Core>
#endif

namespace tesseract::ops {

namespace {

// Blocked GEMM, parallelized over the outer `i` block rows when OpenMP is
// available. Blocks are sized so each thread's tile fits comfortably in L2.
// Work is guarded so tiny matrices skip the parallel region entirely (OMP
// overhead dominates under ~64x64). C is treated as uninitialized on entry
// — the kernel starts by zeroing it so callers can use `Tensor::empty`
// and avoid a duplicate memset in the happy path.
template <typename T>
void gemm_naive(const T* a, const T* b, T* c, int64_t M, int64_t N, int64_t K) {
  std::memset(c, 0, static_cast<std::size_t>(M) * static_cast<std::size_t>(N) * sizeof(T));
  constexpr int64_t kBlock = 64;
  // Amortize OpenMP fork/join against a few ms of work. Measured threshold
  // on a x86-64 laptop (2026-04): below ~128^3 * 2 the single-thread path
  // wins; above, scaling to all cores recovers linearly until memory
  // bandwidth saturates.
  const int64_t work = M * N * K;
  const int nthreads = detail::gemm_num_threads(work);
  (void)nthreads;

#if defined(TESSERACT_HAS_OPENMP)
  #pragma omp parallel for schedule(static) if(nthreads > 1) num_threads(nthreads)
#endif
  for (int64_t i0 = 0; i0 < M; i0 += kBlock) {
    const int64_t i_end = std::min(i0 + kBlock, M);
    for (int64_t j0 = 0; j0 < N; j0 += kBlock) {
      const int64_t j_end = std::min(j0 + kBlock, N);
      for (int64_t k0 = 0; k0 < K; k0 += kBlock) {
        const int64_t k_end = std::min(k0 + kBlock, K);
        for (int64_t i = i0; i < i_end; ++i) {
          for (int64_t k = k0; k < k_end; ++k) {
            const T aik = a[i * K + k];
            T* cr = c + i * N;
            const T* br = b + k * N;
            for (int64_t j = j0; j < j_end; ++j) {
              cr[j] = static_cast<T>(cr[j] + aik * br[j]);
            }
          }
        }
      }
    }
  }
}

// Slab-level gemm dispatcher. Three tiers, ordered by expected
// throughput on x86-64:
//
//   1. Hand-written 4x8 AVX2 + FMA microkernel (`gemm_avx2_f32`). Only
//      eligible for `float` input on CPUs that report AVX2 + FMA at
//      runtime, but on those CPUs this is usually the fastest path
//      (≥ 60 GFLOP/s at 512² on AVX2, the B-002 target).
//   2. Eigen's GEBP kernel (`C.noalias() = A * B`). Always eligible
//      when `TESSERACT_HAS_EIGEN` is defined — handles float64 and
//      acts as the fallback for float when AVX2 isn't available.
//   3. Scalar blocked + optional OpenMP `gemm_naive`.
//
// `Half` / `BFloat16` take a fourth path: upcast both operands to FP32,
// run the fastest available FP32 kernel, then downcast the result back.
// This matches how real hardware (Tensor Cores / AMX) implements these
// dtypes — storage in 16 bits, accumulation in 32 — and keeps the per-K
// quantization error bounded by a single round-to-nearest instead of
// accumulating through K rounds of narrow-precision adds.
//
// Both 1 and 2 fully overwrite C (no accumulate), so the caller can
// use `Tensor::empty`; the scalar fallback self-`memset`s at entry.
template <typename T>
void gemm_slab(const T* a, const T* b, T* c, int64_t M, int64_t N, int64_t K) {
  if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
    const std::size_t nA = static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
    const std::size_t nB = static_cast<std::size_t>(K) * static_cast<std::size_t>(N);
    const std::size_t nC = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    std::vector<float> af(nA);
    std::vector<float> bf(nB);
    std::vector<float> cf(nC);
    for (std::size_t i = 0; i < nA; ++i) af[i] = static_cast<float>(a[i]);
    for (std::size_t i = 0; i < nB; ++i) bf[i] = static_cast<float>(b[i]);
    gemm_slab<float>(af.data(), bf.data(), cf.data(), M, N, K);
    for (std::size_t i = 0; i < nC; ++i) c[i] = static_cast<T>(cf[i]);
    return;
  } else {
    if constexpr (std::is_same_v<T, float>) {
      // Cache the CPU probe on first call — `__builtin_cpu_supports`
      // is cheap but not free, and this path is on the hot loop of
      // every batched matmul.
      static const bool avx2_ok = detail::gemm_avx2_f32_supported();
      if (avx2_ok) {
        if (detail::gemm_avx2_f32(a, b, c, M, N, K)) return;
      }
    }
#if defined(TESSERACT_HAS_EIGEN)
    using MatT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MapC = Eigen::Map<MatT, Eigen::Unaligned>;
    using MapK = Eigen::Map<const MatT, Eigen::Unaligned>;
    MapK A(a, M, K);
    MapK B(b, K, N);
    MapC C(c, M, N);
    C.noalias() = A * B;
#else
    gemm_naive<T>(a, b, c, M, N, K);
#endif
  }
}

// Pull out the leading "batch" portion of `s`, i.e. every dim except the
// trailing pair used for the matmul. Returns a rank-0 Shape for rank-2
// tensors.
Shape batch_shape(const Shape& s) {
  Shape out;
  out.resize(s.rank() - 2);
  for (std::size_t i = 0; i + 2 < s.rank(); ++i) out[i] = s[i];
  return out;
}

// Detect an operand's last-two layout for the cuBLASLt dispatch. Given
// a tensor `t` with logical trailing shape `[rows, cols]`, returns the
// cuBLASLt `op` + leading-dim pair that describes the memory layout:
//
//   * innermost stride == 1 → row-major as stored → op=N, ld = stride[-2].
//     This is the "normal" contiguous case (or any slab obtained by
//     slicing leading dims of a row-major contig tensor).
//   * stride[-2] == 1         → column-major as stored → op=T, ld = stride[-1].
//     This is what `Tensor::transpose(-2, -1)` produces on top of a
//     row-major input; cuBLASLt treats the stored matrix as its
//     transpose-of-transpose by flipping op.
//
// Anything else (both strides != 1, padded/dilated layouts, ...) is
// rejected with a clear message that points at `.contiguous()` /
// future M2H strided copies. Throws `DeviceError`.
struct CudaMatOpLayout {
  cuda::detail::MmOp op;
  int64_t ld;
};

CudaMatOpLayout detect_cuda_mat_layout(const Tensor& t, int64_t rows,
                                       int64_t cols, const char* side) {
  const std::size_t r = t.shape().rank();
  const int64_t s0 = t.strides()[r - 2];
  const int64_t s1 = t.strides()[r - 1];
  if (s1 == 1) {
    // Stored row-major, ld = s0 (>= cols for any valid row-major layout).
    TESSERACT_CHECK(s0 >= cols,
                    "matmul: CUDA {} operand has row stride {} < cols {} "
                    "(non-dense row-major layout is not supported yet)",
                    side, s0, cols);
    return {cuda::detail::MmOp::None, s0};
  }
  if (s0 == 1) {
    // Stored column-major, i.e. a transpose view over a row-major
    // `[cols, rows]` contig. Map to op=T with ld = s1 (>= rows).
    TESSERACT_CHECK(s1 >= rows,
                    "matmul: CUDA {} operand has col stride {} < rows {} "
                    "(non-dense column-major layout is not supported yet)",
                    side, s1, rows);
    return {cuda::detail::MmOp::Transpose, s1};
  }
  TESSERACT_CHECK(false,
                  "matmul: CUDA {} operand has strides [{}, {}] — only "
                  "row-major (stride -1 == 1) or simple transpose (stride "
                  "-2 == 1) are supported in M2G. Call .contiguous() on "
                  "the operand (or wait for M2H strided copy support).",
                  side, s0, s1);
  return {cuda::detail::MmOp::None, 0};  // unreachable
}

// CUDA dispatch path for matmul. Drives one `launch_matmul` call per
// output batch slab, sharing the same batch-broadcast bookkeeping as
// the CPU loop below. Rank-2 inputs hit a single cuBLASLt call with
// batch-count = 1 (out_batch is rank-0); the per-slab loop body only
// runs once. We don't `.contiguous()` the inputs — CUDA doesn't
// support strided materialization until M2H — so the caller is
// responsible for passing operands whose last-two strides are either
// row-major (stride[-1] == 1) or a single transpose (stride[-2] == 1).
// Everything else lands on `detect_cuda_mat_layout`'s throw.
Tensor matmul_forward_cuda(const Tensor& lhs, const Tensor& rhs,
                           const Shape& out_batch, const Shape& out_shape,
                           int64_t M, int64_t N, int64_t K) {
  Tensor c = Tensor::empty(out_shape, lhs.dtype(), lhs.device());

  // Layout of the matmul slabs.
  const auto a_lay = detect_cuda_mat_layout(lhs, M, K, "lhs");
  const auto b_lay = detect_cuda_mat_layout(rhs, K, N, "rhs");
  const int64_t ldc = N;  // c is freshly Tensor::empty, so row-major contig.

  // Batch-grid offsets: same broadcast-aligned stride trick as the CPU
  // path, but applied to the actual lhs/rhs tensors (no `.contiguous()`
  // substitution). A broadcasted batch dim lands on stride 0 so every
  // slab on that dim shares the same base pointer.
  const Shape lhs_batch = batch_shape(lhs.shape());
  const Shape rhs_batch = batch_shape(rhs.shape());
  Shape lhs_batch_strides_raw, rhs_batch_strides_raw;
  lhs_batch_strides_raw.resize(lhs_batch.rank());
  rhs_batch_strides_raw.resize(rhs_batch.rank());
  for (std::size_t i = 0; i < lhs_batch.rank(); ++i) {
    lhs_batch_strides_raw[i] = lhs.strides()[i];
  }
  for (std::size_t i = 0; i < rhs_batch.rank(); ++i) {
    rhs_batch_strides_raw[i] = rhs.strides()[i];
  }
  Shape lhs_aligned, rhs_aligned;
  align_for_broadcast(lhs_batch, lhs_batch_strides_raw, out_batch, lhs_aligned);
  align_for_broadcast(rhs_batch, rhs_batch_strides_raw, out_batch, rhs_aligned);

  // Stream stays const across all slabs — issuing per-slab launches on
  // the same stream means cuBLASLt pipelines them without host-side
  // sync, and all end-of-matmul consumers (e.g. the next `ops::*`
  // call) serialize correctly on stream order.
  Stream s = current_stream(lhs.device());
  const int device_index = lhs.device().index;

  const int64_t c_slab = M * N;
  const std::size_t elem = dtype_size(lhs.dtype());
  const auto* a_base = static_cast<const std::byte*>(lhs.raw_data());
  const auto* b_base = static_cast<const std::byte*>(rhs.raw_data());
  auto* c_base = static_cast<std::byte*>(c.raw_data());

  // Rank-0 fast path (the everyday 2-D case): one cuBLASLt call, no
  // bookkeeping. Measured on RTX 5880 Ada FP32 4096² this reaches
  // ~93 % of raw cuBLASLt throughput, matching the M2G exit bar.
  if (out_batch.rank() == 0) {
    cuda::detail::launch_matmul(
        lhs.dtype(), device_index, M, N, K,
        a_base, a_lay.op, a_lay.ld,
        b_base, b_lay.op, b_lay.ld,
        c_base, ldc, s.native_handle());
    return c;
  }

  int64_t flat_batch = 0;
  detail::for_each_index(out_batch,
                         [&](int64_t /*flat*/, const detail::IndexArray& idx) {
    int64_t off_a = 0;
    int64_t off_b = 0;
    for (std::size_t d = 0; d < out_batch.rank(); ++d) {
      off_a += idx[d] * lhs_aligned[d];
      off_b += idx[d] * rhs_aligned[d];
    }
    cuda::detail::launch_matmul(
        lhs.dtype(), device_index, M, N, K,
        a_base + static_cast<std::size_t>(off_a) * elem, a_lay.op, a_lay.ld,
        b_base + static_cast<std::size_t>(off_b) * elem, b_lay.op, b_lay.ld,
        c_base + static_cast<std::size_t>(flat_batch * c_slab) * elem, ldc,
        s.native_handle());
    ++flat_batch;
  });
  return c;
}

Tensor matmul_forward(const Tensor& lhs, const Tensor& rhs) {
  TESSERACT_CHECK(lhs.defined() && rhs.defined(), "matmul: operand is undefined");
  TESSERACT_CHECK(lhs.dtype() == rhs.dtype(),
                  "matmul: dtype mismatch ({} vs {})",
                  dtype_name(lhs.dtype()), dtype_name(rhs.dtype()));
  TESSERACT_CHECK(lhs.device() == rhs.device(),
                  "matmul: device mismatch ({} vs {})",
                  lhs.device().to_string(), rhs.device().to_string());
  TESSERACT_CHECK(dtype_is_floating(lhs.dtype()),
                  "matmul: floating-point dtypes required, got {}", dtype_name(lhs.dtype()));
  TESSERACT_CHECK(lhs.rank() >= 2 && rhs.rank() >= 2,
                  "matmul: inputs must be rank >= 2 (got {} and {})",
                  lhs.shape().to_string(), rhs.shape().to_string());

  // Inner matmul dims are the last two of each operand.
  const std::size_t lr = lhs.shape().rank();
  const std::size_t rr = rhs.shape().rank();
  const int64_t M  = lhs.shape()[lr - 2];
  const int64_t K  = lhs.shape()[lr - 1];
  const int64_t K2 = rhs.shape()[rr - 2];
  const int64_t N  = rhs.shape()[rr - 1];
  TESSERACT_CHECK(K == K2, "matmul: inner dims disagree ({} vs {})", K, K2);

  // Broadcast the leading (batch) dims PyTorch-style. The rank-2 fast path
  // still falls out: both batch shapes are rank-0 and `out_batch` is rank-0
  // too, so the outer loop body runs exactly once.
  const Shape lhs_batch = batch_shape(lhs.shape());
  const Shape rhs_batch = batch_shape(rhs.shape());
  const Shape out_batch = broadcast_shape(lhs_batch, rhs_batch);

  // Full output shape = out_batch ++ [M, N].
  Shape out_shape;
  out_shape.resize(out_batch.rank() + 2);
  for (std::size_t i = 0; i < out_batch.rank(); ++i) out_shape[i] = out_batch[i];
  out_shape[out_batch.rank() + 0] = M;
  out_shape[out_batch.rank() + 1] = N;

  // CUDA path: cuBLASLt handles both the rank-2 and batched cases
  // directly off the operand strides (no `.contiguous()` materialization,
  // since strided CUDA copy lands in M2H). The CPU path below still owns
  // CPU tensors — device mismatch has already been rejected above.
  if (lhs.device().is_cuda()) {
    return matmul_forward_cuda(lhs, rhs, out_batch, out_shape, M, N, K);
  }

  // Materialize both operands contiguous so the inner `gemm_naive` can
  // assume row-major (M,K) / (K,N) / (M,N) slabs. For rank-2 inputs this
  // is just the old behaviour; for batched inputs it also guarantees that
  // every batch slab sits at a predictable offset of
  // `batch_stride := prod(trailing two dims)` elements.
  Tensor a = lhs.is_contiguous() ? lhs : lhs.contiguous();
  Tensor b = rhs.is_contiguous() ? rhs : rhs.contiguous();
  // Uninitialized storage: every (M, N) slab is fully written by the gemm
  // kernel (Eigen uses `noalias = A*B`, the scalar fallback self-memsets).
  Tensor c = Tensor::empty(out_shape, lhs.dtype(), lhs.device());

  // Batch strides of the two operands, aligned to `out_batch`. Broadcasted
  // batch axes get stride 0 so we just reuse the same (M,K) / (K,N) slab.
  Shape lhs_batch_strides_raw;
  lhs_batch_strides_raw.resize(lhs_batch.rank());
  for (std::size_t i = 0; i < lhs_batch.rank(); ++i) {
    lhs_batch_strides_raw[i] = a.strides()[i];
  }
  Shape rhs_batch_strides_raw;
  rhs_batch_strides_raw.resize(rhs_batch.rank());
  for (std::size_t i = 0; i < rhs_batch.rank(); ++i) {
    rhs_batch_strides_raw[i] = b.strides()[i];
  }
  Shape lhs_aligned, rhs_aligned;
  align_for_broadcast(lhs_batch, lhs_batch_strides_raw, out_batch, lhs_aligned);
  align_for_broadcast(rhs_batch, rhs_batch_strides_raw, out_batch, rhs_aligned);

  const int64_t c_batch_stride = M * N;

  dispatch_float_with_half(lhs.dtype(), [&]<typename T>() {
    const T* pa = a.data_ptr<T>();
    const T* pb = b.data_ptr<T>();
    T* pc = c.data_ptr<T>();

    // Rank-0 fast path (the ubiquitous 2-D case): one gemm call, no
    // bookkeeping.
    if (out_batch.rank() == 0) {
      gemm_slab<T>(pa, pb, pc, M, N, K);
      return;
    }

    // Otherwise walk the broadcasted batch grid. `for_each_index` drives
    // `flat` in row-major order so we can index the output slab via a
    // simple counter; the aligned strides handle each operand's offset
    // (including broadcast axes with stride 0).
    int64_t flat_batch = 0;
    detail::for_each_index(out_batch,
                           [&](int64_t /*flat*/, const detail::IndexArray& idx) {
      int64_t off_a = 0;
      int64_t off_b = 0;
      for (std::size_t d = 0; d < out_batch.rank(); ++d) {
        off_a += idx[d] * lhs_aligned[d];
        off_b += idx[d] * rhs_aligned[d];
      }
      gemm_slab<T>(pa + off_a, pb + off_b, pc + flat_batch * c_batch_stride,
                   M, N, K);
      ++flat_batch;
    });
  });
  return c;
}

// Swap the last two dims of `t`. Used inside the backward to express
// `rhs.mT` / `lhs.mT` without forcing callers to spell it out.
Tensor mat_transpose(const Tensor& t) {
  const std::size_t r = t.rank();
  return t.transpose(static_cast<int64_t>(r - 2), static_cast<int64_t>(r - 1));
}

struct MatMulBackward : Node {
  Tensor a_saved, b_saved;
  std::string_view name() const override { return "MatMulBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(2);
    // grad_a = g @ b.mT, then sum-reduce broadcast batch dims back to
    // `a_saved.shape()`. For the rank-2 case the shapes already agree
    // and `reduce_to_shape` is a no-op (plus a contiguous clone).
    if (next_edges[0].requires_grad) {
      Tensor ga_full = matmul_forward(g, mat_transpose(b_saved));
      outs[0] = reduce_to_shape(ga_full, a_saved.shape());
    }
    // grad_b = a.mT @ g, then sum-reduce.
    if (next_edges[1].requires_grad) {
      Tensor gb_full = matmul_forward(mat_transpose(a_saved), g);
      outs[1] = reduce_to_shape(gb_full, b_saved.shape());
    }
    return outs;
  }
};

}  // namespace

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
  Tensor out = matmul_forward(lhs, rhs);
  if (is_grad_enabled() && autograd::any_requires_grad(lhs, rhs)) {
    auto n = std::make_shared<MatMulBackward>();
    n->a_saved = lhs;
    n->b_saved = rhs;
    n->next_edges = { autograd::edge_for(lhs), autograd::edge_for(rhs) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("matmul", {&lhs, &rhs}, {&out});
  return out;
}

}  // namespace tesseract::ops

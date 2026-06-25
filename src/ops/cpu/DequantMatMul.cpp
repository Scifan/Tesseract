#include "tesseract/ops/Quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "tesseract/autograd/Function.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/detail/DequantMatMul.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

#if defined(TESSERACT_HAS_OPENMP)
#include <omp.h>
#endif

namespace tesseract::ops {

namespace {

// -------------------- CPU reference kernel (FP32 accumulator) --------------------
//
// y[m, n] = scale[n] * sum_k x[m, k] * q_w[n, k]
//
// Identical numerics to the CUDA fused kernel: inner reduction runs in
// FP32, scale multiply in FP32, narrow on store. The CPU path is here
// primarily so the packer and the module can be validated on CPU-only
// CI, and for the autograd-off path when CUDA is not in use.
template <typename Tx>
void dequant_matmul_int8_cpu(const Tx* x, const int8_t* q_w,
                             const float* scale, Tx* y,
                             int64_t M, int64_t N, int64_t K) {
  // Parallelize over output rows (M) when M is large; fall through to
  // single-thread for decode-shape M=1.
#if defined(TESSERACT_HAS_OPENMP)
  const bool parallel = (M * N >= 1024);
  #pragma omp parallel for collapse(2) schedule(static) if(parallel)
#endif
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      const Tx*     rx = x + m * K;
      const int8_t* rw = q_w + n * K;
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        const float xv = static_cast<float>(rx[k]);
        const float wv = static_cast<float>(rw[k]);
        acc += xv * wv;
      }
      y[m * N + n] = static_cast<Tx>(acc * scale[n]);
    }
  }
}

// Narrow CPU-path dispatch: only FP32 / FP16 / BF16. FP64 is rejected at
// the op-layer boundary so we don't silently run a double-precision CPU
// loop against half-precision-only kernels.
void dispatch_cpu(DType dtype, const void* x, const int8_t* q_w,
                  const float* scale, void* y,
                  int64_t M, int64_t N, int64_t K) {
  switch (dtype) {
    case DType::Float32:
      dequant_matmul_int8_cpu(static_cast<const float*>(x), q_w, scale,
                              static_cast<float*>(y), M, N, K);
      return;
    case DType::Float16:
      dequant_matmul_int8_cpu(static_cast<const Half*>(x), q_w, scale,
                              static_cast<Half*>(y), M, N, K);
      return;
    case DType::BFloat16:
      dequant_matmul_int8_cpu(static_cast<const BFloat16*>(x), q_w, scale,
                              static_cast<BFloat16*>(y), M, N, K);
      return;
    default:
      TESSERACT_THROW("dequantize_matmul_int8: CPU path expects "
                      "Float32/Float16/BFloat16, got {}", dtype_name(dtype));
  }
}

// -------------------- Weight dequantization (for the autograd fallback) --------------------
//
// Materialize the full FP weight `W[n, k] = q_w[n, k] * scale[n]` in
// the activation's dtype. Only used when autograd needs to differentiate
// through `x` — the fused forward never touches this path, so the
// allocation + bounce-to-host cost is confined to training setups and
// carefully-constructed autograd probes.
Tensor dequantize_weight(const Tensor& q_w, const Tensor& scale,
                         DType out_dtype, Device out_device) {
  NoGradGuard _;  // the weight tensor is frozen by definition

  Tensor q_cpu = q_w.to(cpu_device()).contiguous();
  Tensor s_cpu = scale.to(cpu_device()).contiguous();
  const int64_t N = q_cpu.shape()[0];
  const int64_t K = q_cpu.shape()[1];

  Tensor W_cpu = Tensor::empty({N, K}, out_dtype, cpu_device());
  const int8_t* qp = q_cpu.data_ptr<int8_t>();
  const float*  sp = s_cpu.data_ptr<float>();

  auto fill = [&]<typename T>() {
    T* wp = W_cpu.data_ptr<T>();
    for (int64_t n = 0; n < N; ++n) {
      const float s = sp[n];
      for (int64_t k = 0; k < K; ++k) {
        wp[n * K + k] = static_cast<T>(static_cast<float>(qp[n * K + k]) * s);
      }
    }
  };
  switch (out_dtype) {
    case DType::Float32:  fill.template operator()<float>();    break;
    case DType::Float16:  fill.template operator()<Half>();     break;
    case DType::BFloat16: fill.template operator()<BFloat16>(); break;
    default:
      TESSERACT_THROW("dequantize_weight: unsupported dtype {}",
                      dtype_name(out_dtype));
  }

  if (out_device.is_cpu()) return W_cpu;
  return W_cpu.to(out_device);
}

}  // namespace

Tensor dequantize_matmul_int8(const Tensor& x,
                              const Tensor& q_w,
                              const Tensor& scale) {
  // ---------- Validation ----------
  TESSERACT_CHECK(x.defined() && q_w.defined() && scale.defined(),
                  "dequantize_matmul_int8: all inputs (x, q_w, scale) "
                  "must be defined");
  TESSERACT_CHECK(x.rank() >= 2,
                  "dequantize_matmul_int8: x must be rank >= 2 "
                  "[..., in_features], got {}", x.shape().to_string());
  TESSERACT_CHECK(q_w.rank() == 2,
                  "dequantize_matmul_int8: q_w must be rank-2 "
                  "[out_features, in_features], got {}",
                  q_w.shape().to_string());
  TESSERACT_CHECK(scale.rank() == 1,
                  "dequantize_matmul_int8: scale must be rank-1 "
                  "[out_features], got {}", scale.shape().to_string());

  const int64_t K = x.shape()[x.rank() - 1];
  const int64_t N = q_w.shape()[0];
  TESSERACT_CHECK(q_w.shape()[1] == K,
                  "dequantize_matmul_int8: q_w[in_features]={} must equal "
                  "x.last_dim={}", q_w.shape()[1], K);
  TESSERACT_CHECK(scale.shape()[0] == N,
                  "dequantize_matmul_int8: scale[out_features]={} must equal "
                  "q_w[out_features]={}", scale.shape()[0], N);

  TESSERACT_CHECK(q_w.dtype() == DType::Int8,
                  "dequantize_matmul_int8: q_w dtype must be Int8, got {}",
                  dtype_name(q_w.dtype()));
  TESSERACT_CHECK(scale.dtype() == DType::Float32,
                  "dequantize_matmul_int8: scale dtype must be Float32, got {}",
                  dtype_name(scale.dtype()));

  const DType dtype = x.dtype();
  TESSERACT_CHECK(dtype == DType::Float32 || dtype == DType::Float16 ||
                  dtype == DType::BFloat16,
                  "dequantize_matmul_int8: x dtype must be "
                  "Float32/Float16/BFloat16, got {}", dtype_name(dtype));

  TESSERACT_CHECK(x.device() == q_w.device() && x.device() == scale.device(),
                  "dequantize_matmul_int8: x / q_w / scale device mismatch "
                  "({} / {} / {})", x.device().to_string(),
                  q_w.device().to_string(), scale.device().to_string());

  // ---------- Autograd fallback ----------
  //
  // Wave 3.1 MVP: gradients through frozen quantized weights are
  // undefined (weights are integers — the only sensible differential
  // is `d/dx`, not `d/dq_w`). When the activation `x` requires grad
  // we dequantize the weight once on the fly and go through the
  // regular `matmul(x, W^T)` path, which wires `ops::matmul`'s
  // `MatMulBackward` node against `x`. `q_w` / `scale` are never
  // attached to the graph. This keeps quantized layers composable
  // inside otherwise-trainable networks.
  if (is_grad_enabled() && x.requires_grad()) {
    Tensor W_fp = dequantize_weight(q_w, scale, dtype, x.device());  // [N, K]
    Tensor Wt   = ops::transpose(W_fp, 0, 1);                         // [K, N]
    Tensor y    = ops::matmul(x, Wt);                                 // [..., N]
    graph::maybe_record("dequant_matmul_int8", {&x, &q_w, &scale},
                        {&y}, {{"autograd_fallback", int64_t{1}}});
    return y;
  }

  // ---------- Inference fast path ----------
  //
  // Flatten leading dims into a single `M` axis so the kernel can treat
  // `x` as `[M, K]`. `x` is required to be contiguous for the fused
  // kernel; non-contiguous activations get a `contiguous()` copy here.
  Tensor xc    = x.is_contiguous()  ? x    : ops::contiguous(x);
  Tensor qwc   = q_w.is_contiguous()? q_w  : ops::contiguous(q_w);
  Tensor sc    = scale.is_contiguous() ? scale : ops::contiguous(scale);

  int64_t M = 1;
  for (int64_t i = 0; i + 1 < static_cast<int64_t>(xc.rank()); ++i) {
    M *= xc.shape()[i];
  }

  // Allocate `y` with the right shape (`[..., N]`). Allocate via a
  // reshape of a flat `[M, N]` buffer so the raw pointer is what the
  // kernel writes into.
  std::vector<int64_t> y_dims;
  y_dims.reserve(xc.rank());
  for (int64_t i = 0; i + 1 < static_cast<int64_t>(xc.rank()); ++i) {
    y_dims.push_back(xc.shape()[i]);
  }
  y_dims.push_back(N);
  Tensor y = Tensor::empty(Shape(y_dims), dtype, xc.device());

  if (xc.device().is_cuda()) {
    Stream s = current_stream(xc.device());
    cuda::detail::launch_dequant_matmul_int8(
        dtype, xc.device().index, M, N, K,
        xc.raw_data(), qwc.raw_data(), sc.raw_data(),
        y.raw_data(), s.native_handle());
  } else {
    dispatch_cpu(dtype, xc.raw_data(),
                 static_cast<const int8_t*>(qwc.raw_data()),
                 static_cast<const float*>(sc.raw_data()),
                 y.raw_data(), M, N, K);
  }

  graph::maybe_record("dequant_matmul_int8", {&x, &q_w, &scale}, {&y});
  return y;
}

// ---------------------------------------------------------------------------
// INT4 per-group (Wave 3.2)
// ---------------------------------------------------------------------------

namespace {

// Sign-extend the low four bits of `nib` (treated as an unsigned 4-bit
// payload in [0..15]) to a signed int in `[-8, 7]`. Our packer trims
// to `[-7, 7]` but we keep the sign-extend range wide to match the
// nibble's raw two's-complement encoding.
inline int signext4_cpu(uint32_t nib) {
  return static_cast<int>((nib ^ 0x8u)) - 8;
}

template <typename Tx>
void dequant_matmul_int4_group_cpu(const Tx* x,
                                   const int8_t* q_packed,
                                   const float* scale,
                                   Tx* y,
                                   int64_t M, int64_t N, int64_t K,
                                   int64_t group_size) {
  const int64_t packed_cols   = K / 2;
  const int64_t groups_per_row = K / group_size;
#if defined(TESSERACT_HAS_OPENMP)
  const bool parallel = (M * N >= 1024);
  #pragma omp parallel for collapse(2) schedule(static) if(parallel)
#endif
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      const Tx*      rx = x         + m * K;
      const int8_t*  rw = q_packed  + n * packed_cols;
      const float*   rs = scale     + n * groups_per_row;
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        const uint8_t byte = static_cast<uint8_t>(rw[k >> 1]);
        const uint32_t nib =
            static_cast<uint32_t>((byte >> ((k & 1) ? 4 : 0)) & 0xFu);
        const int q4 = signext4_cpu(nib);
        const float xv = static_cast<float>(rx[k]);
        const float s  = rs[k / group_size];
        acc += xv * static_cast<float>(q4) * s;
      }
      y[m * N + n] = static_cast<Tx>(acc);
    }
  }
}

void dispatch_cpu_int4_group(DType dtype, const void* x,
                             const int8_t* q_packed, const float* scale,
                             void* y,
                             int64_t M, int64_t N, int64_t K,
                             int64_t group_size) {
  switch (dtype) {
    case DType::Float32:
      dequant_matmul_int4_group_cpu(static_cast<const float*>(x),
                                    q_packed, scale,
                                    static_cast<float*>(y),
                                    M, N, K, group_size);
      return;
    case DType::Float16:
      dequant_matmul_int4_group_cpu(static_cast<const Half*>(x),
                                    q_packed, scale,
                                    static_cast<Half*>(y),
                                    M, N, K, group_size);
      return;
    case DType::BFloat16:
      dequant_matmul_int4_group_cpu(static_cast<const BFloat16*>(x),
                                    q_packed, scale,
                                    static_cast<BFloat16*>(y),
                                    M, N, K, group_size);
      return;
    default:
      TESSERACT_THROW("dequantize_matmul_int4_group: CPU path expects "
                      "Float32/Float16/BFloat16, got {}", dtype_name(dtype));
  }
}

// Materialize the full FP weight `W[n, k] = q4(packed) * scale[n, k/G]`.
// Mirrors `dequantize_weight` but for the group-scaled INT4 layout.
Tensor dequantize_weight_int4_group(const Tensor& q_packed,
                                    const Tensor& scale,
                                    int64_t group_size,
                                    DType out_dtype, Device out_device) {
  NoGradGuard _;

  Tensor q_cpu = q_packed.to(cpu_device()).contiguous();
  Tensor s_cpu = scale.to(cpu_device()).contiguous();
  const int64_t N           = q_cpu.shape()[0];
  const int64_t packed_cols = q_cpu.shape()[1];
  const int64_t K           = packed_cols * 2;
  const int64_t groups_per_row = K / group_size;

  Tensor W_cpu = Tensor::empty({N, K}, out_dtype, cpu_device());
  const int8_t* qp = q_cpu.data_ptr<int8_t>();
  const float*  sp = s_cpu.data_ptr<float>();

  auto fill = [&]<typename T>() {
    T* wp = W_cpu.data_ptr<T>();
    for (int64_t n = 0; n < N; ++n) {
      const float*  rs = sp + n * groups_per_row;
      const int8_t* rw = qp + n * packed_cols;
      for (int64_t k = 0; k < K; ++k) {
        const uint8_t byte = static_cast<uint8_t>(rw[k >> 1]);
        const uint32_t nib =
            static_cast<uint32_t>((byte >> ((k & 1) ? 4 : 0)) & 0xFu);
        const int q4 = signext4_cpu(nib);
        wp[n * K + k] = static_cast<T>(
            static_cast<float>(q4) * rs[k / group_size]);
      }
    }
  };
  switch (out_dtype) {
    case DType::Float32:  fill.template operator()<float>();    break;
    case DType::Float16:  fill.template operator()<Half>();     break;
    case DType::BFloat16: fill.template operator()<BFloat16>(); break;
    default:
      TESSERACT_THROW("dequantize_weight_int4_group: unsupported dtype {}",
                      dtype_name(out_dtype));
  }

  if (out_device.is_cpu()) return W_cpu;
  return W_cpu.to(out_device);
}

}  // namespace

Tensor dequantize_matmul_int4_group(const Tensor& x,
                                    const Tensor& q_packed,
                                    const Tensor& scale,
                                    int64_t group_size) {
  TESSERACT_CHECK(x.defined() && q_packed.defined() && scale.defined(),
                  "dequantize_matmul_int4_group: all inputs must be defined");
  TESSERACT_CHECK(x.rank() >= 2,
                  "dequantize_matmul_int4_group: x must be rank >= 2, got {}",
                  x.shape().to_string());
  TESSERACT_CHECK(q_packed.rank() == 2,
                  "dequantize_matmul_int4_group: q_packed must be rank-2 "
                  "[out_features, in_features/2], got {}",
                  q_packed.shape().to_string());
  TESSERACT_CHECK(scale.rank() == 2,
                  "dequantize_matmul_int4_group: scale must be rank-2 "
                  "[out_features, in_features/group_size], got {}",
                  scale.shape().to_string());

  TESSERACT_CHECK(group_size >= 2 && (group_size % 2) == 0,
                  "dequantize_matmul_int4_group: group_size must be even and "
                  ">=2, got {}", group_size);

  const int64_t K = x.shape()[x.rank() - 1];
  const int64_t N = q_packed.shape()[0];
  TESSERACT_CHECK(q_packed.shape()[1] * 2 == K,
                  "dequantize_matmul_int4_group: q_packed[in_features/2]={} "
                  "but x.last_dim={} (expected {}*2=={})",
                  q_packed.shape()[1], K, q_packed.shape()[1], K);
  TESSERACT_CHECK(K % group_size == 0,
                  "dequantize_matmul_int4_group: K ({}) must be a multiple of "
                  "group_size ({})", K, group_size);
  TESSERACT_CHECK(scale.shape()[0] == N,
                  "dequantize_matmul_int4_group: scale[out_features]={} must "
                  "equal q_packed[out_features]={}", scale.shape()[0], N);
  TESSERACT_CHECK(scale.shape()[1] == K / group_size,
                  "dequantize_matmul_int4_group: scale[groups]={} must equal "
                  "K/group_size={}", scale.shape()[1], K / group_size);

  TESSERACT_CHECK(q_packed.dtype() == DType::Int8,
                  "dequantize_matmul_int4_group: q_packed dtype must be Int8, "
                  "got {}", dtype_name(q_packed.dtype()));
  TESSERACT_CHECK(scale.dtype() == DType::Float32,
                  "dequantize_matmul_int4_group: scale dtype must be Float32, "
                  "got {}", dtype_name(scale.dtype()));

  const DType dtype = x.dtype();
  TESSERACT_CHECK(dtype == DType::Float32 || dtype == DType::Float16 ||
                  dtype == DType::BFloat16,
                  "dequantize_matmul_int4_group: x dtype must be "
                  "Float32/Float16/BFloat16, got {}", dtype_name(dtype));

  TESSERACT_CHECK(x.device() == q_packed.device() &&
                  x.device() == scale.device(),
                  "dequantize_matmul_int4_group: device mismatch "
                  "({} / {} / {})", x.device().to_string(),
                  q_packed.device().to_string(), scale.device().to_string());

  // Autograd fallback — dequantize once, reuse matmul's backward.
  if (is_grad_enabled() && x.requires_grad()) {
    Tensor W_fp = dequantize_weight_int4_group(q_packed, scale, group_size,
                                               dtype, x.device());   // [N, K]
    Tensor Wt   = ops::transpose(W_fp, 0, 1);                         // [K, N]
    Tensor y    = ops::matmul(x, Wt);
    graph::maybe_record("dequant_matmul_int4_group",
                        {&x, &q_packed, &scale}, {&y},
                        {{"autograd_fallback", int64_t{1}},
                         {"group_size", group_size}});
    return y;
  }

  // Inference fast path.
  Tensor xc  = x.is_contiguous()  ? x  : ops::contiguous(x);
  Tensor qwc = q_packed.is_contiguous() ? q_packed : ops::contiguous(q_packed);
  Tensor sc  = scale.is_contiguous()    ? scale    : ops::contiguous(scale);

  int64_t M = 1;
  for (int64_t i = 0; i + 1 < static_cast<int64_t>(xc.rank()); ++i) {
    M *= xc.shape()[i];
  }

  std::vector<int64_t> y_dims;
  y_dims.reserve(xc.rank());
  for (int64_t i = 0; i + 1 < static_cast<int64_t>(xc.rank()); ++i) {
    y_dims.push_back(xc.shape()[i]);
  }
  y_dims.push_back(N);
  Tensor y = Tensor::empty(Shape(y_dims), dtype, xc.device());

  if (xc.device().is_cuda()) {
    Stream s = current_stream(xc.device());
    cuda::detail::launch_dequant_matmul_int4_group(
        dtype, xc.device().index, M, N, K, group_size,
        xc.raw_data(), qwc.raw_data(), sc.raw_data(),
        y.raw_data(), s.native_handle());
  } else {
    dispatch_cpu_int4_group(dtype, xc.raw_data(),
                            static_cast<const int8_t*>(qwc.raw_data()),
                            static_cast<const float*>(sc.raw_data()),
                            y.raw_data(), M, N, K, group_size);
  }

  graph::maybe_record("dequant_matmul_int4_group",
                      {&x, &q_packed, &scale}, {&y},
                      {{"group_size", group_size}});
  return y;
}

}  // namespace tesseract::ops

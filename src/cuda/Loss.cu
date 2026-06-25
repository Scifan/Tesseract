// M2F: CUDA fused cross-entropy (with-logits) forward + backward.
// See `src/cuda/Reduction.cu` and `Softmax.cu` for the broader M2F
// design notes; this TU is simpler because the input is hard-wired
// to rank-2 `[N, C]` and the caller has already materialized
// contiguous buffers.
//
// Kernel strategy:
//   * Forward: one CUDA block per row. Threads cooperate on the
//     row's max + sum(exp(x - max)) via shared-memory tree
//     reductions, then the rank-0 thread computes `log_sum_exp -
//     logits[target]` as the per-row loss contribution. A final
//     atomicAdd into a single fp32/fp64 accumulator produces the
//     sum-of-row-losses; we then launch a tiny finalize kernel
//     that divides by N for the mean.
//
//     NOTE on determinism: `atomicAdd` on floats is *not* bitwise
//     deterministic across runs (the order of contributing blocks
//     varies). For M2F parity we compare against the CPU reference
//     with `WithinAbs(..., 1e-6)`, which stays comfortably clean
//     under row-count dependent ordering up to the shapes we test.
//     If an M3-era training workload needs bitwise reproducibility
//     here we can swap to the two-stage all-reduce from
//     Reduction.cu, but the atomicAdd path has a big latency
//     advantage for the typical `[batch=64, C=10..1000]` shapes
//     that actually matter at training time.
//
//   * Backward: one CUDA block per row. The row is contiguous `[N,
//     C]`, each thread strides over C writing `(probs[c] -
//     (c==target ? 1 : 0)) * (g_scalar / N)`. No reduction required
//     — the CE backward is elementwise per row.
//
// Dtype scope: Float32 / Float64. Target indices are always Int64
// (matches the CPU `targets` dtype check).

#include "KernelUtils.cuh"

#include <math_constants.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/Loss.hpp"

namespace tesseract::cuda::detail {

namespace {

__device__ __forceinline__ float  ce_exp(float  x) { return ::__expf(x); }
__device__ __forceinline__ double ce_exp(double x) { return ::exp(x); }
__device__ __forceinline__ float  ce_log(float  x) { return ::__logf(x); }
__device__ __forceinline__ double ce_log(double x) { return ::log(x); }

template <typename T> __device__ __forceinline__ T neg_inf();
template <> __device__ __forceinline__ float  neg_inf<float>()  { return -CUDART_INF_F; }
template <> __device__ __forceinline__ double neg_inf<double>() { return -CUDART_INF; }

template <typename T>
__device__ __forceinline__ void block_reduce_max(T* sdata, int tid) {
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      T b = sdata[tid + s];
      if (b > sdata[tid]) sdata[tid] = b;
    }
    __syncthreads();
  }
}

template <typename T>
__device__ __forceinline__ void block_reduce_sum(T* sdata, int tid) {
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
    __syncthreads();
  }
}

// Forward kernel. One block per row, `blockDim.x = kBlockSize`. Row
// layout is `[N, C]` contiguous, so `logits + n * C` is the row base
// and stride between classes is 1. `probs` may be null (inference-
// only fast path — see the bridge header). `sum_loss` is a device-
// allocated single-slot accumulator into which each block atomicAdds
// its row's `(log_sum_exp - logits[target])` contribution.
template <typename T>
__global__ void ce_forward_kernel(const T* __restrict__ logits,
                                  const int64_t* __restrict__ targets,
                                  T* __restrict__ probs_or_null,
                                  T* __restrict__ sum_loss,
                                  int64_t N, int64_t C) {
  const int64_t n = blockIdx.x;
  if (n >= N) return;
  extern __shared__ __align__(alignof(T)) unsigned char raw[];
  T* sdata = reinterpret_cast<T*>(raw);

  const T* row = logits + n * C;
  const int64_t target = targets[n];
  const int tid = threadIdx.x;

  // Pass 1: max over C.
  {
    T m = neg_inf<T>();
    for (int64_t c = tid; c < C; c += blockDim.x) {
      const T v = row[c];
      if (v > m) m = v;
    }
    sdata[tid] = m;
    __syncthreads();
    block_reduce_max<T>(sdata, tid);
  }
  const T m = sdata[0];
  __syncthreads();

  // Pass 2: sum(exp(row - m)).
  {
    T s = T(0);
    for (int64_t c = tid; c < C; c += blockDim.x) {
      s = s + ce_exp(row[c] - m);
    }
    sdata[tid] = s;
    __syncthreads();
    block_reduce_sum<T>(sdata, tid);
  }
  const T sum_exp = sdata[0];
  __syncthreads();

  // Optional probs write-back. No reduction needed — each thread
  // handles its own stride through C.
  if (probs_or_null) {
    const T inv_s = T(1) / sum_exp;
    T* prow = probs_or_null + n * C;
    for (int64_t c = tid; c < C; c += blockDim.x) {
      prow[c] = ce_exp(row[c] - m) * inv_s;
    }
  }

  // Per-row loss: log_z - logits[target]. Only thread 0 computes
  // and contributes — cheap atomicAdd once per row.
  if (tid == 0) {
    const T log_z = m + ce_log(sum_exp);
    // Lookup logits[target] the same way the CPU reference does.
    const T row_loss = log_z - row[target];
    atomicAdd(sum_loss, row_loss);
  }
}

// Tiny finalize kernel: `loss_out[0] = sum_loss[0] / N`. Runs on a
// single thread; keeps the host off the critical path.
template <typename T>
__global__ void ce_finalize_mean(const T* __restrict__ sum_loss,
                                 int64_t N,
                                 T* __restrict__ loss_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    loss_out[0] = sum_loss[0] / static_cast<T>(N);
  }
}

// Backward kernel. One block per row. `dlogits[n, c] = (probs[n, c]
// - (c == target[n] ? 1 : 0)) * (g_scalar / N)`. We read the scalar
// grad via a single device pointer.
template <typename T>
__global__ void ce_backward_kernel(const T* __restrict__ probs,
                                   const int64_t* __restrict__ targets,
                                   const T* __restrict__ grad_scalar,
                                   T* __restrict__ dlogits,
                                   int64_t N, int64_t C) {
  const int64_t n = blockIdx.x;
  if (n >= N) return;
  const int64_t target = targets[n];
  const T gs    = grad_scalar[0];
  const T inv_n = T(1) / static_cast<T>(N);
  const T scale = gs * inv_n;
  const T* prow = probs + n * C;
  T* drow = dlogits + n * C;
  const int tid = threadIdx.x;
  for (int64_t c = tid; c < C; c += blockDim.x) {
    const T p = prow[c];
    const T adj = (c == target) ? (p - T(1)) : p;
    drow[c] = adj * scale;
  }
}

template <typename T>
void run_forward(const T* logits, const int64_t* targets,
                 T* probs_or_null, T* loss_out,
                 int64_t N, int64_t C, cudaStream_t stream) {
  // Use a stream-ordered scratch allocation for the sum accumulator.
  // We need to zero-init it before the atomicAdd-ing kernel starts.
  // `cudaMemsetAsync(..., 0, ...)` is cheap (2 bytes → 8 bytes), so
  // it's the right primitive rather than a custom kernel.
  T* sum_loss = nullptr;
  if (cudaMalloc(&sum_loss, sizeof(T)) != cudaSuccess) {
    throw DeviceError(fmt::format(
        "[tesseract] ce_forward: cudaMalloc({} bytes) for sum "
        "accumulator failed", sizeof(T)));
  }
  (void)cudaMemsetAsync(sum_loss, 0, sizeof(T), stream);

  const int blocks = static_cast<int>(std::min<int64_t>(N, 2'147'483'647LL));
  ce_forward_kernel<T><<<blocks, kBlockSize,
                         sizeof(T) * kBlockSize,
                         stream>>>(logits, targets, probs_or_null,
                                   sum_loss, N, C);
  check_launch("ce_forward");

  ce_finalize_mean<T><<<1, 1, 0, stream>>>(sum_loss, N, loss_out);
  check_launch("ce_finalize_mean");

  // Synchronous free — see the matching note in Reduction.cu's
  // `run_all_reduce`. The ~20 µs sync is amortized by the fact that
  // `cross_entropy` is a per-step top-level scalar output.
  (void)cudaFree(sum_loss);
}

template <typename T>
void run_backward(const T* probs, const int64_t* targets,
                  const T* grad_scalar, T* dlogits,
                  int64_t N, int64_t C, cudaStream_t stream) {
  const int blocks = static_cast<int>(std::min<int64_t>(N, 2'147'483'647LL));
  ce_backward_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
      probs, targets, grad_scalar, dlogits, N, C);
  check_launch("ce_backward");
}

}  // namespace

void launch_ce_forward(DType dtype, int device_index,
                       int64_t N, int64_t C,
                       const void* logits,
                       const int64_t* targets,
                       void* loss_out,
                       void* probs_out /*nullable*/,
                       void* stream_handle) {
  TESSERACT_CHECK(N > 0 && C > 0,
                  "[tesseract] launch_ce_forward: N={}, C={} must both "
                  "be > 0", N, C);
  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  switch (dtype) {
    case DType::Float32:
      run_forward<float>(static_cast<const float*>(logits), targets,
                         static_cast<float*>(probs_out),
                         static_cast<float*>(loss_out),
                         N, C, stream);
      break;
    case DType::Float64:
      run_forward<double>(static_cast<const double*>(logits), targets,
                          static_cast<double*>(probs_out),
                          static_cast<double*>(loss_out),
                          N, C, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA cross_entropy on dtype {} is not "
          "implemented at M2F (Float32 / Float64 only).",
          dtype_name(dtype)));
  }
}

void launch_ce_backward(DType dtype, int device_index,
                        int64_t N, int64_t C,
                        const void* probs,
                        const int64_t* targets,
                        const void* grad_scalar,
                        void* dlogits_out,
                        void* stream_handle) {
  TESSERACT_CHECK(N > 0 && C > 0,
                  "[tesseract] launch_ce_backward: N={}, C={} must "
                  "both be > 0", N, C);
  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  switch (dtype) {
    case DType::Float32:
      run_backward<float>(static_cast<const float*>(probs), targets,
                          static_cast<const float*>(grad_scalar),
                          static_cast<float*>(dlogits_out),
                          N, C, stream);
      break;
    case DType::Float64:
      run_backward<double>(static_cast<const double*>(probs), targets,
                           static_cast<const double*>(grad_scalar),
                           static_cast<double*>(dlogits_out),
                           N, C, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA cross_entropy backward on dtype {} is not "
          "implemented at M2F.", dtype_name(dtype)));
  }
}

}  // namespace tesseract::cuda::detail

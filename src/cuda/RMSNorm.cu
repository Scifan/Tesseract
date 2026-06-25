// Wave 2 (B-022): fused RMSNorm / LayerNorm CUDA forward kernels.
//
// The M2K composite path for these ops unrolls into 5-6 per-element
// passes over `x` (mul → mean → add → sqrt → div → mul for RMSNorm,
// +2 for LayerNorm's mean-center + biased-variance). Both ops are
// squarely memory-bound on all realistic transformer shapes
// (D ≤ 16384), so collapsing the chain into a single kernel pass is
// a straight ≥3× bandwidth win per layer and — because every Llama
// block touches normalization twice — a ≥5% end-to-end latency win
// on the full-stack `llama_infer`.
//
// Grid strategy (both kernels):
//   * One block per row. `outer` blocks total, each handling the
//     `[D]` normalization slot for its row. This matches the
//     reduction dialect's dim-reduce: the inner work is a block-
//     scope reduction, the outer work is pure data-parallel rows.
//   * Block width: 256 threads (kBlockSize). Each thread strides
//     over the row with a grid-stride loop, so arbitrarily large
//     `D` is handled without exceeding thread count.
//   * Two passes over `x` per row:
//       pass 1 — accumulate `sum(x^2)` (RMSNorm) or `sum(x)` +
//                `sum(x^2)` (LayerNorm); block-reduce in shared mem;
//                thread 0 computes the normalization factor and
//                broadcasts via shared mem.
//       pass 2 — write `out[j] = (x[j] * inv_rms) * w[j]` (RMSNorm) or
//                `out[j] = ((x[j]-mu) * inv_std) * w[j] + b[j]` (LN).
//
//     Two passes over `x` still beats the composite's ≥5 passes, and
//     avoids the shared-mem staging that would cap `D` at ~12K
//     floats per block.
//
// Dtype policy:
//   * Float32 storage → reduction + math in `float`.
//   * Float64 storage → reduction + math in `double`.
//   * Float16 / BFloat16 storage → FP32-promoted load + FP32 reduction
//     + FP32 math, narrow back on store. Mirrors the B-015 / B-016 /
//     RoPE pattern. The rsqrt itself is always `rsqrtf` (FP32) for
//     half/fp32 storage and `rsqrt` (FP64) for double.

#include "KernelUtils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/RMSNorm.hpp"

namespace tesseract::cuda::detail {

namespace {

// ---- FP32 promotion helpers (same naming as the RoPE kernel). ----

__device__ __forceinline__ float  rn_to_float(float  v) { return v; }
__device__ __forceinline__ float  rn_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float  rn_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T rn_from_float(float v);
template <> __device__ __forceinline__
float rn_from_float<float>(float v) { return v; }
template <> __device__ __forceinline__
__half rn_from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__
__nv_bfloat16 rn_from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// ---- Block-reduction helpers. Mirror `block_reduce` in Reduction.cu
// but specialized to Float32 / Float64 sums for simplicity. We keep
// the scratch at `kBlockSize` + a couple of slots for the broadcast
// of the normalization factor. ----

template <typename Acc>
__device__ __forceinline__ Acc block_sum(Acc val, Acc* sdata, int tid) {
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  return sdata[0];
}

// ---- RMSNorm kernel: FP32 accumulation (handles fp32/fp16/bf16 storage). ----

template <typename Tstorage>
__global__ void rms_norm_kernel_fp32(const Tstorage* __restrict__ x,
                                     const Tstorage* __restrict__ w,
                                     Tstorage* __restrict__ out,
                                     int64_t D, float eps) {
  __shared__ float sdata[kBlockSize];
  __shared__ float s_inv_rms;

  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const Tstorage* row_x = x + row * D;
  Tstorage*       row_o = out + row * D;

  // Pass 1: accumulate sum(x^2) across the row.
  float local_ss = 0.0f;
  for (int64_t j = tid; j < D; j += blockDim.x) {
    const float v = rn_to_float(row_x[j]);
    local_ss += v * v;
  }
  const float total_ss = block_sum(local_ss, sdata, tid);

  if (tid == 0) {
    // `mean(x^2)` then `rsqrt(mean + eps)`. FP32 rsqrt is ~2 ulp,
    // matches the composite's `div(x, sqrt(...))` path to within the
    // tolerances the RMSNorm parity test uses (1e-5 abs).
    const float mean = total_ss / static_cast<float>(D);
    s_inv_rms = rsqrtf(mean + eps);
  }
  __syncthreads();
  const float inv_rms = s_inv_rms;

  // Pass 2: write out = x * w * inv_rms. `w` is read once per thread
  // per element (block-resident in L1 for reasonable D).
  for (int64_t j = tid; j < D; j += blockDim.x) {
    const float xv = rn_to_float(row_x[j]);
    const float wv = rn_to_float(w[j]);
    row_o[j] = rn_from_float<Tstorage>((xv * inv_rms) * wv);
  }
}

// ---- RMSNorm kernel: FP64 variant (double storage). ----

__global__ void rms_norm_kernel_fp64(const double* __restrict__ x,
                                     const double* __restrict__ w,
                                     double* __restrict__ out,
                                     int64_t D, double eps) {
  __shared__ double sdata[kBlockSize];
  __shared__ double s_inv_rms;

  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const double* row_x = x + row * D;
  double*       row_o = out + row * D;

  double local_ss = 0.0;
  for (int64_t j = tid; j < D; j += blockDim.x) {
    const double v = row_x[j];
    local_ss += v * v;
  }
  const double total_ss = block_sum(local_ss, sdata, tid);

  if (tid == 0) {
    const double mean = total_ss / static_cast<double>(D);
    s_inv_rms = rsqrt(mean + eps);
  }
  __syncthreads();
  const double inv_rms = s_inv_rms;

  for (int64_t j = tid; j < D; j += blockDim.x) {
    row_o[j] = (row_x[j] * inv_rms) * w[j];
  }
}

// ---- LayerNorm kernel: FP32 accumulation (fp32/fp16/bf16 storage). ----
//
// Computes `mu` and `var = E[x^2] - mu^2` in one pass using the
// textbook two-statistic accumulator. This is numerically equivalent
// to the composite's `E[(x-mu)^2]` for the ranges transformers see
// (mu ≈ 0, |x| ≪ 100); catastrophic cancellation would only bite for
// very large `mu` with small variance, which is not a regime LN is
// ever used in. Trading a second pass for one-pass-with-two-stats is
// the standard speedup.

template <typename Tstorage>
__global__ void layer_norm_kernel_fp32(const Tstorage* __restrict__ x,
                                       const Tstorage* __restrict__ w,
                                       const Tstorage* __restrict__ b,
                                       Tstorage* __restrict__ out,
                                       int64_t D, float eps,
                                       bool has_bias) {
  // We need two block-reductions (sum and sum-of-squares). Use two
  // independent shared-mem scratches.
  __shared__ float s_sum[kBlockSize];
  __shared__ float s_sq[kBlockSize];
  __shared__ float s_mu;
  __shared__ float s_inv_std;

  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const Tstorage* row_x = x + row * D;
  Tstorage*       row_o = out + row * D;

  float local_sum = 0.0f, local_sq = 0.0f;
  for (int64_t j = tid; j < D; j += blockDim.x) {
    const float v = rn_to_float(row_x[j]);
    local_sum += v;
    local_sq  += v * v;
  }

  // Two serial block-reductions share the same tree shape but need
  // their own scratches so we don't clobber one while finishing the
  // other.
  const float total_sum = block_sum(local_sum, s_sum, tid);
  const float total_sq  = block_sum(local_sq,  s_sq,  tid);

  if (tid == 0) {
    const float mu  = total_sum / static_cast<float>(D);
    const float var = total_sq / static_cast<float>(D) - mu * mu;
    // Clamp negative numerical drift (E[x^2] - mu^2 can go very
    // slightly negative for nearly-constant rows); `var < 0` ⇒ var=0.
    s_mu       = mu;
    s_inv_std  = rsqrtf((var > 0.0f ? var : 0.0f) + eps);
  }
  __syncthreads();
  const float mu      = s_mu;
  const float inv_std = s_inv_std;

  for (int64_t j = tid; j < D; j += blockDim.x) {
    const float xv = rn_to_float(row_x[j]);
    const float wv = rn_to_float(w[j]);
    float y = ((xv - mu) * inv_std) * wv;
    if (has_bias) y += rn_to_float(b[j]);
    row_o[j] = rn_from_float<Tstorage>(y);
  }
}

// ---- LayerNorm kernel: FP64 variant. ----

__global__ void layer_norm_kernel_fp64(const double* __restrict__ x,
                                       const double* __restrict__ w,
                                       const double* __restrict__ b,
                                       double* __restrict__ out,
                                       int64_t D, double eps,
                                       bool has_bias) {
  __shared__ double s_sum[kBlockSize];
  __shared__ double s_sq[kBlockSize];
  __shared__ double s_mu;
  __shared__ double s_inv_std;

  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const double* row_x = x + row * D;
  double*       row_o = out + row * D;

  double local_sum = 0.0, local_sq = 0.0;
  for (int64_t j = tid; j < D; j += blockDim.x) {
    const double v = row_x[j];
    local_sum += v;
    local_sq  += v * v;
  }
  const double total_sum = block_sum(local_sum, s_sum, tid);
  const double total_sq  = block_sum(local_sq,  s_sq,  tid);

  if (tid == 0) {
    const double mu  = total_sum / static_cast<double>(D);
    const double var = total_sq / static_cast<double>(D) - mu * mu;
    s_mu      = mu;
    s_inv_std = rsqrt((var > 0.0 ? var : 0.0) + eps);
  }
  __syncthreads();
  const double mu      = s_mu;
  const double inv_std = s_inv_std;

  for (int64_t j = tid; j < D; j += blockDim.x) {
    double y = ((row_x[j] - mu) * inv_std) * w[j];
    if (has_bias) y += b[j];
    row_o[j] = y;
  }
}

}  // namespace

void launch_rms_norm(DType dtype, int device_index,
                     int64_t outer, int64_t D,
                     const void* x, const void* weight, double eps,
                     void* out, void* stream_handle) {
  TESSERACT_CHECK(D > 0, "[tesseract] rms_norm: D must be > 0, got {}", D);
  if (outer == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  // One block per row; block width is kBlockSize. `outer` is `int64_t`
  // but the `dim3` launch config is `unsigned int` (grid.x ≤ 2^31-1);
  // we cap aggressively to catch programmer error.
  TESSERACT_CHECK(outer <= 2'147'483'647LL,
                  "[tesseract] rms_norm: outer {} exceeds grid.x limit", outer);
  const int blocks = static_cast<int>(outer);
  const dim3 grid(blocks);
  const dim3 block(kBlockSize);
  const float eps_f = static_cast<float>(eps);

  switch (dtype) {
    case DType::Float32:
      rms_norm_kernel_fp32<float><<<grid, block, 0, stream>>>(
          static_cast<const float*>(x),
          static_cast<const float*>(weight),
          static_cast<float*>(out), D, eps_f);
      break;
    case DType::Float64:
      rms_norm_kernel_fp64<<<grid, block, 0, stream>>>(
          static_cast<const double*>(x),
          static_cast<const double*>(weight),
          static_cast<double*>(out), D, eps);
      break;
    case DType::Float16:
      rms_norm_kernel_fp32<__half><<<grid, block, 0, stream>>>(
          static_cast<const __half*>(x),
          static_cast<const __half*>(weight),
          static_cast<__half*>(out), D, eps_f);
      break;
    case DType::BFloat16:
      rms_norm_kernel_fp32<__nv_bfloat16><<<grid, block, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(x),
          static_cast<const __nv_bfloat16*>(weight),
          static_cast<__nv_bfloat16*>(out), D, eps_f);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA rms_norm on dtype {} is not "
          "implemented (Float32 / Float64 / Float16 / BFloat16 only).",
          dtype_name(dtype)));
  }
  check_launch("rms_norm_kernel");
}

void launch_layer_norm(DType dtype, int device_index,
                       int64_t outer, int64_t D,
                       const void* x, const void* weight, const void* bias,
                       double eps,
                       void* out, void* stream_handle) {
  TESSERACT_CHECK(D > 0, "[tesseract] layer_norm: D must be > 0, got {}", D);
  if (outer == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  TESSERACT_CHECK(outer <= 2'147'483'647LL,
                  "[tesseract] layer_norm: outer {} exceeds grid.x limit", outer);
  const int blocks = static_cast<int>(outer);
  const dim3 grid(blocks);
  const dim3 block(kBlockSize);
  const float eps_f = static_cast<float>(eps);
  const bool has_bias = (bias != nullptr);

  switch (dtype) {
    case DType::Float32:
      layer_norm_kernel_fp32<float><<<grid, block, 0, stream>>>(
          static_cast<const float*>(x),
          static_cast<const float*>(weight),
          static_cast<const float*>(bias),
          static_cast<float*>(out), D, eps_f, has_bias);
      break;
    case DType::Float64:
      layer_norm_kernel_fp64<<<grid, block, 0, stream>>>(
          static_cast<const double*>(x),
          static_cast<const double*>(weight),
          static_cast<const double*>(bias),
          static_cast<double*>(out), D, eps, has_bias);
      break;
    case DType::Float16:
      layer_norm_kernel_fp32<__half><<<grid, block, 0, stream>>>(
          static_cast<const __half*>(x),
          static_cast<const __half*>(weight),
          static_cast<const __half*>(bias),
          static_cast<__half*>(out), D, eps_f, has_bias);
      break;
    case DType::BFloat16:
      layer_norm_kernel_fp32<__nv_bfloat16><<<grid, block, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(x),
          static_cast<const __nv_bfloat16*>(weight),
          static_cast<const __nv_bfloat16*>(bias),
          static_cast<__nv_bfloat16*>(out), D, eps_f, has_bias);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA layer_norm on dtype {} is not "
          "implemented (Float32 / Float64 / Float16 / BFloat16 only).",
          dtype_name(dtype)));
  }
  check_launch("layer_norm_kernel");
}

}  // namespace tesseract::cuda::detail

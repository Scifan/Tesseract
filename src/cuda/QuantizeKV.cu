// Wave 9 (B-031): KV-cache INT8 quantization kernels.
//
// Two hot-path kernels backing `nn::QuantizedKVCache`:
//   * quantize   — one thread per row (token·head). Each thread scans
//                  `D_head` for the FP32 absmax, derives scale = absmax/127
//                  (1.0 for an all-zero row), then writes the INT8 row.
//                  D_head is small (64/128) so a per-row thread is fine;
//                  the grid-strides over rows for prefill (many rows).
//   * dequantize — one thread per element, `out[i] = q[i]·scale[i/D_head]`.
//
// Dtype policy mirrors RMSNorm / RoPE: FP32 native, FP16/BF16 via FP32-
// promoted load + FP32 math + narrow-on-store. The scale is always FP32
// so the only quantization error is the INT8 payload, not the scale.

#include "KernelUtils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/QuantizeKV.hpp"

namespace tesseract::cuda::detail {

namespace {

__device__ __forceinline__ float qkv_to_float(float v) { return v; }
__device__ __forceinline__ float qkv_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float qkv_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T qkv_from_float(float v);
template <> __device__ __forceinline__ float qkv_from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ __half qkv_from_float<__half>(float v) {
  return __float2half(v);
}
template <> __device__ __forceinline__ __nv_bfloat16
qkv_from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ int8_t round_clip_i8(float v) {
  const float r = rintf(v);  // round-to-nearest-even, matches std::nearbyint
  if (r >= 127.0f) return static_cast<int8_t>(127);
  if (r <= -127.0f) return static_cast<int8_t>(-127);
  return static_cast<int8_t>(r);
}

template <typename T>
__global__ void quantize_kv_kernel(const T* __restrict__ x,
                                   int8_t* __restrict__ q,
                                   float* __restrict__ scale,
                                   int64_t rows, int64_t dh) {
  const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t row = tid; row < rows; row += stride) {
    const int64_t base = row * dh;
    float max_abs = 0.0f;
    for (int64_t d = 0; d < dh; ++d) {
      const float a = fabsf(qkv_to_float(x[base + d]));
      if (a > max_abs) max_abs = a;
    }
    const float s = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    scale[row] = s;
    const float inv = 1.0f / s;
    for (int64_t d = 0; d < dh; ++d) {
      q[base + d] = round_clip_i8(qkv_to_float(x[base + d]) * inv);
    }
  }
}

template <typename T>
__global__ void dequantize_kv_kernel(const int8_t* __restrict__ q,
                                     const float* __restrict__ scale,
                                     T* __restrict__ out,
                                     int64_t total, int64_t dh) {
  const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < total; i += stride) {
    const float v = static_cast<float>(q[i]) * scale[i / dh];
    out[i] = qkv_from_float<T>(v);
  }
}

}  // namespace

void launch_quantize_kv_per_token(DType dtype, int device_index,
                                   int64_t rows, int64_t head_dim,
                                   const void* x, int8_t* q, float* scale,
                                   void* stream_handle) {
  if (rows == 0 || head_dim == 0) return;
  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  const dim3 grid(launch_grid(rows));
  const dim3 block(kBlockSize);
  switch (dtype) {
    case DType::Float32:
      quantize_kv_kernel<float><<<grid, block, 0, stream>>>(
          static_cast<const float*>(x), q, scale, rows, head_dim);
      break;
    case DType::Float16:
      quantize_kv_kernel<__half><<<grid, block, 0, stream>>>(
          static_cast<const __half*>(x), q, scale, rows, head_dim);
      break;
    case DType::BFloat16:
      quantize_kv_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(x), q, scale, rows, head_dim);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] quantize_kv on dtype {} is not supported "
          "(Float32/Float16/BFloat16 only).", dtype_name(dtype)));
  }
  check_launch("quantize_kv_kernel");
}

void launch_dequantize_kv_per_token(DType dtype, int device_index,
                                     int64_t rows, int64_t head_dim,
                                     const int8_t* q, const float* scale,
                                     void* out, void* stream_handle) {
  const int64_t total = rows * head_dim;
  if (total == 0) return;
  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  const dim3 grid(launch_grid(total));
  const dim3 block(kBlockSize);
  switch (dtype) {
    case DType::Float32:
      dequantize_kv_kernel<float><<<grid, block, 0, stream>>>(
          q, scale, static_cast<float*>(out), total, head_dim);
      break;
    case DType::Float16:
      dequantize_kv_kernel<__half><<<grid, block, 0, stream>>>(
          q, scale, static_cast<__half*>(out), total, head_dim);
      break;
    case DType::BFloat16:
      dequantize_kv_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
          q, scale, static_cast<__nv_bfloat16*>(out), total, head_dim);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] dequantize_kv on dtype {} is not supported "
          "(Float32/Float16/BFloat16 only).", dtype_name(dtype)));
  }
  check_launch("dequantize_kv_kernel");
}

}  // namespace tesseract::cuda::detail

// Wave 4.1 (B-025): fused SwiGLU forward kernel.
//
// `nn::FeedForward` (Llama-style SwiGLU FFN) is structured as
//   gate      = gate_proj(x)           // [..., d_ff]
//   up        = up_proj(x)             // [..., d_ff]
//   silu_gate = gate * sigmoid(gate)   // element-wise
//   hidden    = silu_gate * up         // element-wise
//   y         = down_proj(hidden)
//
// The two inner element-wise steps decompose into four primitives
// (`sigmoid`, `mul`, `mul`) + `sigmoid`'s own exp-based forward, i.e.
// ≥3 passes of HBM traffic on the `[..., d_ff]` intermediate. The
// element-wise portion is 100% memory-bound on every Llama FFN we
// care about (d_ff ∈ [2048, 28672]) — there is no compute reuse
// whatsoever, it's one read and one write per element.
//
// This kernel fuses the two `mul`s and the `sigmoid` into a single
// 1-pass element-wise kernel:
//   out[i] = gate[i] * sigmoid(gate[i]) * up[i]
// reducing the element-wise portion from 3× `(read, write)` to
// 1× `(read gate + read up, write out)`, i.e. 2 reads + 1 write vs.
// the composite's 3 reads + 3 writes. On a memory-bound kernel the
// expected speedup is ≈(3R+3W)/(2R+1W) ≈ 2.0×, matching the B-022
// fused-RMSNorm win ratio.
//
// Grid strategy: plain 1-D grid-stride loop. No reductions, no
// cross-lane communication, so a single block of `kBlockSize`
// threads stamps out as many elements per iteration as it has
// threads and then jumps forward by `gridDim.x * blockDim.x`.
//
// Dtype policy (matches B-015 / B-016 / RMSNorm):
//   * Float32 / Float64 storage → compute in the storage type.
//   * Float16 / BFloat16 storage → load through FP32 accumulator,
//     compute `silu` + `mul` in FP32, narrow back on store. The
//     `expf` used by `sigmoid` always runs in FP32 regardless of
//     storage width so the dynamic range of half precision never
//     limits the saturation tails.

#include "KernelUtils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/SwiGLU.hpp"

namespace tesseract::cuda::detail {

namespace {

// ---- FP32 promotion helpers (same convention as RMSNorm.cu). ----

__device__ __forceinline__ float sg_to_float(float v)           { return v; }
__device__ __forceinline__ float sg_to_float(__half v)          { return __half2float(v); }
__device__ __forceinline__ float sg_to_float(__nv_bfloat16 v)   { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T sg_from_float(float v);
template <> __device__ __forceinline__
float sg_from_float<float>(float v) { return v; }
template <> __device__ __forceinline__
__half sg_from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__
__nv_bfloat16 sg_from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// silu(g) = g * sigmoid(g) = g / (1 + exp(-g)). We compute the
// sigmoid in FP32 regardless of storage width so saturation tails
// (|g| ~ 10+) behave identically across dtypes. Numerically
// equivalent to `g * (1 / (1 + expf(-g)))`; we multiply through so
// the IEEE round-to-nearest fold happens once.
__device__ __forceinline__ float silu_fp32(float g) {
  const float sig = 1.0f / (1.0f + __expf(-g));
  return g * sig;
}

// ---- Float32 / half-precision storage (FP32-accumulated). ----

template <typename Tstorage>
__global__ void swiglu_silu_gate_kernel_fp32(const Tstorage* __restrict__ gate,
                                             const Tstorage* __restrict__ up,
                                             Tstorage* __restrict__ out,
                                             int64_t numel) {
  const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < numel; i += stride) {
    const float g = sg_to_float(gate[i]);
    const float u = sg_to_float(up[i]);
    out[i] = sg_from_float<Tstorage>(silu_fp32(g) * u);
  }
}

// ---- FP64 variant (double storage). ----

__global__ void swiglu_silu_gate_kernel_fp64(const double* __restrict__ gate,
                                             const double* __restrict__ up,
                                             double* __restrict__ out,
                                             int64_t numel) {
  const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < numel; i += stride) {
    const double g = gate[i];
    const double u = up[i];
    // FP64 `exp` for double storage; matches `std::exp(double)` on
    // the CPU reference.
    const double sig = 1.0 / (1.0 + exp(-g));
    out[i] = (g * sig) * u;
  }
}

}  // namespace

void launch_swiglu_silu_gate(DType dtype, int device_index,
                             int64_t numel,
                             const void* gate,
                             const void* up,
                             void* out,
                             void* stream_handle) {
  if (numel == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  // 1-D grid-stride loop. `launch_grid` clamps to the grid.x cap so
  // tensors up to ~0.5T elements still schedule correctly with
  // kBlockSize=256 threads/block.
  const dim3 grid(launch_grid(numel));
  const dim3 block(kBlockSize);

  switch (dtype) {
    case DType::Float32:
      swiglu_silu_gate_kernel_fp32<float><<<grid, block, 0, stream>>>(
          static_cast<const float*>(gate),
          static_cast<const float*>(up),
          static_cast<float*>(out), numel);
      break;
    case DType::Float64:
      swiglu_silu_gate_kernel_fp64<<<grid, block, 0, stream>>>(
          static_cast<const double*>(gate),
          static_cast<const double*>(up),
          static_cast<double*>(out), numel);
      break;
    case DType::Float16:
      swiglu_silu_gate_kernel_fp32<__half><<<grid, block, 0, stream>>>(
          static_cast<const __half*>(gate),
          static_cast<const __half*>(up),
          static_cast<__half*>(out), numel);
      break;
    case DType::BFloat16:
      swiglu_silu_gate_kernel_fp32<__nv_bfloat16><<<grid, block, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(gate),
          static_cast<const __nv_bfloat16*>(up),
          static_cast<__nv_bfloat16*>(out), numel);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA swiglu_silu_gate on dtype {} is not "
          "implemented (Float32 / Float64 / Float16 / BFloat16 only).",
          dtype_name(dtype)));
  }
  check_launch("swiglu_silu_gate_kernel");
}

}  // namespace tesseract::cuda::detail

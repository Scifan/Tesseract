// B-014: CUDA rotary position embedding (RoPE) kernel.
//
// Per-output-element cost: 2 fused multiply-adds, one paired read
// from `x`, one paired read from `cos` / `sin`, one paired write to
// `out`. Total arithmetic intensity is 4 FLOPS per pair (i.e. per
// 2 storage elements), which is firmly memory-bound — the kernel is
// intentionally simple because the critical path here is bandwidth,
// not compute.
//
// Grid strategy:
//   * One thread per output *pair* (2 storage elements). Blocks are
//     `kBlockSize` (256) threads wide; the total grid covers
//     `outer * S * (D/2)` pairs with a grid-stride tail.
//
//   * Input is laid out as `[outer, S, D]` row-major contiguous.
//     Leading batch dims were already flattened into `outer` at the
//     op-layer boundary (`src/ops/cpu/RotaryEmbedding.cpp`). The
//     kernel just peels `(o, p, j)` coords back out of a flat
//     `(outer * S * D/2)` index.
//
//   * `cos` / `sin` are `[S, D]` contiguous. Broadcasting over the
//     leading `outer` dim is implicit in the indexing — same
//     `(p, 2j)` offset used for every `o` slot. We read the
//     odd-column entry from the table at offset `2j+1`; the
//     `nn::RotaryEmbedding` module guarantees `cos[p, 2j] == cos[p,
//     2j+1]` and likewise for sin, but we don't rely on that here —
//     loading both explicitly lets us reuse the kernel against a
//     future "half-width table" (`[S, D/2]`) via a strided view on
//     the caller side.
//
// Dtype policy (same as B-015 / B-016):
//   * Float32 / Float64 — native compute in the storage type.
//   * Float16 / BFloat16 — load → `float`, compute in `float`,
//     narrow back on store. `cos` / `sin` are loaded as the storage
//     type too (so they live in half precision alongside `x` when
//     the caller opts in), and promoted the same way. The extra
//     device cycles for the cast are negligible vs the HBM read.

#include "KernelUtils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/RotaryEmbedding.hpp"

namespace tesseract::cuda::detail {

namespace {

// ---- FP32 promotion helpers (mirroring the B-015 softmax / B-016
// reduction naming convention — `re_to_float` / `re_from_float`). ----

__device__ __forceinline__ float  re_to_float(float  v) { return v; }
__device__ __forceinline__ double re_to_double(double v) { return v; }
__device__ __forceinline__ float  re_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float  re_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T re_from_float(float v);
template <> __device__ __forceinline__
float re_from_float<float>(float v) { return v; }
template <> __device__ __forceinline__
__half re_from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__
__nv_bfloat16 re_from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// ---- FP32 kernel (also handles FP16/BF16 via promotion) ----

template <typename Tstorage>
__global__ void rope_kernel_fp32(const Tstorage* __restrict__ x,
                                 const Tstorage* __restrict__ cs,
                                 const Tstorage* __restrict__ sn,
                                 int64_t outer, int64_t S, int64_t D,
                                 Tstorage* __restrict__ out) {
  const int64_t half_D = D / 2;
  const int64_t total_pairs = outer * S * half_D;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_pairs;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // Decompose flat pair index → (o, p, j) where j ∈ [0, D/2).
    const int64_t j = idx % half_D;
    const int64_t rem_op = idx / half_D;
    const int64_t p = rem_op % S;
    const int64_t o = rem_op / S;

    const int64_t x_base = (o * S + p) * D;
    const int64_t t_base = p * D;

    const float a = re_to_float(x[x_base + 2 * j]);
    const float b = re_to_float(x[x_base + 2 * j + 1]);
    const float c0 = re_to_float(cs[t_base + 2 * j]);
    const float s0 = re_to_float(sn[t_base + 2 * j]);
    const float c1 = re_to_float(cs[t_base + 2 * j + 1]);
    const float s1 = re_to_float(sn[t_base + 2 * j + 1]);

    const float o0 = a * c0 - b * s0;
    const float o1 = a * s1 + b * c1;

    out[x_base + 2 * j]     = re_from_float<Tstorage>(o0);
    out[x_base + 2 * j + 1] = re_from_float<Tstorage>(o1);
  }
}

// FP64 variant — same layout, but accumulates in double so we don't
// throw away mantissa bits against a double cos/sin table.
__global__ void rope_kernel_fp64(const double* __restrict__ x,
                                 const double* __restrict__ cs,
                                 const double* __restrict__ sn,
                                 int64_t outer, int64_t S, int64_t D,
                                 double* __restrict__ out) {
  const int64_t half_D = D / 2;
  const int64_t total_pairs = outer * S * half_D;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_pairs;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t j = idx % half_D;
    const int64_t rem_op = idx / half_D;
    const int64_t p = rem_op % S;
    const int64_t o = rem_op / S;

    const int64_t x_base = (o * S + p) * D;
    const int64_t t_base = p * D;

    const double a = x[x_base + 2 * j];
    const double b = x[x_base + 2 * j + 1];
    const double c0 = cs[t_base + 2 * j];
    const double s0 = sn[t_base + 2 * j];
    const double c1 = cs[t_base + 2 * j + 1];
    const double s1 = sn[t_base + 2 * j + 1];

    out[x_base + 2 * j]     = a * c0 - b * s0;
    out[x_base + 2 * j + 1] = a * s1 + b * c1;
  }
}

// Launch helper — grid size is capped so the stage doesn't emit
// pathological numbers of blocks for tiny shapes. 1024 blocks is
// enough to saturate Ada (144 SMs × ~7 concurrent blocks/SM).
constexpr int kRopeMaxBlocks = 1024;
int pick_rope_blocks(int64_t total_pairs) {
  const int64_t blocks = (total_pairs + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(std::min<int64_t>(blocks, kRopeMaxBlocks));
}

}  // namespace

void launch_rotary_embedding(DType dtype, int device_index,
                             int64_t outer, int64_t seq, int64_t dim,
                             const void* x, const void* cos, const void* sin,
                             void* out, void* stream_handle) {
  TESSERACT_CHECK(dim % 2 == 0,
                  "[tesseract] rotary_embedding: dim must be even, got {}", dim);
  if (outer == 0 || seq == 0 || dim == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  const int64_t total_pairs = outer * seq * (dim / 2);
  const int blocks = pick_rope_blocks(total_pairs);

  switch (dtype) {
    case DType::Float32:
      rope_kernel_fp32<float><<<blocks, kBlockSize, 0, stream>>>(
          static_cast<const float*>(x),
          static_cast<const float*>(cos),
          static_cast<const float*>(sin),
          outer, seq, dim,
          static_cast<float*>(out));
      break;
    case DType::Float64:
      rope_kernel_fp64<<<blocks, kBlockSize, 0, stream>>>(
          static_cast<const double*>(x),
          static_cast<const double*>(cos),
          static_cast<const double*>(sin),
          outer, seq, dim,
          static_cast<double*>(out));
      break;
    case DType::Float16:
      rope_kernel_fp32<__half><<<blocks, kBlockSize, 0, stream>>>(
          static_cast<const __half*>(x),
          static_cast<const __half*>(cos),
          static_cast<const __half*>(sin),
          outer, seq, dim,
          static_cast<__half*>(out));
      break;
    case DType::BFloat16:
      rope_kernel_fp32<__nv_bfloat16><<<blocks, kBlockSize, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(x),
          static_cast<const __nv_bfloat16*>(cos),
          static_cast<const __nv_bfloat16*>(sin),
          outer, seq, dim,
          static_cast<__nv_bfloat16*>(out));
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA rotary_embedding on dtype {} is not "
          "implemented (Float32 / Float64 / Float16 / BFloat16 only).",
          dtype_name(dtype)));
  }
  check_launch("rope_kernel");
}

}  // namespace tesseract::cuda::detail

// Wave 3.1 (B-021): INT8 symmetric weight-only dequantize-matmul CUDA
// kernel. Fuses the "dequantize integer weights + GEMM" pair into a
// single pass over `q_w`, so the dominant memory read is 1 byte per
// weight element (versus 2-4 bytes for FP16/FP32 weights + a staging
// buffer in the naive decompose-then-GEMM path).
//
// Kernel layout:
//   * One block per output pair `(m, n)`, with `grid = (N, M, 1)` and
//     `block = (kBlockSize, 1, 1)`. Each block block-reduces a K-long
//     dot product and writes a single `y[m, n]`.
//   * The block-reduction mirrors `RMSNorm.cu` / `Reduction.cu`'s
//     `block_sum` pattern (shared-mem tree reduction), kept local to
//     this TU so nvcc doesn't have to re-specialize the primitive
//     across translation units.
//
// Dtype policy (matches B-022 / B-015 / RoPE):
//   * FP32 storage: reduction + math in `float`.
//   * FP16 / BF16 storage: FP32-promoted load of `x`, FP32 math, FP32
//     scale multiply, narrow back on store.
//   * INT8 weight byte is always loaded as a signed byte and cast to
//     `float` in the inner loop. The `(int8 → float)` cast compiles
//     to a single `sreg` conversion on Ampere+; the cost is the
//     byte load, which is the whole point of the kernel.
//
// Performance characteristics (Wave 3.1 MVP):
//   * Memory-bound on realistic shapes — each block reads `K` bytes
//     from `q_w`, `K * sizeof(Tx)` bytes from `x`, 4 bytes from
//     `scale`, and writes `sizeof(Tx)` bytes. For INT8 weights with
//     FP16 activations, the weight stream is 2× cheaper than the
//     activation stream; for large `N`, this is roughly a 2-4×
//     bandwidth improvement over the FP16 cuBLASLt path on the
//     decode-phase shapes we care about (`M = 1`).
//   * Big-`M` shapes (training / long-prefill) are suboptimal:
//     `N * M` blocks is a lot. A tiled GEMM rewrite lives in the
//     W3.1b backlog and can share this kernel's dtype plumbing
//     once a benchmark surfaces the gap. The MVP prioritizes the
//     decode path where this kernel directly unlocks the quantized
//     weight storage win.
//
// Numerical accuracy:
//   * The FP32 accumulator is wide enough for any realistic `K`
//     up to ~2^23 terms before catastrophic cancellation is a real
//     concern. Transformer `in_features` top out around 2^15, so we
//     have 8 bits of headroom.
//   * Every multiply is `float * float` (the INT8 weight is promoted
//     to `float` first). That costs one extra FMA cycle per element
//     relative to a hypothetical `int8 * fp16` tensor-core path, but
//     keeps the kernel readable and matches what ATen's reference
//     INT8 matmul does on non-tensor-core paths.

#include "KernelUtils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/DequantMatMul.hpp"

namespace tesseract::cuda::detail {

namespace {

// ---- FP32 promotion helpers (mirror RMSNorm.cu) ----

__device__ __forceinline__ float  dq_to_float(float  v) { return v; }
__device__ __forceinline__ float  dq_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float  dq_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T dq_from_float(float v);
template <> __device__ __forceinline__
float dq_from_float<float>(float v) { return v; }
template <> __device__ __forceinline__
__half dq_from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__
__nv_bfloat16 dq_from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// Block-scope sum reduction — same tree pattern as RMSNorm.cu's
// `block_sum`. Specialized to `float` because the INT8 path always
// accumulates in FP32 regardless of storage dtype.
__device__ __forceinline__ float dq_block_sum(float val, float* sdata, int tid) {
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  return sdata[0];
}

template <typename Tx>
__global__ void dequant_matmul_int8_kernel(const Tx* __restrict__ x,
                                           const int8_t* __restrict__ q_w,
                                           const float* __restrict__ scale,
                                           Tx* __restrict__ y,
                                           int64_t M, int64_t N, int64_t K) {
  __shared__ float sdata[kBlockSize];

  const int tid     = threadIdx.x;
  const int64_t n   = blockIdx.x;
  const int64_t m   = blockIdx.y;
  if (m >= M || n >= N) return;

  const Tx*     row_x = x   + m * K;
  const int8_t* row_w = q_w + n * K;

  // Inner dot product over K. FP32 accumulator; INT8 weights promoted
  // to float on load. Grid-stride loop so arbitrarily large K works
  // with a fixed block width.
  float acc = 0.0f;
  for (int64_t k = tid; k < K; k += blockDim.x) {
    const float xv = dq_to_float(row_x[k]);
    const float wv = static_cast<float>(row_w[k]);
    acc += xv * wv;
  }

  const float total = dq_block_sum(acc, sdata, tid);

  if (tid == 0) {
    // `scale` is FP32 per-output-channel; broadcast once per block.
    const float s = scale[n];
    y[m * N + n]  = dq_from_float<Tx>(total * s);
  }
}

}  // namespace

void launch_dequant_matmul_int8(DType dtype, int device_index,
                                int64_t M, int64_t N, int64_t K,
                                const void* x, const void* q_w,
                                const void* scale, void* y,
                                void* stream_handle) {
  TESSERACT_CHECK(M > 0 && N > 0 && K > 0,
                  "[tesseract] dequant_matmul_int8: non-positive dims "
                  "(M={}, N={}, K={})", M, N, K);
  TESSERACT_CHECK(x && q_w && scale && y,
                  "[tesseract] dequant_matmul_int8: null operand");

  // Grid dims are unsigned 32-bit; cap aggressively. At M * N = 2^31
  // blocks we're already way off any realistic transformer shape
  // (would be ~50k x 50k output, ~10 GiB of activations).
  TESSERACT_CHECK(M <= 2'147'483'647LL && N <= 2'147'483'647LL,
                  "[tesseract] dequant_matmul_int8: M or N exceeds grid "
                  "dim limit (M={}, N={})", M, N);

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  const dim3 grid(static_cast<unsigned int>(N),
                  static_cast<unsigned int>(M), 1);
  const dim3 block(kBlockSize);

  switch (dtype) {
    case DType::Float32:
      dequant_matmul_int8_kernel<float><<<grid, block, 0, stream>>>(
          static_cast<const float*>(x),
          static_cast<const int8_t*>(q_w),
          static_cast<const float*>(scale),
          static_cast<float*>(y), M, N, K);
      break;
    case DType::Float16:
      dequant_matmul_int8_kernel<__half><<<grid, block, 0, stream>>>(
          static_cast<const __half*>(x),
          static_cast<const int8_t*>(q_w),
          static_cast<const float*>(scale),
          static_cast<__half*>(y), M, N, K);
      break;
    case DType::BFloat16:
      dequant_matmul_int8_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(x),
          static_cast<const int8_t*>(q_w),
          static_cast<const float*>(scale),
          static_cast<__nv_bfloat16*>(y), M, N, K);
      break;
    default:
      TESSERACT_THROW("[tesseract] dequant_matmul_int8: unsupported dtype {} "
                      "(expected Float32/Float16/BFloat16)",
                      dtype_name(dtype));
  }
  check_launch("dequant_matmul_int8");
}

// ---------------------------------------------------------------------------
// Wave 3.2: INT4 per-group symmetric
// ---------------------------------------------------------------------------
//
// Same block layout as INT8: one block per `(m, n)`, FP32 accumulator,
// block-reduction over `K`. The inner loop unpacks two signed 4-bit
// nibbles per byte (low nibble = even-k, high nibble = odd-k, each
// sign-extended from its low four bits) and applies the per-group
// scale inside the accumulation so groups with a larger scale are
// weighted correctly.
//
// Why not factor `scale` out across each group? Every `k` carries its
// own group scale lookup; factoring would require the K-loop to run
// in strict `k`-monotonic order inside each thread, and our thread
// stride is `blockDim.x`. Applying the scale element-wise keeps the
// parallel layout unchanged and costs one FMA per element, which is
// dwarfed by the byte/nibble load cost.
namespace {

// Sign-extend the low four bits of `nib` to a signed 32-bit value in
// `[-7, 7]`. Branchless: the top bit of the nibble (0x8) is
// replicated into bits 4..31 of the result via a shift pair.
__device__ __forceinline__ int dq_signext4(uint32_t nib) {
  // nib is in [0..15]. (nib ^ 0x8) - 0x8 gives the signed value
  // (-8..7) via the standard unsigned-to-signed trick. Our packer
  // never emits -8 so the reachable range is [-7..7].
  return static_cast<int>((nib ^ 0x8u)) - 8;
}

template <typename Tx>
__global__ void dequant_matmul_int4_group_kernel(
    const Tx* __restrict__ x,
    const int8_t* __restrict__ q_packed,   // [N, K/2]
    const float* __restrict__ scale,       // [N, K/group_size]
    Tx* __restrict__ y,
    int64_t M, int64_t N, int64_t K, int64_t group_size) {
  __shared__ float sdata[kBlockSize];

  const int tid     = threadIdx.x;
  const int64_t n   = blockIdx.x;
  const int64_t m   = blockIdx.y;
  if (m >= M || n >= N) return;

  const int64_t packed_cols   = K / 2;
  const int64_t groups_per_row = K / group_size;

  const Tx*      row_x = x         + m * K;
  const int8_t*  row_w = q_packed  + n * packed_cols;
  const float*   row_s = scale     + n * groups_per_row;

  // Inner dot product over K. Each thread handles a strided subset of
  // k indices; the nibble selector `(k & 1)` decides high vs low
  // nibble inside the shared byte at `row_w[k / 2]`. The byte load
  // happens once per k because we don't currently cooperate between
  // even/odd threads in the warp (readable > optimal; W3.2b can add
  // a warp-level pairing if a benchmark asks for it).
  float acc = 0.0f;
  for (int64_t k = tid; k < K; k += blockDim.x) {
    const int8_t byte = row_w[k >> 1];
    const uint32_t nib = static_cast<uint32_t>(
        static_cast<uint8_t>(byte) >> ((k & 1) ? 4 : 0)) & 0xFu;
    const int q4 = dq_signext4(nib);

    const float xv = dq_to_float(row_x[k]);
    const float s  = row_s[k / group_size];
    acc += xv * static_cast<float>(q4) * s;
  }

  const float total = dq_block_sum(acc, sdata, tid);
  if (tid == 0) {
    y[m * N + n] = dq_from_float<Tx>(total);
  }
}

}  // namespace

void launch_dequant_matmul_int4_group(DType dtype, int device_index,
                                      int64_t M, int64_t N, int64_t K,
                                      int64_t group_size,
                                      const void* x,
                                      const void* q_packed,
                                      const void* scale,
                                      void* y,
                                      void* stream_handle) {
  TESSERACT_CHECK(M > 0 && N > 0 && K > 0,
                  "[tesseract] dequant_matmul_int4_group: non-positive dims "
                  "(M={}, N={}, K={})", M, N, K);
  TESSERACT_CHECK(x && q_packed && scale && y,
                  "[tesseract] dequant_matmul_int4_group: null operand");
  TESSERACT_CHECK(group_size >= 2 && (group_size % 2) == 0,
                  "[tesseract] dequant_matmul_int4_group: group_size must be "
                  ">=2 and even, got {}", group_size);
  TESSERACT_CHECK(K % group_size == 0,
                  "[tesseract] dequant_matmul_int4_group: K ({}) must be "
                  "a multiple of group_size ({})", K, group_size);
  TESSERACT_CHECK(M <= 2'147'483'647LL && N <= 2'147'483'647LL,
                  "[tesseract] dequant_matmul_int4_group: M or N exceeds grid "
                  "dim limit (M={}, N={})", M, N);

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  const dim3 grid(static_cast<unsigned int>(N),
                  static_cast<unsigned int>(M), 1);
  const dim3 block(kBlockSize);

  switch (dtype) {
    case DType::Float32:
      dequant_matmul_int4_group_kernel<float><<<grid, block, 0, stream>>>(
          static_cast<const float*>(x),
          static_cast<const int8_t*>(q_packed),
          static_cast<const float*>(scale),
          static_cast<float*>(y), M, N, K, group_size);
      break;
    case DType::Float16:
      dequant_matmul_int4_group_kernel<__half><<<grid, block, 0, stream>>>(
          static_cast<const __half*>(x),
          static_cast<const int8_t*>(q_packed),
          static_cast<const float*>(scale),
          static_cast<__half*>(y), M, N, K, group_size);
      break;
    case DType::BFloat16:
      dequant_matmul_int4_group_kernel<__nv_bfloat16>
          <<<grid, block, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(x),
          static_cast<const int8_t*>(q_packed),
          static_cast<const float*>(scale),
          static_cast<__nv_bfloat16*>(y), M, N, K, group_size);
      break;
    default:
      TESSERACT_THROW("[tesseract] dequant_matmul_int4_group: unsupported dtype "
                      "{} (expected Float32/Float16/BFloat16)",
                      dtype_name(dtype));
  }
  check_launch("dequant_matmul_int4_group");
}

}  // namespace tesseract::cuda::detail

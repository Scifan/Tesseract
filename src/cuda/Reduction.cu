// M2F: CUDA reduction kernels (sum / mean / max, both all-reduce and
// along a single dim). Only compiled when TESSERACT_ENABLE_CUDA=ON;
// the CPU-only build gets throwing stubs from `ReductionStub.cpp`.
//
// Design notes — see `docs/m2-plan.md` §M2F for the full rationale:
//
//   * Determinism matches the CPU baseline. We use per-block shared-
//     memory reductions feeding a second-pass reduction rather than
//     `atomicAdd` on a global accumulator, so results are bitwise
//     identical across runs given the same block size (atomicAdd's
//     non-deterministic ordering would otherwise fail a
//     `WithinAbs(..., 1e-6)` parity test on float32).
//
//   * All-reduce is two-stage: one kernel per-block reduces a chunk
//     of `x` into a per-block partial, then a second kernel reduces
//     those partials into the single output scalar. The intermediate
//     lives in a small `cudaMalloc`-allocated workspace that we free
//     before return (it's O(num_blocks) * sizeof(T), typically a few
//     KiB even for very large tensors).
//
//   * Dim-reduce fuses the (outer, inner) iteration into one kernel.
//     Each block handles a single output slot — one (outer, inner)
//     pair — and threads within the block stride over the `D` reduced
//     elements, followed by an intra-block tree reduction in shared
//     memory. This scales well when the reduced dim is long; when
//     `D` is tiny (say `D < 32`) we still pay the block-launch
//     overhead but the overall kernel is still correct and fast
//     enough for the shapes M2F cares about (per-sample loss,
//     feature-wise mean).
//
//   * Strided input. The launcher handles arbitrary `in_strides`
//     (contiguous, broadcast, transposed) by computing the
//     collapsed-non-dim base offset via `flat_to_offset` on the
//     iter-shape (input shape with `dim` removed), then stepping
//     along `in_strides[dim]` inside the block loop. Output is
//     always written as row-major contiguous — `keepdim=true` is
//     reconstructed by a cheap `reshape` at the op layer.
//
//   * Float32 / Float64 natively. Integer dtypes still land on the
//     CPU path at the op layer. `Half` / `BFloat16` — since B-016 —
//     land here via an FP32-promoted accumulator path: inputs are
//     loaded as `__half` / `__nv_bfloat16`, everything downstream
//     (stage-1 partials, shared-memory tree, stage-2 combine,
//     dim-reduce accumulator) runs in `float`, and only the final
//     store narrows back to the storage dtype. Matches the CPU
//     `Acc = float` path in `src/ops/cpu/Reduction.cpp` and the
//     FP32-promoted softmax kernel from B-015 — same motivation:
//     the dynamic range of FP16 (`~6e-5` min, `~6.5e4` max) trips
//     on any non-trivial `sum` over a few hundred elements, which
//     is exactly the regime transformer-backward hits.

#include "KernelUtils.cuh"

#include <math_constants.h>  // CUDART_INF / CUDART_INF_F for max identity
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/Reduction.hpp"

namespace tesseract::cuda::detail {

namespace {

// -------------------- Reduction identity / combine helpers --------------------

// A tiny policy struct per reduction kind. `identity` is the init
// value; `combine` merges two partials; `finalize` handles mean's
// divide-by-N. All device-side, no virtual dispatch.
template <typename T>
struct SumPolicy {
  static __device__ __forceinline__ T identity() { return T(0); }
  static __device__ __forceinline__ T combine(T a, T b) { return a + b; }
  static __device__ __forceinline__ T finalize(T acc, int64_t /*n*/) { return acc; }
};

template <typename T>
struct MeanPolicy {
  static __device__ __forceinline__ T identity() { return T(0); }
  static __device__ __forceinline__ T combine(T a, T b) { return a + b; }
  static __device__ __forceinline__ T finalize(T acc, int64_t n) {
    return static_cast<T>(acc / static_cast<T>(n));
  }
};

template <typename T>
struct MaxPolicy {
  // We can't use `std::numeric_limits<T>::lowest()` inside a device
  // function portably; hand-encode the two floating-point identities.
  static __device__ __forceinline__ T identity();
  static __device__ __forceinline__ T combine(T a, T b) { return a > b ? a : b; }
  static __device__ __forceinline__ T finalize(T acc, int64_t /*n*/) { return acc; }
};

template <>
__device__ __forceinline__ float MaxPolicy<float>::identity() { return -CUDART_INF_F; }
template <>
__device__ __forceinline__ double MaxPolicy<double>::identity() { return -CUDART_INF; }

// -------------------- Shared-memory block reduction --------------------

// Standard log-fan-in tree reduction over `sdata[0..blockDim.x)`,
// leaving the final value in `sdata[0]`. `blockDim.x` must be a
// power of two ≤ 1024; we only ever launch with 256.
template <typename T, typename Policy>
__device__ __forceinline__ void block_reduce(T* sdata, int tid) {
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = Policy::combine(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
}

// -------------------- All-reduce kernels --------------------

// Stage 1: each block reduces a grid-stride slice of `in` into a
// single per-block partial (stored in `partials[blockIdx.x]`).
// We do **not** finalize here — the second stage finalizes so that
// `mean` picks up the total element count rather than per-block N.
template <typename T, typename Policy>
__global__ void reduce_all_stage1(const T* __restrict__ in,
                                  int64_t n,
                                  T* __restrict__ partials) {
  extern __shared__ __align__(alignof(T)) unsigned char raw[];
  T* sdata = reinterpret_cast<T*>(raw);
  const int tid = threadIdx.x;
  T acc = Policy::identity();
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
       i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    acc = Policy::combine(acc, in[i]);
  }
  sdata[tid] = acc;
  __syncthreads();
  block_reduce<T, Policy>(sdata, tid);
  if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// Stage 2: launched with 1 block of `block_size` threads. Reduces
// `partials[0..num_partials)` and writes the finalized result to
// `out[0]`. `total_n` is the original input's element count — only
// `MeanPolicy` actually uses it.
template <typename T, typename Policy>
__global__ void reduce_all_stage2(const T* __restrict__ partials,
                                  int num_partials,
                                  int64_t total_n,
                                  T* __restrict__ out) {
  extern __shared__ __align__(alignof(T)) unsigned char raw[];
  T* sdata = reinterpret_cast<T*>(raw);
  const int tid = threadIdx.x;
  T acc = Policy::identity();
  for (int i = tid; i < num_partials; i += blockDim.x) {
    acc = Policy::combine(acc, partials[i]);
  }
  sdata[tid] = acc;
  __syncthreads();
  block_reduce<T, Policy>(sdata, tid);
  if (tid == 0) out[0] = Policy::finalize(sdata[0], total_n);
}

// -------------------- Dim-reduce kernel --------------------

// One block per output slot. `outer_inner` counts the number of
// (outer, inner) slots (== numel of the input shape with `dim`
// removed). `D` is the size of the reduced dim; `dim_stride` is
// `in_strides[dim]` (in elements, not bytes). `iter_sizes` /
// `iter_strides` describe the input with `dim` removed — used to
// compute the base offset for each slot. `total_n_for_mean` is D
// (used by `MeanPolicy::finalize`).
template <typename T, typename Policy>
__global__ void reduce_dim_kernel(const T* __restrict__ in,
                                  ShapePod iter_sizes,
                                  ShapePod iter_strides,
                                  int64_t outer_inner,
                                  int64_t D,
                                  int64_t dim_stride,
                                  T* __restrict__ out) {
  const int64_t slot = blockIdx.x;
  if (slot >= outer_inner) return;
  extern __shared__ __align__(alignof(T)) unsigned char raw[];
  T* sdata = reinterpret_cast<T*>(raw);

  const int64_t base = flat_to_offset(slot, iter_sizes, iter_strides);
  const int tid = threadIdx.x;
  T acc = Policy::identity();
  for (int64_t m = tid; m < D; m += blockDim.x) {
    acc = Policy::combine(acc, in[base + m * dim_stride]);
  }
  sdata[tid] = acc;
  __syncthreads();
  block_reduce<T, Policy>(sdata, tid);
  if (tid == 0) out[slot] = Policy::finalize(sdata[0], D);
}

// -------------------- Host-side dispatch --------------------

// Pick a sensible grid for the all-reduce stage 1. We want enough
// blocks to saturate the SMs but few enough that the stage-2 final
// reduction stays small (one block, 256 threads). 1024 is plenty for
// the Ada/Hopper SM counts we target (144 SMs × 2 blocks each = 288).
constexpr int kAllReduceStage1MaxBlocks = 1024;

int pick_stage1_blocks(int64_t n) {
  // One block per `kBlockSize` elements, clamped so we don't generate
  // pathological numbers of partials for small inputs.
  const int64_t blocks = (n + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(std::min<int64_t>(blocks, kAllReduceStage1MaxBlocks));
}

template <typename T, typename Policy>
void run_all_reduce(int device_index, int64_t n,
                    const T* x, T* out, cudaStream_t stream) {
  const int blocks = pick_stage1_blocks(n);
  T* partials = nullptr;
  if (cudaMalloc(&partials, sizeof(T) * blocks) != cudaSuccess) {
    throw DeviceError(fmt::format(
        "[tesseract] reduce_all: cudaMalloc({} bytes) for partials failed",
        sizeof(T) * static_cast<std::size_t>(blocks)));
  }
  // Stage 1: n → blocks partials.
  reduce_all_stage1<T, Policy><<<blocks, kBlockSize,
                                 sizeof(T) * kBlockSize,
                                 stream>>>(x, n, partials);
  check_launch("reduce_all_stage1");
  // Stage 2: blocks partials → 1 scalar, finalized.
  reduce_all_stage2<T, Policy><<<1, kBlockSize,
                                 sizeof(T) * kBlockSize,
                                 stream>>>(partials, blocks, n, out);
  check_launch("reduce_all_stage2");
  // We can safely free the partials as soon as the async launch is
  // queued; CUDA keeps the allocation alive until the stream finishes
  // with it (cudaFree on the default stream synchronizes with all
  // streams, but we want the stream-ordered free behavior). Using
  // `cudaFreeAsync` would be cleaner but requires a memory pool;
  // plain `cudaFree` forces a full sync which costs ~20 µs. For M2F
  // we eat that: the all-reduce path isn't a hot inner-loop kernel
  // (it's used for loss scalars).
  (void)cudaFree(partials);
  (void)device_index;  // consumed by the caller's DeviceGuard
}

template <typename T>
void dispatch_all(ReduceKind op, int device_index, int64_t n,
                  const T* x, T* out, cudaStream_t stream) {
  switch (op) {
    case ReduceKind::Sum:
      run_all_reduce<T, SumPolicy<T>>(device_index, n, x, out, stream); break;
    case ReduceKind::Mean:
      run_all_reduce<T, MeanPolicy<T>>(device_index, n, x, out, stream); break;
    case ReduceKind::Max:
      run_all_reduce<T, MaxPolicy<T>>(device_index, n, x, out, stream); break;
  }
}

// Build the iter-shape descriptor (input shape / strides with `dim`
// removed) used by the dim-reduce kernel. The kernel walks the flat
// iter index → input base offset, then strides along `dim_stride`.
struct IterDesc {
  ShapePod iter_sizes;
  ShapePod iter_strides;
  int64_t outer_inner{1};
};

IterDesc build_iter_desc(int ndim, int dim,
                         const int64_t* in_sizes,
                         const int64_t* in_strides) {
  IterDesc d{};
  d.iter_sizes.ndim = ndim - 1;
  d.iter_strides.ndim = ndim - 1;
  int k = 0;
  for (int i = 0; i < ndim; ++i) {
    if (i == dim) continue;
    d.iter_sizes.sizes[k]   = in_sizes[i];
    d.iter_strides.sizes[k] = in_strides[i];
    ++k;
  }
  d.outer_inner = numel_pod(d.iter_sizes);
  return d;
}

template <typename T, typename Policy>
void run_dim_reduce(const T* x, T* out,
                    const IterDesc& d,
                    int64_t D, int64_t dim_stride,
                    cudaStream_t stream) {
  // Zero-slot tensors (any size == 0 outside the reduced dim) are
  // already short-circuited at the op layer — we'd not be called.
  const int blocks = static_cast<int>(std::min<int64_t>(
      d.outer_inner, 2'147'483'647LL));
  reduce_dim_kernel<T, Policy><<<blocks, kBlockSize,
                                 sizeof(T) * kBlockSize,
                                 stream>>>(
      x, d.iter_sizes, d.iter_strides,
      d.outer_inner, D, dim_stride, out);
  check_launch("reduce_dim");
}

template <typename T>
void dispatch_dim(ReduceKind op,
                  const T* x, T* out,
                  const IterDesc& d,
                  int64_t D, int64_t dim_stride,
                  cudaStream_t stream) {
  switch (op) {
    case ReduceKind::Sum:
      run_dim_reduce<T, SumPolicy<T>>(x, out, d, D, dim_stride, stream); break;
    case ReduceKind::Mean:
      run_dim_reduce<T, MeanPolicy<T>>(x, out, d, D, dim_stride, stream); break;
    case ReduceKind::Max:
      run_dim_reduce<T, MaxPolicy<T>>(x, out, d, D, dim_stride, stream); break;
  }
}

// -------------------- B-016: FP32-promoted half-precision path --------------------
//
// Kernels below take `Tstorage` ∈ {`__half`, `__nv_bfloat16`} as the
// input/output dtype and do all arithmetic in `float`. Stage-1
// partials are `float`, not `Tstorage` — we narrow only on the final
// store, which matches the CPU `Acc = float` reference and avoids the
// per-stage double-rounding that would occur if partials stayed in
// half precision. `MaxPolicy<float>::identity()` uses `-CUDART_INF_F`
// which is safely below any finite half-precision value (FP16 max is
// ~6.5e4, BF16 max is ~3.4e38), so the max identity round-trips
// without saturating.

__device__ __forceinline__ float red_to_float(float v) { return v; }
__device__ __forceinline__ float red_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float red_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T red_from_float(float v);
template <> __device__ __forceinline__
float red_from_float<float>(float v) { return v; }
template <> __device__ __forceinline__
__half red_from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__
__nv_bfloat16 red_from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename Tstorage, typename Policy>
__global__ void reduce_all_stage1_promoted(const Tstorage* __restrict__ in,
                                           int64_t n,
                                           float* __restrict__ partials) {
  extern __shared__ __align__(alignof(float)) unsigned char raw[];
  float* sdata = reinterpret_cast<float*>(raw);
  const int tid = threadIdx.x;
  float acc = Policy::identity();
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
       i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    acc = Policy::combine(acc, red_to_float(in[i]));
  }
  sdata[tid] = acc;
  __syncthreads();
  block_reduce<float, Policy>(sdata, tid);
  if (tid == 0) partials[blockIdx.x] = sdata[0];
}

template <typename Tstorage, typename Policy>
__global__ void reduce_all_stage2_promoted(const float* __restrict__ partials,
                                           int num_partials,
                                           int64_t total_n,
                                           Tstorage* __restrict__ out) {
  extern __shared__ __align__(alignof(float)) unsigned char raw[];
  float* sdata = reinterpret_cast<float*>(raw);
  const int tid = threadIdx.x;
  float acc = Policy::identity();
  for (int i = tid; i < num_partials; i += blockDim.x) {
    acc = Policy::combine(acc, partials[i]);
  }
  sdata[tid] = acc;
  __syncthreads();
  block_reduce<float, Policy>(sdata, tid);
  if (tid == 0) {
    out[0] = red_from_float<Tstorage>(Policy::finalize(sdata[0], total_n));
  }
}

template <typename Tstorage, typename Policy>
__global__ void reduce_dim_kernel_promoted(const Tstorage* __restrict__ in,
                                           ShapePod iter_sizes,
                                           ShapePod iter_strides,
                                           int64_t outer_inner,
                                           int64_t D,
                                           int64_t dim_stride,
                                           Tstorage* __restrict__ out) {
  const int64_t slot = blockIdx.x;
  if (slot >= outer_inner) return;
  extern __shared__ __align__(alignof(float)) unsigned char raw[];
  float* sdata = reinterpret_cast<float*>(raw);

  const int64_t base = flat_to_offset(slot, iter_sizes, iter_strides);
  const int tid = threadIdx.x;
  float acc = Policy::identity();
  for (int64_t m = tid; m < D; m += blockDim.x) {
    acc = Policy::combine(acc, red_to_float(in[base + m * dim_stride]));
  }
  sdata[tid] = acc;
  __syncthreads();
  block_reduce<float, Policy>(sdata, tid);
  if (tid == 0) {
    out[slot] = red_from_float<Tstorage>(Policy::finalize(sdata[0], D));
  }
}

template <typename Tstorage, typename Policy>
void run_all_reduce_half(int device_index, int64_t n,
                         const Tstorage* x, Tstorage* out,
                         cudaStream_t stream) {
  const int blocks = pick_stage1_blocks(n);
  float* partials = nullptr;
  if (cudaMalloc(&partials, sizeof(float) * blocks) != cudaSuccess) {
    throw DeviceError(fmt::format(
        "[tesseract] reduce_all (half): cudaMalloc({} bytes) for "
        "partials failed", sizeof(float) * static_cast<std::size_t>(blocks)));
  }
  reduce_all_stage1_promoted<Tstorage, Policy><<<blocks, kBlockSize,
                                                 sizeof(float) * kBlockSize,
                                                 stream>>>(x, n, partials);
  check_launch("reduce_all_stage1_promoted");
  reduce_all_stage2_promoted<Tstorage, Policy><<<1, kBlockSize,
                                                 sizeof(float) * kBlockSize,
                                                 stream>>>(partials, blocks, n, out);
  check_launch("reduce_all_stage2_promoted");
  (void)cudaFree(partials);
  (void)device_index;
}

template <typename Tstorage>
void dispatch_all_half(ReduceKind op, int device_index, int64_t n,
                       const Tstorage* x, Tstorage* out, cudaStream_t stream) {
  switch (op) {
    case ReduceKind::Sum:
      run_all_reduce_half<Tstorage, SumPolicy<float>>(
          device_index, n, x, out, stream); break;
    case ReduceKind::Mean:
      run_all_reduce_half<Tstorage, MeanPolicy<float>>(
          device_index, n, x, out, stream); break;
    case ReduceKind::Max:
      run_all_reduce_half<Tstorage, MaxPolicy<float>>(
          device_index, n, x, out, stream); break;
  }
}

template <typename Tstorage, typename Policy>
void run_dim_reduce_half(const Tstorage* x, Tstorage* out,
                         const IterDesc& d,
                         int64_t D, int64_t dim_stride,
                         cudaStream_t stream) {
  const int blocks = static_cast<int>(std::min<int64_t>(
      d.outer_inner, 2'147'483'647LL));
  reduce_dim_kernel_promoted<Tstorage, Policy><<<blocks, kBlockSize,
                                                 sizeof(float) * kBlockSize,
                                                 stream>>>(
      x, d.iter_sizes, d.iter_strides,
      d.outer_inner, D, dim_stride, out);
  check_launch("reduce_dim_promoted");
}

template <typename Tstorage>
void dispatch_dim_half(ReduceKind op,
                       const Tstorage* x, Tstorage* out,
                       const IterDesc& d,
                       int64_t D, int64_t dim_stride,
                       cudaStream_t stream) {
  switch (op) {
    case ReduceKind::Sum:
      run_dim_reduce_half<Tstorage, SumPolicy<float>>(
          x, out, d, D, dim_stride, stream); break;
    case ReduceKind::Mean:
      run_dim_reduce_half<Tstorage, MeanPolicy<float>>(
          x, out, d, D, dim_stride, stream); break;
    case ReduceKind::Max:
      run_dim_reduce_half<Tstorage, MaxPolicy<float>>(
          x, out, d, D, dim_stride, stream); break;
  }
}

}  // namespace

// -------------------- Public launchers --------------------

void launch_reduce_all(ReduceKind op, DType dtype, int device_index,
                       int64_t nelem,
                       const void* x, void* out,
                       void* stream_handle) {
  if (nelem == 0) {
    // Op layer already rejects empty reductions (matches CPU); this
    // is a defensive no-op so the kernel launch config never goes
    // pathological if a caller bypasses the preflight.
    return;
  }
  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  switch (dtype) {
    case DType::Float32:
      dispatch_all<float>(op, device_index, nelem,
                          static_cast<const float*>(x),
                          static_cast<float*>(out), stream);
      break;
    case DType::Float64:
      dispatch_all<double>(op, device_index, nelem,
                           static_cast<const double*>(x),
                           static_cast<double*>(out), stream);
      break;
    case DType::Float16:
      dispatch_all_half<__half>(op, device_index, nelem,
                                static_cast<const __half*>(x),
                                static_cast<__half*>(out), stream);
      break;
    case DType::BFloat16:
      dispatch_all_half<__nv_bfloat16>(op, device_index, nelem,
                                       static_cast<const __nv_bfloat16*>(x),
                                       static_cast<__nv_bfloat16*>(out), stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA reduction on dtype {} is not implemented "
          "(Float32 / Float64 native; Float16 / BFloat16 via FP32-"
          "promoted accumulator; integer dtypes land via the CPU "
          "reference).", dtype_name(dtype)));
  }
}

void launch_reduce_dim(ReduceKind op, DType dtype, int device_index,
                       int ndim, int dim,
                       const int64_t* in_sizes,
                       const int64_t* in_strides,
                       const void* x, void* out,
                       void* stream_handle) {
  TESSERACT_CHECK(ndim >= 1 && ndim <= kMaxRank,
                  "[tesseract] launch_reduce_dim: ndim={} out of range "
                  "[1, {}]", ndim, kMaxRank);
  TESSERACT_CHECK(dim >= 0 && dim < ndim,
                  "[tesseract] launch_reduce_dim: dim={} out of range "
                  "for ndim={}", dim, ndim);

  const int64_t D          = in_sizes[dim];
  const int64_t dim_stride = in_strides[dim];
  if (D == 0) return;

  IterDesc d = build_iter_desc(ndim, dim, in_sizes, in_strides);
  if (d.outer_inner == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  switch (dtype) {
    case DType::Float32:
      dispatch_dim<float>(op,
                          static_cast<const float*>(x),
                          static_cast<float*>(out),
                          d, D, dim_stride, stream);
      break;
    case DType::Float64:
      dispatch_dim<double>(op,
                           static_cast<const double*>(x),
                           static_cast<double*>(out),
                           d, D, dim_stride, stream);
      break;
    case DType::Float16:
      dispatch_dim_half<__half>(op,
                                static_cast<const __half*>(x),
                                static_cast<__half*>(out),
                                d, D, dim_stride, stream);
      break;
    case DType::BFloat16:
      dispatch_dim_half<__nv_bfloat16>(op,
                                       static_cast<const __nv_bfloat16*>(x),
                                       static_cast<__nv_bfloat16*>(out),
                                       d, D, dim_stride, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA dim-reduction on dtype {} is not "
          "implemented (Float32 / Float64 native; Float16 / BFloat16 "
          "via FP32-promoted accumulator).", dtype_name(dtype)));
  }
}

}  // namespace tesseract::cuda::detail

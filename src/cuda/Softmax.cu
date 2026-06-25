// M2F: CUDA softmax / log_softmax along a single dim. See
// `src/cuda/Reduction.cu` for the broader M2F design notes — this TU
// uses the same `ShapePod` + `flat_to_offset` scaffolding from
// `KernelUtils.cuh` and the same "one block per output slot" layout
// as the dim-reduce kernel.
//
// Kernel strategy:
//   * One CUDA block per "slot" (a single (outer, inner) pair with
//     `dim` collapsed out). Threads in the block cooperate on three
//     passes over the `D` elements along the reduced dim:
//       (1) compute `max(x)` via shared-memory tree reduction,
//       (2) compute `sum(exp(x - max))` via another tree reduction,
//       (3) write `exp(x - max) / sum_exp` (or `x - max - log_sum_exp`
//           for log-softmax) — no tree reduction needed, just a
//           stride-over-D write.
//   * Input strides are honored (the input can be transposed or
//     broadcast); output is always written as row-major contiguous
//     with the same shape.
//   * Float32 / Float64. `__expf`/`__logf` are used on float for the
//     intrinsic fast-math path; doubles go through the precise
//     `exp`/`log`. The ratio of `WithinAbs(..., 1e-6)` we hold
//     ourselves to in the parity test is well inside both paths'
//     tolerance.

#include "KernelUtils.cuh"

#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/Softmax.hpp"

namespace tesseract::cuda::detail {

namespace {

// Precision-typed math helpers, mirroring the ones in `Elementwise.cu`.
__device__ __forceinline__ float  sm_exp(float  x) { return ::__expf(x); }
__device__ __forceinline__ double sm_exp(double x) { return ::exp(x); }
__device__ __forceinline__ float  sm_log(float  x) { return ::__logf(x); }
__device__ __forceinline__ double sm_log(double x) { return ::log(x); }

// `-inf` identity for the max reduction. Hand-encoded so we don't
// need `std::numeric_limits` inside `__global__` code.
template <typename T> __device__ __forceinline__ T neg_inf();
template <> __device__ __forceinline__ float  neg_inf<float>()  { return -CUDART_INF_F; }
template <> __device__ __forceinline__ double neg_inf<double>() { return -CUDART_INF; }

// Log-fan-in tree reductions. We spell out `max` and `+` as separate
// specializations rather than a single `block_reduce<T, Combine>` so
// we don't depend on `--extended-lambda` (C++17 `__device__` lambdas
// are non-default in nvcc). `sdata[blockDim.x]` must already be
// seeded by every thread before the call. Matches the pattern in
// `Reduction.cu`'s block reducer exactly.
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

// Per-slot softmax kernel. `iter_sizes` / `iter_strides` describe the
// input with `dim` collapsed out; `full_sizes` / `full_strides` are
// the full-rank descriptors of the input (unused except for
// reconstructing the base offset — baked into iter_* so we don't even
// pass them). `out_dim_stride` is always `out_strides_contig[dim]`;
// `out_base_stride` uses the contiguous row-major layout for the
// output iter.
template <typename T>
__global__ void softmax_kernel(const T* __restrict__ x,
                               T* __restrict__ out,
                               ShapePod iter_sizes,
                               ShapePod iter_strides,
                               ShapePod out_iter_strides,  // contiguous
                               int64_t outer_inner,
                               int64_t D,
                               int64_t in_dim_stride,
                               int64_t out_dim_stride,
                               bool take_log) {
  const int64_t slot = blockIdx.x;
  if (slot >= outer_inner) return;
  extern __shared__ __align__(alignof(T)) unsigned char raw[];
  T* sdata = reinterpret_cast<T*>(raw);

  const int64_t x_base   = flat_to_offset(slot, iter_sizes, iter_strides);
  const int64_t out_base = flat_to_offset(slot, iter_sizes, out_iter_strides);
  const int tid = threadIdx.x;

  // Pass 1: per-block max over D elements.
  {
    T m = neg_inf<T>();
    for (int64_t d = tid; d < D; d += blockDim.x) {
      const T v = x[x_base + d * in_dim_stride];
      if (v > m) m = v;
    }
    sdata[tid] = m;
    __syncthreads();
    block_reduce_max<T>(sdata, tid);
  }
  const T m = sdata[0];
  __syncthreads();

  // Pass 2: per-block sum(exp(x - m)).
  {
    T s = T(0);
    for (int64_t d = tid; d < D; d += blockDim.x) {
      s = s + sm_exp(x[x_base + d * in_dim_stride] - m);
    }
    sdata[tid] = s;
    __syncthreads();
    block_reduce_sum<T>(sdata, tid);
  }
  const T sum_exp = sdata[0];
  __syncthreads();

  // Pass 3: write output. No reduction needed — each thread handles
  // its stride-assigned elements independently.
  if (take_log) {
    const T log_z = m + sm_log(sum_exp);
    for (int64_t d = tid; d < D; d += blockDim.x) {
      out[out_base + d * out_dim_stride] =
          x[x_base + d * in_dim_stride] - log_z;
    }
  } else {
    const T inv_s = T(1) / sum_exp;
    for (int64_t d = tid; d < D; d += blockDim.x) {
      out[out_base + d * out_dim_stride] =
          sm_exp(x[x_base + d * in_dim_stride] - m) * inv_s;
    }
  }
}

// B-015: FP32-promoted softmax kernel for Float16 / BFloat16 storage.
// Structurally identical to `softmax_kernel<T>` but every intermediate
// — the per-thread running max, the per-thread partial sum, the shared-
// memory block-reduction buffer — is `float`. Load widens on read,
// store narrows on write. Matches PyTorch / cuDNN's "FP32 accum for
// FP16 softmax" convention; integer-add ULP error across D summands
// is kept well below FP16's storage ULP even at S_k = 4096.
template <typename Tstorage>
__device__ __forceinline__ float sm_to_float(Tstorage v);
template <> __device__ __forceinline__ float sm_to_float<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ __forceinline__ float sm_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <typename Tstorage> __device__ __forceinline__ Tstorage sm_from_float(float f);
template <> __device__ __forceinline__ __half sm_from_float<__half>(float f) {
  return __float2half(f);
}
template <> __device__ __forceinline__ __nv_bfloat16 sm_from_float<__nv_bfloat16>(float f) {
  return __float2bfloat16(f);
}

template <typename Tstorage>
__global__ void softmax_kernel_promoted(const Tstorage* __restrict__ x,
                                        Tstorage* __restrict__ out,
                                        ShapePod iter_sizes,
                                        ShapePod iter_strides,
                                        ShapePod out_iter_strides,
                                        int64_t outer_inner,
                                        int64_t D,
                                        int64_t in_dim_stride,
                                        int64_t out_dim_stride,
                                        bool take_log) {
  const int64_t slot = blockIdx.x;
  if (slot >= outer_inner) return;
  extern __shared__ __align__(alignof(float)) unsigned char raw_half[];
  float* sdata = reinterpret_cast<float*>(raw_half);

  const int64_t x_base   = flat_to_offset(slot, iter_sizes, iter_strides);
  const int64_t out_base = flat_to_offset(slot, iter_sizes, out_iter_strides);
  const int tid = threadIdx.x;

  // Pass 1: per-block max (FP32).
  {
    float m = -CUDART_INF_F;
    for (int64_t d = tid; d < D; d += blockDim.x) {
      const float v = sm_to_float(x[x_base + d * in_dim_stride]);
      if (v > m) m = v;
    }
    sdata[tid] = m;
    __syncthreads();
    block_reduce_max<float>(sdata, tid);
  }
  const float m = sdata[0];
  __syncthreads();

  // Pass 2: per-block sum(exp(x - m)) (FP32).
  {
    float s = 0.0f;
    for (int64_t d = tid; d < D; d += blockDim.x) {
      const float v = sm_to_float(x[x_base + d * in_dim_stride]);
      s = s + ::__expf(v - m);
    }
    sdata[tid] = s;
    __syncthreads();
    block_reduce_sum<float>(sdata, tid);
  }
  const float sum_exp = sdata[0];
  __syncthreads();

  // Pass 3: write output (narrow on store).
  if (take_log) {
    const float log_z = m + ::__logf(sum_exp);
    for (int64_t d = tid; d < D; d += blockDim.x) {
      const float v = sm_to_float(x[x_base + d * in_dim_stride]);
      out[out_base + d * out_dim_stride] = sm_from_float<Tstorage>(v - log_z);
    }
  } else {
    const float inv_s = 1.0f / sum_exp;
    for (int64_t d = tid; d < D; d += blockDim.x) {
      const float v = sm_to_float(x[x_base + d * in_dim_stride]);
      out[out_base + d * out_dim_stride] =
          sm_from_float<Tstorage>(::__expf(v - m) * inv_s);
    }
  }
}

// Build the iter-shape descriptor (input shape / strides with `dim`
// removed). Same helper as in Reduction.cu — duplicated here to keep
// the two TUs compile-parallel without a shared .cuh "impl" TU.
struct IterDesc {
  ShapePod iter_sizes;
  ShapePod iter_in_strides;
  ShapePod iter_out_strides;  // contiguous row-major
  int64_t outer_inner{1};
};

IterDesc build_softmax_iter(int ndim, int dim,
                            const int64_t* in_sizes,
                            const int64_t* in_strides) {
  IterDesc d{};
  d.iter_sizes.ndim       = ndim - 1;
  d.iter_in_strides.ndim  = ndim - 1;
  d.iter_out_strides.ndim = ndim - 1;

  // Full-rank contiguous output strides (output shape == input shape).
  // We build these in original-rank space so that each iter position's
  // "output base offset" is the corresponding full-rank contiguous
  // offset — *not* the iter-shape's own row-major offset, which would
  // drop the dim-stride factor and collapse every reduced row onto the
  // first few output elements. This was the bug in the first cut of
  // `build_softmax_iter`: for a [N, C] input reduced along dim=1, the
  // iter shape is just [N] with the iter's own contiguous strides
  // being [1], whereas the output [N, C] contiguous strides at the
  // non-dim positions are [C]. Using [1] made slot `n` write positions
  // `n .. n+C-1` (heavily overlapping between rows) instead of
  // `n*C .. n*C + C - 1`.
  int64_t full_contig_strides[kMaxRank];
  int64_t acc_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    full_contig_strides[i] = acc_stride;
    acc_stride *= in_sizes[i];
  }

  int k = 0;
  for (int i = 0; i < ndim; ++i) {
    if (i == dim) continue;
    d.iter_sizes.sizes[k]       = in_sizes[i];
    d.iter_in_strides.sizes[k]  = in_strides[i];
    d.iter_out_strides.sizes[k] = full_contig_strides[i];
    ++k;
  }
  d.outer_inner = numel_pod(d.iter_sizes);
  return d;
}

// Compute the output's stride along the softmax dim under a fully
// contiguous row-major layout. We'd normally get this from `Shape::
// contiguous_strides` but that lives in C++20-land on the host side;
// the bridge already hands us `in_sizes`, and the output has the same
// sizes, so we reconstruct here.
int64_t contig_stride_for_dim(int ndim, int dim, const int64_t* sizes) {
  int64_t s = 1;
  for (int i = ndim - 1; i > dim; --i) s *= sizes[i];
  return s;
}

template <typename T>
void run_softmax(const T* x, T* out,
                 const IterDesc& d,
                 int64_t D, int64_t in_dim_stride, int64_t out_dim_stride,
                 bool take_log, cudaStream_t stream) {
  const int blocks = static_cast<int>(std::min<int64_t>(
      d.outer_inner, 2'147'483'647LL));
  softmax_kernel<T><<<blocks, kBlockSize,
                      sizeof(T) * kBlockSize,
                      stream>>>(
      x, out,
      d.iter_sizes, d.iter_in_strides, d.iter_out_strides,
      d.outer_inner, D, in_dim_stride, out_dim_stride, take_log);
  check_launch("softmax");
}

// B-015: half-storage softmax. Always allocates the shared buffer as
// `float[kBlockSize]` because the kernel accumulates in FP32.
template <typename Tstorage>
void run_softmax_half(const Tstorage* x, Tstorage* out,
                      const IterDesc& d,
                      int64_t D, int64_t in_dim_stride, int64_t out_dim_stride,
                      bool take_log, cudaStream_t stream) {
  const int blocks = static_cast<int>(std::min<int64_t>(
      d.outer_inner, 2'147'483'647LL));
  softmax_kernel_promoted<Tstorage><<<blocks, kBlockSize,
                                      sizeof(float) * kBlockSize,
                                      stream>>>(
      x, out,
      d.iter_sizes, d.iter_in_strides, d.iter_out_strides,
      d.outer_inner, D, in_dim_stride, out_dim_stride, take_log);
  check_launch("softmax-half");
}

}  // namespace

void launch_softmax(bool take_log, DType dtype, int device_index,
                    int ndim, int dim,
                    const int64_t* in_sizes,
                    const int64_t* in_strides,
                    const void* x, void* out,
                    void* stream_handle) {
  TESSERACT_CHECK(ndim >= 1 && ndim <= kMaxRank,
                  "[tesseract] launch_softmax: ndim={} out of range "
                  "[1, {}]", ndim, kMaxRank);
  TESSERACT_CHECK(dim >= 0 && dim < ndim,
                  "[tesseract] launch_softmax: dim={} out of range "
                  "for ndim={}", dim, ndim);

  const int64_t D          = in_sizes[dim];
  const int64_t in_dim_stride  = in_strides[dim];
  const int64_t out_dim_stride = contig_stride_for_dim(ndim, dim, in_sizes);
  if (D == 0) return;

  IterDesc d = build_softmax_iter(ndim, dim, in_sizes, in_strides);
  if (d.outer_inner == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  switch (dtype) {
    case DType::Float32:
      run_softmax<float>(static_cast<const float*>(x),
                         static_cast<float*>(out),
                         d, D, in_dim_stride, out_dim_stride,
                         take_log, stream);
      break;
    case DType::Float64:
      run_softmax<double>(static_cast<const double*>(x),
                          static_cast<double*>(out),
                          d, D, in_dim_stride, out_dim_stride,
                          take_log, stream);
      break;
    case DType::Float16:
      // B-015: FP32-promoted softmax. Layout is identical to FP32; only
      // the load/store path and the reduction-buffer dtype differ.
      run_softmax_half<__half>(static_cast<const __half*>(x),
                               static_cast<__half*>(out),
                               d, D, in_dim_stride, out_dim_stride,
                               take_log, stream);
      break;
    case DType::BFloat16:
      run_softmax_half<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(x),
                                      static_cast<__nv_bfloat16*>(out),
                                      d, D, in_dim_stride, out_dim_stride,
                                      take_log, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA softmax on dtype {} is not implemented — "
          "only Float32 / Float64 / Float16 / BFloat16 are supported.",
          dtype_name(dtype)));
  }
}

}  // namespace tesseract::cuda::detail

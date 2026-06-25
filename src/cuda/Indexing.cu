// M2H: CUDA indexing kernels (B-003 `index_select` + `gather`, both
// forward and backward). Compiled only when TESSERACT_ENABLE_CUDA=ON;
// the matching CPU-only stubs live in `IndexingStub.cpp`.
//
// The four kernels share a common iteration-over-output-shape
// scaffold with the same `flat_to_offset` trick `Elementwise.cu`
// uses. What varies per op is how the source-side coordinate on
// `dim` is computed:
//
//   * `index_select`  — fancy-indexed dim uses
//                       `indices[coord[dim]]` (K-entry Int64 lookup).
//   * `gather`        — fancy-indexed dim uses
//                       `indices[out_coord]` (full-rank Int64 lookup
//                       with its own strides).
//
// The backward ops use the same coordinate function but scatter via
// `atomicAdd` into the parent tensor. Dtype policy matches
// `Shape.cu`: forward is an itemsize dispatch (dtype-agnostic byte
// copy), backward is a typed dispatch restricted to Float32 /
// Float64 / Int32 / Int64.

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/Indexing.hpp"
#include "tesseract/utils/Logging.hpp"

#include "KernelUtils.cuh"

namespace tesseract::cuda::detail {

namespace {

// ---------------- Device-side coordinate unpack ----------------
//
// Unpacks a flat iteration index into an n-D coordinate, returning
// the coordinate on `dim` separately so the caller can substitute
// the fancy-indexed value. Mirrors `flat_to_offset` but writes the
// intermediate coords out — we need them both for the source-side
// non-`dim` offset and (in the gather case) for the indices-tensor
// lookup.
__device__ __forceinline__ int64_t
unpack_and_offset_except_dim(int64_t flat,
                             const ShapePod& sizes,
                             const ShapePod& strides,
                             int dim,
                             int64_t out_coords[kMaxRank]) {
  int64_t off = 0;
  int64_t rem = flat;
#pragma unroll
  for (int d_rev = 0; d_rev < kMaxRank; ++d_rev) {
    const int d = sizes.ndim - 1 - d_rev;
    if (d < 0) break;
    const int64_t dim_size = sizes.sizes[d];
    const int64_t c = rem % dim_size;
    rem /= dim_size;
    out_coords[d] = c;
    if (d != dim) off += c * strides.sizes[d];
  }
  return off;
}

// ---------------- Kernels ----------------

template <typename Elem>
__global__ void index_select_forward_kernel(
    Elem* __restrict__ out,
    const Elem* __restrict__ src,
    const int64_t* __restrict__ indices,
    ShapePod out_sizes, ShapePod src_str, ShapePod out_str,
    int dim, int64_t total) {
  int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= total) return;
  int64_t coords[kMaxRank];
  const int64_t src_off_non_dim =
      unpack_and_offset_except_dim(k, out_sizes, src_str, dim, coords);
  const int64_t dst_off_non_dim =
      unpack_and_offset_except_dim(k, out_sizes, out_str, dim, coords);
  // `coords[dim]` is the output's coord on dim → index into `indices`.
  const int64_t sel = indices[coords[dim]];
  const int64_t src_off = src_off_non_dim + sel * src_str.sizes[dim];
  const int64_t dst_off = dst_off_non_dim + coords[dim] * out_str.sizes[dim];
  out[dst_off] = src[src_off];
}

template <typename Elem>
__global__ void index_select_scatter_add_kernel(
    Elem* __restrict__ dst,
    const Elem* __restrict__ grad,
    const int64_t* __restrict__ indices,
    ShapePod grad_sizes, ShapePod grad_str, ShapePod dst_str,
    int dim, int64_t total) {
  int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= total) return;
  int64_t coords[kMaxRank];
  const int64_t grad_off_non_dim =
      unpack_and_offset_except_dim(k, grad_sizes, grad_str, dim, coords);
  const int64_t dst_off_non_dim =
      unpack_and_offset_except_dim(k, grad_sizes, dst_str, dim, coords);
  const int64_t sel = indices[coords[dim]];
  const int64_t grad_off = grad_off_non_dim + coords[dim] * grad_str.sizes[dim];
  const int64_t dst_off = dst_off_non_dim + sel * dst_str.sizes[dim];
  atomicAdd(&dst[dst_off], grad[grad_off]);
}

__device__ __forceinline__ void atomic_add_i64(int64_t* addr, int64_t val) {
  unsigned long long* p = reinterpret_cast<unsigned long long*>(addr);
  atomicAdd(p, static_cast<unsigned long long>(val));
}

__global__ void index_select_scatter_add_i64_kernel(
    int64_t* __restrict__ dst,
    const int64_t* __restrict__ grad,
    const int64_t* __restrict__ indices,
    ShapePod grad_sizes, ShapePod grad_str, ShapePod dst_str,
    int dim, int64_t total) {
  int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= total) return;
  int64_t coords[kMaxRank];
  const int64_t grad_off_non_dim =
      unpack_and_offset_except_dim(k, grad_sizes, grad_str, dim, coords);
  const int64_t dst_off_non_dim =
      unpack_and_offset_except_dim(k, grad_sizes, dst_str, dim, coords);
  const int64_t sel = indices[coords[dim]];
  const int64_t grad_off = grad_off_non_dim + coords[dim] * grad_str.sizes[dim];
  const int64_t dst_off = dst_off_non_dim + sel * dst_str.sizes[dim];
  atomic_add_i64(&dst[dst_off], grad[grad_off]);
}

template <typename Elem>
__global__ void gather_forward_kernel(
    Elem* __restrict__ out,
    const Elem* __restrict__ src,
    const int64_t* __restrict__ indices,
    ShapePod out_sizes, ShapePod src_str, ShapePod idx_str, ShapePod out_str,
    int dim, int64_t total) {
  int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= total) return;
  int64_t coords[kMaxRank];
  const int64_t src_off_non_dim =
      unpack_and_offset_except_dim(k, out_sizes, src_str, dim, coords);
  const int64_t dst_off_non_dim =
      unpack_and_offset_except_dim(k, out_sizes, out_str, dim, coords);
  // The full-rank indices tensor lookup uses *all* coords (including
  // the one on `dim`), with its own strides — i.e. same shape as
  // out_sizes, not the source shape.
  int64_t idx_off = 0;
#pragma unroll
  for (int d_rev = 0; d_rev < kMaxRank; ++d_rev) {
    const int d = out_sizes.ndim - 1 - d_rev;
    if (d < 0) break;
    idx_off += coords[d] * idx_str.sizes[d];
  }
  const int64_t sel = indices[idx_off];
  const int64_t src_off = src_off_non_dim + sel * src_str.sizes[dim];
  const int64_t dst_off = dst_off_non_dim + coords[dim] * out_str.sizes[dim];
  out[dst_off] = src[src_off];
}

template <typename Elem>
__global__ void gather_scatter_add_kernel(
    Elem* __restrict__ dst,
    const Elem* __restrict__ grad,
    const int64_t* __restrict__ indices,
    ShapePod grad_sizes, ShapePod grad_str, ShapePod idx_str, ShapePod dst_str,
    int dim, int64_t total) {
  int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= total) return;
  int64_t coords[kMaxRank];
  const int64_t grad_off_non_dim =
      unpack_and_offset_except_dim(k, grad_sizes, grad_str, dim, coords);
  const int64_t dst_off_non_dim =
      unpack_and_offset_except_dim(k, grad_sizes, dst_str, dim, coords);
  int64_t idx_off = 0;
#pragma unroll
  for (int d_rev = 0; d_rev < kMaxRank; ++d_rev) {
    const int d = grad_sizes.ndim - 1 - d_rev;
    if (d < 0) break;
    idx_off += coords[d] * idx_str.sizes[d];
  }
  const int64_t sel = indices[idx_off];
  const int64_t grad_off = grad_off_non_dim + coords[dim] * grad_str.sizes[dim];
  const int64_t dst_off = dst_off_non_dim + sel * dst_str.sizes[dim];
  atomicAdd(&dst[dst_off], grad[grad_off]);
}

__global__ void gather_scatter_add_i64_kernel(
    int64_t* __restrict__ dst,
    const int64_t* __restrict__ grad,
    const int64_t* __restrict__ indices,
    ShapePod grad_sizes, ShapePod grad_str, ShapePod idx_str, ShapePod dst_str,
    int dim, int64_t total) {
  int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= total) return;
  int64_t coords[kMaxRank];
  const int64_t grad_off_non_dim =
      unpack_and_offset_except_dim(k, grad_sizes, grad_str, dim, coords);
  const int64_t dst_off_non_dim =
      unpack_and_offset_except_dim(k, grad_sizes, dst_str, dim, coords);
  int64_t idx_off = 0;
#pragma unroll
  for (int d_rev = 0; d_rev < kMaxRank; ++d_rev) {
    const int d = grad_sizes.ndim - 1 - d_rev;
    if (d < 0) break;
    idx_off += coords[d] * idx_str.sizes[d];
  }
  const int64_t sel = indices[idx_off];
  const int64_t grad_off = grad_off_non_dim + coords[dim] * grad_str.sizes[dim];
  const int64_t dst_off = dst_off_non_dim + sel * dst_str.sizes[dim];
  atomic_add_i64(&dst[dst_off], grad[grad_off]);
}

// ---------------- Host helpers ----------------

template <typename Elem>
void dispatch_index_select_forward_typed(
    void* out, const void* src, const int64_t* indices,
    const ShapePod& out_sizes, const ShapePod& src_str,
    const ShapePod& out_str,
    int dim, int64_t total, cudaStream_t stream) {
  const int grid = launch_grid(total);
  index_select_forward_kernel<Elem><<<grid, kBlockSize, 0, stream>>>(
      static_cast<Elem*>(out),
      static_cast<const Elem*>(src),
      indices,
      out_sizes, src_str, out_str, dim, total);
}

template <typename Elem>
void dispatch_gather_forward_typed(
    void* out, const void* src, const int64_t* indices,
    const ShapePod& out_sizes, const ShapePod& src_str,
    const ShapePod& idx_str, const ShapePod& out_str,
    int dim, int64_t total, cudaStream_t stream) {
  const int grid = launch_grid(total);
  gather_forward_kernel<Elem><<<grid, kBlockSize, 0, stream>>>(
      static_cast<Elem*>(out),
      static_cast<const Elem*>(src),
      indices,
      out_sizes, src_str, idx_str, out_str, dim, total);
}

}  // namespace

// ---------------- Public launchers ----------------

void launch_index_select(DType dtype, int device_index,
                         int ndim, int dim,
                         const int64_t* out_sizes,
                         const int64_t* src_strides,
                         const int64_t* out_strides,
                         const void* src,
                         const int64_t* indices,
                         void* out,
                         void* stream_handle) {
  TESSERACT_CHECK(ndim >= 1 && ndim <= kMaxRank,
                  "[tesseract] launch_index_select: ndim={} out of "
                  "range [1, {}]", ndim, kMaxRank);
  TESSERACT_CHECK(dim >= 0 && dim < ndim,
                  "[tesseract] launch_index_select: dim={} out of "
                  "range [0, {})", dim, ndim);

  const ShapePod s   = pack_raw(out_sizes,    ndim);
  const ShapePod ss  = pack_raw(src_strides,  ndim);
  const ShapePod os  = pack_raw(out_strides,  ndim);
  const int64_t total = numel_pod(s);
  if (total == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  const std::size_t isz = dtype_size(dtype);
  switch (isz) {
    case 1:
      dispatch_index_select_forward_typed<uint8_t>(out, src, indices, s, ss, os, dim, total, stream);
      break;
    case 2:
      dispatch_index_select_forward_typed<uint16_t>(out, src, indices, s, ss, os, dim, total, stream);
      break;
    case 4:
      dispatch_index_select_forward_typed<uint32_t>(out, src, indices, s, ss, os, dim, total, stream);
      break;
    case 8:
      dispatch_index_select_forward_typed<uint64_t>(out, src, indices, s, ss, os, dim, total, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] launch_index_select: unsupported itemsize {} "
          "for dtype {}", isz, dtype_name(dtype)));
  }
  check_launch("index-select-forward");
}

void launch_scatter_add_at_dim(DType dtype, int device_index,
                               int ndim, int dim,
                               const int64_t* grad_sizes,
                               const int64_t* grad_strides,
                               const int64_t* dst_strides,
                               const void* grad,
                               const int64_t* indices,
                               void* dst,
                               void* stream_handle) {
  TESSERACT_CHECK(ndim >= 1 && ndim <= kMaxRank,
                  "[tesseract] launch_scatter_add_at_dim: ndim={} out "
                  "of range [1, {}]", ndim, kMaxRank);
  TESSERACT_CHECK(dim >= 0 && dim < ndim,
                  "[tesseract] launch_scatter_add_at_dim: dim={} out of "
                  "range [0, {})", dim, ndim);

  const ShapePod s   = pack_raw(grad_sizes,   ndim);
  const ShapePod gs  = pack_raw(grad_strides, ndim);
  const ShapePod ds  = pack_raw(dst_strides,  ndim);
  const int64_t total = numel_pod(s);
  if (total == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  const int grid = launch_grid(total);

  switch (dtype) {
    case DType::Float32:
      index_select_scatter_add_kernel<float><<<grid, kBlockSize, 0, stream>>>(
          static_cast<float*>(dst),
          static_cast<const float*>(grad),
          indices, s, gs, ds, dim, total);
      break;
    case DType::Float64:
      index_select_scatter_add_kernel<double><<<grid, kBlockSize, 0, stream>>>(
          static_cast<double*>(dst),
          static_cast<const double*>(grad),
          indices, s, gs, ds, dim, total);
      break;
    case DType::Int32:
      index_select_scatter_add_kernel<int32_t><<<grid, kBlockSize, 0, stream>>>(
          static_cast<int32_t*>(dst),
          static_cast<const int32_t*>(grad),
          indices, s, gs, ds, dim, total);
      break;
    case DType::Int64:
      index_select_scatter_add_i64_kernel<<<grid, kBlockSize, 0, stream>>>(
          static_cast<int64_t*>(dst),
          static_cast<const int64_t*>(grad),
          indices, s, gs, ds, dim, total);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA scatter-add at dim on dtype {} is not "
          "implemented in M2H — cast grads to Float32 first.",
          dtype_name(dtype)));
  }
  check_launch("index-select-scatter-add");
}

void launch_gather(DType dtype, int device_index,
                   int ndim, int dim,
                   const int64_t* out_sizes,
                   const int64_t* src_strides,
                   const int64_t* idx_strides,
                   const int64_t* out_strides,
                   const void* src,
                   const int64_t* indices,
                   void* out,
                   void* stream_handle) {
  TESSERACT_CHECK(ndim >= 1 && ndim <= kMaxRank,
                  "[tesseract] launch_gather: ndim={} out of range "
                  "[1, {}]", ndim, kMaxRank);
  TESSERACT_CHECK(dim >= 0 && dim < ndim,
                  "[tesseract] launch_gather: dim={} out of range "
                  "[0, {})", dim, ndim);

  const ShapePod s   = pack_raw(out_sizes,   ndim);
  const ShapePod ss  = pack_raw(src_strides, ndim);
  const ShapePod is_ = pack_raw(idx_strides, ndim);
  const ShapePod os  = pack_raw(out_strides, ndim);
  const int64_t total = numel_pod(s);
  if (total == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  const std::size_t isz = dtype_size(dtype);
  switch (isz) {
    case 1:
      dispatch_gather_forward_typed<uint8_t>(out, src, indices, s, ss, is_, os, dim, total, stream);
      break;
    case 2:
      dispatch_gather_forward_typed<uint16_t>(out, src, indices, s, ss, is_, os, dim, total, stream);
      break;
    case 4:
      dispatch_gather_forward_typed<uint32_t>(out, src, indices, s, ss, is_, os, dim, total, stream);
      break;
    case 8:
      dispatch_gather_forward_typed<uint64_t>(out, src, indices, s, ss, is_, os, dim, total, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] launch_gather: unsupported itemsize {} for "
          "dtype {}", isz, dtype_name(dtype)));
  }
  check_launch("gather-forward");
}

void launch_gather_scatter_add(DType dtype, int device_index,
                               int ndim, int dim,
                               const int64_t* grad_sizes,
                               const int64_t* grad_strides,
                               const int64_t* idx_strides,
                               const int64_t* dst_strides,
                               const void* grad,
                               const int64_t* indices,
                               void* dst,
                               void* stream_handle) {
  TESSERACT_CHECK(ndim >= 1 && ndim <= kMaxRank,
                  "[tesseract] launch_gather_scatter_add: ndim={} out "
                  "of range [1, {}]", ndim, kMaxRank);
  TESSERACT_CHECK(dim >= 0 && dim < ndim,
                  "[tesseract] launch_gather_scatter_add: dim={} out of "
                  "range [0, {})", dim, ndim);

  const ShapePod s   = pack_raw(grad_sizes,   ndim);
  const ShapePod gs  = pack_raw(grad_strides, ndim);
  const ShapePod is_ = pack_raw(idx_strides,  ndim);
  const ShapePod ds  = pack_raw(dst_strides,  ndim);
  const int64_t total = numel_pod(s);
  if (total == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  const int grid = launch_grid(total);

  switch (dtype) {
    case DType::Float32:
      gather_scatter_add_kernel<float><<<grid, kBlockSize, 0, stream>>>(
          static_cast<float*>(dst),
          static_cast<const float*>(grad),
          indices, s, gs, is_, ds, dim, total);
      break;
    case DType::Float64:
      gather_scatter_add_kernel<double><<<grid, kBlockSize, 0, stream>>>(
          static_cast<double*>(dst),
          static_cast<const double*>(grad),
          indices, s, gs, is_, ds, dim, total);
      break;
    case DType::Int32:
      gather_scatter_add_kernel<int32_t><<<grid, kBlockSize, 0, stream>>>(
          static_cast<int32_t*>(dst),
          static_cast<const int32_t*>(grad),
          indices, s, gs, is_, ds, dim, total);
      break;
    case DType::Int64:
      gather_scatter_add_i64_kernel<<<grid, kBlockSize, 0, stream>>>(
          static_cast<int64_t*>(dst),
          static_cast<const int64_t*>(grad),
          indices, s, gs, is_, ds, dim, total);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA gather scatter-add on dtype {} is not "
          "implemented in M2H — cast grads to Float32 first.",
          dtype_name(dtype)));
  }
  check_launch("gather-scatter-add");
}

}  // namespace tesseract::cuda::detail

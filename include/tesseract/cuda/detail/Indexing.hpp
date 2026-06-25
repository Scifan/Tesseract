#pragma once

// Internal CUDA bridge for the M2H B-003 indexing ops (`index_select`
// and `gather`, both forward and backward). See `Shape.hpp` / ADR-0005
// for the visibility and C++17 conventions shared with the rest of the
// M2 CUDA bridge surface.
//
// Scope (M2H):
//
//   Forward paths (`launch_index_select`, `launch_gather`) are dtype-
//   generic byte copies — the dtype argument is used solely to pick
//   an itemsize. Every numeric + Bool dtype the core layer exposes is
//   supported (1 / 2 / 4 / 8 bytes).
//
//   Backward paths (`launch_scatter_add_at_dim`,
//   `launch_gather_scatter_add`) perform an `atomicAdd` on the
//   destination slot, so the dtype determines the atomic primitive
//   used. Supported dtypes: Float32, Float64, Int32, Int64. Float16 /
//   BFloat16 / Bool / Int8 throw a clear DeviceError; real-world
//   training pipelines cast loss through Float32, so the restriction
//   matches both our cuBLASLt matmul policy (M2G) and PyTorch's own
//   dtype promotion rules for gradient accumulation.
//
// All four launchers skip the call on zero-numel iteration shapes.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// Forward of `ops::index_select(src, dim, indices)`.
//
//   For every flat index `k` over `out_sizes`, with `out_sizes[dim]
//   == K == indices.numel()`, writes
//       out[k] = src[..., indices[idx[dim]], ...]
//   where the trailing `...` is a strided lookup over `src_strides`
//   matching the output's multi-index on every non-`dim` axis, and
//   `dst` is assumed dense row-major contiguous (so `out_strides` is
//   just the contig strides of `out_sizes`; the kernel still accepts
//   it explicitly to keep the bridge symmetric).
//
//   `src_strides` points at `ndim` int64_t element-strides describing
//   the source layout — it is NOT aligned to the output; callers may
//   pass the source's own strides directly. `indices` is a K-entry
//   Int64 buffer on the same CUDA device.
void launch_index_select(DType dtype, int device_index,
                         int ndim, int dim,
                         const int64_t* out_sizes,
                         const int64_t* src_strides,
                         const int64_t* out_strides,
                         const void* src,
                         const int64_t* indices,
                         void* out,
                         void* stream);

// Backward of `ops::index_select` (and `cat`-like scatter patterns
// that funnel a contiguous grad slab into a strided parent on a
// single axis).
//
//   For every flat index `k` over `grad_sizes`, computes
//       dst[..., indices[idx[dim]], ...] += grad[k]
//   using `atomicAdd`. `dst_strides` describes the parent buffer's
//   layout — it may be dense row-major contig (when `dst` was freshly
//   zeroed by `Tensor::zeros`) or any strided layout. The kernel
//   never re-checks the index range; the op-layer already runs the
//   forward range check against the same index tensor, so a bad
//   index has already thrown before the backward runs.
void launch_scatter_add_at_dim(DType dtype, int device_index,
                               int ndim, int dim,
                               const int64_t* grad_sizes,
                               const int64_t* grad_strides,
                               const int64_t* dst_strides,
                               const void* grad,
                               const int64_t* indices,
                               void* dst,
                               void* stream);

// Forward of `ops::gather(src, dim, indices)` (PyTorch GatherElements
// convention). `out_sizes == indices.sizes`; for every flat index `k`
//     out[k] = src[idx[0], ..., indices[...], ..., idx[r-1]]
// where `indices[...]` is the index tensor evaluated at the same
// multi-index as `out[k]` (with its own strides), and the result
// replaces the `dim`-th coord of the src lookup. `indices` layout is
// described by `idx_strides` to support non-contiguous views.
void launch_gather(DType dtype, int device_index,
                   int ndim, int dim,
                   const int64_t* out_sizes,
                   const int64_t* src_strides,
                   const int64_t* idx_strides,
                   const int64_t* out_strides,
                   const void* src,
                   const int64_t* indices,
                   void* out,
                   void* stream);

// Backward of `ops::gather`. Symmetric to the forward: for every flat
// index `k` over `grad_sizes`,
//     dst[idx[0], ..., indices[...], ..., idx[r-1]] += grad[k]
// via `atomicAdd`.
void launch_gather_scatter_add(DType dtype, int device_index,
                               int ndim, int dim,
                               const int64_t* grad_sizes,
                               const int64_t* grad_strides,
                               const int64_t* idx_strides,
                               const int64_t* dst_strides,
                               const void* grad,
                               const int64_t* indices,
                               void* dst,
                               void* stream);

}  // namespace tesseract::cuda::detail

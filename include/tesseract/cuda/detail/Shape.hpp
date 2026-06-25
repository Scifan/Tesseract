#pragma once

// Internal CUDA bridge for the M2H shape / view ops. Same `detail/`
// visibility convention as `Elementwise.hpp` / `Reduction.hpp` — this
// header lives under `include/` so op-layer TUs outside `src/cuda/`
// can include it without private include-dir munging, but it is NOT
// part of the public API. See ADR-0005.
//
// Design contract (C++17 bridge, same as the rest of the M2 CUDA
// surface):
//   * `.cu` TU compiles at `CUDA_STANDARD=17` (CMake 3.22 can't pick
//     `CUDA20`). No `std::span`, no explicit-template lambdas; shape
//     and stride data cross the bridge as raw `int64_t*` + `int ndim`.
//   * Real implementation ships in `src/cuda/Shape.cu` when
//     `TESSERACT_ENABLE_CUDA=ON`; `src/cuda/ShapeStub.cpp` provides
//     throwing stubs for CPU-only builds.
//   * Launchers take an opaque `cudaStream_t` as `void*` so the bridge
//     surface stays free of `<cuda_runtime.h>`. Ordering falls out of
//     stream-order on the caller side; no forced `cudaStreamSynchronize`
//     inside the launchers.
//
// Scope (M2H):
//
//   `launch_strided_copy` — element-wise copy from a strided source
//   into a strided destination over a shared iteration shape. The
//   only constraint is `numel(src_strides) == numel(dst_strides) ==
//   prod(sizes)` (the iteration domain). Dtype is used solely to
//   pick an itemsize (1 / 2 / 4 / 8 bytes) — the copy itself is a
//   byte blit. This single launcher underpins:
//     * `Tensor::contiguous()` on CUDA (strided→dense copy).
//     * `ops::broadcast_to` on CUDA (source strides may be 0 on
//       broadcast dims; iteration shape is the broadcast target).
//     * `cat` forward (copy each contig input into an output slab
//       with dst offset + stride on dim).
//     * `split` / `slice_along_dim` (copy an output slab out of a
//       strided source with src offset + stride on dim).
//     * `SplitChunkBackward` (copy a contig chunk into a zero-filled
//       parent tensor — same pattern as `cat` forward).
//
//   `launch_strided_scatter_add` — element-wise `dst[dst_off] +=
//   src[src_off]` over a shared iteration shape, with per-dim strides
//   on each side. Dst strides may be 0 on "reduced" dims to funnel
//   multiple src elements into a single dst element (the `reduce_to_shape`
//   pattern). Uses `atomicAdd` on the GPU so collisions on the same
//   dst slot are linearized safely. Supported dtypes: Float32, Float64,
//   Int32, Int64. Float16 / BFloat16 / Bool / Int8 throw — training
//   backward paths cast through Float32, so the restriction is
//   invisible in practice.
//
// Both launchers skip the call entirely when the iteration numel is
// zero, matching CPU `for_each_index` no-op behaviour on empty tensors.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

void launch_strided_copy(DType dtype, int device_index,
                         int ndim,
                         const int64_t* sizes,
                         const int64_t* src_strides,
                         const int64_t* dst_strides,
                         const void* src, void* dst,
                         void* stream);

void launch_strided_scatter_add(DType dtype, int device_index,
                                int ndim,
                                const int64_t* sizes,
                                const int64_t* src_strides,
                                const int64_t* dst_strides,
                                const void* src, void* dst,
                                void* stream);

}  // namespace tesseract::cuda::detail

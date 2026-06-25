#pragma once

// Internal CUDA bridge for the M2F reduction suite. Same load-bearing
// `detail/` segment as Elementwise.hpp — despite the `include/` path,
// this is a private header consumed only by `src/ops/cpu/Reduction.cpp`
// and `src/cuda/Reduction.cu`. It lives under `include/` purely so the
// op layer can `#include` it without CMake needing to add a private
// include-dir for `src/cuda/`.
//
// Design contract (see ADR-0005 and the M2E-equivalent in
// Elementwise.hpp for the full rationale):
//   * C++17-parseable surface: the `.cu` TU is nvcc-compiled at
//     `CUDA_STANDARD=17` (CMake 3.22 can't select `CUDA20`), so the
//     kernel side must not parse `std::span` or C++20 lambdas. Shape
//     information crosses the bridge as raw `int64_t*` + `int ndim`.
//   * No circular archive dependency: `tesseract_core` PRIVATE-links
//     `tesseract_cuda`, so the stream handle is passed in as a
//     `void* stream` rather than resolved inside the CUDA TU (where
//     calling `current_stream` would pull us back into `tesseract_core`).
//     Callers resolve `current_stream(device).native_handle()` just
//     before the launcher call.
//   * Stream-sync semantics stay in `Storage::copy_device_bytes` /
//     `zero_device_bytes` (the M2E fix); the launcher itself does
//     *not* force a `cudaStreamSynchronize`. A subsequent `.to(cpu)`
//     or `.item()` crosses the device boundary and that is where the
//     sync happens.
//
// Scope (M2F, floating-point only — matches the CPU `dispatch_float`):
//   * `sum` / `mean` / `max` over the entire tensor → 0-D scalar.
//   * `sum` / `mean` / `max` along a single dim, with `keepdim`
//     handled by the caller (the bridge always writes an output of
//     rank `ndim-1`; the op layer reshapes to ndim when `keepdim`).
//   * Float32 and Float64 only. Half / BFloat16 fall back to the
//     CPU path via explicit host-bounce (see the op-layer dispatch
//     in `src/ops/cpu/Reduction.cpp`). Integer reductions (the CPU
//     max path also accepts integer dtypes at M0) are also handled
//     via the CPU path — there is no integer training hot path that
//     would pay back the kernel-writing cost at M2F.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

enum class ReduceKind : int {
  Sum = 0,
  Mean = 1,
  Max = 2,
};

// All-reduce over a dense numeric buffer. `nelem` > 0 is the number of
// elements in `x`; `out` points at a single dtype-sized scalar on the
// same CUDA device. Internally uses a two-pass reduction (per-block
// partials → final block-local reduction) so the numeric order is
// deterministic given a fixed block size — `atomicAdd`-based single-
// pass reductions would give faster but non-deterministic results, and
// M2F prioritizes parity with the CPU baseline.
void launch_reduce_all(ReduceKind op, DType dtype, int device_index,
                       int64_t nelem,
                       const void* x, void* out,
                       void* stream);

// Reduce along `dim` of a rank-`ndim` tensor. `in_sizes` and
// `in_strides` point at `ndim` int64_t entries each (the full input
// shape and its strides-in-elements, both aligned). `out` is written
// as a row-major contiguous rank-`ndim-1` tensor with `dim` removed —
// the `keepdim=true` reshape is the op layer's responsibility.
//
// Strided inputs are supported: a stride of 0 is permitted (although
// reducing over a broadcasted dim would be unusual — we just compute
// what the CPU reference computes). Non-contiguous inputs therefore
// don't need a `.contiguous()` materialization on the caller side.
void launch_reduce_dim(ReduceKind op, DType dtype, int device_index,
                       int ndim, int dim,
                       const int64_t* in_sizes,
                       const int64_t* in_strides,
                       const void* x, void* out,
                       void* stream);

}  // namespace tesseract::cuda::detail

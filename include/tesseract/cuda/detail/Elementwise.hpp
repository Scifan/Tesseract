#pragma once

// Internal CUDA bridge for the M2E elementwise suite. Despite living under
// `include/tesseract/cuda/detail/`, this header is **not** part of the
// public API — the `detail/` segment is the load-bearing marker. It is
// here (rather than under `src/cuda/`) purely so callers outside
// `src/cuda/` (i.e. `src/ops/...`) can `#include` it without needing
// private include-dirs munging in CMake.
//
// Design contract (matches ADR-0005):
//   * Plain C++17-compatible surface: the `.cu` translation unit is
//     nvcc-compiled with `CUDA_STANDARD=17` (CMake 3.22 can't select
//     `CUDA20`), so this header must stay parseable at C++17. That
//     rules out `std::span`, `Shape` by reference (Shape.hpp uses
//     `std::span`), and explicit-template lambdas — we instead pass
//     shape / strides as raw `int64_t*` + an `int ndim`.
//   * Symbols are defined in one of two mutually-exclusive translation
//     units depending on `TESSERACT_HAS_CUDA`:
//       - `src/cuda/Elementwise.cu` (compiled by nvcc when the backend
//         is ON) provides the real launchers.
//       - `src/cuda/ElementwiseStub.cpp` (always compiled, but its body
//         is `#if !defined(TESSERACT_HAS_CUDA)` gated) provides throwing
//         stubs so a CPU-only build still links.
//   * Launchers enqueue work on the caller-supplied stream handle
//     (the op layer resolves `current_stream(Device{CUDA, device_index})
//     .native_handle()` and passes it through). Keeping the stream as
//     an arg rather than fetching it inside the CUDA TU avoids pulling
//     a `tesseract_core` symbol into `tesseract_cuda` — the two
//     archives stay a-cyclic (see `src/core/CMakeLists.txt` comment).
//     No forced sync; synchronous semantics for `.to(cpu)` / `.item()`
//     fall out of the synchronous `cudaMemcpy` in
//     `Storage::copy_device_bytes`.

#include <cstddef>
#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// Identifies the pointwise arithmetic operation to launch. We dispatch
// on this at the host-side launcher rather than instantiating one
// public launcher per op — keeps the exported surface small and lets
// the `.cu` TU share the strided-kernel scaffolding across ops.
enum class BinaryKind : int {
  Add = 0,
  Sub = 1,
  Mul = 2,
  Div = 3,
};

enum class UnaryKind : int {
  Neg = 0,      // works on any numeric dtype
  Relu = 1,     // works on any numeric dtype
  Sigmoid = 2,  // Float32 / Float64 only
  Tanh = 3,     // Float32 / Float64 only
  Exp = 4,      // Float32 / Float64 only
  Log = 5,      // Float32 / Float64 only
  // M2I: positive-indicator `(x > 0) ? 1 : 0`. Float32 / Float64 only.
  // Used exclusively by `ReluBackward` to materialize the grad mask
  // on the same device as the saved input, so the downstream `mul(g,
  // mask)` stays on-device. Integer step is not exposed — the only
  // caller (Relu backward) always lives in the float world.
  Step = 6,
  // M2K: element-wise `sqrt(x)`. Float32 / Float64 only. Added so that
  // `ops::rms_norm` can stay a pure composite of already-CUDA-resident
  // primitives (no dedicated fused kernel needed for transformer-block
  // parity at this milestone). Integer sqrt is intentionally not
  // exposed — there is no caller today, and integer → float promotion
  // is a separate API concern best handled by `Tensor::to(dtype=...)`.
  Sqrt = 7,
};

// Broadcast-aware binary elementwise.
//
//   `ndim` is the rank of the output. `out_sizes`, `a_strides`,
//   `b_strides` each point to an array of `ndim` int64_t entries. The
//   stride arrays are **already aligned** to `ndim` (see
//   `ops::align_for_broadcast`) with stride-0 entries marking
//   broadcasted dims. `out` / `a` / `b` point into device memory on
//   `device_index`. No null-pointer or element-count checks happen
//   here — the core tensor layer has already validated shapes and
//   dtypes.
//
//   Supported dtypes: `Float32`, `Float64`, `Int32`, `Int64`, and
//   — since B-015 — `Float16` / `BFloat16` via FP32-promotion on the
//   load path. Other dtypes throw a clear "not implemented" error.
void launch_binary_elementwise(BinaryKind op, DType dtype, int device_index,
                               int ndim,
                               const int64_t* out_sizes,
                               const int64_t* a_strides,
                               const int64_t* b_strides,
                               void* out, const void* a, const void* b,
                               void* stream);

// Unary elementwise on a single operand. `sizes` is the output shape
// (equal to the input's shape); `x_strides` is aligned to `ndim` — for
// dense inputs this is just the input's contiguous strides. Output
// `out` is always laid out row-major contiguous.
//
// Supported dtypes (post B-015): `Neg` / `Relu` on
// `{Float32, Float64, Int32, Int64, Float16, BFloat16}`;
// `Sigmoid`, `Tanh`, `Exp`, `Log`, `Step`, `Sqrt` on
// `{Float32, Float64, Float16, BFloat16}`. The half-precision variants
// widen to `float` for the math and narrow on store (matches CPU).
// All other dtypes throw.
void launch_unary_elementwise(UnaryKind op, DType dtype, int device_index,
                              int ndim,
                              const int64_t* sizes,
                              const int64_t* x_strides,
                              void* out, const void* x,
                              void* stream);

// Dense fill. `nelem` contiguous elements of `dtype` at `out` get
// `static_cast<T>(value)` — same numeric contract as CPU `fill_`.
// Replaces the M2D host-scratch + `cudaMemcpy` fallback inside
// `Tensor::fill_` / `ones` / `full`.
//
// Supported dtypes: `Float32`, `Float64`, `Int32`, `Int64`, `Bool`,
// `Float16`, `BFloat16`. Integer dtypes truncate, matching CPU
// semantics; half dtypes narrow via `__float2half` / `__float2bfloat16`.
void launch_fill(DType dtype, int device_index, std::size_t nelem,
                 void* out, double value, void* stream);

}  // namespace tesseract::cuda::detail

#pragma once

// Internal CUDA bridge for the M2F softmax / log_softmax kernels. See
// `detail/Elementwise.hpp` for the broader layering rationale — same
// C++17-parseable surface + `void* stream` convention.
//
// Both `softmax` and `log_softmax` along a dim are served by a single
// launcher that picks the output formula via a boolean flag. Keeping
// them unified makes the fused max+sum+write kernel easy to share
// (the only difference is a single branch at the normalization step).
//
// Scope:
//   * Float32 / Float64, any `ndim >= 1`, any valid `dim`. Strided
//     inputs are supported via the same `(sizes, strides, ndim)`
//     descriptor as the reduction bridge.
//   * Output is contiguous row-major rank-`ndim` (same shape as
//     input). That matches the CPU reference in
//     `src/ops/cpu/Softmax.cpp`.
//   * `Half` / `BFloat16` are intentionally excluded at M2F; the
//     numerically-stable path (`max` + `exp(x-m)`) needs careful
//     handling of the mantissa narrowing that lands with M2G.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// Numerically stable softmax / log-softmax along a single dim.
//
//   * `take_log == false` writes `exp(x - max) / sum_exp`.
//   * `take_log == true`  writes `(x - max) - log(sum_exp)`.
//
// `in_sizes` and `in_strides` are the full-rank descriptor of `x`;
// `dim` is already normalized (`0 <= dim < ndim`). `out` is written
// as a dense row-major contiguous tensor of the same shape — callers
// that need a strided output must allocate separately and copy.
void launch_softmax(bool take_log, DType dtype, int device_index,
                    int ndim, int dim,
                    const int64_t* in_sizes,
                    const int64_t* in_strides,
                    const void* x, void* out,
                    void* stream);

}  // namespace tesseract::cuda::detail

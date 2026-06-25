#pragma once

// Internal CUDA bridge for the rotary position embedding kernel. Same
// layering convention as `detail/Softmax.hpp`: the header stays
// C++17-parseable (no `std::span`, no device types) so it can be
// included from plain `.cpp` translation units, and the entry point
// takes `const void*` / `void*` with an explicit `DType` selector.
//
// Scope:
//
//   * Dtypes: Float32 / Float64 natively; Float16 / BFloat16 via the
//     same FP32-promoted load-compute-store pattern that B-015 /
//     B-016 established for elementwise / softmax / reduction.
//
//   * Shape: `x` is a dense row-major contiguous tensor of shape
//     `[outer, S, D]` (any number of leading dims are collapsed into
//     `outer` at the op-layer boundary — that's why only a flat
//     `outer` count appears here). `cos` / `sin` are `[S, D]`,
//     contiguous. `D` must be even. The kernel writes `out` as a
//     dense row-major `[outer, S, D]`.
//
//   * Convention: adjacent-pair rotation, matching the GPT-NeoX /
//     Llama formulation. `cos[p, 2j]` and `cos[p, 2j+1]` are
//     expected to hold the same value (and likewise for sin) — the
//     caller is responsible for filling the table accordingly. The
//     kernel itself just multiplies element-wise and swaps pairs,
//     which keeps the compute ~2 FMAs per element.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// Forward pass: `out = rope(x, cos, sin)`. For the backward we
// simply pass `-sin` in place of `sin` (rotation by -θ), which the
// autograd node does without needing a separate kernel.
void launch_rotary_embedding(DType dtype, int device_index,
                             int64_t outer, int64_t seq, int64_t dim,
                             const void* x,
                             const void* cos,
                             const void* sin,
                             void* out,
                             void* stream);

}  // namespace tesseract::cuda::detail

#pragma once

// Internal CUDA bridge for the fused SwiGLU activation (Wave 4.1 / B-025).
//
// Scope & layering follow the same pattern as `detail/RMSNorm.hpp` and
// `detail/RotaryEmbedding.hpp`: the header stays C++17-parseable (no
// `std::span`, no CUDA types), and the entry point takes `const void*`
// / `void*` with an explicit `DType` selector so it can be included
// from plain `.cpp` translation units.
//
// Computation (element-wise, same shape):
//     out[i] = silu(gate[i]) * up[i]
//            = (gate[i] / (1 + exp(-gate[i]))) * up[i]
//
// Shape contract (enforced at the op-layer boundary):
//   * `gate`, `up`, `out` are row-major contiguous tensors with the
//     same shape and dtype. All leading dims are collapsed into a
//     single `numel` count (the kernel is a 1-D element-wise pass,
//     no reduction, no cross-lane communication).
//
// Dtype policy (matches B-015 / B-016 / RMSNorm):
//   * Float32 / Float64 — compute in the storage type.
//   * Float16 / BFloat16 — load through an FP32 accumulator, compute
//     `silu` + `mul` in FP32, narrow back on store. The `sigmoid`
//     exponent itself runs in FP32 so the dynamic range of half
//     precision never limits the saturation tails.
//
// Forward-only. Backward continues to flow through the composite
// primitives' autograd nodes (`sigmoid` + `mul` + `mul`), same
// convention as `launch_rms_norm`. The fused path collapses the FFN's
// six-launch `mul(sigmoid(gate)) · mul(up)` tail into a single kernel
// pass — a strict memory-bandwidth win on the memory-bound
// element-wise portion of every Llama FFN.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// Fused forward: `out[i] = silu(gate[i]) * up[i]` for `i ∈ [0, numel)`.
//
// Launches a 1-D grid with a fixed block width and a grid-stride loop,
// so arbitrarily large `numel` is handled without exceeding the
// grid-count limit. `gate`, `up`, `out` must already be on
// `device_index` and contiguous (the op layer validates this).
void launch_swiglu_silu_gate(DType dtype, int device_index,
                             int64_t numel,
                             const void* gate,
                             const void* up,
                             void* out,
                             void* stream);

}  // namespace tesseract::cuda::detail

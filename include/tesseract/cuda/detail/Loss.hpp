#pragma once

// Internal CUDA bridge for the M2F cross-entropy-with-logits kernels.
// See `detail/Elementwise.hpp` for the broader layering rationale.
//
// The CPU reference (`src/ops/cpu/Loss.cpp`) fuses the softmax +
// NLL forward into a single pass over the [N, C] logits matrix, and
// fuses the `(probs - one_hot) * g / N` backward into a single pass.
// The CUDA bridge mirrors that exactly so op-layer dispatch is a
// simple device-index branch.
//
// Scope (M2F):
//   * `logits` shape [N, C], dtype Float32 / Float64, **contiguous**.
//     (The CPU path accepts non-contiguous and materializes with
//     `.contiguous()`; the op layer does the same before reaching us.)
//   * `targets` shape [N], dtype Int64, **contiguous**. Values must
//     lie in `[0, C)` — the kernel does *not* re-check (matching
//     what `dispatch_float` + `TESSERACT_CHECK` already enforce at
//     the op layer for CPU).
//   * `loss` is a single dtype-sized scalar (0-D tensor storage).
//   * `probs_out` is optional (may be null) — populated only when
//     the op layer needs it for the autograd tape. This saves an
//     [N, C] write when `requires_grad == false` (common at
//     inference).
//   * `Half` / `BFloat16` fall back to the CPU path at the op layer.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// Fused forward: numerically-stable softmax over each row of
// `logits` + mean NLL loss against `targets`. `loss_out` receives a
// single `dtype`-sized scalar (the mean loss); `probs_out`, when
// non-null, receives the [N, C] softmax matrix that the backward
// pass needs. `probs_out` may be `nullptr` for the `requires_grad
// == false` fast path; in that case no softmax row-write happens
// and the kernel just accumulates the loss.
void launch_ce_forward(DType dtype, int device_index,
                       int64_t N, int64_t C,
                       const void* logits,
                       const int64_t* targets,
                       void* loss_out,
                       void* probs_out /*nullable*/,
                       void* stream);

// Fused backward: `dlogits = (probs - one_hot(targets)) * g / N`.
// `probs` is the [N, C] softmax output (typically the one saved by
// the forward pass's `probs_out`), `grad` is a 0-D scalar, and
// `dlogits_out` is written as an [N, C] contiguous tensor of the
// same dtype. This launcher is used both by the autograd Node and
// by the standalone graph-mode API
// (`ops::cross_entropy_with_logits_backward`), where the forward
// softmax is recomputed by the CPU op layer before calling us.
void launch_ce_backward(DType dtype, int device_index,
                        int64_t N, int64_t C,
                        const void* probs,
                        const int64_t* targets,
                        const void* grad_scalar,
                        void* dlogits_out,
                        void* stream);

}  // namespace tesseract::cuda::detail

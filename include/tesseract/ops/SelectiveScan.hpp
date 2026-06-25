#pragma once

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// Selective state-space scan — the core primitive of Mamba / S6 (M4 Track A2 /
// B-039). Implements the discretized, input-dependent SSM recurrence per
// (batch b, inner channel d):
//
//   for t in 0..L-1:
//     dA = exp(delta[b,t,d] * A[d,:])            # [N]  (zero-order hold)
//     h  = dA * h + (delta[b,t,d] * B[b,t,:]) * u[b,t,d]   # [N]
//     y[b,t,d] = Σ_n C[b,t,n] * h[n] + D[d] * u[b,t,d]
//
// `A` is the (negative) state matrix `[D, N]`; `B`/`C` are the input-dependent
// `[B, L, N]` projections shared across channels; `D` is the `[D]` skip. The
// recurrence is sequential in `t` (parallel across `b·d`); chunkwise-parallel
// prefill is a perf follow-up — the contract here is that stepping one token at
// a time while threading `state` reproduces a full-sequence call bit-for-bit,
// which is exactly what makes incremental decode equal prefill.
//
// `state_in` is the optional initial hidden state `[B, D, N]` (undefined ⇒
// zeros). The result carries the final hidden state `[B, D, N]` so a decode
// loop can thread it through an `nn::SSMStateCache`.
//
// Forward-only (no autograd node): used under `NoGradGuard` on the inference
// path. Float dtypes only; FP32 interior math.
struct SelectiveScanResult {
  Tensor y;      // [B, L, D]
  Tensor state;  // [B, D, N] final hidden state
};

SelectiveScanResult selective_scan(const Tensor& u, const Tensor& delta,
                                   const Tensor& A, const Tensor& B,
                                   const Tensor& C, const Tensor& D,
                                   const Tensor& state_in = Tensor());

}  // namespace tesseract::ops

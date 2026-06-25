#pragma once

// M4 perf-closeout Phase 4 — device-side MoE top-k routing.
//
// Replaces the host round-trip in `nn::MoEFeedForward` (softmax probs were
// copied D→H, top-k chosen on the CPU, the mask uploaded H→D). The whole
// router post-processing now runs in one kernel on device:
//
//   logits[T,E]  --(this kernel)-->  gates[T,E], mask[T,E]
//
//   gates[t,e] = softmax(logits[t,:])[e]  if e in top-k(logits[t,:])
//                                          renormalized over the k winners
//              = 0                          otherwise
//   mask[t,e]  = 1 if e in top-k else 0
//
// Tie-break matches the host path: value-descending, index-ascending, so the
// k==1 case equals argmax's lowest-index convention and parity with the CPU
// reference is exact up to fp rounding. `E` must be <= 256 (one block per
// token; each thread owns one expert slot). Float32 only — router logits are
// always fp32 in the model configs we ship.

#include <cstdint>

namespace tesseract::cuda::detail {

void launch_moe_route(int device_index, int64_t num_tokens, int64_t num_experts,
                      int64_t top_k, const float* logits, float* gates,
                      float* mask, void* stream);

}  // namespace tesseract::cuda::detail

#pragma once

// M4 perf-closeout Phase 4 — fully fused GPU MoE inference forward.
//
// Replaces the host-bound sparse dispatch (mask→host sort→index_select→
// per-expert FeedForward→cat→gather) with an all-device pipeline:
//
//   1. device permutation: histogram tokens per expert, exclusive prefix
//      sum → offsets, scatter (token,slot) into expert-contiguous order;
//   2. gather x into expert-grouped layout;
//   3. ONE grouped GEMM across all experts for gate_proj and up_proj
//      (cublasGemmGroupedBatchedEx — each expert is its own group with its
//      own row count), fused SiLU·up;
//   4. ONE grouped GEMM for down_proj;
//   5. fused scatter-combine: weight each routed row by its gate and
//      atomic-accumulate back into the token's output row.
//
// Inference only (no autograd); FP32 experts with no bias. The caller
// (nn::MoEFeedForward) falls back to the generic path when grad is enabled,
// the device is CPU, experts are quantized, or a bias is present. Numerically
// matches the generic path to TF32 tolerance (same CUBLAS_COMPUTE_32F the
// dense cuBLASLt path uses).

#include <cstdint>

namespace tesseract::cuda::detail {

// Returns false if the configuration is unsupported (caller should fall back).
//   x      : [T, D]   row-major device
//   gates  : [T, E]   row-major device (renormalized top-k gate weights, 0 elsewhere)
//   mask   : [T, E]   row-major device (0/1, exactly k ones per row)
//   Wg/Wu  : E host pointers, each [dff, D] row-major device  (gate/up weights)
//   Wd     : E host pointers, each [D, dff] row-major device  (down weight)
//   y      : [T, D]   row-major device output (overwritten)
bool launch_moe_grouped_ffn(int device_index, int64_t T, int64_t D,
                            int64_t dff, int64_t E, int64_t k,
                            const float* x, const float* gates,
                            const float* mask, const float* const* Wg,
                            const float* const* Wu, const float* const* Wd,
                            float* y, void* stream);

}  // namespace tesseract::cuda::detail

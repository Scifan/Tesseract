#pragma once

// M4 perf-closeout Phase 5 — causal depthwise conv1d for Mamba.
//
// Replaces the host-orchestrated `cat`/`narrow`/`mul`/`add` op-loop in
// nn::Mamba::conv1d_forward (K separate elementwise passes + a pad-cat) with
// one fused kernel: one thread per (b, t, d) output element accumulates the
// K-tap causal dot product in registers.
//
//   out[b,t,d] = bias[d] + Σ_{k=0..K-1} weight[d,k] · x[b, t-(K-1)+k, d]
//
// (left-padded with zeros; reads with index < 0 contribute nothing). FP32,
// inference only — matches the op-loop to fp rounding.

#include <cstdint>

namespace tesseract::cuda::detail {

void launch_causal_conv1d(int device_index, int64_t B, int64_t L,
                          int64_t channels, int64_t K, const float* x,
                          const float* weight, const float* bias, float* out,
                          void* stream);

}  // namespace tesseract::cuda::detail

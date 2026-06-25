#pragma once

#include <memory>

#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"
#include "tesseract/nn/SSMStateCache.hpp"

namespace tesseract::nn {

// Mamba-1 selective-SSM block (M4 Track A2 / B-039) — the headline
// non-Transformer architecture from `idea.md` §4.2.
//
//   x  ──in_proj──► [x_inner | z]            (d_model → 2·d_inner)
//   x_inner ──causal depthwise conv1d──► SiLU ─► x_conv
//   x_conv ──x_proj──► [dt | B | C]          (d_inner → dt_rank + 2·d_state)
//   delta = softplus(dt_proj(dt))            (dt_rank → d_inner)
//   A = -exp(A_log)                          ([d_inner, d_state])
//   y = selective_scan(x_conv, delta, A, B, C, D)
//   y = y · SiLU(z)
//   out ──out_proj──► [d_model]              (d_inner → d_model)
//
// `forward` runs a full sequence (prefill, zero initial state). `forward_step`
// advances one token threading an `SSMStateCache` (conv ring buffer + SSM
// hidden state), and reproduces the corresponding slice of `forward`
// bit-for-bit — that recurrent≡parallel identity is the A2 exit bar.
//
// Forward-only at A2 (inference path under NoGradGuard); SSM autograd is a
// follow-up. The conv1d / SiLU / softplus tail is expressed as op composition
// so it runs on CPU and CUDA unchanged; the heavy scan is `ops::selective_scan`
// (CPU reference + CUDA kernel).
class Mamba : public Module {
 public:
  Mamba(int64_t d_model, int64_t d_state = 16, int64_t d_conv = 4,
        int64_t expand = 2, int64_t dt_rank = 0,
        DType dtype = DType::Float32);

  // Full-sequence forward. `x` is [B, L, d_model]; returns [B, L, d_model].
  Tensor forward(const Tensor& x) override;

  // Single-token incremental forward. `x` is [B, 1, d_model]; advances `cache`
  // and returns [B, 1, d_model].
  Tensor forward_step(const Tensor& x, SSMStateCache& cache);

  // Allocate a zero-initialized decode cache matching this block's geometry.
  SSMStateCache make_state_cache(int64_t batch) const;

  int64_t d_model() const noexcept { return d_model_; }
  int64_t d_inner() const noexcept { return d_inner_; }
  int64_t d_state() const noexcept { return d_state_; }
  int64_t d_conv() const noexcept { return d_conv_; }
  int64_t dt_rank() const noexcept { return dt_rank_; }

 private:
  // Causal depthwise conv1d over `x` [B, L, d_inner] with the registered
  // weight [d_inner, d_conv] + bias [d_inner], expressed as op composition.
  Tensor conv1d_forward(const Tensor& x);

  int64_t d_model_;
  int64_t d_inner_;
  int64_t d_state_;
  int64_t d_conv_;
  int64_t dt_rank_;
  DType dtype_;

  std::shared_ptr<Linear> in_proj_;
  std::shared_ptr<Linear> x_proj_;
  std::shared_ptr<Linear> dt_proj_;
  std::shared_ptr<Linear> out_proj_;

  Tensor conv_weight_;  // [d_inner, d_conv]
  Tensor conv_bias_;    // [d_inner]
  Tensor a_log_;        // [d_inner, d_state]
  Tensor d_skip_;       // [d_inner]
};

}  // namespace tesseract::nn

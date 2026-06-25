#pragma once

#include <memory>
#include <vector>

#include "tesseract/nn/FeedForward.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// Sparse mixture-of-experts feed-forward (M4 Track A1 / B-038).
//
// The Mixtral / Switch-Transformer recipe: a lightweight `router` Linear
// scores every token over `num_experts`, the top-`k` experts are selected
// per token, their gate weights are the softmax over just those k logits
// (== full softmax then renormalized over the selected set — algebraically
// identical), and the block's output is the gate-weighted sum of the selected
// experts' SwiGLU FFNs:
//
//   logits = router(x)                       [..., E]
//   probs  = softmax(logits, -1)             [..., E]
//   mask   = top_k(probs)                    [..., E]  (0/1, k ones per row)
//   gates  = (probs * mask) / sum(probs*mask, -1)      [..., E]
//   out    = Σ_e gates[..., e] · expert_e(x)           [..., d_model]
//
// Compute is **sparse** (B-038+): tokens are permuted into per-expert groups
// and each expert runs only on its routed rows (Σ_e n_e == T·k), instead of the
// dense "every expert on every token" path. The result is numerically identical
// to the dense reference — each token's output is the same gate-weighted sum of
// the same expert outputs, accumulated in ascending-expert order — but the heavy
// FFN compute scales with the *active* experts (k), not the full set (E). The
// permutation uses autograd-aware `index_select` (gather rows → run experts →
// un-permute) and the combine gathers gate values from the differentiable
// `gates` tensor via `ops::gather`, so the router stays trainable. See
// `bench_moe_sparse` / docs/design/moe-sparse.md for the measured saving.
//
// The top-k selection is computed host-side (a small D→H copy of the router
// probabilities) since we have no device top-k kernel yet; the heavy compute
// (router GEMM, expert FFNs, the weighted combine) stays on-device. Gates flow
// gradient through `probs` (the router) and through the experts, so the module
// is training-capable; `last_aux_loss()` exposes the Switch load-balancing
// auxiliary loss for the training path. A fused grouped-GEMM that runs the
// per-expert slices in one launch (vs the current per-expert loop) is the
// remaining follow-up — it changes only the inner loop, not this contract.
class MoEFeedForward : public Module {
 public:
  MoEFeedForward(int64_t d_model, int64_t d_ff, int64_t num_experts,
                 int64_t num_experts_per_tok, bool use_bias = false,
                 DType dtype = DType::Float32);

  Tensor forward(const Tensor& x) override;

  int64_t d_model() const noexcept { return d_model_; }
  int64_t d_ff() const noexcept { return d_ff_; }
  int64_t num_experts() const noexcept { return num_experts_; }
  int64_t num_experts_per_tok() const noexcept { return top_k_; }

  // Switch-Transformer load-balancing auxiliary loss from the most recent
  // `forward`. Scalar tensor: `E · Σ_e f_e · P_e`, where `f_e` is the fraction
  // of tokens routed to expert e and `P_e` is the mean router probability of e.
  // Differentiable through `P_e` (the router). Undefined before the first
  // forward.
  const Tensor& last_aux_loss() const noexcept { return aux_loss_; }

  const std::shared_ptr<Module>& router() const { return router_; }
  const std::vector<std::shared_ptr<FeedForward>>& experts() const {
    return experts_;
  }

 private:
  int64_t d_model_;
  int64_t d_ff_;
  int64_t num_experts_;
  int64_t top_k_;

  std::shared_ptr<Module> router_;  // d_model -> num_experts (no bias)
  std::vector<std::shared_ptr<FeedForward>> experts_;

  Tensor aux_loss_;
};

}  // namespace tesseract::nn

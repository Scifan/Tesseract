#pragma once

#include <memory>
#include <vector>

#include "tesseract/nn/FeedForward.hpp"
#include "tesseract/nn/KVCacheBase.hpp"
#include "tesseract/nn/MoEFeedForward.hpp"
#include "tesseract/nn/Module.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"
#include "tesseract/nn/RMSNorm.hpp"

namespace tesseract::nn {

// Single Llama-style transformer block:
//
//   h   = x + attn(rms_norm_1(x))        // pre-norm residual 1
//   out = h + ffn(rms_norm_2(h))         // pre-norm residual 2
//
// Pre-norm (RMSNorm before each sub-layer) is the Llama default and
// gives better training stability than post-norm for deep stacks. The
// residual adds are always in the *outer* branch, so the gradient has
// an identity path through the block — RMSNorm, attention, and FFN
// all see a well-scaled gradient even in long chains.
//
// Note on position encoding: Llama uses rotary position embedding
// (RoPE) applied to Q/K inside the attention sub-layer. Since B-014
// the block threads RoPE config through to `MultiHeadAttention` via
// `rope_base` and `rope_max_seq`. Leaving both at their defaults
// (0.0, 0) disables RoPE entirely — the pre-B-014 "pass-your-own
// positional prior" behavior. A positive `rope_base` (Llama uses
// 10000.0) with a `rope_max_seq` ≥ the longest expected input makes
// the block Llama-spec.
class TransformerBlock : public Module {
 public:
  // `num_kv_heads == 0` ⇒ plain MHA (= num_heads); a smaller positive
  // value selects grouped-query attention (must divide num_heads).
  //
  // M4 Track A1 (B-038): `num_experts > 0` swaps the dense SwiGLU FFN for a
  // sparse `nn::MoEFeedForward` (Mixtral-style), keeping the registration slot
  // name "ffn" so param-walk order is unchanged. `0` keeps the dense FFN —
  // fully backward compatible.
  TransformerBlock(int64_t d_model, int64_t num_heads, int64_t d_ff,
                   double norm_eps = 1e-5,
                   bool causal = true,
                   bool use_bias = false,  // Llama default: no biases
                   DType dtype = DType::Float32,
                   double rope_base = 0.0,
                   int64_t rope_max_seq = 0,
                   int64_t num_kv_heads = 0,
                   int64_t num_experts = 0,
                   int64_t num_experts_per_tok = 0);

  Tensor forward(const Tensor& x) override;

  // Wave 5 (B-027) decode-step variant. `x` is `[B, S_new, D_model]`
  // (S_new == 1 for token-by-token decode, larger for chunked prefill).
  // Threads the attention sub-layer through `cache` via
  // `MultiHeadAttention::forward_step`; the RMSNorms, FFN, and residual
  // adds are position-independent so they reuse the eager paths
  // unchanged. Inference-only: the whole step runs under `NoGradGuard`,
  // so no autograd edges are built (the cache stores detached tensors).
  //
  //   h   = x + attn.forward_step(norm_1(x), cache)
  //   out = h + ffn(norm_2(h))
  Tensor forward_step(const Tensor& x, KVCacheBase& cache);

  // Wave 10 (B-032) compute-batched decode. `x` is `[A, S_new, D_model]`
  // (A stacked sequences); `caches` holds one batch-1 cache per sequence.
  // The norms / FFN / residuals are position-independent and run batched
  // over A; the attention threads each sequence through its own cache via
  // `MultiHeadAttention::forward_step_batched`. Inference-only.
  Tensor forward_step_batched(const Tensor& x,
                              const std::vector<KVCacheBase*>& caches);

  int64_t d_model() const noexcept { return d_model_; }
  int64_t num_heads() const noexcept { return num_heads_; }
  int64_t d_ff() const noexcept { return d_ff_; }

  // Pass-throughs that the LlamaModel walker leans on. `attn()` /
  // `ffn()` give it the handles it needs to call
  // `MultiHeadAttention::quantize_` / `FeedForward::quantize_`
  // without adding a block-level quantize method (the block itself
  // has no FP `Linear` children — its RMSNorms are deliberately left
  // in FP, see LlamaModel::quantize_ comment on why).
  const std::shared_ptr<MultiHeadAttention>& attn() const { return attn_; }
  // Dense FFN handle. **Null when this is an MoE block** (`is_moe()` true) —
  // callers that walk projections (e.g. the quantizer) must null-check.
  const std::shared_ptr<FeedForward>& ffn() const { return ffn_; }
  // MoE FFN handle. Null for a dense block.
  const std::shared_ptr<MoEFeedForward>& moe_ffn() const { return moe_ffn_; }
  bool is_moe() const noexcept { return static_cast<bool>(moe_ffn_); }

 private:
  // Routes the FFN sub-layer through whichever variant is live (dense or MoE).
  Tensor ffn_forward(const Tensor& x);

  int64_t d_model_;
  int64_t num_heads_;
  int64_t d_ff_;

  std::shared_ptr<RMSNorm> norm_1_;
  std::shared_ptr<MultiHeadAttention> attn_;
  std::shared_ptr<RMSNorm> norm_2_;
  std::shared_ptr<FeedForward> ffn_;        // null when MoE
  std::shared_ptr<MoEFeedForward> moe_ffn_; // null when dense
};

}  // namespace tesseract::nn

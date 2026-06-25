#include "tesseract/nn/TransformerBlock.hpp"

#include <utility>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

TransformerBlock::TransformerBlock(int64_t d_model, int64_t num_heads, int64_t d_ff,
                                   double norm_eps, bool causal, bool use_bias,
                                   DType dtype, double rope_base,
                                   int64_t rope_max_seq, int64_t num_kv_heads,
                                   int64_t num_experts,
                                   int64_t num_experts_per_tok)
    : d_model_(d_model), num_heads_(num_heads), d_ff_(d_ff) {
  TESSERACT_CHECK(d_model > 0 && num_heads > 0 && d_ff > 0,
                  "TransformerBlock: d_model ({}), num_heads ({}), d_ff ({}) must all be positive",
                  d_model, num_heads, d_ff);
  TESSERACT_CHECK(d_model % num_heads == 0,
                  "TransformerBlock: d_model ({}) must be divisible by num_heads ({})",
                  d_model, num_heads);

  norm_1_ = std::make_shared<RMSNorm>(d_model, norm_eps, dtype);
  attn_   = std::make_shared<MultiHeadAttention>(d_model, num_heads, use_bias,
                                                 causal, dtype,
                                                 rope_base, rope_max_seq,
                                                 num_kv_heads);
  norm_2_ = std::make_shared<RMSNorm>(d_model, norm_eps, dtype);

  register_module("norm_1", norm_1_);
  register_module("attn",   attn_);
  register_module("norm_2", norm_2_);
  // The FFN slot is registered under the same name ("ffn") whether dense or
  // MoE, so `named_parameters()` walk order and any name-based loader stay
  // stable across the two variants.
  if (num_experts > 0) {
    moe_ffn_ = std::make_shared<MoEFeedForward>(d_model, d_ff, num_experts,
                                                num_experts_per_tok, use_bias,
                                                dtype);
    register_module("ffn", moe_ffn_);
  } else {
    ffn_ = std::make_shared<FeedForward>(d_model, d_ff, use_bias, dtype);
    register_module("ffn", ffn_);
  }
}

Tensor TransformerBlock::ffn_forward(const Tensor& x) {
  return moe_ffn_ ? moe_ffn_->forward(x) : ffn_->forward(x);
}

Tensor TransformerBlock::forward(const Tensor& x) {
  TESSERACT_CHECK(x.rank() == 3,
                  "TransformerBlock::forward: expected rank-3 input [B, S, D], got {}",
                  x.shape().to_string());
  TESSERACT_CHECK(x.shape()[2] == d_model_,
                  "TransformerBlock::forward: input last dim {} != d_model {}",
                  x.shape()[2], d_model_);

  // Pre-norm residual structure. Both sub-layers share the same
  // shape-in/shape-out contract so the residual additions are plain
  // elementwise broadcasts with no reshape hiding between them.
  Tensor h   = ops::add(x, attn_->forward(norm_1_->forward(x)));
  Tensor out = ops::add(h, ffn_forward(norm_2_->forward(h)));
  return out;
}

Tensor TransformerBlock::forward_step(const Tensor& x, KVCacheBase& cache) {
  TESSERACT_CHECK(x.rank() == 3,
                  "TransformerBlock::forward_step: expected rank-3 input "
                  "[B, S_new, D], got {}", x.shape().to_string());
  TESSERACT_CHECK(x.shape()[2] == d_model_,
                  "TransformerBlock::forward_step: input last dim {} != "
                  "d_model {}", x.shape()[2], d_model_);

  // Inference-only: the whole step is detached. `MHA::forward_step`
  // installs its own NoGradGuard, but the surrounding norms / FFN /
  // residual adds would still build graph if grad mode were on, so we
  // guard the entire block here.
  NoGradGuard nogg;

  Tensor h   = ops::add(x, attn_->forward_step(norm_1_->forward(x), cache));
  Tensor out = ops::add(h, ffn_forward(norm_2_->forward(h)));
  return out;
}

Tensor TransformerBlock::forward_step_batched(
    const Tensor& x, const std::vector<KVCacheBase*>& caches) {
  TESSERACT_CHECK(x.rank() == 3,
                  "TransformerBlock::forward_step_batched: expected rank-3 "
                  "input [A, S_new, D], got {}", x.shape().to_string());
  TESSERACT_CHECK(x.shape()[2] == d_model_,
                  "TransformerBlock::forward_step_batched: last dim {} != "
                  "d_model {}", x.shape()[2], d_model_);
  NoGradGuard nogg;
  Tensor h = ops::add(
      x, attn_->forward_step_batched(norm_1_->forward(x), caches));
  Tensor out = ops::add(h, ffn_forward(norm_2_->forward(h)));
  return out;
}

}  // namespace tesseract::nn

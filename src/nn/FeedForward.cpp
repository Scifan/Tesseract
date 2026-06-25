#include "tesseract/nn/FeedForward.hpp"

#include <utility>

#include "tesseract/ops/Activation.hpp"
#include "tesseract/quant/Scheme.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

FeedForward::FeedForward(int64_t d_model, int64_t d_ff, bool use_bias, DType dtype)
    : d_model_(d_model), d_ff_(d_ff) {
  TESSERACT_CHECK(d_model > 0 && d_ff > 0,
                  "FeedForward: d_model ({}) and d_ff ({}) must be positive",
                  d_model, d_ff);

  gate_proj_ = std::make_shared<Linear>(d_model, d_ff,    use_bias, dtype);
  up_proj_   = std::make_shared<Linear>(d_model, d_ff,    use_bias, dtype);
  down_proj_ = std::make_shared<Linear>(d_ff,    d_model, use_bias, dtype);

  register_module("gate_proj", gate_proj_);
  register_module("up_proj",   up_proj_);
  register_module("down_proj", down_proj_);
}

Tensor FeedForward::forward(const Tensor& x) {
  TESSERACT_CHECK(x.rank() >= 2,
                  "FeedForward::forward: expected rank >= 2 input [..., D], got {}",
                  x.shape().to_string());
  TESSERACT_CHECK(x.shape()[x.rank() - 1] == d_model_,
                  "FeedForward::forward: input last dim {} != d_model {}",
                  x.shape()[x.rank() - 1], d_model_);

  // SwiGLU's activation tail (`sigmoid(gate)` + `mul` + `mul`) used to
  // unroll into three element-wise kernels per FFN forward, i.e. six
  // launches total when combined with the three projections. Wave 4.1
  // (B-025) introduced `ops::swiglu_silu_gate` which fuses the
  // activation tail into a single memory-bandwidth-bound pass on
  // CUDA (and a single loop on CPU). That drops the FFN to four
  // element-wise-and-matmul launches:
  //
  //   gate_matmul → up_matmul → swiglu_silu_gate(fused) → down_matmul
  //
  // `swiglu_silu_gate` falls back to the composite (`mul(gate,
  // sigmoid(gate)) * up`) transparently when autograd is active or
  // when either operand is non-contiguous, so the forward is
  // numerically identical to the old composite on every tape-building
  // path — the fused kernel is strictly an inference-time
  // optimization, same policy as `rms_norm`.
  Tensor gate = gate_proj_->forward(x);                  // [..., d_ff]
  Tensor up   = up_proj_->forward(x);                    // [..., d_ff]
  Tensor h    = ops::swiglu_silu_gate(gate, up);         // [..., d_ff]
  return down_proj_->forward(h);                          // [..., d_model]
}

void FeedForward::quantize_(const quant::Scheme& scheme) {
  // Same template as MultiHeadAttention::quantize_: dynamic_cast to
  // FP `Linear`, short-circuit if the slot already holds something
  // else (already-quantized / custom), and otherwise swap via
  // `replace_module` under the original registration name.
  auto try_swap = [&](const std::string& slot,
                      std::shared_ptr<Module>& holder) {
    auto as_fp = std::dynamic_pointer_cast<Linear>(holder);
    if (!as_fp) return;
    auto replacement = quant::quantize_linear(*as_fp, scheme);
    holder = replacement;
    replace_module(slot, replacement);
  };
  try_swap("gate_proj", gate_proj_);
  try_swap("up_proj",   up_proj_);
  try_swap("down_proj", down_proj_);
}

}  // namespace tesseract::nn

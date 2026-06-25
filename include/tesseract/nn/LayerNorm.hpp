#pragma once

#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// Standard LayerNorm (Ba et al., 2016) over the last `D` features with
// per-channel learnable affine:
//
//     y = (x - mean(x, dim=-1)) / sqrt(var(x, dim=-1, biased) + eps) * weight + bias
//
// Mirrors `torch.nn.LayerNorm(normalized_shape=D, eps, elementwise_affine=True, bias=True)`:
// - `weight` is registered as a parameter and initialized to 1.
// - `bias` is registered as a parameter and initialized to 0 when `use_bias=true`.
//   Passing `use_bias=false` drops the bias parameter entirely — matching the
//   RoBERTa/PaLM family of models that use bias-free LN (called `elementwise_affine`
//   without bias in PyTorch terminology).
//
// Forward accepts any rank >= 1 input whose last dim equals `normalized_dim`.
// Device-agnostic: `Module::to(Device)` migrates `weight`/`bias` and all
// subsequent forwards run on-device via the composite primitives that
// `ops::layer_norm` dispatches through.
class LayerNorm : public Module {
 public:
  LayerNorm(int64_t normalized_dim, double eps = 1e-5, bool use_bias = true,
            DType dtype = DType::Float32);

  Tensor forward(const Tensor& x) override;

  const Tensor& weight() const { return weight_; }
  const Tensor& bias() const { return bias_; }   // undefined when use_bias=false
  int64_t normalized_dim() const noexcept { return normalized_dim_; }
  double eps() const noexcept { return eps_; }
  bool has_bias() const noexcept { return bias_.defined(); }

 private:
  int64_t normalized_dim_;
  double eps_;
  Tensor weight_;  // [normalized_dim_]
  Tensor bias_;    // [normalized_dim_] or undefined
};

}  // namespace tesseract::nn

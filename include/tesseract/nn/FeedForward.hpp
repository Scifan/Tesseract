#pragma once

#include <memory>

#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"
#include "tesseract/quant/Scheme.hpp"

namespace tesseract::nn {

// Llama-style SwiGLU feed-forward network:
//
//   ffn(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
//   silu(z) = z * sigmoid(z)
//
// No activation on `up_proj`; the gate branch's SiLU provides the
// nonlinearity, and the element-wise product between the two branches
// is the "gated linear unit" (GLU) half of the name. Hidden dim is
// `d_ff`; Llama typically uses `d_ff ≈ 8/3 · d_model` rounded to a
// multiple of 256, but this module exposes it as a raw parameter so
// the caller can tune without extra policy here.
//
// Forward accepts any rank >= 2 input whose last dim equals `d_model`
// (Linear is rank >= 2 since M2K); output shape matches the input.
class FeedForward : public Module {
 public:
  FeedForward(int64_t d_model, int64_t d_ff,
              bool use_bias = true,
              DType dtype = DType::Float32);

  Tensor forward(const Tensor& x) override;

  int64_t d_model() const noexcept { return d_model_; }
  int64_t d_ff() const noexcept { return d_ff_; }

  // Wave 3.3 (B-021) in-place quantization of the three SwiGLU
  // projections. Same contract as `MultiHeadAttention::quantize_`:
  // FP `Linear` children are replaced with the scheme-specific
  // quantized drop-in, already-quantized children are left alone,
  // and `named_parameters()` walk order is preserved.
  void quantize_(const quant::Scheme& scheme);

  // Accessors for testing. Before `quantize_` each returns a
  // `Linear`; afterwards they return the quantized drop-in.
  const std::shared_ptr<Module>& gate_proj() const { return gate_proj_; }
  const std::shared_ptr<Module>& up_proj()   const { return up_proj_; }
  const std::shared_ptr<Module>& down_proj() const { return down_proj_; }

 private:
  int64_t d_model_;
  int64_t d_ff_;

  // Held as `shared_ptr<Module>` so `quantize_` can swap in a
  // `QuantizedLinear*` drop-in; `forward()` goes through
  // `Module::forward` either way. See MultiHeadAttention.hpp for
  // the same rationale.
  std::shared_ptr<Module> gate_proj_;  // d_model -> d_ff
  std::shared_ptr<Module> up_proj_;    // d_model -> d_ff
  std::shared_ptr<Module> down_proj_;  // d_ff -> d_model
};

}  // namespace tesseract::nn

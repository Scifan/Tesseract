#pragma once

#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// Llama-style RMSNorm: `y = (x / sqrt(mean(x²) + eps)) * weight`.
// `weight` is a [D] affine scale, registered as a parameter and
// initialized to 1 (identity scale) — Llama convention.
//
// Wraps `ops::rms_norm`; forward accepts any rank >= 1 input whose
// last dim equals the configured normalized dim `D`. The module is
// device-agnostic: `Module::to(Device)` moves `weight_` onto the
// target device and all subsequent forwards run on-device via the
// composite primitives that `ops::rms_norm` dispatches through.
class RMSNorm : public Module {
 public:
  RMSNorm(int64_t normalized_dim, double eps = 1e-5,
          DType dtype = DType::Float32);

  Tensor forward(const Tensor& x) override;

  const Tensor& weight() const { return weight_; }
  int64_t normalized_dim() const noexcept { return normalized_dim_; }
  double eps() const noexcept { return eps_; }

 private:
  int64_t normalized_dim_;
  double eps_;
  Tensor weight_;  // [normalized_dim_]
};

}  // namespace tesseract::nn

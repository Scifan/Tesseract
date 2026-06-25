#include "tesseract/nn/LayerNorm.hpp"

#include "tesseract/ops/Normalization.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

LayerNorm::LayerNorm(int64_t normalized_dim, double eps, bool use_bias, DType dtype)
    : normalized_dim_(normalized_dim), eps_(eps) {
  TESSERACT_CHECK(normalized_dim > 0,
                  "LayerNorm: normalized_dim must be positive, got {}", normalized_dim);
  TESSERACT_CHECK(eps > 0.0, "LayerNorm: eps must be > 0, got {}", eps);

  // Standard PyTorch init: weight=1, bias=0. `register_parameter` flips
  // `requires_grad=true` so the optimizer updates them like any other leaf.
  weight_ = Tensor::ones({normalized_dim}, dtype);
  register_parameter("weight", weight_);
  if (use_bias) {
    bias_ = Tensor::zeros({normalized_dim}, dtype);
    register_parameter("bias", bias_);
  }
}

Tensor LayerNorm::forward(const Tensor& x) {
  TESSERACT_CHECK(x.rank() >= 1,
                  "LayerNorm::forward: expected rank >= 1 input, got {}",
                  x.shape().to_string());
  TESSERACT_CHECK(x.shape()[x.rank() - 1] == normalized_dim_,
                  "LayerNorm::forward: last dim of input ({}) != normalized_dim ({})",
                  x.shape()[x.rank() - 1], normalized_dim_);
  return ops::layer_norm(x, weight_, bias_, eps_);
}

}  // namespace tesseract::nn

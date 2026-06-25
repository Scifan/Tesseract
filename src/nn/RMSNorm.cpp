#include "tesseract/nn/RMSNorm.hpp"

#include "tesseract/ops/Normalization.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

RMSNorm::RMSNorm(int64_t normalized_dim, double eps, DType dtype)
    : normalized_dim_(normalized_dim), eps_(eps) {
  TESSERACT_CHECK(normalized_dim > 0,
                  "RMSNorm: normalized_dim must be positive, got {}", normalized_dim);
  TESSERACT_CHECK(eps > 0.0, "RMSNorm: eps must be > 0, got {}", eps);

  // Llama convention: weight initialized to 1 so the first forward pass
  // computes a plain RMS normalization (identity affine). Using
  // `Tensor::ones` rather than `empty + fill_` keeps the initializer
  // device-agnostic — nothing reaches back into platform-specific fill
  // code here; `register_parameter` then flips `requires_grad=true` so
  // the optimizer can update it like any other leaf parameter.
  weight_ = Tensor::ones({normalized_dim}, dtype);
  register_parameter("weight", weight_);
}

Tensor RMSNorm::forward(const Tensor& x) {
  TESSERACT_CHECK(x.rank() >= 1,
                  "RMSNorm::forward: expected rank >= 1 input, got {}",
                  x.shape().to_string());
  TESSERACT_CHECK(x.shape()[x.rank() - 1] == normalized_dim_,
                  "RMSNorm::forward: last dim of input ({}) != normalized_dim ({})",
                  x.shape()[x.rank() - 1], normalized_dim_);
  return ops::rms_norm(x, weight_, eps_);
}

}  // namespace tesseract::nn

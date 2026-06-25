#include "tesseract/nn/BatchNorm.hpp"

#include "tesseract/ops/Normalization.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

namespace {

// Shared init: the two BN variants differ only in which ranks they
// accept at forward time, so the parameter + buffer layout lives in one
// helper. `weight` is initialised to 1, `bias` to 0, `running_mean` to
// 0, `running_var` to 1 — verbatim `torch.nn.BatchNorm{1,2}d` defaults.
struct BnInit {
  Tensor weight;
  Tensor bias;
  Tensor running_mean;
  Tensor running_var;
};

BnInit make_bn_state(int64_t C, bool affine, DType dtype, const char* name) {
  TESSERACT_CHECK(C > 0, "{}: num_features must be positive, got {}", name, C);
  BnInit s;
  if (affine) {
    s.weight = Tensor::ones({C},  dtype);
    s.bias   = Tensor::zeros({C}, dtype);
  }
  s.running_mean = Tensor::zeros({C}, dtype);
  s.running_var  = Tensor::ones({C},  dtype);
  return s;
}

}  // namespace

// --- BatchNorm1d -----------------------------------------------------------

BatchNorm1d::BatchNorm1d(int64_t num_features, double eps, double momentum,
                         bool affine, DType dtype)
    : num_features_(num_features), eps_(eps), momentum_(momentum), affine_(affine) {
  TESSERACT_CHECK(eps > 0.0, "BatchNorm1d: eps must be > 0, got {}", eps);
  TESSERACT_CHECK(momentum >= 0.0 && momentum <= 1.0,
                  "BatchNorm1d: momentum must be in [0, 1], got {}", momentum);

  auto s = make_bn_state(num_features_, affine_, dtype, "BatchNorm1d");
  weight_ = std::move(s.weight);
  bias_   = std::move(s.bias);
  running_mean_ = std::move(s.running_mean);
  running_var_  = std::move(s.running_var);

  if (affine_) {
    register_parameter("weight", weight_);
    register_parameter("bias",   bias_);
  }
  register_buffer("running_mean", running_mean_);
  register_buffer("running_var",  running_var_);
}

Tensor BatchNorm1d::forward(const Tensor& x) {
  TESSERACT_CHECK(x.rank() == 2 || x.rank() == 3,
                  "BatchNorm1d::forward: expected rank-2 [N,C] or rank-3 "
                  "[N,C,L] input, got {}", x.shape().to_string());
  TESSERACT_CHECK(x.shape()[1] == num_features_,
                  "BatchNorm1d::forward: channel dim ({}) != num_features ({})",
                  x.shape()[1], num_features_);
  return ops::batch_norm(x, weight_, bias_, running_mean_, running_var_,
                         /*training=*/is_training(), momentum_, eps_);
}

// --- BatchNorm2d -----------------------------------------------------------

BatchNorm2d::BatchNorm2d(int64_t num_features, double eps, double momentum,
                         bool affine, DType dtype)
    : num_features_(num_features), eps_(eps), momentum_(momentum), affine_(affine) {
  TESSERACT_CHECK(eps > 0.0, "BatchNorm2d: eps must be > 0, got {}", eps);
  TESSERACT_CHECK(momentum >= 0.0 && momentum <= 1.0,
                  "BatchNorm2d: momentum must be in [0, 1], got {}", momentum);

  auto s = make_bn_state(num_features_, affine_, dtype, "BatchNorm2d");
  weight_ = std::move(s.weight);
  bias_   = std::move(s.bias);
  running_mean_ = std::move(s.running_mean);
  running_var_  = std::move(s.running_var);

  if (affine_) {
    register_parameter("weight", weight_);
    register_parameter("bias",   bias_);
  }
  register_buffer("running_mean", running_mean_);
  register_buffer("running_var",  running_var_);
}

Tensor BatchNorm2d::forward(const Tensor& x) {
  TESSERACT_CHECK(x.rank() == 4,
                  "BatchNorm2d::forward: expected rank-4 [N,C,H,W] input, got {}",
                  x.shape().to_string());
  TESSERACT_CHECK(x.shape()[1] == num_features_,
                  "BatchNorm2d::forward: channel dim ({}) != num_features ({})",
                  x.shape()[1], num_features_);
  return ops::batch_norm(x, weight_, bias_, running_mean_, running_var_,
                         /*training=*/is_training(), momentum_, eps_);
}

}  // namespace tesseract::nn

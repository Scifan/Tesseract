#pragma once

#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// y = x @ weight^T + bias. Matches PyTorch's convention: `weight` has shape
// [out_features, in_features] so that the call site reads as y = Wx + b and
// fan_in / fan_out based initializers can interpret the layout directly.
// The transpose is materialized by `ops::transpose`, whose backward Node
// (TransposeBackward) keeps the graph connected back to `weight`.
class Linear : public Module {
 public:
  Linear(int64_t in_features, int64_t out_features, bool use_bias = true,
         DType dtype = DType::Float32);

  Tensor forward(const Tensor& x) override;

  const Tensor& weight() const { return weight_; }
  const Tensor& bias() const { return bias_; }
  bool has_bias() const noexcept { return use_bias_; }

 private:
  int64_t in_features_;
  int64_t out_features_;
  bool use_bias_;
  Tensor weight_;  // [out_features, in_features]
  Tensor bias_;    // [out_features]
};

}  // namespace tesseract::nn

#pragma once

#include <vector>

#include "tesseract/core/Tensor.hpp"

namespace tesseract::optim {

// Base class for optimizers. Subclasses implement `step()` to consume each
// parameter's `.grad` and update its storage in-place.
class Optimizer {
 public:
  explicit Optimizer(std::vector<Tensor> params) : params_(std::move(params)) {}
  virtual ~Optimizer() = default;
  Optimizer(const Optimizer&) = delete;
  Optimizer& operator=(const Optimizer&) = delete;

  // Applies an update step using the currently-stored `.grad` on each param.
  virtual void step() = 0;

  // Clears `.grad` on every parameter.
  void zero_grad();

  const std::vector<Tensor>& parameters() const noexcept { return params_; }

 protected:
  std::vector<Tensor> params_;
};

}  // namespace tesseract::optim

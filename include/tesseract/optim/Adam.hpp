#pragma once

#include <vector>

#include "tesseract/optim/Optimizer.hpp"

namespace tesseract::optim {

// Adam (Kingma & Ba, 2015). Maintains first- and second-moment estimates
// per parameter; update rule uses bias-corrected estimates.
class Adam : public Optimizer {
 public:
  Adam(std::vector<Tensor> params, double lr = 1e-3, double beta1 = 0.9,
       double beta2 = 0.999, double eps = 1e-8);

  void step() override;

 private:
  double lr_;
  double beta1_;
  double beta2_;
  double eps_;
  int64_t step_count_{0};
  std::vector<Tensor> m_;  // first moment
  std::vector<Tensor> v_;  // second moment
};

}  // namespace tesseract::optim

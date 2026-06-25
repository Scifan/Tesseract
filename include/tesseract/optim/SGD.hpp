#pragma once

#include <vector>

#include "tesseract/optim/Optimizer.hpp"

namespace tesseract::optim {

// Stochastic gradient descent with optional (heavy-ball) momentum. Update
// rule:
//   v_{t+1} = momentum * v_t + grad
//   w_{t+1} = w_t - lr * v_{t+1}
//
// When momentum == 0 this reduces to vanilla SGD.
class SGD : public Optimizer {
 public:
  SGD(std::vector<Tensor> params, double lr, double momentum = 0.0);

  void step() override;

 private:
  double lr_;
  double momentum_;
  std::vector<Tensor> velocity_;  // one per parameter; lazily allocated
};

}  // namespace tesseract::optim

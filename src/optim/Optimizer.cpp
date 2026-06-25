#include "tesseract/optim/Optimizer.hpp"

#include "tesseract/autograd/AutogradMeta.hpp"

namespace tesseract::optim {

void Optimizer::zero_grad() {
  for (auto& p : params_) {
    auto* am = p.mutable_autograd_meta();
    am->grad = Tensor{};
  }
}

}  // namespace tesseract::optim

#include "tesseract/nn/Loss.hpp"

#include "tesseract/ops/Loss.hpp"

namespace tesseract::nn {

Tensor CrossEntropyLoss::forward(const Tensor& logits, const Tensor& targets) const {
  return ops::cross_entropy_with_logits(logits, targets);
}

}  // namespace tesseract::nn

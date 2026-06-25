#pragma once

#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// Cross-entropy over raw logits (internally: log-softmax + NLL). Mean
// reduction over the batch. Stateless; exists as a Module for symmetry with
// Sequential / training-loop wiring.
class CrossEntropyLoss : public Module {
 public:
  // Not a pure Tensor->Tensor op; dedicated entry point takes targets.
  Tensor forward(const Tensor& logits, const Tensor& targets) const;

 private:
  using Module::forward;
};

}  // namespace tesseract::nn

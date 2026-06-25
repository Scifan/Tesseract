#pragma once

#include <cstdint>
#include <memory>

#include "tesseract/core/Tensor.hpp"

namespace tesseract {

class Node;

// Per-tensor autograd state. Exists only on tensors that either participate in
// a computation graph (via `requires_grad=true`) or are outputs of an op that
// recorded backward information. Leaf tensors with requires_grad accumulate
// their accumulated gradient into `grad`; intermediate tensors carry a
// `grad_fn` that knows how to propagate gradients backward.
//
// For M0 we assume single-output ops, so `output_nr` is always 0 at the
// producer side. It remains as a placeholder for the multi-output extension.
class AutogradMeta {
 public:
  AutogradMeta() = default;

  bool requires_grad{false};
  std::shared_ptr<Node> grad_fn;
  Tensor grad;
  uint32_t output_nr{0};
};

}  // namespace tesseract

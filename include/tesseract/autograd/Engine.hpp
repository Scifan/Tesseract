#pragma once

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"

namespace tesseract {

// Runs the backward pass starting at `root`. Accumulates gradients into the
// `.grad` fields of leaf tensors (those with `requires_grad == true` and no
// `grad_fn`). If `grad` is undefined, a ones-tensor of the same shape / dtype
// as `root` is used (appropriate for scalar losses).
//
// Re-entrant calls overwrite/accumulate into existing `.grad` values; callers
// should reset leaf `.grad` (via optimizer `zero_grad`) between passes.
class Engine {
 public:
  static void backward(const Tensor& root, Tensor grad = Tensor{});
};

}  // namespace tesseract

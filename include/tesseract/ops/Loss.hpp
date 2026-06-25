#pragma once

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// Cross-entropy loss from raw logits. Logits must have shape [N, C] and be
// Float32/Float64. Targets must have shape [N], dtype Int64, with values in
// [0, C).
//
// Returns the mean loss over the batch as a 0-D scalar tensor. The forward
// implementation is numerically stable (log-sum-exp).
Tensor cross_entropy_with_logits(const Tensor& logits, const Tensor& targets);

// Fused CE backward kernel: `d_logits = (softmax(logits) - one_hot(targets))
// * grad / N`. Re-derives the softmax from `logits` instead of saving it,
// since the graph interpreter calls this without any autograd context.
// Does not record into the autograd tape or the active GraphScope.
Tensor cross_entropy_with_logits_backward(const Tensor& logits,
                                          const Tensor& targets,
                                          const Tensor& grad);

}  // namespace tesseract::ops

#pragma once

#include <cstdint>

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// Numerically stable softmax along a single axis (subtract-max trick).
Tensor softmax(const Tensor& x, int64_t dim);

// log-softmax along a single axis; equivalent to log(softmax(x, dim)) but
// numerically stable.
Tensor log_softmax(const Tensor& x, int64_t dim);

}  // namespace tesseract::ops

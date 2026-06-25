#pragma once

#include <initializer_list>
#include <span>

#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"

// Autograd-aware wrappers around Tensor's view-family methods. Forward logic
// is entirely delegated to `Tensor::{view,reshape,permute,transpose,
// contiguous,clone}`; these entry points additionally attach the appropriate
// backward `Node` when gradient recording is enabled. Prefer these over the
// raw `Tensor` methods whenever the input may participate in a graph
// (optimizer parameters, inputs to loss functions, etc.).
namespace tesseract::ops {

Tensor view(const Tensor& self, Shape new_shape);
Tensor reshape(const Tensor& self, Shape new_shape);

Tensor permute(const Tensor& self, std::span<const int64_t> axes);
Tensor permute(const Tensor& self, std::initializer_list<int64_t> axes);

Tensor transpose(const Tensor& self, int64_t dim_a, int64_t dim_b);

Tensor contiguous(const Tensor& self);
Tensor clone(const Tensor& self);

}  // namespace tesseract::ops

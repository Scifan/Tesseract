#pragma once

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// Element-wise binary arithmetic with NumPy-style broadcasting.
// Both operands must share the same dtype; the output dtype equals the input
// dtype. Only implemented dtypes (Float32/Float64/Int32/Int64) are supported.

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// Unary: negate in place of the input (returns a new tensor).
Tensor neg(const Tensor& a);

// Broadcast-aware reduction used by autograd: if `src.shape` is broadcast-
// reducible to `target_shape`, sum-reduce the broadcast axes and return a new
// contiguous tensor of `target_shape`.
Tensor reduce_to_shape(const Tensor& src, const Shape& target_shape);

// Broadcast `src` to `target_shape` by materializing a new contiguous tensor.
// Requires `src.shape` to be broadcast-compatible with `target_shape` (each
// dim equal or 1, with extra leading 1-dims implicit).
Tensor broadcast_to(const Tensor& src, const Shape& target_shape);

}  // namespace tesseract::ops

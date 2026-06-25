#pragma once

#include <cstdint>

#include "tesseract/core/Shape.hpp"

namespace tesseract::ops {

// Compute the NumPy-style broadcast shape of two shapes. Throws on
// incompatible shapes.
Shape broadcast_shape(const Shape& a, const Shape& b);

// Align an input's (shape, strides) to `out_shape`'s rank with stride==0 for
// broadcasted dims. Result has the same rank as out_shape.
//
// Preconditions:
//   in_shape.rank() <= out_shape.rank()
//   each of in_shape's trailing dims matches out_shape or equals 1
void align_for_broadcast(const Shape& in_shape, const Shape& in_strides,
                         const Shape& out_shape, Shape& out_strides);

// True if `a` can be broadcast-reduced to `b` (i.e. `b` is a shape that `a`
// could broadcast from). Used by autograd to decide when to sum-reduce a
// gradient tensor back to the original input's shape.
bool is_broadcastable_to(const Shape& from, const Shape& to) noexcept;

}  // namespace tesseract::ops

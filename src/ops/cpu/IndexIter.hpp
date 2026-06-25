#pragma once

#include <array>
#include <cstdint>

#include "tesseract/core/Shape.hpp"

namespace tesseract::ops::detail {

using IndexArray = std::array<int64_t, Shape::kMaxRank>;

// Iterate over every logical position in `shape` in row-major order, invoking
// `fn(flat_index, const IndexArray& idx)` for each position. `idx[0..rank-1]`
// holds the N-D index.
template <typename F>
inline void for_each_index(const Shape& shape, F&& fn) {
  const std::size_t r = shape.rank();
  IndexArray idx{};
  if (r == 0) {
    fn(int64_t{0}, idx);
    return;
  }
  const int64_t n = shape.numel();
  for (int64_t flat = 0; flat < n; ++flat) {
    fn(flat, idx);
    for (std::size_t d = r; d-- > 0;) {
      if (++idx[d] < shape[d]) break;
      idx[d] = 0;
      if (d == 0) return;
    }
  }
}

// offset = sum(idx[d] * strides[d])  — for 0..strides.rank()-1.
inline int64_t offset_of(const IndexArray& idx, const Shape& strides) noexcept {
  int64_t off = 0;
  const std::size_t r = strides.rank();
  for (std::size_t d = 0; d < r; ++d) off += idx[d] * strides[d];
  return off;
}

}  // namespace tesseract::ops::detail

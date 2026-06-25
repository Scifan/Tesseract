#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace tesseract {

// A small, stack-allocated container of dimension values. Deep-learning
// tensors almost never exceed rank 8 (video + batch = 6, Transformer with
// extra broadcast dims rarely > 6), so we avoid heap allocation entirely.
//
// Used for both shapes (logical sizes along each axis) and strides (in
// elements, not bytes). Values may be negative for strides.
class Shape {
 public:
  static constexpr std::size_t kMaxRank = 8;

  using value_type = int64_t;

  constexpr Shape() = default;

  Shape(std::initializer_list<int64_t> dims);
  explicit Shape(std::span<const int64_t> dims);
  explicit Shape(const std::vector<int64_t>& dims);

  template <typename It>
  Shape(It first, It last) {
    while (first != last) {
      push_back(*first++);
    }
  }

  // Sentinel used for statically-unknown dimensions (aligns with MLIR
  // `ShapedType::kDynamic`). M0 only uses this as a marker; runtime tensors
  // always have concrete shapes.
  static constexpr int64_t kDynamic = -1;

  std::size_t rank() const noexcept { return rank_; }
  bool empty() const noexcept { return rank_ == 0; }

  int64_t operator[](std::size_t i) const noexcept { return dims_[i]; }
  int64_t& operator[](std::size_t i) noexcept { return dims_[i]; }

  int64_t at(std::size_t i) const;

  int64_t front() const noexcept { return dims_[0]; }
  int64_t back() const noexcept { return dims_[rank_ - 1]; }

  const int64_t* data() const noexcept { return dims_.data(); }
  int64_t* data() noexcept { return dims_.data(); }

  std::span<const int64_t> span() const noexcept { return {dims_.data(), rank_}; }

  auto begin() const noexcept { return dims_.data(); }
  auto end() const noexcept { return dims_.data() + rank_; }
  auto begin() noexcept { return dims_.data(); }
  auto end() noexcept { return dims_.data() + rank_; }

  void push_back(int64_t d);
  void pop_back();
  void clear() noexcept { rank_ = 0; }
  void resize(std::size_t n);

  // Product of all dimensions; 0-dim shapes have numel == 1 (scalar).
  int64_t numel() const noexcept;

  // True iff every dim is >= 0 (no kDynamic markers).
  bool is_fully_static() const noexcept;

  // Contiguous row-major strides for this shape.
  Shape contiguous_strides() const;

  // Index accounting for the strided layout described by `strides`.
  int64_t flat_offset(std::span<const int64_t> indices, const Shape& strides) const;

  std::string to_string() const;

  friend bool operator==(const Shape& a, const Shape& b) noexcept;
  friend bool operator!=(const Shape& a, const Shape& b) noexcept { return !(a == b); }

 private:
  std::array<int64_t, kMaxRank> dims_{};
  std::size_t rank_{0};

  void check_rank(std::size_t n) const;
};

std::ostream& operator<<(std::ostream& os, const Shape& s);

}  // namespace tesseract

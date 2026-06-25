#include "tesseract/core/Shape.hpp"

#include <ostream>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "tesseract/utils/Logging.hpp"

namespace tesseract {

Shape::Shape(std::initializer_list<int64_t> dims) {
  TESSERACT_CHECK(dims.size() <= kMaxRank, "shape rank {} exceeds kMaxRank={}", dims.size(),
                  kMaxRank);
  for (int64_t d : dims) {
    dims_[rank_++] = d;
  }
}

Shape::Shape(std::span<const int64_t> dims) {
  TESSERACT_CHECK(dims.size() <= kMaxRank, "shape rank {} exceeds kMaxRank={}", dims.size(),
                  kMaxRank);
  for (int64_t d : dims) {
    dims_[rank_++] = d;
  }
}

Shape::Shape(const std::vector<int64_t>& dims)
    : Shape(std::span<const int64_t>(dims.data(), dims.size())) {}

int64_t Shape::at(std::size_t i) const {
  check_rank(i + 1);
  return dims_[i];
}

void Shape::push_back(int64_t d) {
  TESSERACT_CHECK(rank_ < kMaxRank, "shape rank would exceed kMaxRank={}", kMaxRank);
  dims_[rank_++] = d;
}

void Shape::pop_back() {
  TESSERACT_CHECK(rank_ > 0, "pop_back on empty shape");
  --rank_;
}

void Shape::resize(std::size_t n) {
  TESSERACT_CHECK(n <= kMaxRank, "resize({}) exceeds kMaxRank={}", n, kMaxRank);
  if (n > rank_) {
    for (std::size_t i = rank_; i < n; ++i) dims_[i] = 0;
  }
  rank_ = n;
}

int64_t Shape::numel() const noexcept {
  int64_t n = 1;
  for (std::size_t i = 0; i < rank_; ++i) {
    n *= dims_[i];
  }
  return n;
}

bool Shape::is_fully_static() const noexcept {
  for (std::size_t i = 0; i < rank_; ++i) {
    if (dims_[i] < 0) return false;
  }
  return true;
}

Shape Shape::contiguous_strides() const {
  Shape s;
  s.rank_ = rank_;
  if (rank_ == 0) return s;
  s.dims_[rank_ - 1] = 1;
  for (std::size_t i = rank_ - 1; i > 0; --i) {
    s.dims_[i - 1] = s.dims_[i] * dims_[i];
  }
  return s;
}

int64_t Shape::flat_offset(std::span<const int64_t> indices, const Shape& strides) const {
  TESSERACT_CHECK(indices.size() == rank_, "flat_offset: indices rank {} != shape rank {}",
                  indices.size(), rank_);
  TESSERACT_CHECK(strides.rank() == rank_,
                  "flat_offset: strides rank {} != shape rank {}", strides.rank(), rank_);
  int64_t off = 0;
  for (std::size_t i = 0; i < rank_; ++i) {
    off += indices[i] * strides[i];
  }
  return off;
}

std::string Shape::to_string() const {
  if (rank_ == 0) return "[]";
  std::string out = "[";
  for (std::size_t i = 0; i < rank_; ++i) {
    if (i > 0) out += ", ";
    out += fmt::format("{}", dims_[i]);
  }
  out += "]";
  return out;
}

bool operator==(const Shape& a, const Shape& b) noexcept {
  if (a.rank_ != b.rank_) return false;
  for (std::size_t i = 0; i < a.rank_; ++i) {
    if (a.dims_[i] != b.dims_[i]) return false;
  }
  return true;
}

std::ostream& operator<<(std::ostream& os, const Shape& s) {
  return os << s.to_string();
}

void Shape::check_rank(std::size_t n) const {
  TESSERACT_CHECK(n <= rank_, "index {} out of range for rank {}", n - 1, rank_);
}

}  // namespace tesseract

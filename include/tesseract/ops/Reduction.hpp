#pragma once

#include <cstdint>
#include <optional>

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// Reductions. When `dim` is std::nullopt, reduce across every element and
// return a 0-D (scalar) tensor. When `dim` is provided, reduce along that
// single axis. `keepdim` controls whether the reduced axis stays as size 1
// in the output (for broadcast-friendly downstream use).

Tensor sum(const Tensor& x);
Tensor sum(const Tensor& x, int64_t dim, bool keepdim = false);

Tensor mean(const Tensor& x);
Tensor mean(const Tensor& x, int64_t dim, bool keepdim = false);

// `max` only returns the values (no indices) at M0. Index support lands with
// argmax (T3+).
Tensor max(const Tensor& x);
Tensor max(const Tensor& x, int64_t dim, bool keepdim = false);

}  // namespace tesseract::ops

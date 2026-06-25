#pragma once

#include "tesseract/core/DType.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract {

// Dispatch a templated lambda over all implemented scalar dtypes (including
// Bool). The lambda must be invocable as `f.template operator()<T>()`.
template <typename F>
inline void dispatch_all(DType dt, F&& f) {
  switch (dt) {
    case DType::Float32: f.template operator()<float>(); return;
    case DType::Float64: f.template operator()<double>(); return;
    case DType::Int32:   f.template operator()<int32_t>(); return;
    case DType::Int64:   f.template operator()<int64_t>(); return;
    case DType::Bool:    f.template operator()<bool>(); return;
    default:
      TESSERACT_THROW("dispatch_all: unsupported dtype {}", dtype_name(dt));
  }
}

// Dispatch over numeric (non-bool) dtypes.
template <typename F>
inline void dispatch_numeric(DType dt, F&& f) {
  switch (dt) {
    case DType::Float32: f.template operator()<float>(); return;
    case DType::Float64: f.template operator()<double>(); return;
    case DType::Int32:   f.template operator()<int32_t>(); return;
    case DType::Int64:   f.template operator()<int64_t>(); return;
    default:
      TESSERACT_THROW("dispatch_numeric: unsupported dtype {}", dtype_name(dt));
  }
}

// Dispatch over floating-point dtypes only.
template <typename F>
inline void dispatch_float(DType dt, F&& f) {
  switch (dt) {
    case DType::Float32: f.template operator()<float>(); return;
    case DType::Float64: f.template operator()<double>(); return;
    default:
      TESSERACT_THROW("dispatch_float: expected floating-point dtype, got {}",
                      dtype_name(dt));
  }
}

// Variants that additionally cover the software-emulated half-precision
// types. These are intentionally *opt-in* — most numeric kernels do not
// want to be instantiated with `Half` / `BFloat16` because they rely on
// things like `std::exp<T>` that only work for built-in floats. The
// elementwise arithmetic path and matmul use these explicitly.
template <typename F>
inline void dispatch_numeric_with_half(DType dt, F&& f) {
  switch (dt) {
    case DType::Float32:  f.template operator()<float>();    return;
    case DType::Float64:  f.template operator()<double>();   return;
    case DType::Float16:  f.template operator()<Half>();     return;
    case DType::BFloat16: f.template operator()<BFloat16>(); return;
    case DType::Int32:    f.template operator()<int32_t>();  return;
    case DType::Int64:    f.template operator()<int64_t>();  return;
    default:
      TESSERACT_THROW("dispatch_numeric_with_half: unsupported dtype {}", dtype_name(dt));
  }
}

template <typename F>
inline void dispatch_float_with_half(DType dt, F&& f) {
  switch (dt) {
    case DType::Float32:  f.template operator()<float>();    return;
    case DType::Float64:  f.template operator()<double>();   return;
    case DType::Float16:  f.template operator()<Half>();     return;
    case DType::BFloat16: f.template operator()<BFloat16>(); return;
    default:
      TESSERACT_THROW("dispatch_float_with_half: expected floating-point dtype, got {}",
                      dtype_name(dt));
  }
}

}  // namespace tesseract

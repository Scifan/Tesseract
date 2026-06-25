#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string_view>

#include "tesseract/core/Float16.hpp"

namespace tesseract {

// Enumeration of scalar element types a Tensor may carry.
// Ordering is stable; new dtypes are appended. Values are explicit so that
// serialized representations do not drift across versions.
enum class DType : uint8_t {
  // Currently implemented
  Float32 = 0,
  Float64 = 1,
  Int32 = 2,
  Int64 = 3,
  Bool = 4,

  // Reserved for future milestones (kept as enumerators so that switches can
  // be made exhaustive ahead of time). Do not use before implementation.
  Float16 = 5,
  BFloat16 = 6,
  Float8_E4M3 = 7,
  Float8_E5M2 = 8,
  Int8 = 9,
  UInt8 = 10,
  Int4 = 11,

  kNumDTypes,
};

// Size of one element of the given dtype in bytes.
size_t dtype_size(DType dt) noexcept;

// Human-readable short name (e.g. "f32", "i64").
std::string_view dtype_name(DType dt) noexcept;

// True if the dtype is currently implemented by the runtime.
bool dtype_is_implemented(DType dt) noexcept;

// True for floating-point dtypes (including Float16/BFloat16/FP8).
bool dtype_is_floating(DType dt) noexcept;

// True for signed/unsigned integer dtypes (including Bool).
bool dtype_is_integral(DType dt) noexcept;

std::ostream& operator<<(std::ostream& os, DType dt);

// Compile-time mapping from C++ type to DType.
template <typename T>
struct CppTypeToDType;

#define TESSERACT_DEFINE_CPP_TO_DTYPE(CppT, EnumV) \
  template <>                                      \
  struct CppTypeToDType<CppT> {                    \
    static constexpr DType value = DType::EnumV;   \
  }

TESSERACT_DEFINE_CPP_TO_DTYPE(float, Float32);
TESSERACT_DEFINE_CPP_TO_DTYPE(double, Float64);
TESSERACT_DEFINE_CPP_TO_DTYPE(int32_t, Int32);
TESSERACT_DEFINE_CPP_TO_DTYPE(int64_t, Int64);
TESSERACT_DEFINE_CPP_TO_DTYPE(bool, Bool);
TESSERACT_DEFINE_CPP_TO_DTYPE(Half, Float16);
TESSERACT_DEFINE_CPP_TO_DTYPE(BFloat16, BFloat16);
TESSERACT_DEFINE_CPP_TO_DTYPE(int8_t, Int8);

#undef TESSERACT_DEFINE_CPP_TO_DTYPE

}  // namespace tesseract

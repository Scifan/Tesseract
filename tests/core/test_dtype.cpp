#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"

using tesseract::DType;

TEST_CASE("DType sizes and names", "[dtype]") {
  REQUIRE(tesseract::dtype_size(DType::Float32) == 4);
  REQUIRE(tesseract::dtype_size(DType::Float64) == 8);
  REQUIRE(tesseract::dtype_size(DType::Int32) == 4);
  REQUIRE(tesseract::dtype_size(DType::Int64) == 8);
  REQUIRE(tesseract::dtype_size(DType::Bool) == 1);

  REQUIRE(tesseract::dtype_name(DType::Float32) == "f32");
  REQUIRE(tesseract::dtype_name(DType::Int64) == "i64");
}

TEST_CASE("DType classification predicates", "[dtype]") {
  REQUIRE(tesseract::dtype_is_implemented(DType::Float32));
  // B-005: half-precision types are now software-emulated on CPU.
  REQUIRE(tesseract::dtype_is_implemented(DType::Float16));
  REQUIRE(tesseract::dtype_is_implemented(DType::BFloat16));
  // FP8 / Int4 are still reserved-but-not-implemented.
  REQUIRE_FALSE(tesseract::dtype_is_implemented(DType::Float8_E4M3));
  REQUIRE_FALSE(tesseract::dtype_is_implemented(DType::Int4));

  REQUIRE(tesseract::dtype_is_floating(DType::Float64));
  REQUIRE(tesseract::dtype_is_floating(DType::Float16));
  REQUIRE(tesseract::dtype_is_floating(DType::BFloat16));
  REQUIRE_FALSE(tesseract::dtype_is_floating(DType::Int64));

  REQUIRE(tesseract::dtype_is_integral(DType::Int32));
  REQUIRE(tesseract::dtype_is_integral(DType::Bool));
  REQUIRE_FALSE(tesseract::dtype_is_integral(DType::Float32));
}

TEST_CASE("CppTypeToDType mapping", "[dtype]") {
  static_assert(tesseract::CppTypeToDType<float>::value == DType::Float32);
  static_assert(tesseract::CppTypeToDType<double>::value == DType::Float64);
  static_assert(tesseract::CppTypeToDType<int32_t>::value == DType::Int32);
  static_assert(tesseract::CppTypeToDType<int64_t>::value == DType::Int64);
  static_assert(tesseract::CppTypeToDType<bool>::value == DType::Bool);
  static_assert(tesseract::CppTypeToDType<tesseract::Half>::value == DType::Float16);
  static_assert(tesseract::CppTypeToDType<tesseract::BFloat16>::value == DType::BFloat16);
  SUCCEED();
}

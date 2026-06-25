#include <catch2/catch_test_macros.hpp>

#include <fmt/core.h>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/DType.hpp"

TEST_CASE("smoke: library links and fmt is available", "[smoke]") {
  const tesseract::Device d = tesseract::cpu_device();
  REQUIRE(d.is_cpu());
  REQUIRE(d.index == 0);

  const auto msg = fmt::format("dtype={} device={}", tesseract::dtype_name(tesseract::DType::Float32),
                               d.to_string());
  REQUIRE(msg == "dtype=f32 device=cpu:0");
}

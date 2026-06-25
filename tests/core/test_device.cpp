#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/Device.hpp"

using tesseract::Device;
using tesseract::DeviceType;

TEST_CASE("Device equality and default", "[device]") {
  Device a;
  REQUIRE(a.is_cpu());
  REQUIRE(a.index == 0);

  Device b{DeviceType::CPU, 0};
  REQUIRE(a == b);

  Device cuda0{DeviceType::CUDA, 0};
  Device cuda1{DeviceType::CUDA, 1};
  REQUIRE(cuda0 != cuda1);
  REQUIRE(cuda0.is_cuda());
}

TEST_CASE("Device to_string", "[device]") {
  REQUIRE(Device{DeviceType::CPU, 0}.to_string() == "cpu:0");
  REQUIRE(Device{DeviceType::CUDA, 3}.to_string() == "cuda:3");
  REQUIRE(Device{DeviceType::Metal, 0}.to_string() == "metal:0");
}

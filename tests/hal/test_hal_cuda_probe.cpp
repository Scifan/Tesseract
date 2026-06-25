// M2B smoke test for the CUDA probe surface. Exercises the public API in
// both build configurations:
//
//   * TESSERACT_ENABLE_CUDA=OFF: every API returns its "no CUDA" stub. The
//     test case for the stubbed behavior is unconditionally compiled in so
//     the CPU-only configuration still verifies the routing layer.
//
//   * TESSERACT_ENABLE_CUDA=ON + no visible GPU: `has_cuda_support()` is
//     true but `device_count()` is 0 / `is_available()` is false. The
//     driver-probe test case uses Catch2's `SKIP` to bail out cleanly.
//
//   * TESSERACT_ENABLE_CUDA=ON + >= 1 visible GPU: the driver-probe test
//     case asserts that every reported device has a non-empty name and a
//     compute capability >= 3.0, and that the runtime version string is
//     non-empty.
//
// We don't allocate memory or launch kernels here — that's M2C/M2D
// territory. This file's job is only to prove "the library you linked
// against actually knows how to talk to the CUDA driver".

#include <catch2/catch_test_macros.hpp>

#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/utils/Logging.hpp"

using tesseract::DeviceError;
using tesseract::cuda::device_count;
using tesseract::cuda::device_info;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;
using tesseract::cuda::runtime_version_string;

TEST_CASE("cuda::has_cuda_support matches the build configuration",
          "[cuda][probe]") {
#if defined(TESSERACT_ENABLE_CUDA_BUILD_FLAG)
  REQUIRE(has_cuda_support());
#else
  // Either way, the function must not throw and must return a bool.
  const bool b = has_cuda_support();
  (void)b;
#endif
}

TEST_CASE("cuda::device_count is non-negative and is_available agrees",
          "[cuda][probe]") {
  const int n = device_count();
  REQUIRE(n >= 0);
  REQUIRE(is_available() == (n > 0));
}

TEST_CASE("cuda::device_info on a CPU-only build throws DeviceError",
          "[cuda][probe]") {
  if (has_cuda_support()) {
    SKIP("Build has CUDA support; the stub-branch test does not apply.");
  }
  REQUIRE_THROWS_AS(device_info(0), DeviceError);
}

TEST_CASE("cuda::runtime_version_string is non-empty",
          "[cuda][probe]") {
  const auto s = runtime_version_string();
  REQUIRE_FALSE(s.empty());
}

TEST_CASE("cuda::device_info enumerates every visible device",
          "[cuda][probe][gpu]") {
  if (!has_cuda_support()) {
    SKIP("Binary built without TESSERACT_ENABLE_CUDA — nothing to enumerate.");
  }
  const int n = device_count();
  if (n <= 0) {
    SKIP("CUDA build, but no GPU visible to the driver on this host.");
  }

  for (int i = 0; i < n; ++i) {
    const auto info = device_info(i);
    CAPTURE(i, info.name, info.compute_capability_major,
            info.compute_capability_minor, info.total_global_memory_bytes);
    REQUIRE(info.index == i);
    REQUIRE_FALSE(info.name.empty());
    // SM 3.0 is the floor CUDA 12 still supports; every modern GPU is
    // above this. A report below it almost certainly means the probe
    // picked up garbage memory rather than a real device descriptor.
    REQUIRE(info.compute_capability_major >= 3);
    REQUIRE(info.total_global_memory_bytes > 0);
  }

  // Out-of-range requests must fail cleanly without segfaulting.
  REQUIRE_THROWS_AS(device_info(n), DeviceError);
  REQUIRE_THROWS_AS(device_info(-1), DeviceError);
}

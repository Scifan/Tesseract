// M2C smoke test for the CUDA allocator.
//
// Two build configurations are covered:
//
//   * TESSERACT_ENABLE_CUDA=OFF: every call that would hand out device
//     memory must throw `DeviceError` with a clear "rebuild with
//     -DTESSERACT_ENABLE_CUDA=ON" message. This is the "happy stub"
//     contract that lets higher-level code (like
//     `default_allocator_for(cuda_device())`) fail with a single
//     catchable exception rather than a link error or a segfault.
//
//   * TESSERACT_ENABLE_CUDA=ON + >= 1 visible GPU:
//       - `default_allocator_for(cuda_device(0))` returns a non-null
//         `CudaAllocator*` whose `device()` matches.
//       - A single 64-byte round-trip (cudaMalloc → cudaMemset(0) →
//         cudaMemcpy D→H → cudaFree) returns zeros.
//       - A 10 000-iteration alloc/free stress loop survives without
//         leaking (verified externally with compute-sanitizer; see
//         `docs/m2-plan.md` progress log for the run log). The loop
//         itself just checks that no call throws.
//
// No kernels are launched here; that's M2D / M2E. We only need to
// confirm the allocator actually reaches the driver and hands back
// usable device pointers.

#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/Allocator.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/cuda/CudaAllocator.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/utils/Logging.hpp"

#if defined(TESSERACT_HAS_CUDA)
#include <cuda_runtime.h>
#endif

using tesseract::default_allocator_for;
using tesseract::Device;
using tesseract::DeviceError;
using tesseract::DeviceType;
using tesseract::cuda::CudaAllocator;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

TEST_CASE("default_allocator_for(cuda) throws cleanly on CPU-only builds",
          "[cuda][allocator]") {
  if (has_cuda_support()) {
    SKIP("Binary has CUDA support; the stub-branch test does not apply.");
  }
  REQUIRE_THROWS_AS(default_allocator_for(Device{DeviceType::CUDA, 0}),
                    DeviceError);
  REQUIRE_THROWS_AS(CudaAllocator::instance_for(0), DeviceError);
}

TEST_CASE("CudaAllocator hands out real device pointers",
          "[cuda][allocator][gpu]") {
  if (!has_cuda_support()) {
    SKIP("Binary built without TESSERACT_ENABLE_CUDA.");
  }
  if (!is_available()) {
    SKIP("CUDA build, but no GPU visible on this host.");
  }

  auto* a = default_allocator_for(Device{DeviceType::CUDA, 0});
  REQUIRE(a != nullptr);
  REQUIRE(a->device() == Device{DeviceType::CUDA, 0});

  // 16-byte round-trip: zero on device, copy back, assert zeros.
  constexpr std::size_t kBytes = 64;
  void* dev = a->allocate(kBytes, 64);
  REQUIRE(dev != nullptr);

#if defined(TESSERACT_HAS_CUDA)
  REQUIRE(cudaMemset(dev, 0, kBytes) == cudaSuccess);
  std::vector<unsigned char> host(kBytes, 0xAB);
  REQUIRE(cudaMemcpy(host.data(), dev, kBytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (unsigned char b : host) {
    REQUIRE(static_cast<int>(b) == 0);
  }
#endif

  a->deallocate(dev, kBytes);

  // Zero-sized allocs must be cheap and return nullptr, not a driver call.
  REQUIRE(a->allocate(0, 64) == nullptr);
}

TEST_CASE("CudaAllocator survives a 10000-cycle alloc/free stress run",
          "[cuda][allocator][gpu][stress]") {
  if (!has_cuda_support()) {
    SKIP("Binary built without TESSERACT_ENABLE_CUDA.");
  }
  if (!is_available()) {
    SKIP("CUDA build, but no GPU visible on this host.");
  }

  auto* a = default_allocator_for(Device{DeviceType::CUDA, 0});
  REQUIRE(a != nullptr);

  // A mix of small and medium allocations to exercise at least two
  // internal bins if a pool allocator ever lands here. 64B and 64KiB
  // are the two bins the first-pass cuBLASLt autotuner is expected
  // to use; keeping the test shape stable lets the B-010 caching
  // pool reuse this harness as its soak test.
  for (int i = 0; i < 10000; ++i) {
    const std::size_t nbytes = (i % 2 == 0) ? std::size_t{64}
                                            : std::size_t{64 * 1024};
    void* p = a->allocate(nbytes, 64);
    REQUIRE(p != nullptr);
    a->deallocate(p, nbytes);
  }
}

TEST_CASE("CudaAllocator rejects out-of-range device indices",
          "[cuda][allocator][gpu]") {
  if (!has_cuda_support()) {
    SKIP("Binary built without TESSERACT_ENABLE_CUDA.");
  }
  REQUIRE_THROWS_AS(CudaAllocator::instance_for(-1), DeviceError);
  REQUIRE_THROWS_AS(CudaAllocator::instance_for(1024), DeviceError);
}

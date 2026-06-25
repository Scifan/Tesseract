// M2D smoke tests for the HAL byte-copy primitives. Three build
// configurations are exercised:
//
//   * TESSERACT_ENABLE_CUDA=OFF: the CUDA bridge stubs throw; the
//     public helpers in `Storage` that don't touch CUDA still work
//     (CPU→CPU memcpy, CPU memset).
//
//   * TESSERACT_ENABLE_CUDA=ON with a visible GPU:
//       - H → D → H round-trip preserves every byte.
//       - D → D round-trip (on the same device) copies the bytes.
//       - `zero_device_bytes` on a CUDA buffer reports all zeros
//         when copied back.
//
// Kernel launches and elementwise kernels are deliberately out of
// scope — this layer only handles the DMA bytes. Op-level parity
// tests live under `tests/ops/`.

#include <cstddef>
#include <cstring>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/Allocator.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Storage.hpp"
#include "tesseract/cuda/CudaAllocator.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/utils/Logging.hpp"

using tesseract::default_allocator_for;
using tesseract::Device;
using tesseract::DeviceError;
using tesseract::DeviceType;
using tesseract::Storage;
using tesseract::cpu_device;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

namespace {

std::vector<unsigned char> make_pattern(std::size_t n) {
  std::vector<unsigned char> v(n);
  // 0xA5, 0x5A, 0xA5, ... alternating gives a pattern that exposes
  // both half-byte nibbles and won't accidentally compare equal to
  // zero-initialized memory. std::iota wouldn't stress nibbles, a
  // memset constant wouldn't prove both halves copied.
  for (std::size_t i = 0; i < n; ++i) v[i] = (i & 1) ? 0x5A : 0xA5;
  return v;
}

}  // namespace

TEST_CASE("copy_device_bytes CPU→CPU matches std::memcpy", "[copy][cpu]") {
  constexpr std::size_t kN = 128;
  auto src = make_pattern(kN);
  std::vector<unsigned char> dst(kN, 0x00);
  Storage::copy_device_bytes(dst.data(), cpu_device(),
                             src.data(), cpu_device(), kN);
  for (std::size_t i = 0; i < kN; ++i) REQUIRE(dst[i] == src[i]);
}

TEST_CASE("zero_device_bytes on CPU zeroes the target", "[copy][cpu]") {
  std::vector<unsigned char> buf(64, 0xFF);
  Storage::zero_device_bytes(buf.data(), cpu_device(), buf.size());
  for (unsigned char b : buf) REQUIRE(static_cast<int>(b) == 0);
}

TEST_CASE("copy_device_bytes rejects unknown devices", "[copy][cpu]") {
  std::vector<unsigned char> a(8), b(8);
  // Metal is declared but not wired up as of M2; it's the cleanest
  // "unsupported device" proxy that doesn't depend on the build
  // having CUDA enabled.
  Device metal{DeviceType::Metal, 0};
  REQUIRE_THROWS(Storage::copy_device_bytes(a.data(), metal,
                                            b.data(), cpu_device(), 8));
}

TEST_CASE("copy_device_bytes round-trips through CUDA (H→D→H)",
          "[copy][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for copy tests.");
  }

  Device dev{DeviceType::CUDA, 0};
  auto* alloc = default_allocator_for(dev);
  REQUIRE(alloc != nullptr);

  constexpr std::size_t kN = 4096;
  auto src = make_pattern(kN);
  std::vector<unsigned char> dst(kN, 0x00);

  void* device_buf = alloc->allocate(kN, 64);
  REQUIRE(device_buf != nullptr);

  Storage::copy_device_bytes(device_buf, dev, src.data(), cpu_device(), kN);
  Storage::copy_device_bytes(dst.data(), cpu_device(), device_buf, dev, kN);

  for (std::size_t i = 0; i < kN; ++i) {
    REQUIRE(static_cast<int>(dst[i]) == static_cast<int>(src[i]));
  }

  alloc->deallocate(device_buf, kN);
}

TEST_CASE("copy_device_bytes supports D→D copy", "[copy][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for copy tests.");
  }

  Device dev{DeviceType::CUDA, 0};
  auto* alloc = default_allocator_for(dev);

  constexpr std::size_t kN = 1024;
  auto src = make_pattern(kN);
  std::vector<unsigned char> dst(kN, 0xFF);

  void* dev_a = alloc->allocate(kN, 64);
  void* dev_b = alloc->allocate(kN, 64);
  REQUIRE(dev_a != nullptr);
  REQUIRE(dev_b != nullptr);

  // Seed dev_a from host, then D→D into dev_b, then pull dev_b back.
  Storage::copy_device_bytes(dev_a, dev, src.data(), cpu_device(), kN);
  Storage::copy_device_bytes(dev_b, dev, dev_a, dev, kN);
  Storage::copy_device_bytes(dst.data(), cpu_device(), dev_b, dev, kN);

  for (std::size_t i = 0; i < kN; ++i) {
    REQUIRE(static_cast<int>(dst[i]) == static_cast<int>(src[i]));
  }

  alloc->deallocate(dev_a, kN);
  alloc->deallocate(dev_b, kN);
}

TEST_CASE("zero_device_bytes on CUDA produces an all-zero buffer",
          "[copy][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for copy tests.");
  }

  Device dev{DeviceType::CUDA, 0};
  auto* alloc = default_allocator_for(dev);

  constexpr std::size_t kN = 256;
  // Seed the device buffer with a non-zero pattern so a no-op
  // "zero" implementation would still leave the pattern visible.
  auto pattern = make_pattern(kN);
  void* buf = alloc->allocate(kN, 64);
  Storage::copy_device_bytes(buf, dev, pattern.data(), cpu_device(), kN);

  Storage::zero_device_bytes(buf, dev, kN);

  std::vector<unsigned char> host(kN, 0xFF);
  Storage::copy_device_bytes(host.data(), cpu_device(), buf, dev, kN);
  for (unsigned char b : host) REQUIRE(static_cast<int>(b) == 0);

  alloc->deallocate(buf, kN);
}

TEST_CASE("copy_device_bytes to a CUDA device in a CPU-only build throws",
          "[copy][cpu-only]") {
  if (has_cuda_support()) {
    SKIP("Binary has CUDA support; stub-branch test does not apply.");
  }
  std::vector<unsigned char> host(16, 0);
  Device cuda0{DeviceType::CUDA, 0};
  // The bridge symbol is defined but throws. Going through the
  // Storage helper gives us the same user-facing exception type as
  // `default_allocator_for(cuda)`.
  REQUIRE_THROWS_AS(Storage::copy_device_bytes(host.data(), cuda0,
                                                host.data(), cpu_device(), 16),
                    DeviceError);
}

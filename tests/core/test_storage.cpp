#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "tesseract/core/Allocator.hpp"
#include "tesseract/core/Storage.hpp"
#include "tesseract/utils/Logging.hpp"

using tesseract::CpuAllocator;
using tesseract::Storage;

TEST_CASE("CpuAllocator singleton and zero-byte behavior", "[allocator]") {
  auto* a = CpuAllocator::instance();
  REQUIRE(a != nullptr);
  REQUIRE(a->device().is_cpu());
  REQUIRE(a->allocate(0, 64) == nullptr);
  // deallocate of nullptr must not crash.
  a->deallocate(nullptr, 0);
}

TEST_CASE("Storage owning mode allocates and frees", "[storage]") {
  auto* a = CpuAllocator::instance();
  auto s = Storage::make_owning(1024, a);
  REQUIRE(s->nbytes() == 1024);
  REQUIRE(s->data() != nullptr);
  REQUIRE(s->is_owning());
  REQUIRE(s->device().is_cpu());
  // Writing to the buffer should be safe.
  auto* bytes = static_cast<uint8_t*>(s->data());
  bytes[0] = 0x42;
  REQUIRE(bytes[0] == 0x42);
}

TEST_CASE("Storage borrowed mode does not free the buffer", "[storage]") {
  std::vector<float> buf(16, 3.14f);
  auto s = Storage::make_borrowed(buf.data(), buf.size() * sizeof(float), tesseract::cpu_device());
  REQUIRE_FALSE(s->is_owning());
  REQUIRE(s->data() == buf.data());
  REQUIRE(s->nbytes() == buf.size() * sizeof(float));
  s.reset();
  // Underlying buffer must still be intact.
  REQUIRE(buf[0] == 3.14f);
}

TEST_CASE("Storage rejects non-power-of-two alignment", "[storage][allocator]") {
  auto* a = CpuAllocator::instance();
  REQUIRE_THROWS_AS(Storage(64, a, /*alignment=*/3), tesseract::Error);
}

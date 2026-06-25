#pragma once

#include <cstddef>

#include "tesseract/core/Device.hpp"

namespace tesseract {

// Default alignment (bytes) for tensor storages. 64 bytes accommodates AVX-512
// loads and is coarse enough to be safe for future SIMD widths.
inline constexpr std::size_t kDefaultTensorAlignment = 64;

// Abstract allocator interface. All tensor-facing allocations go through a
// subclass so that the runtime can swap in GPU, pinned, or pool allocators
// without touching the tensor layer.
class Allocator {
 public:
  virtual ~Allocator() = default;

  // Allocate `nbytes` aligned to at least `alignment`. Returns nullptr on
  // zero-sized requests (which never fault).
  virtual void* allocate(std::size_t nbytes, std::size_t alignment) = 0;

  // Free the pointer returned by a previous allocate() call with matching
  // `nbytes`. No-op when `ptr` is nullptr.
  virtual void deallocate(void* ptr, std::size_t nbytes) noexcept = 0;

  virtual Device device() const noexcept = 0;
};

// Default CPU allocator backed by aligned malloc. Stateless singleton.
class CpuAllocator final : public Allocator {
 public:
  static CpuAllocator* instance() noexcept;

  void* allocate(std::size_t nbytes, std::size_t alignment) override;
  void deallocate(void* ptr, std::size_t nbytes) noexcept override;
  Device device() const noexcept override { return cpu_device(); }

 private:
  CpuAllocator() = default;
};

// Selector: the default Allocator* for the given device. At M0 only CPU is
// backed; other devices throw.
Allocator* default_allocator_for(Device device);

}  // namespace tesseract

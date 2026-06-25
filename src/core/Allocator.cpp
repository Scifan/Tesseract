#include "tesseract/core/Allocator.hpp"

#include <cstdlib>

#include "tesseract/utils/Logging.hpp"

namespace tesseract {

// Strong-symbol hook that the CUDA HAL (`tesseract_cuda`) provides. In a
// CPU-only build the definition comes from `src/cuda/Allocator.cpp`
// compiled without TESSERACT_HAS_CUDA and throws a clean DeviceError
// that says "rebuild with -DTESSERACT_ENABLE_CUDA=ON". In a CUDA
// build the definition comes from the same file compiled with
// TESSERACT_HAS_CUDA=1 and returns the per-device `CudaAllocator`
// singleton.
//
// Declaring the symbol here (inside `tesseract_core`) creates a
// strong reference that pulls `tesseract_cuda`'s Allocator.cpp into
// every binary that links the meta target — no `--whole-archive`
// linker tricks, no registry, no static-init ordering.
namespace detail {
Allocator* cuda_default_allocator(int device_index);
}  // namespace detail

CpuAllocator* CpuAllocator::instance() noexcept {
  static CpuAllocator inst;
  return &inst;
}

void* CpuAllocator::allocate(std::size_t nbytes, std::size_t alignment) {
  if (nbytes == 0) return nullptr;
  TESSERACT_CHECK(alignment > 0 && (alignment & (alignment - 1)) == 0,
                  "alignment must be a positive power of two, got {}", alignment);
  if (alignment < sizeof(void*)) alignment = sizeof(void*);

  // posix_memalign requires nbytes to be a multiple of alignment in some libcs
  // (and size == 0 returns EINVAL). Round up silently; the caller asked for
  // "at least nbytes".
  const std::size_t rem = nbytes % alignment;
  if (rem != 0) nbytes += alignment - rem;

  void* ptr = nullptr;
  const int rc = ::posix_memalign(&ptr, alignment, nbytes);
  TESSERACT_CHECK(rc == 0 && ptr != nullptr,
                  "CpuAllocator::allocate(nbytes={}, align={}) failed with rc={}", nbytes,
                  alignment, rc);
  return ptr;
}

void CpuAllocator::deallocate(void* ptr, std::size_t /*nbytes*/) noexcept {
  std::free(ptr);
}

Allocator* default_allocator_for(Device device) {
  switch (device.type) {
    case DeviceType::CPU:
      return CpuAllocator::instance();
    case DeviceType::CUDA:
      return detail::cuda_default_allocator(device.index);
    case DeviceType::Metal:
    case DeviceType::NPU:
      TESSERACT_THROW("device {} has no allocator registered yet",
                      device.to_string());
    default:
      TESSERACT_THROW("unknown device type {}", static_cast<int>(device.type));
  }
}

}  // namespace tesseract

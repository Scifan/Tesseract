#include "tesseract/cuda/PinnedHostAllocator.hpp"

#include <fmt/format.h>

#include "tesseract/utils/Logging.hpp"

#if defined(TESSERACT_HAS_CUDA)
#include <cuda_runtime.h>
#endif

// Always-compiled TU, same pattern as `Allocator.cpp` / `Stream.cpp` /
// `CudaGraph.cpp`:
//
//   * CPU-only build: every method throws a clean DeviceError (upstream
//     callers are guarded by `is_cuda()` / `device_count() > 0` checks
//     in the hot path, so hitting a throwing stub means a routing bug).
//   * CUDA build: each method forwards to the CUDA runtime.
//
// Kept deliberately simple on the allocation path — no bucketed cache,
// no per-size free-list. Pinned memory is expensive to allocate
// (page-locking has to touch every page under the OS lock) but we
// expect call sites to allocate a small number of long-lived buffers
// (one per weight staging, one per input-token buffer) rather than
// the high-churn pattern the `CudaAllocator` bucketed cache exists to
// serve. If the allocation profile shifts toward many small shortlived
// pinned buffers we can bolt on the same bucket strategy the device
// allocator uses; until then the simplest thing is the right thing.

namespace tesseract::cuda {

#if defined(TESSERACT_HAS_CUDA)

namespace {

// Flags for `cudaHostAlloc`. We use `cudaHostAllocPortable` so the
// pinned region is usable from every CUDA context in the process
// (multi-GPU futures); `cudaHostAllocWriteCombined` is NOT set —
// WC memory is faster for H→D streaming but uncacheable by the CPU,
// which would tank the read-side when callers (tokenizer, loader,
// benches) want to peek at the buffer from the host.
constexpr unsigned int kPinnedFlags = cudaHostAllocPortable;

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw DeviceError(fmt::format("[tesseract] {} failed: {}", what,
                                  cudaGetErrorString(err)));
  }
}

}  // namespace

PinnedHostAllocator& PinnedHostAllocator::instance() {
  static PinnedHostAllocator inst;
  return inst;
}

void* PinnedHostAllocator::allocate(std::size_t nbytes,
                                    std::size_t /*alignment*/) {
  if (nbytes == 0) return nullptr;
  void* ptr = nullptr;
  check(cudaHostAlloc(&ptr, nbytes, kPinnedFlags), "cudaHostAlloc");
  return ptr;
}

void PinnedHostAllocator::deallocate(void* ptr, std::size_t /*nbytes*/) noexcept {
  if (ptr == nullptr) return;
  // `cudaFreeHost` is synchronous with respect to the CPU but does
  // not require an active device context, and any in-flight
  // `cudaMemcpyAsync` using this buffer would have to have
  // synchronized before the pointer can be freed (otherwise the
  // caller is wrong regardless of which allocator we use). We
  // intentionally swallow the return code here — `deallocate` is
  // noexcept per the Allocator contract, and a failing free
  // terminates the buffer's lifetime either way. The driver will
  // reclaim everything at process exit.
  (void)cudaFreeHost(ptr);
}

void PinnedHostAllocator::release_all_cached() noexcept {
  // Current implementation doesn't cache. Symmetric placeholder
  // for future bucketed caching if usage pattern demands it.
}

Allocator* pinned_host_allocator() {
  return &PinnedHostAllocator::instance();
}

#else  // !TESSERACT_HAS_CUDA

namespace {
[[noreturn]] void throw_not_built(const char* op) {
  throw DeviceError(fmt::format(
      "[tesseract] PinnedHostAllocator::{} called but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)", op));
}
}  // namespace

PinnedHostAllocator& PinnedHostAllocator::instance() {
  // We still need something to return — a static stub object gives
  // callers a stable address to hold. Every non-const method throws
  // below, so the stub can never actually hand out memory.
  static PinnedHostAllocator inst;
  return inst;
}

void* PinnedHostAllocator::allocate(std::size_t, std::size_t) {
  throw_not_built("allocate");
}

void PinnedHostAllocator::deallocate(void*, std::size_t) noexcept {
  // Allocator::deallocate is noexcept; a CPU-only binary should
  // never reach here (the allocate path already threw), but be
  // defensive and no-op in case a caller holds onto a freed pointer.
}

void PinnedHostAllocator::release_all_cached() noexcept {
  // Nothing cached to release; no-op is the correct behavior here
  // so CPU-only tests can `release_all_cached()` without guards.
}

Allocator* pinned_host_allocator() {
  return &PinnedHostAllocator::instance();
}

#endif  // TESSERACT_HAS_CUDA

}  // namespace tesseract::cuda

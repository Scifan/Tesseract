#include "tesseract/cuda/CudaAllocator.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "tesseract/core/Allocator.hpp"
#include "tesseract/utils/Logging.hpp"

#if defined(TESSERACT_HAS_CUDA)
#include "Internal.hpp"
#endif

// CUDA allocator implementation + the routing glue that
// `tesseract_core` reaches for from `default_allocator_for(cuda)`.
//
// This TU is always compiled (both in CPU-only and CUDA-enabled
// builds). That's load-bearing: `src/core/Allocator.cpp` declares
// `tesseract::detail::cuda_default_allocator` as an extern function
// and every call to `default_allocator_for(cuda_device(i))` dispatches
// through it. If this symbol weren't defined everywhere,
// CPU-only links of `tesseract_core` would fail. Keeping the plumbing
// always-compiled also gives us a single, clear point of failure for
// "CUDA not compiled in" — users never see a link error or a strange
// crash, only a `DeviceError` they can catch and recover from.
//
// When TESSERACT_HAS_CUDA is defined the CudaAllocator actually calls
// `cudaMalloc` / `cudaFree` via the bridge functions in `Internal.hpp`
// (which are in turn implemented in `Probe.cu` / this file's
// TESSERACT_HAS_CUDA branch below). When it's not defined every
// method throws a clear DeviceError.

namespace tesseract::cuda {

namespace {

// One singleton per device index. The small-array cap matches the
// maximum a single server board is likely to expose (8× A100 / H100 /
// RTX 5880 Ada etc.). We refuse to hand out allocators for indices
// beyond that rather than heap-allocate a dynamic table, which would
// complicate teardown order during static destruction.
//
// `mutex` guards construction only; `allocate` / `deallocate` don't
// take the lock because `CudaAllocator` itself is stateless.
struct AllocatorSlot {
  std::once_flag flag;
  CudaAllocator* ptr{nullptr};
};

constexpr std::size_t kMaxCudaDevices = 16;
std::array<AllocatorSlot, kMaxCudaDevices> g_slots{};

// ---- Bucketed caching pool -----------------------------------------------
//
// Every on-path `ops::*` call on CUDA materializes its output via
// `Tensor::empty`, which used to call straight into `cudaMalloc` / `cudaFree`.
// Driver-level allocation is expensive (often 100–200 µs) and, on small
// matmuls / elementwise kernels, it completely dominates the dispatch
// budget. Caching blocks by rounded size collapses that cost to a
// mutex-protected vector pop in the steady state, which the M2L.1
// ≥99% cuBLASLt and ≥95% memcpy targets all depend on.
//
// Policy:
//   * Small blocks (≤ 1 MiB): round up to 512-byte multiples. This
//     keeps the bucket count bounded for many small allocations
//     (metadata, scales, scratch tensors).
//   * Large blocks: round up to the next power of two. Square FP32
//     4096² = 64 MiB maps cleanly; elementwise {1,16,64,256} MiB buffers
//     all fall on distinct buckets.
//   * Never `cudaFree` on the fast path — the driver reclaims everything
//     at process exit, which is the only time the cache would need to
//     drain. `release_all_cached` is exposed for tests that want to
//     force-drop every block.
//
// Thread-safety: `g_caches[d].map_mu` guards insertion into the bucket
// map; each Bucket has its own mutex guarding its free-list. Allocate /
// deallocate on different-sized blocks never contend.
constexpr std::size_t kSmallBlockThreshold = 1 << 20;   // 1 MiB
constexpr std::size_t kSmallBlockGranularity = 512;

std::size_t round_up(std::size_t nbytes) {
  if (nbytes <= kSmallBlockGranularity) return kSmallBlockGranularity;
  if (nbytes <= kSmallBlockThreshold) {
    return (nbytes + kSmallBlockGranularity - 1) & ~(kSmallBlockGranularity - 1);
  }
  std::size_t p = 1;
  while (p < nbytes) p <<= 1;
  return p;
}

struct Bucket {
  std::mutex mu;
  std::vector<void*> free_list;
};

struct DeviceCache {
  std::mutex map_mu;
  std::unordered_map<std::size_t, std::unique_ptr<Bucket>> buckets;

  Bucket* get_or_create(std::size_t rounded) {
    std::lock_guard<std::mutex> g(map_mu);
    auto it = buckets.find(rounded);
    if (it == buckets.end()) {
      it = buckets.emplace(rounded, std::make_unique<Bucket>()).first;
    }
    return it->second.get();
  }
};

std::array<DeviceCache, kMaxCudaDevices> g_caches{};

// ---- CUDA-graph capture: deferred frees ----------------------------------
//
// While a graph is being captured on this thread, freed blocks are
// parked here instead of going straight back to the bucket free-lists.
// That guarantees every `allocate` during capture hands out a distinct
// block (no intra-capture reuse), eliminating the address-aliasing
// hazard that breaks replay bit-exactness. See CudaAllocator.hpp.
//
// Thread-local because `cudaStreamCaptureModeThreadLocal` scopes the
// capture to one host thread; other threads keep recycling normally.
thread_local int g_defer_device = -1;
thread_local std::vector<std::pair<void*, std::size_t>> g_deferred;

void flush_deferred_locked() {
  for (auto& [ptr, rounded] : g_deferred) {
    Bucket* b = g_caches[static_cast<std::size_t>(g_defer_device)]
                    .get_or_create(rounded);
    std::lock_guard<std::mutex> g(b->mu);
    b->free_list.push_back(ptr);
  }
  g_deferred.clear();
}

}  // namespace

CudaAllocator& CudaAllocator::instance_for(int device_index) {
  if (device_index < 0 || static_cast<std::size_t>(device_index) >= kMaxCudaDevices) {
    throw DeviceError(fmt::format(
        "[tesseract] CudaAllocator::instance_for: device_index {} is out of "
        "the supported range [0, {})",
        device_index, kMaxCudaDevices));
  }

#if !defined(TESSERACT_HAS_CUDA)
  // Even the creation path reports the missing backend. Deferring the
  // throw to allocate() would be just as safe but would let callers
  // hold a dangling reference; throwing here keeps the contract
  // "CudaAllocator objects only exist if CUDA was compiled in".
  throw DeviceError(
      "[tesseract] CudaAllocator requested but the CUDA backend was not "
      "compiled in. Rebuild with -DTESSERACT_ENABLE_CUDA=ON to enable GPU "
      "allocation.");
#else
  auto& slot = g_slots[static_cast<std::size_t>(device_index)];
  std::call_once(slot.flag, [&slot, device_index]() {
    slot.ptr = new CudaAllocator(device_index);
  });
  return *slot.ptr;
#endif
}

void* CudaAllocator::allocate(std::size_t nbytes, std::size_t alignment) {
  (void)alignment;  // cudaMalloc returns 256-byte-aligned pointers.
#if defined(TESSERACT_HAS_CUDA)
  if (nbytes == 0) return nullptr;
  const std::size_t rounded = round_up(nbytes);
  auto& cache = g_caches[static_cast<std::size_t>(device_index_)];
  Bucket* b = cache.get_or_create(rounded);
  {
    std::lock_guard<std::mutex> g(b->mu);
    if (!b->free_list.empty()) {
      void* ptr = b->free_list.back();
      b->free_list.pop_back();
      return ptr;
    }
  }
  // Cache miss — hit the driver for a block sized to the bucket so a
  // later deallocate with the same rounded size lands back here.
  return detail::real_cuda_malloc(device_index_, rounded);
#else
  (void)nbytes;
  throw DeviceError(
      "[tesseract] CudaAllocator::allocate called in a CPU-only build. This "
      "should be unreachable; the CudaAllocator constructor refuses to hand "
      "out instances when TESSERACT_ENABLE_CUDA is OFF.");
#endif
}

void CudaAllocator::deallocate(void* ptr, std::size_t nbytes) noexcept {
#if defined(TESSERACT_HAS_CUDA)
  if (ptr == nullptr) return;
  const std::size_t rounded = round_up(nbytes);
  // Graph-capture path: park the block instead of recycling it so no
  // address is reused within the capture (replay-correctness, see
  // CudaAllocator.hpp). Only applies to frees on the device currently
  // being captured by this thread.
  if (g_defer_device == device_index_) {
    g_deferred.emplace_back(ptr, rounded);
    return;
  }
  auto& cache = g_caches[static_cast<std::size_t>(device_index_)];
  Bucket* b = cache.get_or_create(rounded);
  std::lock_guard<std::mutex> g(b->mu);
  b->free_list.push_back(ptr);
  // No cudaFree — pool lives until the driver reclaims at process exit
  // (or release_all_cached is called explicitly, e.g. from tests).
#else
  (void)ptr;
  (void)nbytes;
  // noexcept: cannot throw on a stub call. In practice this is also
  // unreachable (no allocator to even produce a pointer), but we keep
  // the no-op safe rather than abort().
#endif
}

void CudaAllocator::release_all_cached() noexcept {
#if defined(TESSERACT_HAS_CUDA)
  auto& cache = g_caches[static_cast<std::size_t>(device_index_)];
  std::lock_guard<std::mutex> g(cache.map_mu);
  for (auto& kv : cache.buckets) {
    Bucket* b = kv.second.get();
    std::lock_guard<std::mutex> bg(b->mu);
    for (void* ptr : b->free_list) {
      detail::real_cuda_free(device_index_, ptr);
    }
    b->free_list.clear();
    b->free_list.shrink_to_fit();
  }
#endif
}

namespace detail {

#if defined(TESSERACT_HAS_CUDA)

void cuda_alloc_begin_capture(int device_index) noexcept {
  // Flush any stale parked blocks from a prior (aborted) capture first.
  if (g_defer_device >= 0) flush_deferred_locked();
  g_defer_device = device_index;
  g_deferred.clear();
}

void cuda_alloc_flush_deferred(int /*device_index*/) noexcept {
  if (g_defer_device >= 0) flush_deferred_locked();
}

void cuda_alloc_end_capture(int /*device_index*/) noexcept {
  if (g_defer_device >= 0) flush_deferred_locked();
  g_defer_device = -1;
}

#else

void cuda_alloc_begin_capture(int) noexcept {}
void cuda_alloc_flush_deferred(int) noexcept {}
void cuda_alloc_end_capture(int) noexcept {}

#endif

}  // namespace detail

}  // namespace tesseract::cuda

// ---- Routing symbol for default_allocator_for(cuda) -----------------------

namespace tesseract::detail {

Allocator* cuda_default_allocator(int device_index) {
#if defined(TESSERACT_HAS_CUDA)
  return &cuda::CudaAllocator::instance_for(device_index);
#else
  (void)device_index;
  throw DeviceError(
      "[tesseract] default_allocator_for(cuda) called but the CUDA backend "
      "was not compiled in. Rebuild with -DTESSERACT_ENABLE_CUDA=ON or use "
      "a CPU device.");
#endif
}

}  // namespace tesseract::detail

// ---- Bridge implementations (compiled only when TESSERACT_HAS_CUDA) ------

#if defined(TESSERACT_HAS_CUDA)

#include <cuda_runtime.h>

namespace tesseract::cuda::detail {

namespace {

// RAII guard: sets the current CUDA device for the lifetime of the
// object, restoring whatever was current before. This is the standard
// "device scope" idiom from PyTorch's c10::cuda::CUDAGuard. We don't
// ship it as a public type in M2C because the only consumers are
// these bridge functions — when M2D introduces multi-device tensors
// it'll graduate into tesseract/core/Device.hpp.
struct DeviceGuard {
  int previous{-1};

  explicit DeviceGuard(int target) {
    cudaError_t err = cudaGetDevice(&previous);
    if (err != cudaSuccess) {
      previous = -1;
    }
    err = cudaSetDevice(target);
    if (err != cudaSuccess) {
      throw DeviceError(fmt::format(
          "[tesseract] cudaSetDevice({}) failed: {}", target, cudaGetErrorString(err)));
    }
  }

  ~DeviceGuard() {
    if (previous >= 0) {
      (void)cudaSetDevice(previous);
    }
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
};

}  // namespace

void* real_cuda_malloc(int device_index, std::size_t nbytes) {
  DeviceGuard g(device_index);
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, nbytes);
  if (err != cudaSuccess) {
    throw DeviceError(fmt::format(
        "[tesseract] cudaMalloc({} bytes) on device {} failed: {}",
        nbytes, device_index, cudaGetErrorString(err)));
  }
  return ptr;
}

void real_cuda_free(int device_index, void* ptr) noexcept {
  if (ptr == nullptr) return;
  int previous = -1;
  if (cudaGetDevice(&previous) != cudaSuccess) previous = -1;
  (void)cudaSetDevice(device_index);
  (void)cudaFree(ptr);
  if (previous >= 0) (void)cudaSetDevice(previous);
}

}  // namespace tesseract::cuda::detail

#else  // TESSERACT_HAS_CUDA

// CPU-only stubs for the bridge functions. They're unreachable from
// the allocator (the CudaAllocator constructor throws before allocate
// is ever called) but we provide them so that any other code path
// that accidentally takes the symbol address still links cleanly.
namespace tesseract::cuda::detail {

void* real_cuda_malloc(int /*device_index*/, std::size_t /*nbytes*/) {
  throw tesseract::DeviceError(
      "[tesseract] real_cuda_malloc called in a CPU-only build");
}

void real_cuda_free(int /*device_index*/, void* /*ptr*/) noexcept {
  // Unreachable in practice; see above.
}

}  // namespace tesseract::cuda::detail

#endif  // TESSERACT_HAS_CUDA

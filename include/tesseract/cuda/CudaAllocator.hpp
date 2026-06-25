#pragma once

// CUDA device-memory allocator. Implements the `Allocator` interface from
// `tesseract/core/Allocator.hpp` for `Device(DeviceType::CUDA, i)`.
//
// Like `Float16.hpp` and the probe surface in `CudaRuntime.hpp`, this
// header is intentionally plain C++20 — it does NOT include
// `<cuda_runtime.h>` or use any CUDA-specific type in its signatures.
// That lets any translation unit in the project reach for a CUDA
// allocator without pulling the CUDA Toolkit into its include path,
// and it keeps CPU-only builds (`TESSERACT_ENABLE_CUDA=OFF`)
// compilable without `nvcc`.
//
// Semantics:
//   * In a CPU-only build, every method of this class throws a
//     `DeviceError` with a clear "CUDA backend not compiled in" message.
//     The stub definitions live in the `tesseract_cuda` library so the
//     rest of the core can reference the class without an `#ifdef`.
//   * In a CUDA-enabled build, `allocate` rounds the request to a
//     bucket size (small ≤ 1 MiB: 512-byte granularity; large: next
//     power of two) and hands out a block from a per-device free-list,
//     falling back to `cudaMalloc` on a cache miss. `deallocate` pushes
//     the block back into its bucket — it NEVER calls `cudaFree` on the
//     fast path. The driver reclaims the pool at process exit, or
//     `release_all_cached` can be called explicitly (tests, debugging).
//     This caching layer is load-bearing for the M2L.1 performance
//     targets: without it, every small `ops::*` call loses 100-200 µs
//     to driver-level allocation.
//
// The allocator itself is stream-agnostic: `cudaMalloc` / `cudaFree`
// are synchronizing operations on the device-global context. When the
// M2C stream infrastructure lands next to this file we add a
// stream-aware counterpart (likely switching to `cudaMallocAsync` once
// the caching-pool work starts, tracked as a follow-up in B-010).

#include <cstddef>

#include "tesseract/core/Allocator.hpp"
#include "tesseract/core/Device.hpp"

namespace tesseract::cuda {

class CudaAllocator final : public Allocator {
 public:
  // Per-device singleton. Creating a fresh `CudaAllocator` every call
  // would be pointless (the allocator holds no mutable state in M2C);
  // sharing one instance per device index also matches the way
  // `CpuAllocator::instance()` works. Throws if `device_index` is out of
  // range or if the CUDA runtime reports no such device (CPU-only
  // build: always throws).
  static CudaAllocator& instance_for(int device_index);

  // Allocate `nbytes` of device memory on `device_index_`. Throws
  // `DeviceError` on driver failure (including OOM).
  void* allocate(std::size_t nbytes, std::size_t alignment) override;

  // Free a pointer previously returned by `allocate`. Swallows any
  // driver error because `deallocate` is noexcept — a failed free is
  // logged at the CUDA level but never unwinds.
  void deallocate(void* ptr, std::size_t nbytes) noexcept override;

  Device device() const noexcept override {
    return Device{DeviceType::CUDA, device_index_};
  }

  int device_index() const noexcept { return device_index_; }

  // Drop every block currently held in the cache back to the driver.
  // Intended for test teardown and memory-pressure recovery; the steady
  // state should never need this. Safe to call in a CPU-only build
  // (no-op).
  void release_all_cached() noexcept;

 private:
  explicit CudaAllocator(int device_index) noexcept : device_index_(device_index) {}
  int device_index_{0};
};

namespace detail {

// ---- CUDA-graph capture support (deferred frees) -------------------------
//
// A bucketed caching allocator that recycles freed blocks via a LIFO
// free-list is unsafe to drive *inside* a `cudaStreamBeginCapture`
// region: a block freed early in the captured closure can be popped
// again for an unrelated tensor later in the same closure. The two
// uses end up sharing one device address, but the recorded graph has
// no edge that forces the second writer to wait for the first reader —
// so on replay the kernels race and the result diverges from eager
// (bit-exactness loss). PyTorch solves this with a private pool per
// graph; we take the lighter-weight equivalent: while capture is in
// flight, `deallocate` does *not* return blocks to the shared
// free-list. Instead it parks them on a thread-local "deferred" list
// so each allocation during capture gets a distinct, never-reused
// block. The deferred blocks are flushed back to the free-lists once
// capture (and its warmup) completes.
//
// All three are no-ops in a CPU-only build.

// Begin deferring frees for `device_index` on the calling thread.
void cuda_alloc_begin_capture(int device_index) noexcept;

// Move every block parked since `begin_capture` back into the shared
// free-lists, but keep deferring (used between the warmup passes and
// the capture pass so the capture has a fully primed pool to pop from).
void cuda_alloc_flush_deferred(int device_index) noexcept;

// Flush parked blocks and stop deferring on the calling thread.
void cuda_alloc_end_capture(int device_index) noexcept;

}  // namespace detail

}  // namespace tesseract::cuda

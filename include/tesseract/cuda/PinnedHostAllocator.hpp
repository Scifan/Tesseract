#pragma once

// Wave 2.4 (B-011): page-locked host memory allocator.
//
// `CpuAllocator` serves pageable host memory (the kernel is free to
// relocate or swap out any page). When a pageable buffer is used as
// the source of `cudaMemcpyAsync(H→D, ..., stream)`, the CUDA driver
// internally stages the bytes through a hidden, tiny pinned bounce
// buffer — effectively serializing the copy. Translated to the
// decode-latency bottleneck this gives us no overlap between
// "token id → device-side embedding input" and the next step's
// attention + norm compute.
//
// Page-locked ("pinned") host memory lives in a region the CUDA driver
// has pinned with the OS, so `cudaMemcpyAsync(H→D, pinned_src, ...)`
// can stream directly from host DRAM into the GPU's PCIe copy engine
// without the bounce buffer. Two visible consequences:
//
//   * The copy **overlaps** with other work on other streams (the
//     primary win — this is the infrastructure that lets us issue a
//     new input embedding copy in parallel with the previous step's
//     CUDA Graph replay in B-023).
//   * The copy **latency** per byte is 10–30 % lower on typical PCIe
//     Gen4 / Gen5 rigs because there's one fewer memcpy in the
//     path.
//
// Public surface — same pattern as `CudaAllocator`:
//
//   * Plain C++20 header (no `<cuda_runtime.h>`) so any TU can reach
//     for this allocator without pulling CUDA toolkit into its include
//     path. Opaque `Allocator*` return keeps the interface stable
//     whether the CUDA backend is compiled in or not.
//   * `PinnedHostAllocator::instance()` is a singleton identical in
//     ownership semantics to `CpuAllocator::instance()`. Pinned memory
//     is a *host* resource (not bound to a specific GPU index after
//     the portable flag is set), so there is no per-device sharding.
//   * `device()` returns `cpu_device()` — a pinned-memory Tensor is
//     still a CPU-resident Tensor for every routing / dispatch check;
//     only the underlying byte source is different. This means no
//     kernel, no op, and no autograd path needs to know about pinning
//     — you get the speed-up purely by swapping allocators.
//   * In a CPU-only build (`TESSERACT_HAS_CUDA` undefined) every
//     method throws `DeviceError` with the same "rebuild with
//     -DTESSERACT_ENABLE_CUDA=ON" wording used by the rest of the
//     `tesseract/cuda` stubs.
//
// See `Tensor::empty_pinned` / `Tensor::to_async` in `Tensor.hpp` for
// the high-level API that wires this allocator into tensor factories
// and async transfers.

#include <cstddef>

#include "tesseract/core/Allocator.hpp"
#include "tesseract/core/Device.hpp"

namespace tesseract::cuda {

class PinnedHostAllocator final : public Allocator {
 public:
  // Process-wide singleton. Pinned memory is a host-global resource
  // once allocated with `cudaHostAllocPortable`, so unlike
  // `CudaAllocator` there's no per-device index to shard on.
  static PinnedHostAllocator& instance();

  // Allocate `nbytes` of page-locked host memory. Throws
  // `DeviceError` on driver failure (OOM, no CUDA context, ...).
  //
  // The `alignment` parameter is accepted for `Allocator` API
  // compatibility but ignored: `cudaHostAlloc` already returns
  // 256-byte-aligned pointers, which satisfies any M2 kernel
  // alignment requirement (AVX-512 needs 64, cuBLAS needs 16).
  void* allocate(std::size_t nbytes, std::size_t alignment) override;

  // Free a pointer previously returned by `allocate`. Swallows any
  // driver error because `deallocate` is noexcept — a failed
  // `cudaFreeHost` is logged at the CUDA level and the pointer is
  // leaked (which is still preferable to unwinding through a dtor).
  void deallocate(void* ptr, std::size_t nbytes) noexcept override;

  // Every Tensor backed by pinned memory still reports CPU device
  // identity. That's what lets `Tensor::to(cuda_dev)` continue to
  // route through the existing H→D memcpy dispatch — the pinning
  // is discovered by the async copy primitive at transfer time,
  // not by a device-type check on the source.
  Device device() const noexcept override { return cpu_device(); }

  // Drop every cached pinned block back to the driver. Intended for
  // test teardown / memory-pressure recovery. In the current simple
  // allocator this is a no-op because we don't cache — each
  // `deallocate` hits `cudaFreeHost` directly. Kept in the API for
  // symmetry with `CudaAllocator::release_all_cached` and to avoid
  // having to evolve the surface when we add caching later.
  void release_all_cached() noexcept;

 private:
  PinnedHostAllocator() = default;
};

// Convenience helper mirroring `cpu_device()` / `cuda_device()`: an
// `Allocator*` for pinned host memory. Equivalent to
// `&PinnedHostAllocator::instance()`; kept as a free function so call
// sites don't have to reach for the class header every time.
Allocator* pinned_host_allocator();

}  // namespace tesseract::cuda

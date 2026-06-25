#pragma once

#include <cstddef>
#include <memory>

#include "tesseract/core/Allocator.hpp"
#include "tesseract/core/Device.hpp"

namespace tesseract {

// Opaque byte buffer that owns (or borrows) a region of memory sitting on a
// particular Device. Tensors hold a shared pointer to a Storage so that
// multiple views can share the same underlying bytes cheaply.
//
// The distinction between Storage and Tensor is identical to PyTorch's ATen:
// Storage deals in bytes + device + allocator; Tensor layers shape, stride,
// dtype, and autograd metadata on top.
class Storage {
 public:
  // Owning constructor: allocates `nbytes` via `alloc` (must be non-null and
  // must outlive the Storage).
  Storage(std::size_t nbytes, Allocator* alloc, std::size_t alignment = kDefaultTensorAlignment);

  // Non-owning constructor: wraps an externally managed buffer. The caller is
  // responsible for ensuring the pointer outlives every Tensor that refers to
  // this Storage. The custom deleter will NOT be invoked.
  struct BorrowTag {};
  Storage(BorrowTag, void* data, std::size_t nbytes, Device device);

  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) noexcept = delete;
  Storage& operator=(Storage&&) noexcept = delete;

  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }

  std::size_t nbytes() const noexcept { return nbytes_; }
  Device device() const noexcept { return device_; }
  Allocator* allocator() const noexcept { return allocator_; }
  bool is_owning() const noexcept { return allocator_ != nullptr; }

  // Factory helper for callers that want a shared-ownership Storage.
  static std::shared_ptr<Storage> make_owning(std::size_t nbytes, Allocator* alloc,
                                              std::size_t alignment = kDefaultTensorAlignment);
  static std::shared_ptr<Storage> make_borrowed(void* data, std::size_t nbytes, Device device);

  // Bulk device-aware copy primitives. These sit on Storage rather
  // than on a separate `Copy.hpp` so ops, factories, and
  // `Tensor::to(Device)` can all reach for them without a new include.
  //
  // `copy_device_bytes` performs a synchronous byte copy between any
  // two memory spaces the HAL understands. CPU↔CPU falls through to
  // `std::memcpy`; any CUDA-touching combination forwards to the
  // CUDA HAL, which selects the right `cudaMemcpyKind` and blocks
  // until the transfer is complete (same semantics as PyTorch's
  // `.to(device)` for non-pinned buffers). `nbytes == 0` is a no-op.
  //
  // `zero_device_bytes` is the contiguous zero-fill used by
  // `Tensor::zeros`. CPU path is `std::memset`; CUDA path is
  // `cudaMemset`. This intentionally does NOT support non-zero
  // fills — those require either a CUDA kernel (M2E) or a host-side
  // scratch buffer; `Tensor::fill_` picks the latter today.
  static void copy_device_bytes(void* dst, Device dst_device,
                                const void* src, Device src_device,
                                std::size_t nbytes);

  // Asynchronous byte copy variant (Wave 2.4 / B-011).
  //
  // Semantics — same dispatch as `copy_device_bytes`, but:
  //   * CPU↔CPU falls through to `std::memcpy`, the result is
  //     observable immediately on return. `stream` is ignored for
  //     this case to match the "host work is always synchronous"
  //     contract of the HAL.
  //   * Any CUDA-touching combination forwards to
  //     `cudaMemcpyAsync(..., stream.native_handle())`. The call
  //     RETURNS IMMEDIATELY after enqueuing the transfer; the
  //     caller must `stream.synchronize()` (or add an `Event` and
  //     `wait` on it from a consumer stream) before reading the
  //     destination from the host, **and** must keep both the src
  //     and dst buffers alive until the transfer completes.
  //   * Effective async behavior requires the host side to be
  //     backed by `PinnedHostAllocator`. With pageable host memory
  //     `cudaMemcpyAsync` silently degrades to a synchronous path
  //     (a hidden staging buffer copy): correctness is preserved,
  //     the overlap benefit is lost. `Tensor::to_async` is the
  //     high-level wrapper that pairs the pinned allocator with
  //     this primitive.
  //   * `nbytes == 0` is a no-op (matches the sync variant).
  static void copy_device_bytes_async(void* dst, Device dst_device,
                                      const void* src, Device src_device,
                                      std::size_t nbytes,
                                      const class Stream& stream);

  // Strided device→device async copy: `height` rows of `width` contiguous
  // bytes each, source rows packed at `spitch`, destination rows at
  // `dpitch`. Collapses what would otherwise be a per-row memcpy loop
  // into a single `cudaMemcpy2DAsync` — both faster and capture-friendly
  // (one graph node instead of `height`). CPU falls back to a row loop.
  static void copy_device_bytes_2d_async(void* dst, std::size_t dpitch,
                                         const void* src, std::size_t spitch,
                                         std::size_t width, std::size_t height,
                                         Device device, const class Stream& stream);

  static void zero_device_bytes(void* ptr, Device device, std::size_t nbytes);

 private:
  void* data_{nullptr};
  std::size_t nbytes_{0};
  Allocator* allocator_{nullptr};  // nullptr means non-owning / borrowed
  Device device_{};
};

}  // namespace tesseract

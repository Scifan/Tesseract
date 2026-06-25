#include <cstddef>
#include <cstring>

#include <fmt/format.h>

#include "tesseract/utils/Logging.hpp"

#if defined(TESSERACT_HAS_CUDA)
#include <cuda_runtime.h>
#include "Internal.hpp"
#endif

// CUDA bridge implementation for the M2D copy primitives. Same always-
// compiled pattern as Allocator.cpp / Stream.cpp: one TU provides strong
// definitions for `real_cuda_memcpy` and `real_cuda_memset_zero` in both
// build configurations so `tesseract_core` can reach for them from
// `src/core/Storage.cpp` without any `#ifdef` leaking out of the CUDA
// subdirectory. In a CPU-only build the definitions are throwing stubs
// (callers are guarded upstream, so reaching the stub means there's a
// bug in the device-dispatch switch).

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA memcpy primitive called but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void real_cuda_memcpy(void* /*dst*/, int /*dst_device*/,
                      const void* /*src*/, int /*src_device*/,
                      std::size_t /*nbytes*/) {
  throw_not_built();
}

void real_cuda_memcpy_async(void* /*dst*/, int /*dst_device*/,
                            const void* /*src*/, int /*src_device*/,
                            std::size_t /*nbytes*/,
                            void* /*stream_handle*/) {
  throw_not_built();
}

void real_cuda_memcpy_2d_async(void* /*dst*/, std::size_t /*dpitch*/,
                               const void* /*src*/, std::size_t /*spitch*/,
                               std::size_t /*width*/, std::size_t /*height*/,
                               int /*device_index*/, void* /*stream_handle*/) {
  throw_not_built();
}

void real_cuda_memset_zero(int /*device_index*/, void* /*ptr*/,
                           std::size_t /*nbytes*/) {
  throw_not_built();
}

void real_cuda_memset_zero_async(int /*device_index*/, void* /*ptr*/,
                                 std::size_t /*nbytes*/,
                                 void* /*stream_handle*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#else  // TESSERACT_HAS_CUDA

namespace tesseract::cuda::detail {

namespace {

struct DeviceGuard {
  int previous{-1};

  explicit DeviceGuard(int target) {
    cudaError_t err = cudaGetDevice(&previous);
    if (err != cudaSuccess) previous = -1;
    err = cudaSetDevice(target);
    if (err != cudaSuccess) {
      throw DeviceError(fmt::format(
          "[tesseract] cudaSetDevice({}) failed: {}", target, cudaGetErrorString(err)));
    }
  }

  ~DeviceGuard() {
    if (previous >= 0) (void)cudaSetDevice(previous);
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
};

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw DeviceError(fmt::format(
        "[tesseract] {} failed: {}", what, cudaGetErrorString(err)));
  }
}

}  // namespace

void real_cuda_memcpy(void* dst, int dst_device,
                      const void* src, int src_device,
                      std::size_t nbytes) {
  if (nbytes == 0) return;

  // H → H would be a routing bug at the core layer (Storage.cpp handles
  // it with std::memcpy before reaching this function), but we stay
  // defensive and forward it correctly anyway.
  if (dst_device < 0 && src_device < 0) {
    std::memcpy(dst, src, nbytes);
    return;
  }

  // Choose the cudaMemcpyKind and the device context where the copy
  // must be issued. The CUDA runtime requires cudaSetDevice to match
  // the non-host side; when both sides are CUDA we set to the
  // destination device (matches cuMemcpyDtoD API expectations).
  cudaMemcpyKind kind{};
  int ctx_device = 0;
  if (dst_device >= 0 && src_device < 0) {
    kind = cudaMemcpyHostToDevice;
    ctx_device = dst_device;
  } else if (dst_device < 0 && src_device >= 0) {
    kind = cudaMemcpyDeviceToHost;
    ctx_device = src_device;
  } else {
    kind = cudaMemcpyDeviceToDevice;
    ctx_device = dst_device;
  }

  DeviceGuard g(ctx_device);
  // We use synchronous `cudaMemcpy` rather than `cudaMemcpyAsync` +
  // `cudaStreamSynchronize` because the extra stream argument would
  // force a decision about *which* stream to use, and the synchronous
  // variant is already exactly what PyTorch's `.to(device)` guarantees
  // for non-pinned host memory (it blocks until the transfer is
  // complete). The async variant with an explicit stream argument
  // lives in `real_cuda_memcpy_async` below (Wave 2.4 / B-011).
  check(cudaMemcpy(dst, src, nbytes, kind), "cudaMemcpy");
}

void real_cuda_memcpy_async(void* dst, int dst_device,
                            const void* src, int src_device,
                            std::size_t nbytes,
                            void* stream_handle) {
  if (nbytes == 0) return;

  // CPU↔CPU would be a routing bug at the core layer (`Storage::
  // copy_device_bytes_async` short-circuits it to `std::memcpy`
  // before reaching this bridge). Stay defensive anyway — the
  // stream argument is simply ignored in that case, same as
  // `cudaMemcpyAsync` would do.
  if (dst_device < 0 && src_device < 0) {
    std::memcpy(dst, src, nbytes);
    return;
  }

  cudaMemcpyKind kind{};
  int ctx_device = 0;
  if (dst_device >= 0 && src_device < 0) {
    kind = cudaMemcpyHostToDevice;
    ctx_device = dst_device;
  } else if (dst_device < 0 && src_device >= 0) {
    kind = cudaMemcpyDeviceToHost;
    ctx_device = src_device;
  } else {
    kind = cudaMemcpyDeviceToDevice;
    ctx_device = dst_device;
  }

  DeviceGuard g(ctx_device);
  // `stream_handle` is the `cudaStream_t` returned by
  // `Stream::native_handle()`. Passing it directly lets the driver
  // enqueue the transfer onto that stream; the call returns as soon
  // as the transfer is submitted. Caller is responsible for
  // `stream.synchronize()` (or Event-based cross-stream ordering)
  // before reading the destination — see Storage.hpp for the
  // contract.
  //
  // Note: if `src` is pageable (non-pinned) host memory on an H→D
  // copy, the CUDA driver silently stages through a tiny pinned
  // bounce buffer, which makes the copy effectively synchronous.
  // Correctness is preserved; overlap is not. Pair with
  // `PinnedHostAllocator` to get the real async benefit.
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  check(cudaMemcpyAsync(dst, src, nbytes, kind, stream),
        "cudaMemcpyAsync");
}

void real_cuda_memcpy_2d_async(void* dst, std::size_t dpitch,
                               const void* src, std::size_t spitch,
                               std::size_t width, std::size_t height,
                               int device_index, void* stream_handle) {
  if (width == 0 || height == 0) return;
  DeviceGuard g(device_index);
  // Single strided D→D copy: `height` rows of `width` contiguous bytes,
  // src rows packed at `spitch`, dst rows at `dpitch`. Replaces a
  // per-row memcpy loop with one call (one graph node under capture),
  // which is both faster and capture-friendly (fewer nodes, no host
  // sync). `cudaMemcpyDeviceToDevice` keeps it on-device.
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  check(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                          cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DAsync");
}

void real_cuda_memset_zero(int device_index, void* ptr, std::size_t nbytes) {
  if (nbytes == 0) return;
  DeviceGuard g(device_index);
  check(cudaMemset(ptr, 0, nbytes), "cudaMemset");
}

void real_cuda_memset_zero_async(int device_index, void* ptr,
                                 std::size_t nbytes, void* stream_handle) {
  if (nbytes == 0) return;
  DeviceGuard g(device_index);
  // `cudaMemsetAsync` on the (possibly capturing) stream is recorded as a
  // graph node when capture is active, and otherwise behaves like an async
  // memset — ordering is preserved for subsequent ops on the same stream.
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  check(cudaMemsetAsync(ptr, 0, nbytes, stream), "cudaMemsetAsync");
}

}  // namespace tesseract::cuda::detail

#endif  // TESSERACT_HAS_CUDA

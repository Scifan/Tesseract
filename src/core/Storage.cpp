#include "tesseract/core/Storage.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tesseract/core/Stream.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract {

// Strong-symbol bridge into tesseract_cuda. Same pattern as the
// allocator hook in `Allocator.cpp`: the definitions live in
// `src/cuda/Memcpy.cpp`, compiled unconditionally (stub-throws in the
// CPU-only build, real `cudaMemcpy` / `cudaMemset` in the CUDA build).
// Declaring them as strong references here creates a hard link-time
// dependency that flips the static-library order so ld resolves them
// on the first left-to-right scan; see `src/core/CMakeLists.txt` for
// the matching PRIVATE `tesseract_cuda` link dep that keeps the
// graph acyclic.
namespace cuda::detail {
void real_cuda_memcpy(void* dst, int dst_device,
                      const void* src, int src_device,
                      std::size_t nbytes);
void real_cuda_memcpy_async(void* dst, int dst_device,
                            const void* src, int src_device,
                            std::size_t nbytes,
                            void* stream_handle);
void real_cuda_memcpy_2d_async(void* dst, std::size_t dpitch,
                               const void* src, std::size_t spitch,
                               std::size_t width, std::size_t height,
                               int device_index, void* stream_handle);
void real_cuda_memset_zero(int device_index, void* ptr, std::size_t nbytes);
void real_cuda_memset_zero_async(int device_index, void* ptr,
                                 std::size_t nbytes, void* stream_handle);
bool real_stream_is_capturing(int device_index, void* stream) noexcept;
}  // namespace cuda::detail

Storage::Storage(std::size_t nbytes, Allocator* alloc, std::size_t alignment)
    : nbytes_(nbytes), allocator_(alloc) {
  TESSERACT_CHECK(alloc != nullptr, "Storage requires a non-null Allocator for owning mode");
  device_ = alloc->device();
  data_ = alloc->allocate(nbytes, alignment);
}

Storage::Storage(BorrowTag, void* data, std::size_t nbytes, Device device)
    : data_(data), nbytes_(nbytes), allocator_(nullptr), device_(device) {}

Storage::~Storage() {
  if (allocator_ != nullptr) {
    allocator_->deallocate(data_, nbytes_);
  }
}

std::shared_ptr<Storage> Storage::make_owning(std::size_t nbytes, Allocator* alloc,
                                              std::size_t alignment) {
  return std::make_shared<Storage>(nbytes, alloc, alignment);
}

std::shared_ptr<Storage> Storage::make_borrowed(void* data, std::size_t nbytes, Device device) {
  return std::make_shared<Storage>(BorrowTag{}, data, nbytes, device);
}

void Storage::copy_device_bytes(void* dst, Device dst_device,
                                const void* src, Device src_device,
                                std::size_t nbytes) {
  if (nbytes == 0) return;
  TESSERACT_CHECK(dst != nullptr && src != nullptr,
                  "Storage::copy_device_bytes: null pointer (dst={}, src={})",
                  fmt::ptr(dst), fmt::ptr(src));

  const bool dst_cpu = dst_device.is_cpu();
  const bool src_cpu = src_device.is_cpu();

  if (dst_cpu && src_cpu) {
    std::memcpy(dst, src, nbytes);
    return;
  }

  // Anything touching CUDA goes through the CUDA HAL. We convert the
  // Device-space indices to the sentinel-encoded int the bridge
  // expects (`-1` for CPU, `>= 0` for CUDA device).
  const bool dst_cuda = dst_device.is_cuda();
  const bool src_cuda = src_device.is_cuda();
  TESSERACT_CHECK(dst_cuda || dst_cpu,
                  "copy_device_bytes: unsupported dst device {}", dst_device.to_string());
  TESSERACT_CHECK(src_cuda || src_cpu,
                  "copy_device_bytes: unsupported src device {}", src_device.to_string());

  const int dst_idx = dst_cuda ? dst_device.index : -1;
  const int src_idx = src_cuda ? src_device.index : -1;

  // Capture-safe fast path: while a CUDA graph is being captured on the
  // current stream, a synchronous `cudaMemcpy` (and the drain
  // `synchronize()` below) is illegal. Issue the copy as
  // `cudaMemcpyAsync` on the capturing stream instead — it's recorded as a
  // graph node and ordering is preserved for everything else on that
  // stream. This only triggers during capture; the non-capture path below
  // is byte-for-byte the prior behavior.
  {
    const Device cuda_side = dst_cuda ? dst_device : src_device;
    const Stream cs = current_stream(cuda_side);
    if (cuda::detail::real_stream_is_capturing(cuda_side.index,
                                               cs.native_handle())) {
      if (std::getenv("TS_CAPTURE_DEBUG"))
        std::fprintf(stderr, "[cap] copy dst_idx=%d src_idx=%d nbytes=%zu\n",
                     dst_idx, src_idx, nbytes);
      cuda::detail::real_cuda_memcpy_async(dst, dst_idx, src, src_idx, nbytes,
                                           cs.native_handle());
      return;
    }
  }

  // Drain any in-flight kernels on the device(s) involved before we
  // issue the synchronous `cudaMemcpy`. The M2E elementwise kernels
  // run on the per-device / per-thread non-blocking stream returned
  // by `current_stream(device)`, and `cudaMemcpy` (synchronous) only
  // waits for work on the legacy *null* stream — not on arbitrary
  // user streams. Without this, a kernel that is still in flight
  // when the copy starts can be read past, producing stale (often
  // all-zero) bytes on the host. `compute-sanitizer --tool memcheck`
  // surfaces this as a wrong-result regression on D→H copies (the
  // sanitizer perturbs timing enough for the race to manifest every
  // run). Syncing here — rather than inside every kernel launcher —
  // keeps CUDA ops themselves truly async and pays the sync cost
  // only when crossing the device boundary. D→D copies still need
  // the sync so a chain `kernel → copy → kernel` on two different
  // streams doesn't drop bytes.
  if (src_cuda) current_stream(src_device).synchronize();
  if (dst_cuda && !(src_cuda && src_idx == dst_idx)) {
    current_stream(dst_device).synchronize();
  }
  cuda::detail::real_cuda_memcpy(dst, dst_idx, src, src_idx, nbytes);
}

void Storage::copy_device_bytes_async(void* dst, Device dst_device,
                                      const void* src, Device src_device,
                                      std::size_t nbytes,
                                      const Stream& stream) {
  if (nbytes == 0) return;
  TESSERACT_CHECK(dst != nullptr && src != nullptr,
                  "Storage::copy_device_bytes_async: null pointer (dst={}, src={})",
                  fmt::ptr(dst), fmt::ptr(src));

  const bool dst_cpu = dst_device.is_cpu();
  const bool src_cpu = src_device.is_cpu();

  if (dst_cpu && src_cpu) {
    // CPU↔CPU has no async variant in our HAL — the caller almost
    // certainly wants immediate visibility after the call returns.
    // Matches the no-async contract documented in Storage.hpp.
    std::memcpy(dst, src, nbytes);
    return;
  }

  const bool dst_cuda = dst_device.is_cuda();
  const bool src_cuda = src_device.is_cuda();
  TESSERACT_CHECK(dst_cuda || dst_cpu,
                  "copy_device_bytes_async: unsupported dst device {}",
                  dst_device.to_string());
  TESSERACT_CHECK(src_cuda || src_cpu,
                  "copy_device_bytes_async: unsupported src device {}",
                  src_device.to_string());

  // Stream validation. The stream must live on the CUDA-side device;
  // for H↔D that's the non-CPU endpoint, for D↔D it has to be the
  // destination's device (cudaMemcpyAsync uses the calling context).
  if (src_cuda && dst_cuda) {
    TESSERACT_CHECK(stream.device().is_cuda() &&
                    stream.device().index == dst_device.index,
                    "copy_device_bytes_async: D→D requires stream on dst "
                    "device, got stream on {} but dst on {}",
                    stream.device().to_string(), dst_device.to_string());
  } else {
    const Device cuda_side = src_cuda ? src_device : dst_device;
    TESSERACT_CHECK(stream.device().is_cuda() &&
                    stream.device().index == cuda_side.index,
                    "copy_device_bytes_async: H↔D requires stream on the "
                    "CUDA endpoint's device, got stream on {} but "
                    "CUDA side is {}",
                    stream.device().to_string(), cuda_side.to_string());
  }

  const int dst_idx = dst_cuda ? dst_device.index : -1;
  const int src_idx = src_cuda ? src_device.index : -1;

  // Unlike the synchronous path we do NOT drain `current_stream` here
  // — the whole point of the async primitive is to preserve ordering
  // via the explicit `stream` argument. Callers that need to see the
  // result on the host must `stream.synchronize()` before reading;
  // callers that want to feed the result into a kernel on a different
  // stream must record an Event and `wait` on it. That's the standard
  // CUDA overlap contract and we preserve it verbatim.
  cuda::detail::real_cuda_memcpy_async(dst, dst_idx, src, src_idx, nbytes,
                                       stream.native_handle());
}

void Storage::copy_device_bytes_2d_async(void* dst, std::size_t dpitch,
                                         const void* src, std::size_t spitch,
                                         std::size_t width, std::size_t height,
                                         Device device, const Stream& stream) {
  if (width == 0 || height == 0) return;
  TESSERACT_CHECK(dst != nullptr && src != nullptr,
                  "Storage::copy_device_bytes_2d_async: null pointer (dst={}, src={})",
                  fmt::ptr(dst), fmt::ptr(src));
  if (device.is_cpu()) {
    // Host strided copy: row-by-row memcpy. No async variant on CPU.
    auto* d = static_cast<std::byte*>(dst);
    const auto* s = static_cast<const std::byte*>(src);
    for (std::size_t r = 0; r < height; ++r) {
      std::memcpy(d + r * dpitch, s + r * spitch, width);
    }
    return;
  }
  TESSERACT_CHECK(device.is_cuda(),
                  "copy_device_bytes_2d_async: unsupported device {}",
                  device.to_string());
  TESSERACT_CHECK(stream.device().is_cuda() &&
                  stream.device().index == device.index,
                  "copy_device_bytes_2d_async: stream device {} != target {}",
                  stream.device().to_string(), device.to_string());
  cuda::detail::real_cuda_memcpy_2d_async(dst, dpitch, src, spitch, width,
                                          height, device.index,
                                          stream.native_handle());
}

void Storage::zero_device_bytes(void* ptr, Device device, std::size_t nbytes) {
  if (nbytes == 0) return;
  TESSERACT_CHECK(ptr != nullptr, "Storage::zero_device_bytes: null pointer");
  if (device.is_cpu()) {
    std::memset(ptr, 0, nbytes);
    return;
  }
  TESSERACT_CHECK(device.is_cuda(),
                  "zero_device_bytes: unsupported device {}", device.to_string());
  // Capture-safe fast path: a synchronous `cudaMemset` (and the drain
  // `synchronize()` below) is illegal while the current stream is
  // capturing a CUDA graph. Record an async memset on the capturing
  // stream instead — ordering is preserved for subsequent same-stream ops.
  // Only triggers during capture; the non-capture path is unchanged.
  {
    const Stream cs = current_stream(device);
    if (cuda::detail::real_stream_is_capturing(device.index,
                                               cs.native_handle())) {
      if (std::getenv("TS_CAPTURE_DEBUG"))
        std::fprintf(stderr, "[cap] memset device=%d nbytes=%zu\n",
                     device.index, nbytes);
      cuda::detail::real_cuda_memset_zero_async(device.index, ptr, nbytes,
                                                cs.native_handle());
      return;
    }
  }
  // Same rationale as in `copy_device_bytes`: `cudaMemset` issued on
  // the null stream doesn't wait for pending work on `current_stream`,
  // so a subsequent host read (or a later kernel on the same stream
  // expecting the zeroed buffer) could race with an older in-flight
  // kernel. The sync is cheap compared to the allocation that
  // usually precedes this call.
  current_stream(device).synchronize();
  cuda::detail::real_cuda_memset_zero(device.index, ptr, nbytes);
}

}  // namespace tesseract

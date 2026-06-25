#include <cstddef>

#include <fmt/format.h>

#include "tesseract/utils/Logging.hpp"

#if defined(TESSERACT_HAS_CUDA)
#include <cuda_runtime.h>
#include "Internal.hpp"
#endif

// CUDA-side implementation of the stream / event bridge declared in
// `src/cuda/Internal.hpp` and consumed by `src/core/Stream.cpp`.
//
// Mirroring the pattern used for `Allocator.cpp`, this TU is always
// compiled. The CPU-only build keeps only stub bodies that throw a
// clear DeviceError — unreachable in practice (Stream::create refuses
// to produce a CUDA stream without a CUDA backend), but it keeps
// `tesseract_core` linkable without conditional CMake magic.

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA stream / event primitive called but the CUDA "
      "backend was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void* real_stream_create(int /*device_index*/) { throw_not_built(); }
void real_stream_destroy(int /*device_index*/, void* /*stream*/) noexcept {}
void real_stream_synchronize(int /*device_index*/, void* /*stream*/) { throw_not_built(); }
void real_stream_wait_event(int /*device_index*/, void* /*stream*/, void* /*event*/) {
  throw_not_built();
}

void* real_event_create(int /*device_index*/) { throw_not_built(); }
void real_event_destroy(void* /*event*/) noexcept {}
void real_event_record(int /*device_index*/, void* /*event*/, void* /*stream*/) {
  throw_not_built();
}
void real_event_synchronize(void* /*event*/) { throw_not_built(); }
bool real_event_query(void* /*event*/) noexcept { return true; }
bool real_stream_is_capturing(int /*device_index*/, void* /*stream*/) noexcept {
  return false;
}

}  // namespace tesseract::cuda::detail

#else  // TESSERACT_HAS_CUDA

namespace tesseract::cuda::detail {

namespace {

// Small device-scoping RAII helper; same shape as the one in
// `Allocator.cpp`. Kept file-local here so the two compilation units
// don't race on ODR — they never share a symbol.
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

void* real_stream_create(int device_index) {
  DeviceGuard g(device_index);
  cudaStream_t s{};
  // `cudaStreamNonBlocking` ensures ops on `s` don't implicitly
  // synchronize with the legacy default stream. Without this the
  // kernel launches we issue on `s` would still block on cuBLAS /
  // cuRAND internals that are hardcoded to the default stream.
  check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  return static_cast<void*>(s);
}

void real_stream_destroy(int device_index, void* stream) noexcept {
  if (stream == nullptr) return;
  int previous = -1;
  if (cudaGetDevice(&previous) != cudaSuccess) previous = -1;
  (void)cudaSetDevice(device_index);
  (void)cudaStreamDestroy(static_cast<cudaStream_t>(stream));
  if (previous >= 0) (void)cudaSetDevice(previous);
}

void real_stream_synchronize(int device_index, void* stream) {
  DeviceGuard g(device_index);
  check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
}

bool real_stream_is_capturing(int device_index, void* stream) noexcept {
  int previous = -1;
  if (cudaGetDevice(&previous) != cudaSuccess) previous = -1;
  if (cudaSetDevice(device_index) != cudaSuccess) {
    if (previous >= 0) (void)cudaSetDevice(previous);
    return false;
  }
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cudaError_t err =
      cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status);
  if (previous >= 0) (void)cudaSetDevice(previous);
  // Clear the error flag so a benign query doesn't poison later calls.
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return status == cudaStreamCaptureStatusActive;
}

void real_stream_wait_event(int device_index, void* stream, void* event) {
  DeviceGuard g(device_index);
  check(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),
                            static_cast<cudaEvent_t>(event), 0),
        "cudaStreamWaitEvent");
}

void* real_event_create(int device_index) {
  DeviceGuard g(device_index);
  cudaEvent_t e{};
  // `cudaEventDisableTiming` halves the cost of a record because the
  // driver doesn't have to stamp a microsecond clock. M2 doesn't need
  // timing events (that's a profiler concern); if we ever add a
  // profiler event type we'll expose a second factory.
  check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreateWithFlags");
  return static_cast<void*>(e);
}

void real_event_destroy(void* event) noexcept {
  if (event == nullptr) return;
  (void)cudaEventDestroy(static_cast<cudaEvent_t>(event));
}

void real_event_record(int device_index, void* event, void* stream) {
  DeviceGuard g(device_index);
  check(cudaEventRecord(static_cast<cudaEvent_t>(event),
                        static_cast<cudaStream_t>(stream)),
        "cudaEventRecord");
}

void real_event_synchronize(void* event) {
  check(cudaEventSynchronize(static_cast<cudaEvent_t>(event)), "cudaEventSynchronize");
}

bool real_event_query(void* event) noexcept {
  cudaError_t err = cudaEventQuery(static_cast<cudaEvent_t>(event));
  return err == cudaSuccess;
}

}  // namespace tesseract::cuda::detail

#endif  // TESSERACT_HAS_CUDA

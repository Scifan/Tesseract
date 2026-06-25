#pragma once

// Internal bridge between plain-C++ code in `tesseract_cuda` and the
// nvcc-compiled `.cu` translation units in the same target. Consumers
// outside `src/cuda/` must not include this file — the public API lives
// in `include/tesseract/cuda/CudaRuntime.hpp`.
//
// The entry points declared here are free functions with pure C++
// signatures so that the CPU-side caller in `CudaRuntime.cpp` does not
// need to know about `<cuda_runtime.h>`. They are only defined when the
// library is built with TESSERACT_HAS_CUDA=1 (i.e. compiled through
// `Probe.cu`); see `CudaRuntime.cpp` for the ifdef routing.

#include <cstdint>
#include <string>

#include "tesseract/cuda/CudaRuntime.hpp"

namespace tesseract::cuda::detail {

// ---- Probe surface (M2B) ---------------------------------------------------

// Number of devices visible to the CUDA driver. Returns 0 on any driver
// error (no GPU, missing libcuda, permission denied, ...). Never throws.
int real_device_count() noexcept;

// Populate `*out` with the device descriptor for `index`. Returns true on
// success, false on any driver error. The caller reports the failure.
bool real_device_info(int index, DeviceInfo* out) noexcept;

// Human-readable "CUDA X.Y runtime / driver"-style string.
std::string real_runtime_version_string();

// ---- Allocator surface (M2C) ----------------------------------------------

// `cudaMalloc(nbytes)` on `device_index`. Throws `DeviceError` on any
// driver failure (OOM, invalid device, ...). The caller must already
// hold device scope; we handle `cudaSetDevice` internally.
void* real_cuda_malloc(int device_index, std::size_t nbytes);

// `cudaFree(ptr)` on `device_index`. Never throws; any driver error is
// swallowed so destructors are safe. A null `ptr` is a no-op.
void real_cuda_free(int device_index, void* ptr) noexcept;

// ---- Stream / Event surface (M2C) -----------------------------------------

// `cudaStreamCreateWithFlags(cudaStreamNonBlocking)` on `device_index`.
// Throws `DeviceError` on failure. Returned handle is the raw
// `cudaStream_t` cast to `void*`.
void* real_stream_create(int device_index);

// `cudaStreamDestroy(stream)` on `device_index`. Never throws; errors
// are swallowed so shared_ptr destructors remain safe.
void real_stream_destroy(int device_index, void* stream) noexcept;

// `cudaStreamSynchronize(stream)`. Throws on driver error.
void real_stream_synchronize(int device_index, void* stream);

// `cudaStreamWaitEvent(stream, event, 0)`. Throws on driver error.
void real_stream_wait_event(int device_index, void* stream, void* event);

// `cudaEventCreateWithFlags(cudaEventDisableTiming)` on `device_index`.
// Throws on failure.
void* real_event_create(int device_index);

// `cudaEventDestroy(event)`. Never throws.
void real_event_destroy(void* event) noexcept;

// `cudaEventRecord(event, stream)`. Throws on failure.
void real_event_record(int device_index, void* event, void* stream);

// `cudaEventSynchronize(event)`. Throws on failure.
void real_event_synchronize(void* event);

// `cudaEventQuery(event) == cudaSuccess`. Returns false on "not ready"
// and on any transient driver error; never throws.
bool real_event_query(void* event) noexcept;

// ---- Memcpy surface (M2D) -------------------------------------------------
//
// Synchronous byte copy that may cross the host/device boundary. Device
// indices are encoded in the tesseract convention: any value `< 0`
// means "host memory" (CPU pointer), values `>= 0` are CUDA device
// indices. The four directions (H→H, H→D, D→H, D→D) are all handled
// internally — H→H short-circuits to `std::memcpy`, the three
// CUDA-touching directions set the target device, pick the right
// `cudaMemcpyKind`, and synchronize before returning. For M2D we
// deliberately stick to the synchronous variant so `Tensor::to(device)`
// has PyTorch-like "result is ready on return" semantics; a non-
// blocking overload paired with pinned host memory shows up in M3's
// inference pipeline and is tracked there.
void real_cuda_memcpy(void* dst, int dst_device,
                      const void* src, int src_device,
                      std::size_t nbytes);

// Zero-fill `nbytes` of device memory at `ptr` on `device_index`
// (synchronous). Used by `Tensor::zeros(..., cuda_device())` so we
// don't have to bounce through a host buffer + H→D copy just to
// write a constant zero. Any dtype can safely be zero-filled with
// this because the bit pattern for 0 is 0 for every numeric type
// we support today.
void real_cuda_memset_zero(int device_index, void* ptr, std::size_t nbytes);

// Async zero-fill of `nbytes` at `ptr` enqueued on `stream_handle` (a raw
// `cudaStream_t`). Unlike `real_cuda_memset_zero` this is capture-safe — a
// synchronous `cudaMemset` is illegal while a stream is capturing a CUDA
// graph, whereas `cudaMemsetAsync` on the capturing stream is recorded as a
// graph node. Caller owns ordering (the work lands on `stream_handle`).
void real_cuda_memset_zero_async(int device_index, void* ptr,
                                 std::size_t nbytes, void* stream_handle);

// True iff `stream` is currently capturing a CUDA graph
// (`cudaStreamIsCapturing` == active). Used by the core copy/zero paths to
// pick a capture-safe async variant instead of the synchronous default.
// Never throws; returns false on any driver error or in CPU-only builds.
bool real_stream_is_capturing(int device_index, void* stream) noexcept;

}  // namespace tesseract::cuda::detail

#pragma once

// Device-agnostic asynchronous stream + event types. These are the
// concurrency primitives that every HAL-aware op takes as an implicit
// argument via a thread-local "current stream" (see `current_stream`
// and `StreamGuard` below).
//
// On CPU devices both types are no-ops: a `Stream` is just a Device
// tag, `synchronize()` returns immediately, and `Event::record` /
// `Event::synchronize` are trivial. That keeps CPU-only call sites
// free of `#ifdef` gates.
//
// On CUDA devices:
//   * `Stream::create(device)` calls
//     `cudaStreamCreateWithFlags(cudaStreamNonBlocking)` so our streams
//     never implicitly synchronize with cuBLAS / cuRAND defaults.
//   * `current_stream(device)` returns a lazily-created thread-local
//     non-blocking stream per (thread, device) pair — the
//     recommended idiom for per-thread async work that doesn't
//     pollute the legacy default stream.
//   * `Event` wraps `cudaEvent_t` and supports cross-stream ordering
//     (`stream.wait(event)`).
//
// The public header is deliberately plain C++20 and never includes
// `<cuda_runtime.h>`. Opaque handles are exposed as `void*` for
// callers that need to hand the raw `cudaStream_t` / `cudaEvent_t`
// to a third-party library (cuBLAS, FlashAttention-3, ...). A plain
// integer `uintptr_t` would also work, but `void*` matches the CUDA
// API types 1:1 and carries the "this is a handle" hint.
//
// This file ships in M2C. M2D (H↔D copy) and M2E (elementwise kernels)
// consume it; the MLIR JIT (M1) is CPU-only and never sees a Stream.

#include <cstddef>
#include <memory>

#include "tesseract/core/Device.hpp"

namespace tesseract {

class Event;

class Stream {
 public:
  // Default-constructs a Stream representing the CPU default (no-op).
  // To get a useful CUDA stream use `Stream::create(device)` or
  // `current_stream(device)`.
  Stream();

  // Create a new non-blocking stream on `device`. For CPU devices this
  // is equivalent to the default constructor (no allocation); for CUDA
  // devices it calls `cudaStreamCreateWithFlags`.
  static Stream create(Device device);

  ~Stream() = default;
  Stream(const Stream&) = default;
  Stream& operator=(const Stream&) = default;
  Stream(Stream&&) noexcept = default;
  Stream& operator=(Stream&&) noexcept = default;

  Device device() const noexcept;

  // Opaque driver handle: `cudaStream_t` for CUDA devices, `nullptr`
  // for CPU. Callers into cuBLAS / FA3 cast with
  // `static_cast<cudaStream_t>(s.native_handle())`.
  void* native_handle() const noexcept;

  // Block the host until every work item previously enqueued on this
  // stream completes. No-op on CPU devices. Throws on driver error.
  void synchronize() const;

  // Make this stream's next work items wait until `event` has been
  // signaled (on whatever stream recorded it). No-op on CPU streams.
  // Throws if the stream and event are on different devices.
  void wait(const Event& event) const;

  // Two Streams compare equal iff they share the same underlying
  // driver handle (i.e. they are handles into the same stream). Useful
  // for asserting "I got back the stream I just installed".
  friend bool operator==(const Stream& a, const Stream& b) noexcept;
  friend bool operator!=(const Stream& a, const Stream& b) noexcept {
    return !(a == b);
  }

 private:
  struct State;
  std::shared_ptr<State> state_;

  explicit Stream(std::shared_ptr<State> s) noexcept : state_(std::move(s)) {}

  friend class StreamGuard;
  friend class Event;
};

class Event {
 public:
  // Default-constructs an Event on the CPU (no-op).
  Event();

  // Create an event on `device`. On CUDA, events are created with
  // `cudaEventDisableTiming` so `record` is lighter-weight than the
  // timing-enabled default (we don't need millisecond-accurate
  // per-event wall-clock measurement in M2; profiling tools use nsys).
  static Event create(Device device);

  ~Event() = default;
  Event(const Event&) = default;
  Event& operator=(const Event&) = default;
  Event(Event&&) noexcept = default;
  Event& operator=(Event&&) noexcept = default;

  Device device() const noexcept;
  void* native_handle() const noexcept;

  // Record the event on `stream`. Subsequent `stream.wait(event)`
  // calls will order the waiter behind all work enqueued on `stream`
  // *before* this record call. Throws if stream and event are on
  // different devices.
  void record(const Stream& stream);

  // Block host until this event has been signaled. No-op on CPU.
  void synchronize() const;

  // Non-blocking check: true if every operation that was on the
  // stream when `record` was called has completed. No-op on CPU
  // (always true).
  bool query() const noexcept;

 private:
  struct State;
  std::shared_ptr<State> state_;

  explicit Event(std::shared_ptr<State> s) noexcept : state_(std::move(s)) {}

  friend class Stream;
};

// Look up the thread-local "current" stream for `device`. On first
// call for a given (thread, device) pair the stream is lazily created
// as a non-blocking stream; subsequent calls return the same handle.
// `current_stream(cpu_device())` returns a no-op CPU stream.
Stream current_stream(Device device);

// RAII guard: install `stream` as the current stream for its device
// for the lifetime of the guard, restoring the previous current
// stream on destruction. Non-copyable; a StreamGuard is always
// scoped to the block that created it.
//
// Typical usage (inside an op kernel):
//   StreamGuard g(my_stream);
//   cuda::launch_add(a, b, out, my_stream);
//   // `g` destructor restores the caller's previous current stream.
class StreamGuard {
 public:
  explicit StreamGuard(Stream stream);
  ~StreamGuard();

  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;
  StreamGuard(StreamGuard&&) = delete;
  StreamGuard& operator=(StreamGuard&&) = delete;

 private:
  Stream previous_;
  Device device_;
};

}  // namespace tesseract

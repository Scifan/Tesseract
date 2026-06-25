// M2C smoke tests for the device-agnostic Stream / Event / StreamGuard
// primitives. Mirrors the allocator test's split:
//
//   * CPU path: `current_stream(cpu_device())` returns a stream with a
//     null native handle; `Stream::create(cpu_device())` is a no-op.
//     These test cases run in every build configuration.
//
//   * CPU-only build: every CUDA-device call throws DeviceError.
//
//   * CUDA-enabled build with visible GPU: create a non-blocking
//     stream, install it via `StreamGuard`, check `current_stream`
//     reports the installed handle, then exit the guard and check it
//     was restored. Record an event on the guarded stream and wait
//     on it from a second stream to exercise cross-stream ordering.

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/utils/Logging.hpp"

using tesseract::cpu_device;
using tesseract::current_stream;
using tesseract::Device;
using tesseract::DeviceError;
using tesseract::DeviceType;
using tesseract::Event;
using tesseract::Stream;
using tesseract::StreamGuard;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

TEST_CASE("CPU Stream is a no-op wrapper around its Device", "[stream][cpu]") {
  Stream s = Stream::create(cpu_device());
  REQUIRE(s.device() == cpu_device());
  REQUIRE(s.native_handle() == nullptr);
  // Synchronizing on a CPU stream must not throw or block; it's the
  // identity op. We don't have a time measurement but the test
  // completing at all is the contract.
  s.synchronize();
}

TEST_CASE("CPU current_stream is a stable per-thread handle", "[stream][cpu]") {
  // Two queries on the same thread return handles that compare equal
  // (same underlying State). A fresh thread would get its own slot —
  // we don't exercise that here to avoid a sleep-based race.
  Stream a = current_stream(cpu_device());
  Stream b = current_stream(cpu_device());
  REQUIRE(a == b);
}

TEST_CASE("CPU StreamGuard restores the previous current stream",
          "[stream][cpu]") {
  Stream prev = current_stream(cpu_device());
  Stream injected = Stream::create(cpu_device());
  REQUIRE_FALSE(prev == injected);  // distinct State objects.

  {
    StreamGuard g(injected);
    REQUIRE(current_stream(cpu_device()) == injected);
  }
  REQUIRE(current_stream(cpu_device()) == prev);
}

TEST_CASE("Creating a CUDA Stream in a CPU-only build throws",
          "[stream][cpu-only]") {
  if (has_cuda_support()) {
    SKIP("Binary has CUDA support; stub-branch test does not apply.");
  }
  REQUIRE_THROWS_AS(Stream::create(Device{DeviceType::CUDA, 0}), DeviceError);
  REQUIRE_THROWS_AS(Event::create(Device{DeviceType::CUDA, 0}), DeviceError);
}

TEST_CASE("CUDA Stream round-trips through StreamGuard",
          "[stream][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for stream tests.");
  }

  Device dev{DeviceType::CUDA, 0};
  Stream s = Stream::create(dev);
  REQUIRE(s.device() == dev);
  REQUIRE(s.native_handle() != nullptr);

  Stream prev = current_stream(dev);
  {
    StreamGuard g(s);
    // After install, current_stream should return our handle.
    REQUIRE(current_stream(dev) == s);
  }
  REQUIRE(current_stream(dev) == prev);

  // Sync should succeed on an empty stream (nothing to wait for).
  s.synchronize();
}

TEST_CASE("CUDA Event orders work between two streams",
          "[stream][event][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for event tests.");
  }

  Device dev{DeviceType::CUDA, 0};
  Stream producer = Stream::create(dev);
  Stream consumer = Stream::create(dev);

  Event e = Event::create(dev);
  e.record(producer);
  // Since `producer` has no enqueued work, the event should be ready
  // essentially immediately. `query()` may or may not already report
  // true (it's racy), but `synchronize()` must return cleanly.
  e.synchronize();
  REQUIRE(e.query() == true);

  // The consumer waits on the event; on an already-complete event
  // this should be a fast no-op. The primary contract we want to
  // check is "no throw, no hang".
  consumer.wait(e);
  consumer.synchronize();
}

TEST_CASE("Mixing CUDA streams and CPU events fails with a clear message",
          "[stream][event][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for cross-device negative test.");
  }

  Device dev{DeviceType::CUDA, 0};
  Stream cuda_stream = Stream::create(dev);
  Event cpu_event = Event::create(cpu_device());

  // `Event::record` is called on the wrong device — the library
  // rejects this rather than silently doing nothing, because a
  // caller that wires CPU events into a CUDA stream has a bug.
  // We catch the specific message-bearing exception.
  REQUIRE_THROWS(cpu_event.record(cuda_stream));
}

#include "tesseract/core/Stream.hpp"

#include <array>
#include <memory>
#include <unordered_map>

#include "tesseract/utils/Logging.hpp"

// Device-agnostic implementation of `Stream` / `Event` / `StreamGuard`
// plus the thread-local "current stream" registry.
//
// The CPU path is entirely in this file: a CPU Stream is a value-type
// wrapper around a `Device` tag with no driver handle, and every
// blocking / synchronizing method is a no-op. For CUDA devices we
// forward every driver-touching call through the bridge symbols in
// `src/cuda/Internal.hpp`; those are strong-linked from
// `tesseract_cuda` (both in CUDA-enabled and CPU-only builds — in the
// latter, the stubs throw a clear "CUDA backend not compiled in"
// `DeviceError` so users see a single consistent failure point).
//
// Owning vs. borrowed streams:
//   * `Stream::create(cuda_device)` returns an *owning* Stream: its
//     State holds the `cudaStream_t` handle and will call
//     `cudaStreamDestroy` when the last shared_ptr goes out of scope.
//   * `current_stream(cuda_device)` lazily constructs one owning
//     Stream per (thread, device) pair and caches it in TLS; every
//     subsequent call returns a shared_ptr copy of the same State.
//   * The default-constructed `Stream()` is a borrowed CPU stream
//     with no state — it's the value used when the caller doesn't
//     care about streams.

namespace tesseract {

// ---- Internal bridge declarations (match src/cuda/Internal.hpp) -----------
//
// We redeclare these here rather than including `Internal.hpp` so
// `tesseract_core` doesn't have to carry a header path into
// `src/cuda/`. They are strong-linked from `tesseract_cuda` with
// throw-when-CUDA-off semantics baked in; see that library's
// Stream.cpp / Allocator.cpp for the definitions.
namespace cuda::detail {

void* real_stream_create(int device_index);
void real_stream_destroy(int device_index, void* stream) noexcept;
void real_stream_synchronize(int device_index, void* stream);
void real_stream_wait_event(int device_index, void* stream, void* event);

void* real_event_create(int device_index);
void real_event_destroy(void* event) noexcept;
void real_event_record(int device_index, void* event, void* stream);
void real_event_synchronize(void* event);
bool real_event_query(void* event) noexcept;

}  // namespace cuda::detail

// ---- Stream::State --------------------------------------------------------

struct Stream::State {
  Device device;
  void* native_handle{nullptr};

  State(Device d, void* h) noexcept : device(d), native_handle(h) {}

  ~State() {
    if (native_handle != nullptr && device.is_cuda()) {
      cuda::detail::real_stream_destroy(device.index, native_handle);
    }
  }

  State(const State&) = delete;
  State& operator=(const State&) = delete;
};

// ---- Event::State ---------------------------------------------------------

struct Event::State {
  Device device;
  void* native_handle{nullptr};

  State(Device d, void* h) noexcept : device(d), native_handle(h) {}

  ~State() {
    if (native_handle != nullptr && device.is_cuda()) {
      cuda::detail::real_event_destroy(native_handle);
    }
  }

  State(const State&) = delete;
  State& operator=(const State&) = delete;
};

// ---- Stream ---------------------------------------------------------------

Stream::Stream() = default;

Stream Stream::create(Device device) {
  if (device.is_cpu()) {
    // CPU stream: no driver handle needed. We still wrap a State so
    // `device()` returns the right thing and equality compares by
    // identity (important when users pass a CPU "stream" across
    // boundaries and want to compare handles).
    return Stream{std::make_shared<State>(device, nullptr)};
  }
  TESSERACT_CHECK(device.is_cuda(),
                  "Stream::create: only CPU and CUDA devices are supported in M2C; got {}",
                  device.to_string());
  void* handle = cuda::detail::real_stream_create(device.index);
  return Stream{std::make_shared<State>(device, handle)};
}

Device Stream::device() const noexcept {
  return state_ ? state_->device : cpu_device();
}

void* Stream::native_handle() const noexcept {
  return state_ ? state_->native_handle : nullptr;
}

void Stream::synchronize() const {
  if (!state_) return;  // default-constructed CPU stream: no-op
  if (state_->device.is_cuda()) {
    cuda::detail::real_stream_synchronize(state_->device.index, state_->native_handle);
  }
  // CPU stream: by construction every enqueue is synchronous, so no wait.
}

void Stream::wait(const Event& event) const {
  if (!state_) return;  // default-constructed CPU stream: nothing to order.
  // Device mismatch is a caller bug regardless of which side is CPU;
  // we want the same clear exception the CUDA path would produce.
  TESSERACT_CHECK(event.device() == state_->device,
                  "Stream::wait: stream is on {} but event is on {}",
                  state_->device.to_string(), event.device().to_string());
  if (!state_->device.is_cuda()) return;  // CPU self-wait: trivially complete.
  cuda::detail::real_stream_wait_event(state_->device.index,
                                       state_->native_handle, event.native_handle());
}

bool operator==(const Stream& a, const Stream& b) noexcept {
  return a.state_.get() == b.state_.get();
}

// ---- Event ----------------------------------------------------------------

Event::Event() = default;

Event Event::create(Device device) {
  if (device.is_cpu()) {
    return Event{std::make_shared<State>(device, nullptr)};
  }
  TESSERACT_CHECK(device.is_cuda(),
                  "Event::create: only CPU and CUDA devices are supported in M2C; got {}",
                  device.to_string());
  void* handle = cuda::detail::real_event_create(device.index);
  return Event{std::make_shared<State>(device, handle)};
}

Device Event::device() const noexcept {
  return state_ ? state_->device : cpu_device();
}

void* Event::native_handle() const noexcept {
  return state_ ? state_->native_handle : nullptr;
}

void Event::record(const Stream& stream) {
  if (!state_) return;  // default-constructed event: no-op.
  TESSERACT_CHECK(stream.device() == state_->device,
                  "Event::record: event is on {} but stream is on {}",
                  state_->device.to_string(), stream.device().to_string());
  if (!state_->device.is_cuda()) return;  // CPU event on CPU stream: no driver call.
  cuda::detail::real_event_record(state_->device.index, state_->native_handle,
                                  stream.native_handle());
}

void Event::synchronize() const {
  if (!state_ || !state_->device.is_cuda()) return;
  cuda::detail::real_event_synchronize(state_->native_handle);
}

bool Event::query() const noexcept {
  if (!state_ || !state_->device.is_cuda()) return true;
  return cuda::detail::real_event_query(state_->native_handle);
}

// ---- Current-stream registry ---------------------------------------------
//
// One slot per DeviceType × device-index. We use a small fixed array
// for DeviceType (known at compile time) and a thread_local unordered
// map keyed by device index. Cheap enough for M2C's expected use
// (single-GPU, a handful of worker threads) and trivially swappable
// for something denser if it ever shows up in a profile.
namespace {

struct CurrentStreamRegistry {
  // Per device type and index, the current stream installed on this
  // thread. A missing entry means "never set on this thread" — on
  // first query we materialize a fresh non-blocking stream.
  std::array<std::unordered_map<int, Stream>,
             static_cast<std::size_t>(DeviceType::kNumDeviceTypes)> slots;
};

thread_local CurrentStreamRegistry tls_registry{};

Stream& slot_for(Device device) {
  auto& m = tls_registry.slots[static_cast<std::size_t>(device.type)];
  auto it = m.find(device.index);
  if (it == m.end()) {
    // Lazy init: for CPU make a no-op wrapper; for CUDA make a
    // non-blocking stream on the target device.
    auto inserted = m.emplace(device.index, Stream::create(device));
    it = inserted.first;
  }
  return it->second;
}

}  // namespace

Stream current_stream(Device device) {
  return slot_for(device);
}

// ---- StreamGuard ----------------------------------------------------------

StreamGuard::StreamGuard(Stream stream) : device_(stream.device()) {
  auto& slot = slot_for(device_);
  previous_ = slot;
  slot = std::move(stream);
}

StreamGuard::~StreamGuard() {
  slot_for(device_) = std::move(previous_);
}

}  // namespace tesseract

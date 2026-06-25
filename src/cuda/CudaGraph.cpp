#include "tesseract/cuda/CudaGraph.hpp"

#include <fmt/format.h>
#include <cstdio>
#include <cstdlib>

#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/CudaAllocator.hpp"
#include "tesseract/utils/Logging.hpp"

#if defined(TESSERACT_HAS_CUDA)
#include <cuda_runtime.h>
#endif

// CUDA Graph implementation. This TU is always compiled; when
// TESSERACT_HAS_CUDA is not defined every method throws a clean
// DeviceError (same pattern as Allocator.cpp / Stream.cpp).

namespace tesseract::cuda {

#if defined(TESSERACT_HAS_CUDA)

struct CudaGraph::Impl {
  cudaGraph_t     graph{nullptr};       // raw captured graph
  cudaGraphExec_t graph_exec{nullptr};  // instantiated exec handle
  bool            instantiated{false};

  // Release any currently held driver objects, resetting this Impl to
  // "empty" state. Safe to call multiple times.
  void reset_driver_objects() noexcept {
    if (graph_exec != nullptr) {
      (void)cudaGraphExecDestroy(graph_exec);
      graph_exec = nullptr;
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
      graph = nullptr;
    }
    instantiated = false;
  }
};

namespace {

// Same RAII device-scoping helper as Allocator / Stream; kept
// TU-local so we don't race on ODR.
struct DeviceGuard {
  int previous{-1};
  explicit DeviceGuard(int target) {
    if (cudaGetDevice(&previous) != cudaSuccess) previous = -1;
    cudaError_t err = cudaSetDevice(target);
    if (err != cudaSuccess) {
      throw DeviceError(fmt::format(
          "[tesseract] CudaGraph: cudaSetDevice({}) failed: {}",
          target, cudaGetErrorString(err)));
    }
  }
  ~DeviceGuard() {
    if (previous >= 0) (void)cudaSetDevice(previous);
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
};

}  // namespace

CudaGraph::CudaGraph(int device_index)
    : impl_(std::make_unique<Impl>()), device_index_(device_index) {}

CudaGraph::~CudaGraph() {
  if (impl_) impl_->reset_driver_objects();
}

CudaGraph::CudaGraph(CudaGraph&&) noexcept = default;
CudaGraph& CudaGraph::operator=(CudaGraph&&) noexcept = default;

bool CudaGraph::instantiated() const noexcept {
  return impl_ && impl_->instantiated;
}

void CudaGraph::capture(const Stream& stream,
                        const std::function<void()>& fn) {
  TESSERACT_CHECK(fn != nullptr,
                  "CudaGraph::capture: callable must not be null");
  TESSERACT_CHECK(stream.device().is_cuda(),
                  "CudaGraph::capture: stream must be on a CUDA device, got {}",
                  stream.device().to_string());
  TESSERACT_CHECK(stream.device().index == device_index_,
                  "CudaGraph::capture: stream device ({}) != graph device ({})",
                  stream.device().index, device_index_);

  DeviceGuard dg(device_index_);

  // Rebuild from scratch — supports re-capture after workload
  // shape changes.
  impl_->reset_driver_objects();

  auto* raw_stream = static_cast<cudaStream_t>(stream.native_handle());

  // Install `stream` as the thread-local current stream for both the
  // warmup and capture phases. Every op inside `fn` resolves its
  // target stream via `current_stream(device)` — without this guard
  // the closure's kernels would be issued to whatever stream happened
  // to be current on the calling thread, which is almost certainly
  // NOT the one `cudaStreamBeginCapture` is watching, and the capture
  // would come back empty (or, worse, partially empty with an
  // `cudaErrorStreamCaptureInvalidated` at end time).
  StreamGuard sg(stream);

  // Defer allocator frees for the duration of warmup + capture so no
  // device block is recycled *within* the captured closure. Recycling
  // a block mid-capture makes two logically distinct tensors share one
  // address with no graph edge between them; on replay the kernels race
  // and bit-exactness is lost. The RAII guard guarantees we always
  // flush parked blocks and re-enable normal recycling, even if capture
  // throws. See CudaAllocator.hpp for the full rationale.
  struct CaptureAllocGuard {
    int dev;
    explicit CaptureAllocGuard(int d) : dev(d) {
      cuda::detail::cuda_alloc_begin_capture(d);
    }
    ~CaptureAllocGuard() { cuda::detail::cuda_alloc_end_capture(dev); }
    CaptureAllocGuard(const CaptureAllocGuard&) = delete;
    CaptureAllocGuard& operator=(const CaptureAllocGuard&) = delete;
  } cap_alloc_guard(device_index_);

  // --- Phase 1: warmup ----------------------------------------------------
  //
  // Drives the closure TWICE outside capture so the bucketed allocator
  // caches every size the closure needs — including the closure's
  // *output* slot. Why two passes:
  //
  //   * The typical capture closure writes its final result into a
  //     caller-owned slot (e.g. `out = ops::add(...)`). On the first
  //     warmup pass the result buffer is handed to that slot and
  //     stays alive; only intermediate temporaries are recycled. If
  //     we moved directly into capture at that point the closure
  //     would ask for N buffers of the same size but the cache only
  //     holds (N − 1).
  //   * On the second warmup pass the closure's first output buffer
  //     is freed when the slot is reassigned, returning the "last
  //     live" buffer to the cache as well. After this sync() the
  //     cache is fully primed and the subsequent capture pass never
  //     has to fall through to `cudaMalloc` — which is exactly the
  //     property stream capture requires.
  //
  // A full synchronize() between phases guarantees no in-flight
  // kernel is holding the GPU at capture-begin time; stream capture
  // is strict about that and will fail the whole capture if the
  // stream has pending work.
  fn();
  stream.synchronize();
  fn();
  stream.synchronize();

  // Return every block parked during warmup to the bucket free-lists so
  // the capture pass below pops from a fully primed pool (and never
  // falls through to a `cudaMalloc`, which is illegal under capture).
  // Frees made *during* the capture pass stay deferred — that's what
  // prevents intra-capture address reuse.
  cuda::detail::cuda_alloc_flush_deferred(device_index_);

  // --- Phase 2: capture ---------------------------------------------------
  //
  // Mode choice: `cudaStreamCaptureModeThreadLocal` restricts the
  // capture state to the current host thread. That's the safest
  // default — we don't block any other thread from issuing CUDA
  // work on other streams while this capture is in flight.
  {
    cudaError_t err = cudaStreamBeginCapture(
        raw_stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
      throw DeviceError(fmt::format(
          "[tesseract] CudaGraph::capture: cudaStreamBeginCapture failed: {}",
          cudaGetErrorString(err)));
    }
  }

  // Wrap the user closure so that any exception it throws still
  // ends the capture cleanly (otherwise the stream stays in capture
  // mode and every subsequent CUDA call fails). We collect
  // `user_exc` and rethrow after we've closed the capture + driver
  // objects.
  std::exception_ptr user_exc;
  try {
    fn();
  } catch (...) {
    user_exc = std::current_exception();
  }

  cudaGraph_t captured = nullptr;
  cudaError_t end_err = cudaStreamEndCapture(raw_stream, &captured);
  if (user_exc) {
    if (captured != nullptr) (void)cudaGraphDestroy(captured);
    std::rethrow_exception(user_exc);
  }
  if (end_err != cudaSuccess) {
    if (captured != nullptr) (void)cudaGraphDestroy(captured);
    throw DeviceError(fmt::format(
        "[tesseract] CudaGraph::capture: cudaStreamEndCapture failed: {}. "
        "This typically means the captured closure triggered a "
        "`cudaMalloc` (allocator cache miss) or another forbidden "
        "host-synchronizing call. Pre-allocate long-lived tensors "
        "and ensure the warmup pass covered every intermediate size.",
        cudaGetErrorString(end_err)));
  }
  impl_->graph = captured;

  // --- Phase 3: instantiate ----------------------------------------------
  //
  // Instantiate once; subsequent `launch` calls reuse `graph_exec`.
  cudaGraphExec_t exec = nullptr;
  cudaError_t inst_err = cudaGraphInstantiate(
      &exec, impl_->graph, /*pErrorNode=*/nullptr,
      /*pLogBuffer=*/nullptr, /*bufferSize=*/0);
  if (inst_err != cudaSuccess) {
    (void)cudaGraphDestroy(impl_->graph);
    impl_->graph = nullptr;
    throw DeviceError(fmt::format(
        "[tesseract] CudaGraph::capture: cudaGraphInstantiate failed: {}",
        cudaGetErrorString(inst_err)));
  }
  impl_->graph_exec = exec;
  impl_->instantiated = true;
}

void CudaGraph::launch(const Stream& stream) const {
  TESSERACT_CHECK(instantiated(),
                  "CudaGraph::launch: graph has not been captured yet");
  TESSERACT_CHECK(stream.device().is_cuda(),
                  "CudaGraph::launch: stream must be on a CUDA device, got {}",
                  stream.device().to_string());
  TESSERACT_CHECK(stream.device().index == device_index_,
                  "CudaGraph::launch: stream device ({}) != graph device ({})",
                  stream.device().index, device_index_);

  DeviceGuard dg(device_index_);
  auto* raw_stream = static_cast<cudaStream_t>(stream.native_handle());
  cudaError_t err = cudaGraphLaunch(impl_->graph_exec, raw_stream);
  if (err != cudaSuccess) {
    throw DeviceError(fmt::format(
        "[tesseract] CudaGraph::launch: cudaGraphLaunch failed: {}",
        cudaGetErrorString(err)));
  }
}

#else  // TESSERACT_HAS_CUDA

struct CudaGraph::Impl { int unused{0}; };

CudaGraph::CudaGraph(int device_index) : device_index_(device_index) {}
CudaGraph::~CudaGraph() = default;
CudaGraph::CudaGraph(CudaGraph&&) noexcept = default;
CudaGraph& CudaGraph::operator=(CudaGraph&&) noexcept = default;
bool CudaGraph::instantiated() const noexcept { return false; }

[[noreturn]] static void throw_not_built(const char* op) {
  throw DeviceError(fmt::format(
      "[tesseract] CudaGraph::{} called but the CUDA backend was not "
      "compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)", op));
}

void CudaGraph::capture(const Stream&, const std::function<void()>&) {
  throw_not_built("capture");
}
void CudaGraph::launch(const Stream&) const { throw_not_built("launch"); }

#endif  // TESSERACT_HAS_CUDA

}  // namespace tesseract::cuda

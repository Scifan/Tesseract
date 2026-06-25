#pragma once

// Wave 2.3 (B-023): CUDA Graph capture + replay primitive.
//
// Motivation: every M2 CUDA op submits one or more kernel launches
// through cuBLASLt / our bespoke kernels / cudaMemcpyAsync. For the
// decode-phase `MultiHeadAttention::forward_step` that's ~20+ launches
// per decoded token — each paying ~5–15 µs of host-side overhead
// (driver API + queueing) on top of the actual GPU work. At very
// small shapes (Llama-3.2-1B, one-token decode) the launch overhead
// stops being negligible and starts matching the real compute time.
//
// CUDA Graphs let us capture the full sequence of launches once,
// instantiate into a single `cudaGraphExec_t`, and replay with a
// single driver call. That collapses the per-step host work into
// one `cudaGraphLaunch` regardless of how many kernels the step
// actually runs — which is the infrastructure that unlocks the
// "roofline-throughput decode" target for Wave 2.
//
// Design contract — safety model for allocations inside capture:
//
//   * Stream capture rejects any call that would synchronize the
//     host with the capturing stream. That includes *every* new
//     `cudaMalloc` / `cudaFree`. Our `CudaAllocator` is a bucketed
//     cache on top of `cudaMalloc`: a cache *hit* is pure userspace
//     (pop from a mutex-guarded vector) and is safe inside capture;
//     a cache *miss* falls through to `cudaMalloc` and will break
//     the capture with `cudaErrorStreamCaptureUnsupported`.
//   * The caller is therefore responsible for *warming* the bucket
//     cache before `capture(fn)`: run the same closure once (or a
//     few times) on the target stream *before* entering capture so
//     every `Tensor::empty` the closure issues finds a cached block.
//     `capture(fn)` automatically runs the closure once in its own
//     warmup phase to guarantee this; users only need to ensure
//     that `fn` is deterministic w.r.t. allocation sizes.
//   * Long-lived inputs / outputs (weights, KVCache, token embedding
//     table, the returned logits buffer) **must** be allocated
//     outside capture. The graph records raw device pointers at
//     capture time; those pointers must stay valid for the lifetime
//     of the `CudaGraph`.
//
// Usage pattern (single-token decode step):
//
//     CudaGraph g(device_index);
//     g.capture(stream, [&]() {
//       out = mha.forward_step(x, cache);
//     });
//     for (int t = 0; t < num_tokens; ++t) {
//       copy_new_token_into(x);  // mutate *the same* buffer x points at
//       g.launch(stream);
//       stream.synchronize();
//       emit(out);
//     }
//
// On CPU-only builds every public method throws `DeviceError` with
// a clear "CUDA not compiled in" message, mirroring the rest of
// `tesseract/cuda`.

#include <functional>
#include <memory>

#include "tesseract/core/Stream.hpp"

namespace tesseract::cuda {

class CudaGraph {
 public:
  // Construct an empty, not-yet-captured graph bound to `device_index`.
  // Actual CUDA driver objects are lazily created on the first
  // `capture(...)` call.
  explicit CudaGraph(int device_index = 0);

  ~CudaGraph();

  CudaGraph(const CudaGraph&) = delete;
  CudaGraph& operator=(const CudaGraph&) = delete;

  CudaGraph(CudaGraph&&) noexcept;
  CudaGraph& operator=(CudaGraph&&) noexcept;

  // Capture `fn()` running on `stream` into a new `cudaGraphExec_t`.
  //
  // Semantics:
  //   1. A warmup pass: `fn()` is invoked once on `stream` with
  //      capture OFF. This primes the bucketed allocator cache with
  //      every block size that `fn` will request during capture —
  //      which is what makes subsequent in-capture allocations safe.
  //      After the warmup returns, `stream.synchronize()` is called
  //      so any transient tensors' storage has been recycled into
  //      the cache.
  //   2. Capture phase: `cudaStreamBeginCapture(stream,
  //      cudaStreamCaptureModeThreadLocal)` → `fn()` again →
  //      `cudaStreamEndCapture`. All `Tensor::empty` calls inside
  //      `fn` are expected to hit the warmed cache.
  //   3. Instantiate: `cudaGraphInstantiate` (auto-launch errors
  //      surface as a `DeviceError` with the driver message).
  //
  // If the graph has already been captured, this tears down the
  // previous `cudaGraphExec_t` and `cudaGraph_t` and rebuilds from
  // scratch — supports re-capture when the workload shape changes
  // (e.g. switching batch size).
  void capture(const Stream& stream, const std::function<void()>& fn);

  // Replay the captured graph on `stream`. Returns immediately after
  // enqueuing; caller must `stream.synchronize()` to block until the
  // replay completes. Throws if the graph hasn't been captured or if
  // `stream.device()` doesn't match the graph's device.
  void launch(const Stream& stream) const;

  // True iff a successful capture has produced an executable graph.
  bool instantiated() const noexcept;

  // Device the graph lives on.
  int device_index() const noexcept { return device_index_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int device_index_;
};

}  // namespace tesseract::cuda

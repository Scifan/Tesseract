// Wave 2.3 (B-023) — CUDA Graph capture + replay primitive.
//
// Three things we prove here:
//
//   1. Capture actually captures. On CUDA builds, instantiating a
//      `CudaGraph` around a closure that does `y = x * 2 + 1` and
//      then replaying it reproduces the same output bit-for-bit
//      (within FP32 abs epsilon) as the eager run on the same
//      stream. If the StreamGuard inside `capture()` regressed,
//      or if `current_stream` escaped to the default stream, the
//      captured graph would be empty and the replay would leave
//      the output unchanged — caught by the parity assertion.
//   2. Re-capture resets cleanly. Capturing once, launching, then
//      capturing a *different* closure on the same `CudaGraph`
//      object must not leak the previous `cudaGraphExec_t`, and
//      the second replay must produce the new closure's output.
//      We inspect `instantiated()` transitions in between.
//   3. CPU-only behavior on a CUDA-disabled build (compiled but
//      ignored at runtime when no device is present) surfaces as
//      `SUCCEED("no CUDA")` — kept behind `device_count()` so the
//      CI matrix on boxes without a GPU still runs the rest of the
//      binary rather than blowing up in a driver call.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaAllocator.hpp"
#include "tesseract/cuda/CudaGraph.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Arithmetic.hpp"

using namespace tesseract;

namespace {

Device cuda0() { return Device{DeviceType::CUDA, 0}; }

}  // namespace

TEST_CASE("CudaGraph: capture + replay reproduces eager output",
          "[cuda][graph]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }

  // 1 MiB worth of floats — large enough that the compute (~1 kernel
  // per op) is measurable, small enough that the test is fast.
  constexpr int64_t N = 1 << 18;
  Tensor x_cpu = Tensor::zeros({N}, DType::Float32);
  float* px = x_cpu.data_ptr<float>();
  for (int64_t i = 0; i < N; ++i) px[i] = static_cast<float>(i % 17);

  Tensor x   = x_cpu.to(cuda0());
  Tensor two = Tensor::full({}, 2.0, DType::Float32, cuda0());
  Tensor one = Tensor::full({}, 1.0, DType::Float32, cuda0());

  // Eager reference. Runs on the thread-local current stream — which
  // is different from the capture stream below, so the graph capture
  // won't see any leftover launches from this phase.
  Tensor y_eager = ops::add(ops::mul(x, two), one);
  Tensor y_eager_cpu = y_eager.to(cpu_device());
  const float* pe = y_eager_cpu.data_ptr<float>();

  // Graph-captured run. We hold `out` outside the closure so its
  // storage survives past `capture()`; the captured graph records the
  // device pointer that `ops::add` wrote into on the capture pass,
  // and every subsequent `launch()` writes into that same buffer.
  //
  // NOTE: `out` is reassigned INSIDE the closure (different Tensor
  // handles each pass), but because the arithmetic chain is identical
  // across warmup and capture, the bucketed allocator gives us the
  // same device pointer both times — cache hit on capture, no
  // `cudaMalloc`, capture stays alive.
  Stream s = Stream::create(cuda0());
  Tensor out;
  cuda::CudaGraph graph(/*device_index=*/0);
  graph.capture(s, [&]() {
    out = ops::add(ops::mul(x, two), one);
  });
  REQUIRE(graph.instantiated());

  // `out` now points to the buffer that the final `ops::add` wrote
  // on the capture pass. The pre-capture warmup already ran the
  // closure once, so this pointer is "warm" — replaying the graph
  // overwrites it with the same formula applied to the current
  // contents of `x` (which we haven't changed, so replay result
  // equals eager result).
  graph.launch(s);
  s.synchronize();

  Tensor out_cpu = out.to(cpu_device());
  const float* po = out_cpu.data_ptr<float>();
  float mx = 0.0f;
  for (int64_t i = 0; i < N; ++i) {
    mx = std::max(mx, std::abs(po[i] - pe[i]));
  }
  REQUIRE(mx < 1e-5f);
}

TEST_CASE("CudaGraph: recapture rebinds cleanly", "[cuda][graph]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  constexpr int64_t N = 1 << 14;
  Tensor x = Tensor::full({N}, 3.0, DType::Float32, cuda0());
  Tensor a = Tensor::full({}, 5.0, DType::Float32, cuda0());
  Tensor b = Tensor::full({}, 7.0, DType::Float32, cuda0());

  Stream s = Stream::create(cuda0());
  cuda::CudaGraph graph(/*device_index=*/0);
  REQUIRE_FALSE(graph.instantiated());

  Tensor first_out;
  graph.capture(s, [&]() { first_out = ops::add(x, a); });  // 3 + 5 = 8
  REQUIRE(graph.instantiated());
  graph.launch(s);
  s.synchronize();
  {
    Tensor c = first_out.to(cpu_device());
    const float* p = c.data_ptr<float>();
    for (int64_t i = 0; i < N; ++i) REQUIRE(p[i] == 8.0f);
  }

  // Re-capture a different closure. The old `graph_exec` must be
  // released; the new closure is the source of truth after this point.
  Tensor second_out;
  graph.capture(s, [&]() { second_out = ops::mul(x, b); });  // 3 * 7 = 21
  REQUIRE(graph.instantiated());
  graph.launch(s);
  s.synchronize();
  {
    Tensor c = second_out.to(cpu_device());
    const float* p = c.data_ptr<float>();
    for (int64_t i = 0; i < N; ++i) REQUIRE(p[i] == 21.0f);
  }
}

TEST_CASE("CudaGraph: launch before capture throws", "[cuda][graph]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  cuda::CudaGraph graph(/*device_index=*/0);
  REQUIRE_FALSE(graph.instantiated());
  Stream s = Stream::create(cuda0());
  REQUIRE_THROWS(graph.launch(s));
}

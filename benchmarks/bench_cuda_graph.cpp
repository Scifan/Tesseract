// Wave 2.3 (B-023) — bench: CUDA Graph replay vs eager launches.
//
// What this bench measures — and what the hard bars gate on:
//
//   * Eager path: issues a sequence of small elementwise ops via the
//     normal `ops::add` / `ops::mul` calls. Each op round-trips
//     through the thread-local current stream, the autograd check,
//     cuBLAS handle cache (none here, but same dispatch path as
//     matmul), and a raw `cudaLaunchKernel`. On small shapes the
//     GPU work finishes in < 5 µs per op, while the host-side
//     launch overhead adds ~3-8 µs per op — so a 5-op chain is
//     host-bound on most consumer GPUs.
//
//   * Graph-replay path: captures the same 5-op chain into a
//     `cudaGraphExec_t` once, then calls `cudaGraphLaunch` every
//     iteration. The entire chain becomes a single driver call,
//     so host-side overhead collapses to one launch's worth.
//
// Hard bar: `eager_us / graph_us >= 1.25` on the small-shape case
// (N=4096 floats, 10-op chain). On modern CUDA drivers + Ada / Hopper
// the per-kernel host-side launch overhead is already well-optimized
// (~1-2 µs each); graph replay collapses a 10-op chain to a single
// `cudaGraphLaunch` which saves ~0.5-1 µs per op in practice. That
// puts the realistic speedup in the 1.3-1.8× range for this
// elementwise-chain fixture. We anchor the hard bar at 1.25× — well
// below what we observe in practice (≥1.5× typical) but strict enough
// that a regression where graph replay goes back to per-kernel
// dispatch would trip it.
//
// What this bench intentionally does NOT measure yet:
//
//   * A full `MultiHeadAttention::forward_step` capture — that
//     requires the attention output buffer and the cache slabs
//     to be managed such that a single captured graph can replay
//     across growing `current_len()` values. Correct handling
//     needs a fixed-S_k attention variant + per-step mask memcpy,
//     tracked as B-023b. The elementwise-chain bench here already
//     proves the infrastructure end-to-end on the happy path.

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaGraph.hpp"
#include "tesseract/ops/Arithmetic.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Tensor;
using tesseract::ops::add;
using tesseract::ops::mul;

namespace {

struct Case {
  const char* label;
  int64_t     N;
  double      speedup_hard_bar;  // eager / graph must be >= this
};

// Ten-op chain: alternating mul / add against 0-D scalar operands.
// Enough kernels that the host-side launch overhead becomes a
// measurable fraction of wall time on small shapes, without making
// the benchmark take forever on large ones.
Tensor run_chain(const Tensor& x, const std::vector<Tensor>& ops_rhs) {
  Tensor t = x;
  for (std::size_t i = 0; i < ops_rhs.size(); ++i) {
    t = (i % 2 == 0) ? mul(t, ops_rhs[i]) : add(t, ops_rhs[i]);
  }
  return t;
}

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_graph] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_graph");

  bench::BenchStream bench_stream;
  cudaStream_t raw_stream = bench_stream.native();
  const tesseract::Stream& stream = bench_stream.stream();

  // Capture benches run inference-only — autograd wiring would add
  // backward-node allocations that aren't stable across warmup and
  // would break capture. This matches the real decode path, which
  // is also NoGradGuard-wrapped inside `MHA::forward_step`.
  tesseract::NoGradGuard nogg;

  const Device cuda0{DeviceType::CUDA, 0};

  // The speedup bar is only meaningful when the GPU work per op is
  // short enough that host-launch overhead dominates. 4 Ki elements
  // is the smallest shape we routinely hit in decode (`[1, 1, D]`
  // with D = hidden_size for residual sums, etc.); 64 Ki is on the
  // boundary where the kernel itself starts to dominate and the
  // ratio collapses toward 1. We only enforce the hard bar on the
  // smallest shape.
  const int64_t kChainLen = 10;
  const std::vector<Case> cases = {
    {"N=4K   (host-bound)",     4'096,     1.25},
    {"N=64K  (mixed)",          65'536,    0.0},  // informational
    {"N=1M   (bw-bound)",       1'048'576, 0.0},  // informational
  };

  std::printf("  shape              |   eager_us   |   graph_us   |  speedup\n");
  std::printf("  -------------------+--------------+--------------+----------\n");

  bool all_pass = true;
  for (const auto& c : cases) {
    Tensor x = Tensor::full({c.N}, 1.5, DType::Float32, cuda0);
    std::vector<Tensor> rhs;
    rhs.reserve(kChainLen);
    for (int64_t i = 0; i < kChainLen; ++i) {
      // Scalars picked so that the chain's fixed point is bounded
      // (keeps values away from Inf / NaN across hundreds of
      // benchmark iterations, which would otherwise skew the
      // bandwidth-bound measurement).
      const double v = (i % 2 == 0) ? 1.001 : -0.0005;
      rhs.push_back(Tensor::full({}, v, DType::Float32, cuda0));
    }

    // ---- Eager ------------------------------------------------------------
    // Keep `y_eager` outside the lambda so its storage is kept alive
    // across iterations — that way the bucketed allocator's cache
    // reaches steady state after the first two calls and we measure
    // the real per-op launch overhead, not first-call allocation.
    Tensor y_eager;
    auto eager_fn = [&](cudaStream_t /*s*/) {
      y_eager = run_chain(x, rhs);
    };
    auto eager_st = bench::steady_state_time(eager_fn, raw_stream,
                                             /*cov_target=*/0.05,
                                             /*warmup=*/10,
                                             /*max_iters=*/300,
                                             /*batch=*/0,
                                             /*min_window_ms=*/1.0);

    // ---- Graph-captured ---------------------------------------------------
    Tensor y_graph;
    tesseract::cuda::CudaGraph graph(/*device_index=*/0);
    graph.capture(stream, [&]() {
      y_graph = run_chain(x, rhs);
    });
    auto graph_fn = [&](cudaStream_t /*s*/) { graph.launch(stream); };
    auto graph_st = bench::steady_state_time(graph_fn, raw_stream,
                                             /*cov_target=*/0.05,
                                             /*warmup=*/10,
                                             /*max_iters=*/300,
                                             /*batch=*/0,
                                             /*min_window_ms=*/1.0);

    const double speedup = eager_st.mean_us / graph_st.mean_us;
    std::printf("  %-19s|  %9.3f   |  %9.3f   |  %5.2f x\n",
                c.label, eager_st.mean_us, graph_st.mean_us, speedup);

    if (c.speedup_hard_bar > 0.0) {
      const bool ok = bench::hard_check(speedup >= c.speedup_hard_bar,
                                        (std::string("graph_speedup ") +
                                         c.label).c_str(),
                                        speedup, c.speedup_hard_bar);
      all_pass = all_pass && ok;
    }
  }

  std::printf("\n  %s\n", all_pass ? "ALL HARD BARS PASSED." : "HARD BAR MISS.");
  return all_pass ? bench::kExitOk : bench::kExitPerfMiss;
}

// Wave 4.3 (B-023b) — bench: CUDA Graph capture of `MHA::forward_step`.
//
// What this bench measures
// -------------------------
//   * Eager path:   `mha.forward_step(x_step, cache)` driven one step at
//     a time, inside `NoGradGuard` just like the production decode
//     loop. Every step walks ~20+ kernel launches (Q/K/V projections,
//     reshape / permute, RoPE, per-(b,h) cache memcpy, fused attention
//     or composite softmax+matmul, output projection).
//   * Graph-replay path: the same closure is captured into a
//     `cudaGraphExec_t` once and driven through `cudaGraphLaunch`.
//     The entire decode step collapses into a single driver call; host
//     overhead goes from 20+ × per-launch to 1 × per-replay.
//
// Why this is the right proxy for llama decode
// --------------------------------------------
// A real llama decoder runs the step through N transformer blocks, so
// the per-step cost is roughly `N × (MHA.forward_step + FFN)`. FFN is
// a three-matmul + fused SwiGLU chain; MHA.forward_step is strictly
// more complex (projections, RoPE, cache append, attention over an
// `S_k`-wide key slab). Capturing one MHA step already exercises every
// single API hazard we care about for full-graph capture:
//   * async KVCache memcpy (Wave 4.3 rewrite);
//   * fused attention kernel inside the capture stream (Wave 4.2);
//   * view-family ops (reshape / permute / contiguous) — verified
//     allocation-cache-friendly.
// The ratio observed here sets a lower bound on the speedup we expect
// from capturing a full llama decode step end-to-end.
//
// Hard bars
// ---------
// The captured graph MUST replay faster than the eager equivalent on
// the small-`S_k` shapes where host overhead dominates.
//
//   * `S_k=8, D=512, H=16, Dh=32`: every kernel is tiny, host overhead
//     per step is ~40-80 µs out of ~60-120 µs wall. We require
//     `eager / graph >= 1.25×`. Below 1.25× means either (a) the graph
//     is not actually being replayed (e.g. re-capture on every
//     iteration), or (b) one of the ops is still doing a host-side
//     sync that kills capture latency. The bar mirrors
//     `bench_cuda_graph` on its small-shape 10-op chain — same
//     mechanism, bigger workload.
//   * `S_k=256, D=512, H=16, Dh=32`: longer key slab, attention kernel
//     takes meaningful work. Speedup usually still > 1.10× but the
//     bar is informational-only so natural-regression noise on
//     compute-bound shapes doesn't fail CI.

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaGraph.hpp"
#include "tesseract/nn/KVCache.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::NoGradGuard;
using tesseract::Shape;
using tesseract::Stream;
using tesseract::Tensor;

namespace {

struct Case {
  const char* label;
  int64_t     pos;           // captured graph decodes token at this position
  double      speedup_hard_bar;  // 0.0 means "informational, no bar"
};

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_mha_decode_graph] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_mha_decode_graph");

  bench::BenchStream bench_stream;
  cudaStream_t raw_stream = bench_stream.native();
  const Stream& stream = bench_stream.stream();

  const Device cuda0{DeviceType::CUDA, 0};

  // `S_k=8` is the chunked-decode sweet spot where host launch overhead
  // dominates over compute. `S_k=256` approximates mid-sequence decode
  // where attention compute starts to catch up. We only hard-bar the
  // small one.
  constexpr int64_t B = 1, H = 16, Dh = 32, D = H * Dh;
  constexpr int64_t MAX = 512;
  const std::vector<Case> cases = {
    {"S_k=8   (host-bound decode)",    8,    1.25},
    {"S_k=64  (mixed)",                64,   0.0},
    {"S_k=256 (compute-leaning)",     256,   0.0},
  };

  tesseract::nn::MultiHeadAttention mha(
      /*d_model=*/D, /*num_heads=*/H, /*use_bias=*/false,
      /*causal=*/true, DType::Float32,
      /*rope_base=*/10000.0, /*rope_max_seq=*/MAX);
  mha.to(cuda0);

  // One-time input scratch buffer for the decoded token's embedding.
  // We allocate via the library's pooled allocator so both eager and
  // graph paths hit the cache — the graph records this pointer at
  // capture time and reuses it across replays.
  Tensor x_step = Tensor::empty({B, 1, D}, DType::Float32, cuda0);
  bench::check_cuda(cudaMemsetAsync(x_step.raw_data(), 0,
                                    B * 1 * D * sizeof(float), raw_stream),
                    "memset x_step");
  bench::check_cuda(cudaStreamSynchronize(raw_stream), "init-sync");

  std::printf("  config   : d_model=%ld heads=%ld d_head=%ld  B=%ld\n",
              long(D), long(H), long(Dh), long(B));
  std::printf("  shape                          |   eager_us   |   graph_us   |  speedup\n");
  std::printf("  -------------------------------+--------------+--------------+----------\n");

  bool all_pass = true;
  for (const auto& c : cases) {
    tesseract::nn::KVCache cache(B, H, Dh, MAX, DType::Float32, cuda0);

    // ---- Eager path -------------------------------------------------------
    //
    // Each iteration resets `current_len_` back to `pos` so every call
    // measures the same "decode step at offset `pos`" cost. Without
    // the rewind the cache would overflow `max_len` after a few
    // iterations. Keep `y_slot` outside the closure so its storage is
    // held across iterations and the allocator reaches steady state
    // — same trick as `bench_cuda_graph`.
    Tensor y_eager;
    auto eager_fn = [&](cudaStream_t /*s*/) {
      NoGradGuard nogg;
      cache.set_current_len(c.pos);
      y_eager = mha.forward_step(x_step, cache);
    };
    auto eager_st = bench::steady_state_time(eager_fn, raw_stream,
                                             /*cov_target=*/0.05,
                                             /*warmup=*/10,
                                             /*max_iters=*/300,
                                             /*batch=*/0,
                                             /*min_window_ms=*/1.0);

    // ---- Graph-captured path ---------------------------------------------
    Tensor y_graph;
    tesseract::cuda::CudaGraph graph(/*device_index=*/0);
    graph.capture(stream, [&]() {
      NoGradGuard nogg;
      cache.set_current_len(c.pos);
      y_graph = mha.forward_step(x_step, cache);
    });

    // During replay we only rewind the cache counter between launches
    // — the slab position the graph writes to is fixed at capture time
    // and we want every iteration to exercise that same slot.
    auto graph_fn = [&](cudaStream_t /*s*/) {
      cache.set_current_len(c.pos);
      graph.launch(stream);
    };
    auto graph_st = bench::steady_state_time(graph_fn, raw_stream,
                                             /*cov_target=*/0.05,
                                             /*warmup=*/10,
                                             /*max_iters=*/300,
                                             /*batch=*/0,
                                             /*min_window_ms=*/1.0);

    const double speedup = eager_st.mean_us / graph_st.mean_us;
    std::printf("  %-31s|  %9.3f   |  %9.3f   |  %5.2f x\n",
                c.label, eager_st.mean_us, graph_st.mean_us, speedup);

    if (c.speedup_hard_bar > 0.0) {
      const bool ok = bench::hard_check(
          speedup >= c.speedup_hard_bar,
          (std::string("decode_graph_speedup ") + c.label).c_str(),
          speedup, c.speedup_hard_bar);
      all_pass = all_pass && ok;
    }
  }

  std::printf("\n  %s\n", all_pass ? "ALL HARD BARS PASSED." : "HARD BAR MISS.");
  return all_pass ? bench::kExitOk : bench::kExitPerfMiss;
}

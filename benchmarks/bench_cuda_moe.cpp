// M4 perf-closeout Phase 4 — GPU MoE benchmark.
//
// Two things to show, both on a clean isolated card:
//   1. The MoE architectural advantage on GPU: the sparse top-k dispatch
//      (nn::MoEFeedForward::forward, routing now on-device) costs ≈ k/E of
//      the dense "every expert on every token" path that an equivalent dense
//      model would pay. This is the whole point of MoE and it now holds on
//      GPU with device-side routing (no host round-trip).
//   2. Absolute latency of the full MoE layer at Mixtral/Switch shapes, so it
//      can be compared head-to-head with the PyTorch MoE baseline
//      (`bench/external/torch_baseline.py --bench moe`).
//
// The fused GPU MoE path (device permute + grouped GEMM + fused SiLU +
// scatter-combine, in src/cuda/MoeForward.cu) makes the sparse dispatch
// genuinely cheaper than the dense all-experts path on GPU — the k/E FLOP
// advantage finally translates to wall-clock.
//
// Hard bar: sparse (fused) faster than dense at every measured shape.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/MoEFeedForward.hpp"
#include "tesseract/ops/Arithmetic.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using namespace tesseract;

namespace {

Tensor rand_tokens(int64_t T, int64_t D, uint64_t seed) {
  std::vector<float> v(static_cast<std::size_t>(T * D));
  uint64_t s = seed;
  for (auto& x : v) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    x = static_cast<float>((s >> 33) % 2000) / 1000.0f - 1.0f;
  }
  return Tensor::from_vector(v, Shape({T, D}));
}

bool run(int64_t T, int64_t D, int64_t dff, int64_t E, int64_t k) {
  NoGradGuard nogg;
  const Device cuda0{DeviceType::CUDA, 0};
  nn::MoEFeedForward moe(D, dff, E, k);
  moe.to(cuda0);
  Tensor x = rand_tokens(T, D, 0xA5A5u).to(cuda0);

  cudaStream_t stream = nullptr;

  auto sparse = [&](cudaStream_t) { (void)moe.forward(x); };
  auto dense = [&](cudaStream_t) {
    Tensor acc;
    for (int64_t e = 0; e < E; ++e) {
      Tensor y = moe.experts()[static_cast<std::size_t>(e)]->forward(x);
      acc = acc.defined() ? ops::add(acc, y) : y;
    }
    (void)acc;
  };

  sparse(stream); dense(stream);
  bench::check_cuda(cudaDeviceSynchronize(), "warm");

  auto ss = bench::steady_state_time(sparse, stream);
  auto sd = bench::steady_state_time(dense, stream);
  const double ratio = ss.mean_us / sd.mean_us;
  const double ideal = double(k) / double(E);
  std::printf("T=%-5ld D=%-4ld dff=%-5ld E=%-3ld k=%-2ld | "
              "sparse=%9.2f us  dense=%9.2f us | ratio=%.3f ideal=%.3f "
              "speedup=%.2fx\n",
              long(T), long(D), long(dff), long(E), long(k),
              ss.mean_us, sd.mean_us, ratio, ideal, sd.mean_us / ss.mean_us);
  return ratio < 1.0;
}

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_moe] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_moe (sparse top-k dispatch vs dense, GPU)");

  bool ok = true;
  // Mixtral-like E=8 k=2 (ideal 0.25). Sweep token count.
  ok &= run(512,  1024, 2048, 8, 2);
  ok &= run(2048, 1024, 2048, 8, 2);
  ok &= run(4096, 1024, 4096, 8, 2);
  // Switch-like E=16 k=1 (ideal 0.0625).
  ok &= run(4096, 1024, 2048, 16, 1);
  std::printf("\n");

  bool pass = bench::hard_check(ok, "fused sparse MoE cheaper than dense",
                                ok ? 1.0 : 0.0, 1.0);
  return pass ? bench::kExitOk : bench::kExitPerfMiss;
}

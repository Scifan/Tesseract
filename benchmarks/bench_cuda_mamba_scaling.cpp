// M4 perf-closeout Phase 5 — GPU Mamba (O(1) decode) vs attention (O(L)
// decode) scaling. The CUDA counterpart of the CPU-only
// bench_mamba_vs_llama_scaling: it sweeps the context length L and measures
// per-step decode latency for a stack of Mamba layers (constant-width SSM
// state, cost independent of L) against a stack of multi-head-attention
// layers (must score against the full KV prefix, cost growing with L).
//
// Layer-level (like bench_cuda_llama_decode) so the embedding / lm_head cancel
// and the only moving part is SSM-vs-attention. Both stacks share d_model /
// layers and run FP32 on one isolated card.
//
// Hard bar: at the largest L, Mamba's per-step decode is faster than
// attention's (the O(1)-vs-O(L) asymptotic win, realized on GPU).

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/KVCache.hpp"
#include "tesseract/nn/Mamba.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using namespace tesseract;

namespace {

constexpr int64_t kDModel = 1024;
constexpr int64_t kLayers = 8;
constexpr int64_t kHeads = 16;          // head_dim = 64
constexpr int64_t kDState = 16;
constexpr int64_t kDConv = 4;
constexpr int64_t kExpand = 2;

bool run(const std::vector<int64_t>& lengths) {
  NoGradGuard nogg;
  const Device cuda0{DeviceType::CUDA, 0};
  const int64_t B = 1, Dh = kDModel / kHeads;

  // Build the two layer stacks once.
  std::vector<std::shared_ptr<nn::Mamba>> mamba;
  std::vector<std::shared_ptr<nn::MultiHeadAttention>> attn;
  for (int64_t l = 0; l < kLayers; ++l) {
    auto m = std::make_shared<nn::Mamba>(kDModel, kDState, kDConv, kExpand);
    m->to(cuda0);
    mamba.push_back(m);
    auto a = std::make_shared<nn::MultiHeadAttention>(
        kDModel, kHeads, /*use_bias=*/false, /*causal=*/true, DType::Float32);
    a->to(cuda0);
    attn.push_back(a);
  }

  Tensor x = Tensor::empty({B, 1, kDModel}, DType::Float32, cuda0);
  cudaStream_t stream = nullptr;

  std::printf("\n%-7s | %14s %14s | %12s %12s | %8s\n", "L",
              "mamba_us/step", "attn_us/step", "mamba_tok/s", "attn_tok/s",
              "speedup");
  std::printf("%s\n", "--------------------------------------------------"
                      "----------------------------------");

  bool win_at_max = false;
  const int64_t Lmax = lengths.back();
  for (int64_t L : lengths) {
    // Mamba: O(1) state caches, no prefill needed.
    std::vector<nn::SSMStateCache> mcache;
    for (auto& m : mamba) mcache.push_back(m->make_state_cache(B));
    auto mamba_step = [&](cudaStream_t) {
      Tensor h = x;
      for (int64_t l = 0; l < kLayers; ++l) h = mamba[l]->forward_step(h, mcache[l]);
      (void)h;
    };

    // Attention: each step decodes at a fixed context length L. Pin the cache
    // length to L before every step (set_current_len) so the step scores
    // against L keys and overwrites slot L instead of growing unboundedly —
    // this isolates the O(L) per-step cost without an O(L) re-prefill.
    std::vector<std::shared_ptr<nn::KVCache>> acache;
    for (int64_t l = 0; l < kLayers; ++l)
      acache.push_back(std::make_shared<nn::KVCache>(
          B, kHeads, Dh, L + 4, DType::Float32, cuda0));
    auto attn_step = [&](cudaStream_t) {
      Tensor h = x;
      for (int64_t l = 0; l < kLayers; ++l) {
        acache[l]->set_current_len(L);
        h = attn[l]->forward_step(h, *acache[l]);
      }
      (void)h;
    };

    mamba_step(stream); attn_step(stream);
    bench::check_cuda(cudaDeviceSynchronize(), "warm");
    auto sm = bench::steady_state_time(mamba_step, stream);
    auto sa = bench::steady_state_time(attn_step, stream);
    const double sp = sa.mean_us / sm.mean_us;
    std::printf("%-7ld | %14.2f %14.2f | %12.1f %12.1f | %7.2fx\n",
                long(L), sm.mean_us, sa.mean_us, 1e6 / sm.mean_us,
                1e6 / sa.mean_us, sp);
    if (L == Lmax) win_at_max = (sp > 1.0);
  }
  return win_at_max;
}

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_mamba_scaling] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_mamba_scaling (Mamba O(1) vs attention O(L), GPU)");
  std::printf("  d_model=%ld layers=%ld | attn heads=%ld | mamba d_state=%ld expand=%ld\n",
              long(kDModel), long(kLayers), long(kHeads), long(kDState),
              long(kExpand));

  const bool ok = run({128, 512, 1024, 2048, 4096});
  std::printf("\n");
  bool pass = bench::hard_check(ok, "Mamba decode faster than attention at max L",
                                ok ? 1.0 : 0.0, 1.0);
  return pass ? bench::kExitOk : bench::kExitPerfMiss;
}

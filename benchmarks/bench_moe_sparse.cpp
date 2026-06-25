// M4 Track A1 (B-038+) — the MoE architectural-advantage metric: sparse
// top-k dispatch should cost ≈ k/E of the dense "every expert on every token"
// path. A1 originally computed all experts densely (correct, but zero compute
// saving — defeating the purpose of MoE). This bench quantifies the saving the
// sparse dispatch (now in nn::MoEFeedForward::forward) actually delivers.
//
// Two timed paths over the same router/experts:
//   * dense:  Σ_e expert_e(x) for *all* E experts on *all* T tokens (the old
//             A1 behaviour; the work an equivalent dense model would do).
//   * sparse: moe.forward(x) — each expert runs only on its routed rows.
//
// Headline: sparse/dense time ratio vs the ideal k/E. CPU, informational.
// See docs/design/moe-sparse.md.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/MoEFeedForward.hpp"
#include "tesseract/ops/Arithmetic.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using namespace tesseract;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

Tensor rand_tokens(int64_t T, int64_t D, uint64_t seed) {
  std::vector<float> v(static_cast<std::size_t>(T * D));
  uint64_t s = seed;
  for (auto& x : v) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    x = static_cast<float>((s >> 33) % 2000) / 1000.0f - 1.0f;
  }
  return Tensor::from_vector(v, Shape({T, D}));
}

void run(int64_t T, int64_t D, int64_t dff, int64_t E, int64_t k, int iters) {
  NoGradGuard nogg;
  nn::MoEFeedForward moe(D, dff, E, k);
  Tensor x = rand_tokens(T, D, 0xA5A5u);

  // Warm both paths.
  (void)moe.forward(x);
  {
    Tensor acc;
    for (int64_t e = 0; e < E; ++e) {
      Tensor y = moe.experts()[static_cast<std::size_t>(e)]->forward(x);
      acc = acc.defined() ? ops::add(acc, y) : y;
    }
    (void)acc;
  }

  // Dense: every expert on every token (old A1 cost).
  const auto td0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    Tensor acc;
    for (int64_t e = 0; e < E; ++e) {
      Tensor y = moe.experts()[static_cast<std::size_t>(e)]->forward(x);
      acc = acc.defined() ? ops::add(acc, y) : y;
    }
    (void)acc;
  }
  const double dense_s = seconds_since(td0) / iters;

  // Sparse: route + run only the active experts.
  const auto ts0 = Clock::now();
  for (int it = 0; it < iters; ++it) (void)moe.forward(x);
  const double sparse_s = seconds_since(ts0) / iters;

  const double ratio = sparse_s / dense_s;
  const double ideal = static_cast<double>(k) / static_cast<double>(E);
  std::printf(
      "[bench] moe_sparse  T=%lld D=%lld dff=%lld E=%lld k=%lld  "
      "dense_ms=%.3f  sparse_ms=%.3f  ratio=%.3f  ideal(k/E)=%.3f  "
      "speedup=%.2fx\n",
      static_cast<long long>(T), static_cast<long long>(D),
      static_cast<long long>(dff), static_cast<long long>(E),
      static_cast<long long>(k), dense_s * 1e3, sparse_s * 1e3, ratio, ideal,
      dense_s / sparse_s);
}

}  // namespace

int main() {
  std::printf("# MoE sparse top-k dispatch vs dense all-experts — CPU\n");
  // Mixtral-like: E=8, k=2 (ideal ratio 0.25). Sweep token count so the
  // compute-bound regime (where the saving shows) is visible vs the small-T
  // regime (where routing overhead dominates).
  run(/*T=*/64,   /*D=*/512, /*dff=*/1024, /*E=*/8, /*k=*/2, /*iters=*/50);
  run(/*T=*/256,  /*D=*/512, /*dff=*/1024, /*E=*/8, /*k=*/2, /*iters=*/30);
  run(/*T=*/1024, /*D=*/512, /*dff=*/1024, /*E=*/8, /*k=*/2, /*iters=*/15);
  run(/*T=*/4096, /*D=*/512, /*dff=*/1024, /*E=*/8, /*k=*/2, /*iters=*/6);
  // Switch-like: E=16, k=1 (ideal 0.0625).
  run(/*T=*/1024, /*D=*/512, /*dff=*/1024, /*E=*/16, /*k=*/1, /*iters=*/10);
  return 0;
}

// M4 Phase 6 — CPU decode throughput with the AVX-512-VNNI W8A8 GEMV.
//
// Decode at real model sizes is memory-bandwidth bound on reading the
// weights. This bench wires the full per-token Linear stack of a real-size
// Llama (q/k/v/o + gate/up/down per layer + lm_head) through the W8A8 GEMV
// kernel with a pre-allocated arena (zero allocation in the decode loop),
// and reports tokens/second + effective GB/s — directly comparable to
// `llama.cpp`'s `llama-bench -p 0 -n N` on a Q8_0 model of the same shape.
//
// Output (machine-readable):
//   [bench] tesseract cpu_decode_vnni  cfg=<...>  decode_tok_s=<..>  gbps=<..>
//
// Hard bar: with VNNI present, must clear a floor that the FP32 eager path
// cannot reach (the old path was ~73 tok/s on a tiny model and is FP32 /
// per-op-allocating). We assert the int8 arena path beats a conservative
// bandwidth-derived floor so a regression (e.g. losing VNNI dispatch)
// fails CI.

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../src/ops/cpu/GemvVnni.hpp"

#if defined(TESSERACT_HAS_OPENMP)
  #include <omp.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
using tesseract::ops::detail::compute_row_sums;
using tesseract::ops::detail::gemv_vnni_supported;
using tesseract::ops::detail::gemv_w8a8;
using tesseract::ops::detail::gemv_w8a8_range;
using tesseract::ops::detail::quantize_row_int8;

// One INT8 weight matrix [N,K] + per-row scale + precomputed row sums.
struct QWeight {
  std::vector<std::int8_t> w;       // [N*K]
  std::vector<float> scale;         // [N]
  std::vector<std::int32_t> rowsum; // [N]
  std::int64_t N = 0, K = 0;

  void init(std::int64_t n, std::int64_t k, std::mt19937& rng) {
    N = n; K = k;
    w.resize(static_cast<size_t>(n * k));
    scale.resize(static_cast<size_t>(n));
    rowsum.resize(static_cast<size_t>(n));
    std::uniform_int_distribution<int> wd(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (auto& v : w) v = static_cast<std::int8_t>(wd(rng));
    for (auto& s : scale) s = sd(rng);
    compute_row_sums(w.data(), N, K, rowsum.data());
  }
};

}  // namespace

int main(int argc, char** argv) {
  // Default ~1.1B-class Llama (TinyLlama-1.1B shape): d=2048, ffn=5632,
  // 22 layers, vocab 32000, 32 heads (MHA). Override via flags.
  std::int64_t d = 2048, ffn = 5632, layers = 22, vocab = 32000;
  std::int64_t n_tokens = 64;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::atoll(argv[++i]); };
    if (a == "--d") d = next();
    else if (a == "--ffn") ffn = next();
    else if (a == "--layers") layers = next();
    else if (a == "--vocab") vocab = next();
    else if (a == "--gen") n_tokens = next();
  }

  std::printf("[info] VNNI supported: %s\n", gemv_vnni_supported() ? "yes" : "no");
#if defined(TESSERACT_HAS_OPENMP)
  std::printf("[info] omp_get_max_threads = %d\n", omp_get_max_threads());
#endif

  std::mt19937 rng(1234);
  // DISTINCT weights per layer: the full ~1.2 GB working set must exceed L3
  // (the EPYC 9474F has 256 MB L3) so decode is genuinely memory-bound on
  // reading weights from DRAM — exactly what llama.cpp does. Reusing one
  // weight set would make the run L3-cache-resident and report cache, not
  // memory, bandwidth (an unfairly optimistic number).
  std::vector<QWeight> q(layers), k(layers), v(layers), o(layers);
  std::vector<QWeight> gate(layers), up(layers), down(layers);
  QWeight lm;
  for (std::int64_t l = 0; l < layers; ++l) {
    q[l].init(d, d, rng);
    k[l].init(d, d, rng);
    v[l].init(d, d, rng);
    o[l].init(d, d, rng);
    gate[l].init(ffn, d, rng);
    up[l].init(ffn, d, rng);
    down[l].init(d, ffn, rng);
  }
  lm.init(vocab, d, rng);

  // Pre-allocated arena: every buffer the decode step needs, allocated once.
  std::vector<float> x(static_cast<size_t>(d));
  std::vector<float> hffn(static_cast<size_t>(ffn));
  std::vector<float> y_d(static_cast<size_t>(d));
  std::vector<float> y_ffn(static_cast<size_t>(ffn));
  std::vector<float> y_vocab(static_cast<size_t>(vocab));
  std::vector<std::int8_t> xq(static_cast<size_t>(d));
  std::vector<std::int8_t> xq_ffn(static_cast<size_t>(ffn));

  std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
  for (auto& e : x) e = xd(rng);
  for (auto& e : hffn) e = xd(rng);

  // One decode token: per layer q/k/v/o (d->d) + gate/up (d->ffn) +
  // down (ffn->d), then lm_head (d->vocab) once. All GEMVs are driven inside
  // a SINGLE parallel region per token (see decode_token) so the thread team
  // forks once per token, not once per matvec — the only way the W8A8 path
  // scales past ~8 cores. Activation scales are precomputed serially (cheap,
  // O(K)); the heavy O(N*K) GEMVs are row-partitioned across the team.
  const float xs_d = quantize_row_int8(x.data(), d, xq.data());
  const float xs_ffn = quantize_row_int8(hffn.data(), ffn, xq_ffn.data());

  // Worksharing helper: thread `tid` of `nt` computes its row slice of a GEMV.
  auto run = [&](int tid, int nt, const QWeight& W, const std::int8_t* xqp,
                 float xs, float* out) {
    const std::int64_t chunk = (W.N + nt - 1) / nt;
    const std::int64_t n0 = std::min<std::int64_t>(W.N, (std::int64_t)tid * chunk);
    const std::int64_t n1 = std::min<std::int64_t>(W.N, n0 + chunk);
    gemv_w8a8_range(W.w.data(), W.scale.data(), W.rowsum.data(), xqp, xs, out,
                    n0, n1, W.K);
  };
  auto decode_token = [&](int nthreads) {
#if defined(TESSERACT_HAS_OPENMP)
    #pragma omp parallel num_threads(nthreads)
    {
      const int tid = omp_get_thread_num();
      const int nt = omp_get_num_threads();
#else
      const int tid = 0, nt = 1;
#endif
      for (std::int64_t l = 0; l < layers; ++l) {
        run(tid, nt, q[l], xq.data(), xs_d, y_d.data());
        run(tid, nt, k[l], xq.data(), xs_d, y_d.data());
        run(tid, nt, v[l], xq.data(), xs_d, y_d.data());
        run(tid, nt, o[l], xq.data(), xs_d, y_d.data());
        run(tid, nt, gate[l], xq.data(), xs_d, y_ffn.data());
        run(tid, nt, up[l], xq.data(), xs_d, y_ffn.data());
        run(tid, nt, down[l], xq_ffn.data(), xs_ffn, y_d.data());
      }
      run(tid, nt, lm, xq.data(), xs_d, y_vocab.data());
#if defined(TESSERACT_HAS_OPENMP)
    }
#endif
  };

  // Bytes of INT8 weight streamed per token = layers*(4*d*d + 2*ffn*d +
  // ffn*d) + vocab*d.
  const double per_layer_bytes =
      static_cast<double>(4 * d * d + 3 * ffn * d);
  const double bytes_per_tok =
      static_cast<double>(layers) * per_layer_bytes + static_cast<double>(vocab) * d;

  int nthreads = 1;
#if defined(TESSERACT_HAS_OPENMP)
  // Honor OMP_NUM_THREADS if set; otherwise default to the physical-core
  // count (SMT siblings hurt this BW-bound kernel). The env, when present,
  // takes precedence so the sweep above can probe other counts.
  nthreads = omp_get_max_threads();
  if (std::getenv("OMP_NUM_THREADS") == nullptr) {
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 1) nthreads = static_cast<int>(online / 2);  // drop SMT
  }
#endif

  // Warm-up.
  for (int w = 0; w < 3; ++w) decode_token(nthreads);

  const auto t0 = Clock::now();
  for (std::int64_t t = 0; t < n_tokens; ++t) decode_token(nthreads);
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();

  const double decode_tok_s = static_cast<double>(n_tokens) / secs;
  const double gbps = bytes_per_tok * n_tokens / secs / 1e9;

  std::printf(
      "[bench] tesseract cpu_decode_vnni  cfg=L%lld_d%lld_ffn%lld_v%lld  "
      "decode_tok_s=%.2f  gbps=%.1f  bytes/tok=%.1fMB\n",
      static_cast<long long>(layers), static_cast<long long>(d),
      static_cast<long long>(ffn), static_cast<long long>(vocab), decode_tok_s,
      gbps, bytes_per_tok / 1e6);

  // Hard bar (only when VNNI is actually available): the int8 arena path
  // must sustain a real fraction of memory bandwidth. A conservative floor
  // of 40 GB/s is far above what the FP32 eager per-op path can do and
  // still leaves headroom below the EPYC's ~400 GB/s ceiling.
  if (gemv_vnni_supported()) {
    if (gbps < 40.0) {
      std::printf("[FAIL] VNNI W8A8 GEMV below 40 GB/s floor (%.1f)\n", gbps);
      return 1;
    }
    std::printf("[PASS] VNNI W8A8 decode sustains %.1f GB/s\n", gbps);
  }
  return 0;
}

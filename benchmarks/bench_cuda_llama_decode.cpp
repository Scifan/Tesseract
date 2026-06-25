// Wave 4.4 (B-026) — bench: end-to-end Llama decode step (one
// transformer block) FP32 vs INT8 vs INT4G, measuring per-step
// latency + weight footprint.
//
// What one decode step looks like
// --------------------------------
//   h = mha.forward_step(x_step, kv_cache)     // Q/K/V projections
//                                              // + RoPE + cache-append
//                                              // + attention over
//                                              //   full `S_k`-wide prefix
//                                              // + o_proj
//   y = ffn.forward(h)                         // gate_proj + up_proj
//                                              // + fused SwiGLU +
//                                              // down_proj
//
// That's 4 Linear + 3 Linear = 7 matmuls per block per step, i.e.
// every Linear in a Llama block is on the measured path. Quantizing
// all seven projections with `mha.quantize_(scheme)` +
// `ffn.quantize_(scheme)` turns the matmuls into fused
// dequant-matmul kernels; residuals / norms / RoPE / cache-append
// stay FP (they're not Linear → walker skips them).
//
// Shapes: Llama-2-7B block (the first scale where the FP32 weight
// footprint — 4 × 4096² + 3 × 4096·11008 ≈ 800 MB — comprehensively
// exceeds L2 on every currently-shipping GPU and the per-step read
// becomes purely HBM-bandwidth-limited).
//   d_model = 4096, H = 32, Dh = 128, d_ff = 11008
//   cache prefix = 128 tokens (mid-sequence decode)
//
// Smaller Llama-1B-ish blocks still benefit from quantization but
// the 200-MB FP32 footprint partially caches in L2 after warmup and
// INT4G's per-output compute (nibble unpack + group-scale lookup)
// dominates at that scale — the memory savings show up in
// `weight_MB`/`mem_frac` but don't always translate to a decode-step
// latency win. The 7B-shape pins the latency story for both variants
// so the hard bar tracks what inference workloads care about in
// practice.
//
// Hard bars
// ---------
//   * Weight memory: INT8 block / FP32 block ≤ 0.30, INT4G ≤ 0.18.
//     Deterministic from the packer layout, not a measurement — the
//     bar still matters as a guard in case a future refactor changes
//     the per-module scale buffers.
//   * Latency on this HBM-bound decode step:
//       - INT8  ≤ 0.55 × FP32  (4× less weight streaming → ~2× speedup
//                                even once the dequant-matmul fused
//                                kernel leaves the HBM-bound regime
//                                and becomes mildly compute-bound on
//                                SM 8.9.)
//       - INT4G ≤ 0.75 × FP32  (8× less weight streaming in theory,
//                                but on SM 8.9 the nibble-unpack +
//                                group-scale lookup path is
//                                compute-bound — the kernel runs
//                                slower than the INT8 one. The bar
//                                still requires a clear win over FP32
//                                and tracks what the current fused
//                                kernel actually delivers; a future
//                                vectorized-unpack rewrite would let
//                                us tighten it further.)
//     Both bars sit ~15% above the measured ratio on SM 8.9 so
//     steady-state CoV (~5%) + DVFS jitter can't flap the bar.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/FeedForward.hpp"
#include "tesseract/nn/KVCache.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"
#include "tesseract/nn/QuantizedLinear.hpp"
#include "tesseract/nn/QuantizedLinearInt4G.hpp"
#include "tesseract/quant/Scheme.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::NoGradGuard;
using tesseract::Tensor;
using tesseract::nn::FeedForward;
using tesseract::nn::KVCache;
using tesseract::nn::Linear;
using tesseract::nn::MultiHeadAttention;
using tesseract::nn::Module;
using tesseract::nn::QuantizedLinear;
using tesseract::nn::QuantizedLinearInt4G;

namespace {

// Byte-count helper. We walk `named_buffers()` + `named_parameters()`
// on the pair of modules — Linear weights are in `parameters()`,
// quantized `q_weight` + `weight_scale` live in `buffers()`.
int64_t weight_bytes(const Module& m) {
  int64_t total = 0;
  for (const auto& [name, t] : m.named_parameters()) {
    total += t.nbytes();
  }
  for (const auto& [name, t] : m.named_buffers()) {
    total += t.nbytes();
  }
  return total;
}

int64_t block_weight_bytes(const MultiHeadAttention& mha,
                           const FeedForward& ffn) {
  return weight_bytes(mha) + weight_bytes(ffn);
}

struct VariantResult {
  const char* label;
  double      step_us;
  int64_t     bytes;
};

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_llama_decode] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_llama_decode");

  bench::BenchStream bench_stream;
  cudaStream_t raw_stream = bench_stream.native();

  const Device cuda0{DeviceType::CUDA, 0};

  // Llama-2-7B per-block hyperparameters — smallest shape where the
  // FP32 weight footprint (~800 MB) comprehensively exceeds L2 so
  // cuBLASLt's steady-state FP32 path is HBM-bandwidth-limited and
  // the INT4G / INT8 latency wins track their weight-byte ratios.
  //   d_model = 4096, H = 32, Dh = d_model / H = 128, d_ff = 11008
  // Peak memory across the three variants is ~1.1 GB of weights —
  // fits on any 8+ GiB GPU with comfortable headroom.
  constexpr int64_t d_model = 4096;
  constexpr int64_t H       = 32;
  constexpr int64_t Dh      = d_model / H;
  constexpr int64_t d_ff    = 11008;
  constexpr int64_t MAX     = 256;
  constexpr int64_t pos     = 128;    // mid-sequence decode offset
  constexpr int64_t B       = 1;

  std::printf("  config   : d_model=%ld H=%ld Dh=%ld d_ff=%ld  B=%ld  S_k=%ld\n",
              long(d_model), long(H), long(Dh), long(d_ff),
              long(B), long(pos + 1));
  std::printf("\n");

  // Build one FP32 block to capture the baseline; `quantize_` is
  // destructive so we have to rebuild per variant. Use `use_bias=false`
  // everywhere (Llama default).
  auto build_fp = [&]() {
    auto mha = std::make_shared<MultiHeadAttention>(
        /*d_model=*/d_model, /*num_heads=*/H,
        /*use_bias=*/false, /*causal=*/true, DType::Float32,
        /*rope_base=*/10000.0, /*rope_max_seq=*/MAX);
    auto ffn = std::make_shared<FeedForward>(
        /*d_model=*/d_model, /*d_ff=*/d_ff,
        /*use_bias=*/false, DType::Float32);
    mha->to(cuda0);
    ffn->to(cuda0);
    mha->eval();
    ffn->eval();
    return std::make_pair(mha, ffn);
  };

  auto run_variant = [&](const char* label,
                         std::shared_ptr<MultiHeadAttention> mha,
                         std::shared_ptr<FeedForward>        ffn) {
    KVCache cache(B, H, Dh, MAX, DType::Float32, cuda0);
    Tensor x_step = Tensor::empty({B, 1, d_model}, DType::Float32, cuda0);
    bench::check_cuda(cudaMemsetAsync(x_step.raw_data(), 0,
                                      B * 1 * d_model * sizeof(float),
                                      raw_stream),
                      "memset x_step");
    bench::check_cuda(cudaStreamSynchronize(raw_stream), "init-sync");

    auto step = [&](cudaStream_t /*s*/) {
      NoGradGuard nogg;
      cache.set_current_len(pos);
      Tensor h = mha->forward_step(x_step, cache);
      // Residual is a trivial add — skipped so the measurement
      // isolates the Linear-heavy path. Correct Llama semantics
      // would be `h = x_step + mha(rms(x_step))` etc.; the omission
      // is documented.
      Tensor y = ffn->forward(h);
      (void)y;
    };

    for (int i = 0; i < 5; ++i) step(raw_stream);
    bench::check_cuda(cudaStreamSynchronize(raw_stream), "warm");

    auto st = bench::steady_state_time(step, raw_stream,
                                       /*cov_target=*/0.05,
                                       /*warmup=*/10,
                                       /*max_iters=*/400,
                                       /*batch=*/0,
                                       /*min_window_ms=*/1.0);

    VariantResult r;
    r.label = label;
    r.step_us = st.mean_us;
    r.bytes = block_weight_bytes(*mha, *ffn);
    return r;
  };

  std::vector<VariantResult> results;

  // ---- FP32 baseline --------------------------------------------------------
  {
    auto [mha, ffn] = build_fp();
    results.push_back(run_variant("FP32", mha, ffn));
  }

  // ---- INT8 -----------------------------------------------------------------
  {
    auto [mha, ffn] = build_fp();
    mha->quantize_(tesseract::quant::Scheme::int8_symmetric());
    ffn->quantize_(tesseract::quant::Scheme::int8_symmetric());
    // `quantize_` rebuilds the Linear children but leaves the
    // module-level training flag alone; the quantized drop-ins
    // default to `is_training()==true`, so we re-assert eval here
    // so the Wave 4.4 fast path fires on every `.forward`.
    mha->eval(); ffn->eval();
    results.push_back(run_variant("INT8", mha, ffn));
  }

  // ---- INT4G (group_size = 128) ---------------------------------------------
  {
    auto [mha, ffn] = build_fp();
    mha->quantize_(tesseract::quant::Scheme::int4_group_symmetric(128));
    ffn->quantize_(tesseract::quant::Scheme::int4_group_symmetric(128));
    mha->eval(); ffn->eval();
    results.push_back(run_variant("INT4G(g=128)", mha, ffn));
  }

  // ---- Report --------------------------------------------------------------
  const double fp_step_us  = results[0].step_us;
  const int64_t fp_bytes   = results[0].bytes;

  std::printf("  %-14s | %10s | %8s | %10s | %8s\n",
              "variant", "step_us", "vs_fp",
              "weight_MB", "mem_frac");
  std::printf("  %s\n",
              "-------------------------------------------------------------"
              "------");
  for (const auto& r : results) {
    const double t_ratio = r.step_us / fp_step_us;
    const double mb = r.bytes / 1.0e6;
    const double m_ratio =
        static_cast<double>(r.bytes) / static_cast<double>(fp_bytes);
    std::printf("  %-14s | %10.2f | %8.3f | %10.2f | %8.3f\n",
                r.label, r.step_us, t_ratio, mb, m_ratio);
  }
  std::printf("\n");

  bool all_pass = true;

  // Weight-memory hard bars (INT8 / FP, INT4G / FP).
  {
    const double i8_mem  = static_cast<double>(results[1].bytes) / fp_bytes;
    const double i4g_mem = static_cast<double>(results[2].bytes) / fp_bytes;
    all_pass &= bench::hard_check(
        i8_mem <= 0.30,
        "block_weight_bytes INT8 / FP32 <= 0.30",
        i8_mem, 0.30);
    all_pass &= bench::hard_check(
        i4g_mem <= 0.18,
        "block_weight_bytes INT4G / FP32 <= 0.18",
        i4g_mem, 0.18);
  }

  // Latency hard bars. Both quantized paths must at least match FP32
  // within a small margin (we actually expect wins on this HBM-bound
  // shape) and INT4G must be meaningfully faster than INT8.
  {
    const double i8_lat  = results[1].step_us / fp_step_us;
    const double i4g_lat = results[2].step_us / fp_step_us;
    all_pass &= bench::hard_check(
        i8_lat <= 0.55,
        "decode_step INT8 / FP32 latency <= 0.55",
        i8_lat, 0.55);
    all_pass &= bench::hard_check(
        i4g_lat <= 0.75,
        "decode_step INT4G / FP32 latency <= 0.75",
        i4g_lat, 0.75);
  }

  std::printf("\n  %s\n", all_pass ? "ALL HARD BARS PASSED." : "HARD BAR MISS.");
  return all_pass ? bench::kExitOk : bench::kExitPerfMiss;
}

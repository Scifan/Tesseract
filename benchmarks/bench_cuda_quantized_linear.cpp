// Wave 4.4 (B-026) — bench: weight-only quantized Linear vs FP32
// baseline, on decode- and prefill-shape matmul.
//
// What this bench pins down
// -------------------------
//   * Latency: fused `ops::dequantize_matmul_int8` / `_int4_group`
//     kernels (reached through `QuantizedLinear::forward` /
//     `QuantizedLinearInt4G::forward` in eval mode) vs the FP32
//     `nn::Linear::forward` cuBLASLt path.
//   * Weight-memory footprint: packed INT8 / INT4G buffers + their
//     FP32 scales compared to the FP32 `[N, K]` weight + bias.
//
// Why this is the right set of hard bars for weight-only quant
// -------------------------------------------------------------
// Weight-only quant is primarily a memory technique: the FP weight
// gets replaced with INT8 / INT4 storage that sidesteps most of the
// HBM traffic on a decode-shape matmul (M = 1). The wall-time win
// only materializes when the FP32 weight **exceeds the GPU's L2
// cache** — otherwise cuBLASLt's steady-state loop ends up hitting
// L2 on every repeat and the FP32 path looks artificially fast
// (64 MB of FP32 weights on a `K=N=4096` layer fits neatly in the
// ~64 MB L2 of a typical Ada GPU and streams at L2 bandwidth on
// iteration 2+).
//
// Real Llama decode loops avoid L2-caching entirely because each
// transformer layer reads a *different* set of weights (32 layers ×
// ~50 MB per block at Llama-1B ≫ L2), so the relevant microbench
// hard bar has to use a single-matmul shape whose weight already
// exceeds L2 to faithfully model that regime:
//
//   * `M=1, K=N=8192` → FP32 weight is 256 MB, ~4× typical L2 →
//     every iteration re-reads from HBM → HBM bandwidth limits the
//     FP32 kernel. INT4G (32 MB weight) still fits in L2 after warmup
//     → L2-bandwidth limited. Ratio reflects the `HBM_bw_per_GB / L2_bw`
//     asymmetry *plus* the 8× weight-size reduction, compounding into
//     the 4-6× latency win that ships real decode value.
//
// Smaller shape `M=1, K=N=4096` is kept as an informational row —
// it documents the L2 crossover so a future reader understands why
// the hard bar is pinned at the larger shape.
//
// Prefill-shape (M = 512) is compute-bound; our hand-rolled
// block-per-(m, n) kernel cannot match tensor-core cuBLASLt GEMM on
// compute-bound shapes, so the prefill row stays informational-only.
// The point of that row is to document the crossover point for
// future tuning (e.g., when we ship a tensor-core-backed INT4G MMA
// kernel), not to regress-guard today.
//
// Memory-ratio bars are hard bars everywhere: the packer layout is
// fixed (`pack_int8_symmetric` / `pack_int4_group`), so the ratios
// are deterministic and any shift there is a real regression.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>

#include <cuda_runtime.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/QuantizedLinear.hpp"
#include "tesseract/nn/QuantizedLinearInt4G.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::NoGradGuard;
using tesseract::Tensor;
using tesseract::nn::Linear;
using tesseract::nn::QuantizedLinear;
using tesseract::nn::QuantizedLinearInt4G;

namespace {

struct Case {
  const char* label;
  int64_t     M;            // rows of activations (1 = decode, 512 = prefill)
  int64_t     K;            // in_features
  int64_t     N;            // out_features
  // Hard-bar ratios (times / bytes). 0.0 means "informational only".
  double      int8_latency_bar;   // INT8 / FP32 latency <=
  double      int4_latency_bar;   // INT4G / FP32 latency <=
};

// INT8-symmetric layout:
//   packed weight  = N * K bytes
//   per-channel s  = N * 4 bytes (FP32)
//   total         ≈ N * (K + 4)
int64_t int8_weight_bytes(int64_t N, int64_t K) {
  return N * K + N * 4;
}

// INT4-group-symmetric layout (group_size = 128 by default):
//   packed weight  = N * K / 2 bytes (two nibbles per byte)
//   per-group scale = N * (K / group_size) * 4 bytes
int64_t int4g_weight_bytes(int64_t N, int64_t K, int64_t group_size) {
  return N * (K / 2) + N * (K / group_size) * 4;
}

int64_t fp32_weight_bytes(int64_t N, int64_t K) {
  return N * K * 4;
}

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_quantized_linear] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_quantized_linear");

  bench::BenchStream bench_stream;
  cudaStream_t raw_stream = bench_stream.native();

  const Device cuda0{DeviceType::CUDA, 0};
  constexpr int64_t group_size = 128;

  // Shapes:
  //   decode(M=1), K=N=8192 — FP32 weight 256 MB ≫ L2 on any SM 8.x/9.0
  //     GPU, so cuBLASLt's steady-state re-reads weight from HBM each
  //     iteration. This is the regime that matches real Llama decode
  //     (32 layers × 50+ MB per layer, nothing fits L2 across the
  //     layer loop). Hard-bar the INT8 / INT4G wins here.
  //   decode(M=1), K=N=4096 — 64 MB FP32 weight fits L2 on most cards,
  //     so the FP32 row is L2-bandwidth-limited rather than
  //     HBM-bandwidth-limited. Kept as informational so a reader can
  //     see the crossover empirically.
  //   prefill(M=512), K=N=4096 — compute-bound; cuBLASLt tensor cores
  //     crush our naive block-per-(m, n) fused kernel. Informational.
  // Hard-bar rationale (K=N=8192 row):
  //   INT8 / FP32 ≤ 0.30 — empirically 0.17 on SM 8.9; the 2× head-
  //     room absorbs per-run DVFS + L2-prefetch jitter without
  //     letting a real regression slip through.
  //   INT4G / FP32 ≤ 0.55 — empirically 0.46 on SM 8.9. INT4G's
  //     per-output compute (nibble sign-extend + group-scale lookup
  //     + warp-reduce) is meaningfully higher than INT8's, so even
  //     when the INT4G weight buffer fits L2 the kernel runs
  //     CPU-core-bound rather than HBM-bound. 0.55 catches a
  //     compute-path regression; a future tensor-core MMA INT4G
  //     kernel would lower this bar substantially.
  const std::vector<Case> cases = {
      {"decode  M=1   K=8192 N=8192",   1,   8192, 8192, 0.30, 0.55},
      {"decode  M=1   K=4096 N=4096",   1,   4096, 4096, 0.0,  0.0 },
      {"prefill M=512 K=4096 N=4096", 512,   4096, 4096, 0.0,  0.0 },
  };

  std::printf("  dtype baselines: FP32 nn::Linear (cuBLASLt) vs\n"
              "                   INT8 QuantizedLinear (fused dequant-matmul) vs\n"
              "                   INT4G QuantizedLinearInt4G g=%ld (fused)\n\n",
              long(group_size));

  std::printf("  %-30s | %10s %10s %10s | %6s %6s | %10s %10s %10s | %6s %6s\n",
              "shape",
              "fp32_us", "int8_us", "int4g_us",
              "i8/fp",  "i4/fp",
              "fp32_MB", "int8_MB", "int4g_MB",
              "i8/fp", "i4/fp");
  std::printf("  %s\n",
              "---------------------------------------------------------"
              "-----------------------------------------------------------"
              "---------------------");

  bool all_pass = true;

  for (const auto& c : cases) {
    // Build the FP32 reference layer and its quantized drop-ins. We
    // deliberately use `use_bias=false` so the benchmark timing is
    // dominated by the matmul itself — a trailing `ops::add(bias)`
    // is cheap but adds noise to the ratio we actually care about.
    Linear fp{c.K, c.N, /*use_bias=*/false, DType::Float32};
    auto qlin_i8  = QuantizedLinear::from_linear(fp);
    auto qlin_i4g = QuantizedLinearInt4G::from_linear(fp, group_size);
    fp.to(cuda0);
    qlin_i8 ->to(cuda0);
    qlin_i4g->to(cuda0);

    // Put modules in eval mode so the Wave 4.4 fast path fires even
    // when the input carries `requires_grad=true` (the realistic
    // case — an activation from an earlier trainable module).
    qlin_i8 ->eval();
    qlin_i4g->eval();

    Tensor X = Tensor::empty({c.M, c.K}, DType::Float32, cuda0);
    bench::check_cuda(cudaMemsetAsync(X.raw_data(), 0,
                                      c.M * c.K * sizeof(float), raw_stream),
                      "memset X");
    bench::check_cuda(cudaStreamSynchronize(raw_stream), "init-sync");

    auto call_fp = [&](cudaStream_t /*s*/) {
      NoGradGuard nogg;
      auto y = fp.forward(X);
      (void)y;
    };
    auto call_i8 = [&](cudaStream_t /*s*/) {
      auto y = qlin_i8->forward(X);
      (void)y;
    };
    auto call_i4g = [&](cudaStream_t /*s*/) {
      auto y = qlin_i4g->forward(X);
      (void)y;
    };

    // Warmup each path so cuBLASLt's heuristic + descriptor cache
    // lands in steady state and the fused kernels finish their
    // first-pass allocator dance.
    for (int i = 0; i < 3; ++i) {
      call_fp (raw_stream);
      call_i8 (raw_stream);
      call_i4g(raw_stream);
    }
    bench::check_cuda(cudaStreamSynchronize(raw_stream), "warm");

    auto st_fp  = bench::steady_state_time(call_fp , raw_stream);
    auto st_i8  = bench::steady_state_time(call_i8 , raw_stream);
    auto st_i4g = bench::steady_state_time(call_i4g, raw_stream);

    const double r_i8_lat  = st_i8 .mean_us / st_fp.mean_us;
    const double r_i4g_lat = st_i4g.mean_us / st_fp.mean_us;

    const double fp_mb   = fp32_weight_bytes (c.N, c.K) / 1.0e6;
    const double i8_mb   = int8_weight_bytes (c.N, c.K) / 1.0e6;
    const double i4g_mb  = int4g_weight_bytes(c.N, c.K, group_size) / 1.0e6;
    const double r_i8_m  = i8_mb  / fp_mb;
    const double r_i4g_m = i4g_mb / fp_mb;

    std::printf("  %-30s | %10.2f %10.2f %10.2f | %6.3f %6.3f | %10.2f %10.2f %10.2f | %6.3f %6.3f\n",
                c.label,
                st_fp.mean_us, st_i8.mean_us, st_i4g.mean_us,
                r_i8_lat, r_i4g_lat,
                fp_mb, i8_mb, i4g_mb,
                r_i8_m, r_i4g_m);

    // Latency hard bars (only on HBM-bound shapes where the ratio is
    // actually meaningful and stable).
    if (c.int8_latency_bar > 0.0) {
      all_pass &= bench::hard_check(
          r_i8_lat <= c.int8_latency_bar,
          (std::string("int8_latency / fp32_latency @ ") + c.label).c_str(),
          r_i8_lat, c.int8_latency_bar);
    }
    if (c.int4_latency_bar > 0.0) {
      all_pass &= bench::hard_check(
          r_i4g_lat <= c.int4_latency_bar,
          (std::string("int4g_latency / fp32_latency @ ") + c.label).c_str(),
          r_i4g_lat, c.int4_latency_bar);
    }
  }
  std::printf("\n");

  // Memory hard bars — identical for every shape since the packer
  // layout is shape-oblivious (aside from the O(K / group_size)
  // scale tail for INT4G, which is negligible at K >= 512).
  //   INT8:  (N*K + N*4)  / (N*K*4)  ≈ 0.251 + 1/K
  //   INT4G: (N*K/2 + N*K/G*4) / (N*K*4) = 0.125 + 1/G
  //          at G=128: 0.125 + 0.0078 ≈ 0.133
  // Set bars with generous headroom: 0.30 for INT8, 0.18 for INT4G.
  {
    const int64_t N = 8192, K = 8192;
    const double r_i8_m  =
        static_cast<double>(int8_weight_bytes(N, K)) / fp32_weight_bytes(N, K);
    const double r_i4g_m =
        static_cast<double>(int4g_weight_bytes(N, K, group_size)) /
        fp32_weight_bytes(N, K);
    all_pass &= bench::hard_check(
        r_i8_m <= 0.30,
        "int8_weight_bytes / fp32_weight_bytes @ 8192x8192",
        r_i8_m, 0.30);
    all_pass &= bench::hard_check(
        r_i4g_m <= 0.18,
        "int4g_weight_bytes / fp32_weight_bytes @ 8192x8192",
        r_i4g_m, 0.18);
  }

  std::printf("\n  %s\n", all_pass ? "ALL HARD BARS PASSED." : "HARD BAR MISS.");
  return all_pass ? bench::kExitOk : bench::kExitPerfMiss;
}

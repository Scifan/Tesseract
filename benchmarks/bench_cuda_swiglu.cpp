// Wave 4.1 (B-025) — bench: fused `ops::swiglu_silu_gate` vs composite
// `mul(gate, sigmoid(gate)) * up` vs memcpy D2D roofline.
//
// Same three-row layout as `bench_cuda_rms_norm`:
//
//   composite — sigmoid(gate) → mul(gate, sig) → mul(silu_gate, up)
//               (3 launches, 5 reads + 3 writes of the
//                [..., d_ff] intermediate tensor).
//   fused     — `launch_swiglu_silu_gate` (1 launch, 2 reads + 1 write).
//   memcpy    — theoretical roofline reference for one read + one write
//               of the [..., d_ff] tensor.
//
// Effective-bandwidth counts one read of `gate`, one read of `up`, and
// one write of `out` (i.e. 3 × same-sized tensor), matching the fused
// kernel's actual traffic. The composite's additional 5 traffic units
// show up as a lower `eff_GB/s`.
//
// Hard bars (largest shape — where launch-overhead is most amortized):
//   * fused speedup over composite ≥ 2.0×
//   * fused eff_GB/s / memcpy ≥ 0.80
//
// Both bars are conservative: on SM 8.9 Ada with `launch_swiglu_silu_gate`'s
// single-pass FP32-promoted pattern we should sit north of 0.85 × roofline
// on FP32 and well above 2.3× over the composite on d_ff=4096 shapes.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;

// Composite reference — rebuilds the pre-B-025 tail that used to live
// in `nn::FeedForward::forward`. We bypass `ops::swiglu_silu_gate`
// (which now routes to the fused kernel under NoGradGuard) and instead
// reconstruct the composite explicitly for a clean apples-to-apples
// measurement against the fused path.
static void composite_swiglu(const tesseract::Tensor& gate,
                             const tesseract::Tensor& up) {
  using tesseract::Tensor;
  namespace t_ops = tesseract::ops;
  Tensor sig  = t_ops::sigmoid(gate);
  Tensor silu = t_ops::mul(gate, sig);
  Tensor y    = t_ops::mul(silu, up);
  (void)y;
}

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_swiglu] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_swiglu (fused vs composite, FP32)");

  bench::BenchStream bench_stream;
  cudaStream_t stream = bench_stream.native();

  const double memcpy_gbs = bench::measure_d2d_memcpy_gbs(stream);
  std::printf("  memcpy D2D roofline : %7.1f GB/s\n\n", memcpy_gbs);

  using tesseract::Tensor;
  using tesseract::DType;
  using tesseract::Device;
  using tesseract::DeviceType;
  using tesseract::NoGradGuard;
  Device cuda0{DeviceType::CUDA, 0};

  // (B, S, d_ff): B=8/32, S=512/2048/4096, d_ff spanning 1k-4k to
  // cover both small and Llama-decode-sized FFN intermediates. d_ff
  // is the element-wise axis so the kernel's grid-stride loop sees a
  // realistic range of elements-per-row.
  struct Case { int64_t B, S, D; };
  Case cases[] = {
      {  8,  512, 1024 },
      { 32, 2048, 4096 },
      { 16, 4096, 4096 },
  };

  std::printf("%-14s | %12s %12s %10s | %9s %9s | %9s\n",
              "(B,S,d_ff)", "fused_us", "composite_us", "speedup",
              "fused_GB/s", "comp_GB/s", "fused_frac");
  std::printf("%s\n",
              "---------------------------------------------------------------"
              "---------------------------------");

  double largest_fused_frac = 0.0;
  double largest_speedup    = 0.0;
  Case largest = cases[0];
  for (const auto& c : cases) {
    if (static_cast<int64_t>(c.B) * c.S * c.D >
        static_cast<int64_t>(largest.B) * largest.S * largest.D) {
      largest = c;
    }
  }

  for (const auto& c : cases) {
    auto G = Tensor::empty({c.B, c.S, c.D}, DType::Float32, cuda0);
    auto U = Tensor::empty({c.B, c.S, c.D}, DType::Float32, cuda0);
    const std::size_t xb = static_cast<std::size_t>(c.B) * c.S * c.D * 4;
    bench::check_cuda(cudaMemsetAsync(G.raw_data(), 0, xb, stream),
                      "memset G");
    bench::check_cuda(cudaMemsetAsync(U.raw_data(), 0, xb, stream),
                      "memset U");
    bench::check_cuda(cudaStreamSynchronize(stream), "init-sync");

    auto call_fused = [&](cudaStream_t /*s*/) {
      NoGradGuard nogg;
      auto y = tesseract::ops::swiglu_silu_gate(G, U);
      (void)y;
    };
    auto call_comp = [&](cudaStream_t /*s*/) {
      NoGradGuard nogg;
      composite_swiglu(G, U);
    };
    call_fused(stream);
    call_comp(stream);
    bench::check_cuda(cudaStreamSynchronize(stream), "warm");

    auto st_fused = bench::steady_state_time(call_fused, stream);
    auto st_comp  = bench::steady_state_time(call_comp,  stream);

    // Effective bandwidth for the fused kernel: 1 read(gate) +
    // 1 read(up) + 1 write(out) = 3 × xb. The composite reads/writes
    // more, but we attribute the same denominator to both so the
    // *ratio* reflects only the launch-level improvement — the GB/s
    // column then tells you how close each approach sits to the
    // memcpy roofline.
    const double bytes   = 3.0 * static_cast<double>(xb);
    const double gbs_f   = bytes / (st_fused.mean_us * 1e-6) / 1e9;
    const double gbs_c   = bytes / (st_comp.mean_us  * 1e-6) / 1e9;
    const double speedup = st_comp.mean_us / st_fused.mean_us;
    const double frac_f  = gbs_f / memcpy_gbs;
    char label[32];
    std::snprintf(label, sizeof(label), "(%ld,%ld,%ld)",
                  long(c.B), long(c.S), long(c.D));
    std::printf("%-14s | %12.2f %12.2f %10.2f | %9.1f %9.1f | %9.3f\n",
                label, st_fused.mean_us, st_comp.mean_us, speedup,
                gbs_f, gbs_c, frac_f);

    if (c.B == largest.B && c.S == largest.S && c.D == largest.D) {
      largest_fused_frac = frac_f;
      largest_speedup    = speedup;
    }
  }
  std::printf("\n");

  // Hard bars: enforce on the largest shape. 2.0× over composite is
  // the conservative lower bound (composite does 5R+3W of traffic vs
  // our 2R+1W, i.e. memcpy-bounded expectation is ~2.67×). 0.80 ×
  // memcpy tracks the `rms_norm` bar since both kernels are
  // element-wise-with-reduction-at-most and hit the same HBM ceiling.
  constexpr double kMinFusedFracLargest = 0.80;
  constexpr double kMinSpeedupLargest   = 2.00;

  const bool ok_bw = bench::hard_check(
      largest_fused_frac >= kMinFusedFracLargest,
      "fused eff_GB/s / memcpy @ largest shape  >= 0.80",
      largest_fused_frac, kMinFusedFracLargest);
  const bool ok_su = bench::hard_check(
      largest_speedup >= kMinSpeedupLargest,
      "fused speedup over composite @ largest shape >= 2.0",
      largest_speedup, kMinSpeedupLargest);

  return (ok_bw && ok_su) ? bench::kExitOk : bench::kExitPerfMiss;
}

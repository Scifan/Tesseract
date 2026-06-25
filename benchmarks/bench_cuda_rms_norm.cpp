// Wave 2 (B-022) — bench: fused `ops::rms_norm` vs primitive composite vs
// memcpy D2D roofline.
//
// M2L.1 shipped this bench as a baseline; Wave 2 adds the fused
// one-block-per-row kernel that collapses the composite's 5-6
// per-element passes into a single 2-pass kernel. We now print
// **three** rows per shape:
//
//   composite  — mul(x,x) → mean → add(eps) → sqrt → div(x) → mul(w)
//                (6 CUDA kernel launches, ≥5 global-mem read passes
//                over `x`, one pass for `weight`, one write).
//   fused      — `launch_rms_norm` (1 launch, 2 passes over `x`,
//                one pass over `w`, one write).
//   memcpy     — theoretical roofline reference.
//
// Effective-bandwidth counts one read of `x`, one read of `w`, and
// one write of `y`. The composite's extra passes show up as a lower
// `eff_GB/s` (traffic per useful op goes up). The hard bar:
//
//   * fused `eff_GB/s` ≥ 0.80 × memcpy D2D roofline on the largest
//     shape (`D=4096` is where we get the most compute-launch
//     amortization).
//   * fused speedup over composite ≥ 3× on the largest shape.
//
// Both bars are conservative: on SM 8.9 Ada with `launch_rms_norm`'s
// FP32-promoted single-pass pattern, we should sit closer to 0.9 ×
// roofline on FP32 and well above 3× over the composite.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/detail/RMSNorm.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Normalization.hpp"
#include "tesseract/ops/Reduction.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;

// Composite launcher — rebuilds the pre-Wave-2 path by calling the
// primitive chain directly. We bypass `ops::rms_norm` here (which
// now routes to the fused kernel under NoGradGuard) and instead
// reconstruct the composite to keep a clean apples-to-apples number.
static void composite_rms_norm(const tesseract::Tensor& x,
                               const tesseract::Tensor& w,
                               double eps) {
  using tesseract::Tensor;
  namespace t_ops = tesseract::ops;
  Tensor sq    = t_ops::mul(x, x);
  Tensor ms    = t_ops::mean(sq, static_cast<int64_t>(x.rank()) - 1,
                             /*keepdim=*/true);
  Tensor eps_t = Tensor::full({}, eps, x.dtype(), x.device());
  Tensor denom = t_ops::sqrt(t_ops::add(ms, eps_t));
  Tensor yhat  = t_ops::div(x, denom);
  Tensor y     = t_ops::mul(yhat, w);
  (void)y;
}

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_rms_norm] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_rms_norm (fused vs composite, FP32)");

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

  struct Case { int64_t B, S, D; };
  Case cases[] = {
      {  8,  512, 1024 },
      { 32, 2048, 4096 },
      { 16, 4096, 4096 },
  };

  std::printf("%-14s | %12s %12s %10s | %9s %9s | %9s\n",
              "(B,S,D)", "fused_us", "composite_us", "speedup",
              "fused_GB/s", "comp_GB/s", "fused_frac");
  std::printf("%s\n",
              "---------------------------------------------------------------"
              "---------------------------------");

  // Hard bars — enforced on the last (largest) shape only. The
  // smaller shapes are launch-overhead-sensitive; reporting only
  // (per_us, GB/s, fraction) is enough.
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
    auto X = Tensor::empty({c.B, c.S, c.D}, DType::Float32, cuda0);
    auto W = Tensor::ones ({c.D},            DType::Float32, cuda0);
    const std::size_t xb = static_cast<std::size_t>(c.B) * c.S * c.D * 4;
    bench::check_cuda(cudaMemsetAsync(X.raw_data(), 0, xb, stream),
                      "memset X");
    bench::check_cuda(cudaStreamSynchronize(stream), "init-sync");

    auto call_fused = [&](cudaStream_t /*s*/) {
      NoGradGuard nogg;
      auto y = tesseract::ops::rms_norm(X, W, 1e-5);
      (void)y;
    };
    auto call_comp = [&](cudaStream_t /*s*/) {
      NoGradGuard nogg;
      composite_rms_norm(X, W, 1e-5);
    };
    call_fused(stream);
    call_comp(stream);
    bench::check_cuda(cudaStreamSynchronize(stream), "warm");

    auto st_fused = bench::steady_state_time(call_fused, stream);
    auto st_comp  = bench::steady_state_time(call_comp,  stream);

    // Effective bandwidth: 1 read X + 1 write Y = 2·xb (ignore W as
    // negligible). Both variants get the same denominator so the
    // *ratio* reflects only the kernel-level improvement.
    const double bytes   = 2.0 * static_cast<double>(xb);
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

  // Hard bars: enforce on the largest shape. Picking up 0.80 × roofline
  // is already conservative for a one-block-per-row 2-pass kernel;
  // the 3× composite speedup is what the Wave-2 design note promises.
  constexpr double kMinFusedFracLargest = 0.80;
  constexpr double kMinSpeedupLargest   = 3.00;

  const bool ok_bw = bench::hard_check(
      largest_fused_frac >= kMinFusedFracLargest,
      "fused eff_GB/s / memcpy @ largest shape  >= 0.80",
      largest_fused_frac, kMinFusedFracLargest);
  const bool ok_su = bench::hard_check(
      largest_speedup >= kMinSpeedupLargest,
      "fused speedup over composite @ largest shape >= 3.0",
      largest_speedup, kMinSpeedupLargest);

  return (ok_bw && ok_su) ? bench::kExitOk : bench::kExitPerfMiss;
}

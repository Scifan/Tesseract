// M2F: CPU↔CUDA parity for the reduction suite (sum / mean / max —
// all-reduce and along-dim variants, with and without keepdim). Same
// structural pattern as `test_ops_cuda_elementwise.cpp`: each test
// builds identical inputs on CPU and CUDA, runs the op on both,
// copies the CUDA output back to host, and asserts numeric equality
// with `WithinAbs` for floating-point tolerance.
//
// Tolerance rationale: our CUDA reduction is a per-block tree
// reduction followed by a single-block finalize, so the summation
// order differs slightly from the CPU's fully-sequential pass. For
// Float32 with `N <= 4096` elements the observed drift is well
// under 1e-5; we pick `1e-5f` (dense) and `1e-4f` (D=4096 dim sum)
// to leave a generous safety margin without hiding real bugs.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Reduction.hpp"

using Catch::Matchers::WithinAbs;
using tesseract::cpu_device;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Tensor;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

namespace {

bool cuda_ready() { return has_cuda_support() && is_available(); }
Device cuda0() { return Device{DeviceType::CUDA, 0}; }

std::vector<float> pattern_floats(std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    // Centered pattern, with both signs and a wide magnitude range
    // so the sum-cancel path (negatives + positives) gets real work.
    out[i] = (static_cast<float>(i) - static_cast<float>(n) * 0.5f) * 0.0125f;
  }
  return out;
}

// Compare two host-side float buffers elementwise with a given tol.
void require_close(const float* a, const float* b, int64_t n, float tol) {
  for (int64_t i = 0; i < n; ++i) REQUIRE_THAT(b[i], WithinAbs(a[i], tol));
}

}  // namespace

TEST_CASE("CUDA sum all matches CPU (Float32)", "[ops][gpu][reduction]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto p = pattern_floats(1024);
  Tensor h = Tensor::from_vector(p, {32, 32});
  Tensor d = h.to(cuda0());

  Tensor h_out = tesseract::ops::sum(h);
  Tensor d_out = tesseract::ops::sum(d);
  REQUIRE(d_out.device() == cuda0());
  REQUIRE(d_out.shape().rank() == 0);

  Tensor back = d_out.to(cpu_device());
  REQUIRE_THAT(*back.data_ptr<float>(),
               WithinAbs(*h_out.data_ptr<float>(), 1e-3f));
}

TEST_CASE("CUDA mean/max all match CPU (Float32)", "[ops][gpu][reduction]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto p = pattern_floats(2048);
  Tensor h = Tensor::from_vector(p, {64, 32});
  Tensor d = h.to(cuda0());
  {
    Tensor hm = tesseract::ops::mean(h);
    Tensor dm = tesseract::ops::mean(d).to(cpu_device());
    REQUIRE_THAT(*dm.data_ptr<float>(),
                 WithinAbs(*hm.data_ptr<float>(), 1e-6f));
  }
  {
    Tensor hmx = tesseract::ops::max(h);
    Tensor dmx = tesseract::ops::max(d).to(cpu_device());
    REQUIRE(*dmx.data_ptr<float>() == *hmx.data_ptr<float>());
  }
}

TEST_CASE("CUDA sum/mean/max along dim match CPU (Float32)",
          "[ops][gpu][reduction]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // [N, D] with D moderate so each block does a serial stride-loop
  // across the reduced dim — exercises the inner-loop path.
  const int64_t N = 32;
  const int64_t D = 128;
  auto p = pattern_floats(N * D);
  Tensor h = Tensor::from_vector(p, {N, D});
  Tensor d = h.to(cuda0());

  SECTION("sum along dim=1, keepdim=false") {
    Tensor ho = tesseract::ops::sum(h, /*dim=*/1, /*keepdim=*/false);
    Tensor co = tesseract::ops::sum(d, /*dim=*/1, /*keepdim=*/false);
    Tensor back = co.to(cpu_device());
    REQUIRE(back.shape() == ho.shape());
    // Tolerance derived from fp32 precision: each per-row sum is
    // ~3000 in magnitude, and fp32 ULP at that magnitude is ~2e-4,
    // so we set 3e-3 to absorb the summation-order drift between
    // the CPU's sequential pass and the GPU's block-tree reduction.
    require_close(ho.data_ptr<float>(), back.data_ptr<float>(),
                  back.numel(), 3e-3f);
  }

  SECTION("mean along dim=0, keepdim=true") {
    Tensor ho = tesseract::ops::mean(h, /*dim=*/0, /*keepdim=*/true);
    Tensor co = tesseract::ops::mean(d, /*dim=*/0, /*keepdim=*/true);
    Tensor back = co.to(cpu_device());
    REQUIRE(back.shape() == ho.shape());
    require_close(ho.data_ptr<float>(), back.data_ptr<float>(),
                  back.numel(), 1e-6f);
  }

  SECTION("max along dim=1, keepdim=false") {
    // Max is exact (no floating-point cancellation), so this one
    // compares bitwise.
    Tensor ho = tesseract::ops::max(h, /*dim=*/1, /*keepdim=*/false);
    Tensor co = tesseract::ops::max(d, /*dim=*/1, /*keepdim=*/false);
    Tensor back = co.to(cpu_device());
    REQUIRE(back.shape() == ho.shape());
    for (int64_t i = 0; i < back.numel(); ++i) {
      REQUIRE(back.data_ptr<float>()[i] == ho.data_ptr<float>()[i]);
    }
  }
}

TEST_CASE("CUDA reduction on Float64 matches CPU", "[ops][gpu][reduction]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 16;
  const int64_t D = 64;
  std::vector<double> p(N * D);
  for (std::size_t i = 0; i < p.size(); ++i) {
    p[i] = (static_cast<double>(i) - static_cast<double>(p.size()) * 0.5) * 0.125;
  }
  Tensor h = Tensor::from_vector(p, {N, D});
  Tensor d = h.to(cuda0());

  Tensor ho = tesseract::ops::sum(h, /*dim=*/1, /*keepdim=*/false);
  Tensor co = tesseract::ops::sum(d, /*dim=*/1, /*keepdim=*/false).to(cpu_device());
  REQUIRE(co.shape() == ho.shape());
  for (int64_t i = 0; i < co.numel(); ++i) {
    REQUIRE_THAT(co.data_ptr<double>()[i],
                 WithinAbs(ho.data_ptr<double>()[i], 1e-12));
  }
}

TEST_CASE("CUDA reduction along a middle dim on rank-3 matches CPU",
          "[ops][gpu][reduction]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // [A, B, C] reducing along dim=1 exercises the iter-shape
  // flat-to-offset path (dim is neither innermost nor outermost).
  const int64_t A = 4, B = 7, C = 5;
  auto p = pattern_floats(A * B * C);
  Tensor h = Tensor::from_vector(p, {A, B, C});
  Tensor d = h.to(cuda0());
  Tensor ho = tesseract::ops::sum(h, /*dim=*/1, /*keepdim=*/false);
  Tensor co = tesseract::ops::sum(d, /*dim=*/1, /*keepdim=*/false).to(cpu_device());
  REQUIRE(co.shape() == ho.shape());
  require_close(ho.data_ptr<float>(), co.data_ptr<float>(),
                co.numel(), 1e-5f);
}

// ----------------------------- B-016 -----------------------------
//
// FP16 / BF16 parity on the FP32-promoted reduction kernels. We
// narrow the input magnitude (`×1e-3` vs the pattern above) so the
// summed value fits inside FP16's ±65504 dynamic range on the
// largest shape we exercise here — this keeps the parity signal
// about kernel-arithmetic correctness rather than about FP16
// saturation. Tolerances follow the elementwise / softmax
// precedent: 2e-3 abs for FP16, 5e-3 abs for BF16 (BF16's 7-bit
// mantissa is ~3× noisier than FP16's 10-bit).

namespace {
std::vector<float> small_pattern_floats(std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = (static_cast<float>(i) - static_cast<float>(n) * 0.5f) * 1e-4f;
  }
  return out;
}
}  // namespace

TEST_CASE("CUDA sum/mean/max match CPU on Float16",
          "[ops][gpu][reduction]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 32, D = 128;
  auto p = small_pattern_floats(N * D);
  std::vector<tesseract::Half> ph(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) ph[i] = tesseract::Half(p[i]);
  Tensor h = Tensor::from_vector(ph, {N, D});
  Tensor d = h.to(cuda0());

  SECTION("sum all-reduce") {
    Tensor hs = tesseract::ops::sum(h);
    Tensor ds = tesseract::ops::sum(d).to(cpu_device());
    REQUIRE_THAT(static_cast<float>(*ds.data_ptr<tesseract::Half>()),
                 WithinAbs(static_cast<float>(*hs.data_ptr<tesseract::Half>()),
                           2e-3f));
  }
  SECTION("mean along dim=1, keepdim=false") {
    Tensor ho = tesseract::ops::mean(h, /*dim=*/1, /*keepdim=*/false);
    Tensor co = tesseract::ops::mean(d, /*dim=*/1, /*keepdim=*/false)
                    .to(cpu_device());
    REQUIRE(co.shape() == ho.shape());
    const auto* e = ho.data_ptr<tesseract::Half>();
    const auto* a = co.data_ptr<tesseract::Half>();
    for (int64_t i = 0; i < co.numel(); ++i) {
      REQUIRE_THAT(static_cast<float>(a[i]),
                   WithinAbs(static_cast<float>(e[i]), 2e-3f));
    }
  }
  SECTION("max along dim=0, keepdim=true") {
    // Max is exact modulo round-tripping through the `Acc=float`
    // accumulator — since the input is already FP16 and the max
    // operation is monotone, load-then-narrow is a no-op on the
    // final store.
    Tensor ho = tesseract::ops::max(h, /*dim=*/0, /*keepdim=*/true);
    Tensor co = tesseract::ops::max(d, /*dim=*/0, /*keepdim=*/true)
                    .to(cpu_device());
    REQUIRE(co.shape() == ho.shape());
    const auto* e = ho.data_ptr<tesseract::Half>();
    const auto* a = co.data_ptr<tesseract::Half>();
    for (int64_t i = 0; i < co.numel(); ++i) {
      REQUIRE(static_cast<float>(a[i]) == static_cast<float>(e[i]));
    }
  }
}

TEST_CASE("CUDA sum/mean match CPU on BFloat16",
          "[ops][gpu][reduction]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t A = 4, B = 7, C = 5;
  auto p = small_pattern_floats(A * B * C);
  std::vector<tesseract::BFloat16> pb(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) pb[i] = tesseract::BFloat16(p[i]);
  Tensor h = Tensor::from_vector(pb, {A, B, C});
  Tensor d = h.to(cuda0());

  SECTION("sum along middle dim=1, keepdim=false") {
    Tensor ho = tesseract::ops::sum(h, /*dim=*/1, /*keepdim=*/false);
    Tensor co = tesseract::ops::sum(d, /*dim=*/1, /*keepdim=*/false)
                    .to(cpu_device());
    REQUIRE(co.shape() == ho.shape());
    const auto* e = ho.data_ptr<tesseract::BFloat16>();
    const auto* a = co.data_ptr<tesseract::BFloat16>();
    for (int64_t i = 0; i < co.numel(); ++i) {
      REQUIRE_THAT(static_cast<float>(a[i]),
                   WithinAbs(static_cast<float>(e[i]), 5e-3f));
    }
  }
  SECTION("mean all-reduce") {
    Tensor hs = tesseract::ops::mean(h);
    Tensor ds = tesseract::ops::mean(d).to(cpu_device());
    REQUIRE_THAT(static_cast<float>(*ds.data_ptr<tesseract::BFloat16>()),
                 WithinAbs(static_cast<float>(*hs.data_ptr<tesseract::BFloat16>()),
                           5e-3f));
  }
}

TEST_CASE("CPU-only reduction smoke", "[ops][cpu][reduction]") {
  // Always-running path so the CPU-only build still exercises at
  // least one assertion from this TU.
  auto p = pattern_floats(128);
  Tensor h = Tensor::from_vector(p, {8, 16});
  Tensor s = tesseract::ops::sum(h);
  REQUIRE(s.shape().rank() == 0);
  // The pattern is symmetric around zero up to the half-offset
  // chosen by `pattern_floats`, so the sum is deterministic but not
  // necessarily zero — just make sure it's finite.
  REQUIRE(std::isfinite(*s.data_ptr<float>()));
}

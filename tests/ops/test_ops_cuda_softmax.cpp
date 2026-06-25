// M2F: CPU↔CUDA parity for softmax / log_softmax. Same test harness
// structure as `test_ops_cuda_reduction.cpp`; every GPU-touching
// case SKIPs cleanly when no CUDA is available.
//
// Tolerance rationale: softmax on CUDA uses `__expf`/`__logf` (the
// single-precision fast-math intrinsics, max ULP ~2 on Ada). We
// compare with `WithinAbs(..., 1e-5f)`, which is loose enough to
// absorb the intrinsic-vs-libm drift and tight enough to catch any
// real numerical mistake (like a missing max-subtract).

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Softmax.hpp"

using Catch::Matchers::WithinAbs;
using tesseract::cpu_device;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::Tensor;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

namespace {

bool cuda_ready() { return has_cuda_support() && is_available(); }
Device cuda0() { return Device{DeviceType::CUDA, 0}; }

// Pick a pattern that (a) has large magnitudes so numerically-naive
// implementations overflow `exp` and (b) includes negatives.
std::vector<float> logit_pattern(std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = (static_cast<float>(i) - static_cast<float>(n) * 0.5f) * 0.25f;
  }
  return out;
}

}  // namespace

TEST_CASE("CUDA softmax matches CPU (Float32, rank-2, dim=-1)",
          "[ops][gpu][softmax]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 16;
  const int64_t C = 32;
  auto p = logit_pattern(N * C);
  Tensor h = Tensor::from_vector(p, {N, C});
  Tensor d = h.to(cuda0());
  Tensor ho = tesseract::ops::softmax(h, /*dim=*/-1);
  Tensor co = tesseract::ops::softmax(d, /*dim=*/-1).to(cpu_device());
  REQUIRE(co.shape() == ho.shape());
  for (int64_t i = 0; i < co.numel(); ++i) {
    REQUIRE_THAT(co.data_ptr<float>()[i],
                 WithinAbs(ho.data_ptr<float>()[i], 1e-5f));
  }
}

TEST_CASE("CUDA log_softmax matches CPU (Float32)",
          "[ops][gpu][softmax]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 32;
  const int64_t C = 16;
  auto p = logit_pattern(N * C);
  Tensor h = Tensor::from_vector(p, {N, C});
  Tensor d = h.to(cuda0());
  Tensor ho = tesseract::ops::log_softmax(h, /*dim=*/1);
  Tensor co = tesseract::ops::log_softmax(d, /*dim=*/1).to(cpu_device());
  REQUIRE(co.shape() == ho.shape());
  for (int64_t i = 0; i < co.numel(); ++i) {
    REQUIRE_THAT(co.data_ptr<float>()[i],
                 WithinAbs(ho.data_ptr<float>()[i], 1e-5f));
  }
}

TEST_CASE("CUDA softmax along a non-last dim on rank-3",
          "[ops][gpu][softmax]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // [B, T, C] with softmax over T — the non-innermost path hits
  // non-trivial `in_strides[dim]` in the kernel.
  const int64_t B = 3, T = 12, C = 5;
  auto p = logit_pattern(B * T * C);
  Tensor h = Tensor::from_vector(p, {B, T, C});
  Tensor d = h.to(cuda0());
  Tensor ho = tesseract::ops::softmax(h, /*dim=*/1);
  Tensor co = tesseract::ops::softmax(d, /*dim=*/1).to(cpu_device());
  REQUIRE(co.shape() == ho.shape());
  for (int64_t i = 0; i < co.numel(); ++i) {
    REQUIRE_THAT(co.data_ptr<float>()[i],
                 WithinAbs(ho.data_ptr<float>()[i], 1e-5f));
  }
}

TEST_CASE("CUDA softmax on Float64 matches CPU",
          "[ops][gpu][softmax]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 8;
  const int64_t C = 16;
  std::vector<double> p(N * C);
  for (std::size_t i = 0; i < p.size(); ++i) {
    p[i] = (static_cast<double>(i) - static_cast<double>(p.size()) * 0.5) * 0.25;
  }
  Tensor h = Tensor::from_vector(p, {N, C});
  Tensor d = h.to(cuda0());
  Tensor ho = tesseract::ops::log_softmax(h, /*dim=*/-1);
  Tensor co = tesseract::ops::log_softmax(d, /*dim=*/-1).to(cpu_device());
  REQUIRE(co.shape() == ho.shape());
  for (int64_t i = 0; i < co.numel(); ++i) {
    REQUIRE_THAT(co.data_ptr<double>()[i],
                 WithinAbs(ho.data_ptr<double>()[i], 1e-12));
  }
}

TEST_CASE("CUDA softmax produces a normalized distribution",
          "[ops][gpu][softmax]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 8;
  const int64_t C = 11;
  auto p = logit_pattern(N * C);
  Tensor d = Tensor::from_vector(p, {N, C}).to(cuda0());
  Tensor sm = tesseract::ops::softmax(d, /*dim=*/-1).to(cpu_device());
  const float* s = sm.data_ptr<float>();
  for (int64_t n = 0; n < N; ++n) {
    float acc = 0.0f;
    for (int64_t c = 0; c < C; ++c) {
      const float v = s[n * C + c];
      REQUIRE(v >= 0.0f);
      REQUIRE(v <= 1.0f);
      acc += v;
    }
    REQUIRE_THAT(acc, WithinAbs(1.0f, 1e-5f));
  }
}

// B-015: FP16 / BF16 parity on the FP32-promoted softmax kernel.
// Tolerance is the normal half-precision envelope (2e-3 abs) — the
// CUDA kernel accumulates in FP32 and only narrows on the final
// store, so the intermediate max/sum/exp/log are bit-identical to
// the CPU `Acc = float` path.
TEST_CASE("CUDA softmax matches CPU on Float16", "[ops][gpu][softmax]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 16, C = 32;
  auto p = logit_pattern(N * C);
  std::vector<tesseract::Half> ph(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) ph[i] = tesseract::Half(p[i]);
  Tensor h = Tensor::from_vector(ph, {N, C});
  Tensor d = h.to(cuda0());
  Tensor ho = tesseract::ops::softmax(h, /*dim=*/-1);
  Tensor co = tesseract::ops::softmax(d, /*dim=*/-1).to(cpu_device());
  REQUIRE(co.shape() == ho.shape());
  const tesseract::Half* e = ho.data_ptr<tesseract::Half>();
  const tesseract::Half* a = co.data_ptr<tesseract::Half>();
  for (int64_t i = 0; i < co.numel(); ++i) {
    REQUIRE_THAT(static_cast<float>(a[i]),
                 WithinAbs(static_cast<float>(e[i]), 2e-3f));
  }
}

TEST_CASE("CUDA log_softmax matches CPU on BFloat16",
          "[ops][gpu][softmax]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 8, C = 24;
  auto p = logit_pattern(N * C);
  std::vector<tesseract::BFloat16> pb(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) pb[i] = tesseract::BFloat16(p[i]);
  Tensor h = Tensor::from_vector(pb, {N, C});
  Tensor d = h.to(cuda0());
  Tensor ho = tesseract::ops::log_softmax(h, /*dim=*/-1);
  Tensor co = tesseract::ops::log_softmax(d, /*dim=*/-1).to(cpu_device());
  REQUIRE(co.shape() == ho.shape());
  const tesseract::BFloat16* e = ho.data_ptr<tesseract::BFloat16>();
  const tesseract::BFloat16* a = co.data_ptr<tesseract::BFloat16>();
  for (int64_t i = 0; i < co.numel(); ++i) {
    REQUIRE_THAT(static_cast<float>(a[i]),
                 WithinAbs(static_cast<float>(e[i]), 5e-3f));
  }
}

TEST_CASE("CPU-only softmax smoke", "[ops][cpu][softmax]") {
  // Always-running sanity check — keeps the CPU-only build with at
  // least one asserted path in this TU.
  Tensor h = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f},
                                 {1, 4});
  Tensor s = tesseract::ops::softmax(h, /*dim=*/-1);
  // Uniform logits → uniform softmax.
  for (int64_t i = 0; i < s.numel(); ++i) {
    REQUIRE_THAT(s.data_ptr<float>()[i], WithinAbs(0.25f, 1e-6f));
  }
}

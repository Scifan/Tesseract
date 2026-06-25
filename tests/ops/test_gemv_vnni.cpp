// M4 Phase 6 — correctness of the AVX-512-VNNI W8A8 decode GEMV.
//
// The kernel quantizes the activation to INT8 (per-vector symmetric) and
// reduces against per-row-quantized INT8 weights via vpdpbusd, with a
// +128/u8-offset correction. We check it against an exact integer reference
// (same quantization, scalar accumulation) — they must match to the bit on
// the integer dot product, and to a tight float tolerance after scaling —
// and against the FP32 dequantized matmul within activation-quant error.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "../../src/ops/cpu/GemvVnni.hpp"

using tesseract::ops::detail::compute_row_sums;
using tesseract::ops::detail::gemv_w8a8;
using tesseract::ops::detail::quantize_row_int8;

namespace {

// Exact integer reference for y[n] = xscale*wscale[n]*sum_k xq[k]*W[n,k].
void ref_w8a8(const std::vector<std::int8_t>& W, const std::vector<float>& wscale,
              const std::vector<std::int8_t>& xq, float xscale,
              std::vector<float>& y, std::int64_t N, std::int64_t K) {
  y.assign(static_cast<size_t>(N), 0.0f);
  for (std::int64_t n = 0; n < N; ++n) {
    std::int64_t acc = 0;
    for (std::int64_t k = 0; k < K; ++k)
      acc += static_cast<std::int64_t>(xq[static_cast<size_t>(k)]) *
             static_cast<std::int64_t>(W[static_cast<size_t>(n * K + k)]);
    y[static_cast<size_t>(n)] = xscale * wscale[static_cast<size_t>(n)] *
                                static_cast<float>(acc);
  }
}

}  // namespace

TEST_CASE("VNNI W8A8 GEMV matches integer reference across shapes",
          "[ops][cpu][gemv][vnni]") {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> wd(-127, 127);
  std::uniform_real_distribution<float> sd(0.001f, 0.05f);
  std::uniform_real_distribution<float> xd(-3.0f, 3.0f);

  // K values straddle the 256/64 vectorized spans and the scalar tail so the
  // u8-offset correction is exercised in every regime.
  struct Cfg { std::int64_t N, K; };
  for (const Cfg c : {Cfg{17, 64}, Cfg{32, 128}, Cfg{8, 200}, Cfg{64, 256},
                      Cfg{40, 511}, Cfg{96, 513}, Cfg{1, 2048}, Cfg{129, 4096}}) {
    const std::int64_t N = c.N, K = c.K;
    std::vector<std::int8_t> W(static_cast<size_t>(N * K));
    std::vector<float> wscale(static_cast<size_t>(N));
    std::vector<std::int32_t> rowsum(static_cast<size_t>(N));
    for (auto& v : W) v = static_cast<std::int8_t>(wd(rng));
    for (auto& s : wscale) s = sd(rng);
    compute_row_sums(W.data(), N, K, rowsum.data());

    std::vector<float> x(static_cast<size_t>(K));
    for (auto& e : x) e = xd(rng);
    std::vector<std::int8_t> xq(static_cast<size_t>(K));
    const float xscale = quantize_row_int8(x.data(), K, xq.data());

    std::vector<float> y_ref;
    ref_w8a8(W, wscale, xq, xscale, y_ref, N, K);

    std::vector<float> y(static_cast<size_t>(N), -1234.0f);
    gemv_w8a8(W.data(), wscale.data(), rowsum.data(), xq.data(), xscale,
              y.data(), N, K);

    for (std::int64_t n = 0; n < N; ++n) {
      // Integer dot is exact; the only slack is FP rounding of the final
      // scale multiply, so a relative tol of 1e-4 is generous.
      const float a = y[static_cast<size_t>(n)];
      const float b = y_ref[static_cast<size_t>(n)];
      const float tol = 1e-4f * (std::fabs(b) + 1e-3f);
      REQUIRE(std::fabs(a - b) <= tol);
    }
  }
}

TEST_CASE("VNNI W8A8 GEMV approximates the FP32 matmul (W8A8 quant error)",
          "[ops][cpu][gemv][vnni]") {
  std::mt19937 rng(11);
  std::normal_distribution<float> g(0.0f, 1.0f);
  const std::int64_t N = 256, K = 1024;

  // Float weights, then per-row symmetric INT8 quantization.
  std::vector<float> Wf(static_cast<size_t>(N * K));
  for (auto& e : Wf) e = g(rng) * 0.05f;
  std::vector<std::int8_t> W(static_cast<size_t>(N * K));
  std::vector<float> wscale(static_cast<size_t>(N));
  for (std::int64_t n = 0; n < N; ++n) {
    float amax = 0.0f;
    for (std::int64_t k = 0; k < K; ++k)
      amax = std::max(amax, std::fabs(Wf[static_cast<size_t>(n * K + k)]));
    const float s = amax / 127.0f;
    wscale[static_cast<size_t>(n)] = s;
    for (std::int64_t k = 0; k < K; ++k)
      W[static_cast<size_t>(n * K + k)] = static_cast<std::int8_t>(
          std::nearbyint(Wf[static_cast<size_t>(n * K + k)] / s));
  }
  std::vector<std::int32_t> rowsum(static_cast<size_t>(N));
  compute_row_sums(W.data(), N, K, rowsum.data());

  std::vector<float> x(static_cast<size_t>(K));
  for (auto& e : x) e = g(rng);
  std::vector<std::int8_t> xq(static_cast<size_t>(K));
  const float xscale = quantize_row_int8(x.data(), K, xq.data());

  std::vector<float> y(static_cast<size_t>(N));
  gemv_w8a8(W.data(), wscale.data(), rowsum.data(), xq.data(), xscale, y.data(),
            N, K);

  // Reference: full FP32 y[n] = sum_k x[k]*Wf[n,k]. W8A8 error is O(1/127)
  // relative on a K=1024 reduction; assert a comfortably loose 3% RMS.
  double se = 0.0, sref = 0.0;
  for (std::int64_t n = 0; n < N; ++n) {
    double acc = 0.0;
    for (std::int64_t k = 0; k < K; ++k)
      acc += static_cast<double>(x[static_cast<size_t>(k)]) *
             static_cast<double>(Wf[static_cast<size_t>(n * K + k)]);
    const double diff = static_cast<double>(y[static_cast<size_t>(n)]) - acc;
    se += diff * diff;
    sref += acc * acc;
  }
  const double rrms = std::sqrt(se / sref);
  REQUIRE(rrms < 0.03);
}

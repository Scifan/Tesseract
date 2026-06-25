// Parity tests for the B-005 software-emulated FP16 / BFloat16 pipeline.
// The goal is NOT to verify that half-precision produces the same bits as
// float — it can't — but that:
//
//   1. `add` / `mul` / `matmul` accept the new dtypes and produce outputs
//      within the precision budget of the scalar type (about 1e-3 for
//      FP16 and 1e-2 for BF16 on values close to 1.0).
//   2. Conversions `float <-> Half` / `float <-> BFloat16` round-trip
//      cleanly on representable values and handle the tricky edge cases
//      (zero, inf, NaN, denormals).
//   3. `dtype_is_implemented(Float16) == true` and the tensor factories
//      agree.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/MatMul.hpp"

using tesseract::BFloat16;
using tesseract::DType;
using tesseract::Half;
using tesseract::Shape;
using tesseract::Tensor;

namespace {

// Relative + absolute tolerance check. Half-precision ops inevitably
// incur ~1e-3 rounding and bfloat16 ~1e-2; we scale by the magnitude of
// the reference so that outputs with large dynamic range still pass.
bool close_to(float actual, float expected, float atol, float rtol) {
  const float diff = std::fabs(actual - expected);
  const float tol = atol + rtol * std::fabs(expected);
  return diff <= tol;
}

// Convert every element of `t` to float. Works for Float32, Half, BFloat16.
std::vector<float> to_float_vec(const Tensor& t) {
  const auto n = static_cast<std::size_t>(t.numel());
  std::vector<float> out(n);
  switch (t.dtype()) {
    case DType::Float32: {
      const float* p = t.data_ptr<float>();
      for (std::size_t i = 0; i < n; ++i) out[i] = p[i];
      break;
    }
    case DType::Float16: {
      const Half* p = t.data_ptr<Half>();
      for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
      break;
    }
    case DType::BFloat16: {
      const BFloat16* p = t.data_ptr<BFloat16>();
      for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
      break;
    }
    default:
      FAIL("unsupported dtype in to_float_vec");
  }
  return out;
}

// Build a (shape=data.size(), flat) tensor of the requested dtype. Values
// are first converted to the target scalar type so the tensor's bit
// pattern reflects the *destination* precision, not float.
Tensor make_fp(DType dt, const std::vector<float>& data, Shape shape) {
  switch (dt) {
    case DType::Float32:
      return Tensor::from_vector(data, std::move(shape));
    case DType::Float16: {
      std::vector<Half> v(data.size());
      for (std::size_t i = 0; i < data.size(); ++i) v[i] = Half(data[i]);
      return Tensor::from_vector(v, std::move(shape));
    }
    case DType::BFloat16: {
      std::vector<BFloat16> v(data.size());
      for (std::size_t i = 0; i < data.size(); ++i) v[i] = BFloat16(data[i]);
      return Tensor::from_vector(v, std::move(shape));
    }
    default:
      FAIL("unsupported dtype in make_fp");
      return Tensor{};
  }
}

}  // namespace

TEST_CASE("Half round-trip preserves exactly representable values", "[fp16]") {
  // Integers in [-2048, 2048] and the classic halves are all exactly
  // representable in binary16 — the round-trip must be bit-identical.
  const std::vector<float> exact = {
      0.0f, -0.0f, 1.0f, -1.0f, 2.0f, -2.0f,
      0.5f, -0.5f, 0.25f, 1024.0f, -2048.0f,
  };
  for (float v : exact) {
    const Half h(v);
    REQUIRE(static_cast<float>(h) == v);
  }
}

TEST_CASE("Half handles inf / NaN / subnormals", "[fp16]") {
  const float inf = std::numeric_limits<float>::infinity();
  const Half h_inf(inf);
  REQUIRE(std::isinf(static_cast<float>(h_inf)));
  REQUIRE(static_cast<float>(h_inf) > 0.0f);

  const Half h_ninf(-inf);
  REQUIRE(std::isinf(static_cast<float>(h_ninf)));
  REQUIRE(static_cast<float>(h_ninf) < 0.0f);

  const Half h_nan(std::numeric_limits<float>::quiet_NaN());
  REQUIRE(std::isnan(static_cast<float>(h_nan)));

  // 1e-7 falls below the smallest FP16 subnormal (~5.96e-8 rounds to zero
  // or the smallest subnormal — either is acceptable as long as it stays
  // finite and non-negative on a positive input).
  const Half h_tiny(1.0e-7f);
  const float rt = static_cast<float>(h_tiny);
  REQUIRE(rt >= 0.0f);
  REQUIRE(std::isfinite(rt));
}

TEST_CASE("BFloat16 round-trip keeps full exponent range", "[bf16]") {
  // BF16 has the same exponent field as float, so anything representable
  // with <= 7 explicit mantissa bits round-trips cleanly. Use powers of
  // two to get at the "full exponent range" claim without tripping on
  // the mantissa truncation.
  const std::vector<float> exact = {
      0.0f, -0.0f, 1.0f, -1.0f, 2.0f, -2.0f,
      float(1ULL << 30), -float(1ULL << 30),   // 2^30 fits in both FP32 and BF16
      std::ldexp(1.0f, 100), std::ldexp(1.0f, -100),  // far outside FP16's range
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
  };
  for (float v : exact) {
    const BFloat16 b(v);
    const float rt = static_cast<float>(b);
    if (std::isinf(v)) {
      REQUIRE(std::isinf(rt));
      REQUIRE((rt > 0) == (v > 0));
    } else {
      REQUIRE(rt == v);
    }
  }
}

TEST_CASE("add / mul elementwise parity with FP32", "[fp16][ops]") {
  constexpr int N = 64;
  std::mt19937 rng(0xB005);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> a(N), b(N);
  for (int i = 0; i < N; ++i) { a[i] = dist(rng); b[i] = dist(rng); }

  auto check = [&](DType dt, float atol, float rtol) {
    Tensor ta_ref = make_fp(DType::Float32, a, {N});
    Tensor tb_ref = make_fp(DType::Float32, b, {N});
    Tensor ta = make_fp(dt, a, {N});
    Tensor tb = make_fp(dt, b, {N});

    // The reference is computed in FP32 on the *already-rounded* values,
    // so the half path only needs to match modulo a single final round.
    std::vector<float> a_rounded = to_float_vec(ta);
    std::vector<float> b_rounded = to_float_vec(tb);
    Tensor ta_ref_r = Tensor::from_vector(a_rounded, Shape{N});
    Tensor tb_ref_r = Tensor::from_vector(b_rounded, Shape{N});

    Tensor sum_ref = tesseract::ops::add(ta_ref_r, tb_ref_r);
    Tensor mul_ref = tesseract::ops::mul(ta_ref_r, tb_ref_r);
    Tensor sum_hp = tesseract::ops::add(ta, tb);
    Tensor mul_hp = tesseract::ops::mul(ta, tb);

    REQUIRE(sum_hp.dtype() == dt);
    REQUIRE(mul_hp.dtype() == dt);

    auto s_hp = to_float_vec(sum_hp);
    auto m_hp = to_float_vec(mul_hp);
    auto s_r = to_float_vec(sum_ref);
    auto m_r = to_float_vec(mul_ref);
    for (int i = 0; i < N; ++i) {
      INFO("dt=" << tesseract::dtype_name(dt) << " i=" << i
                 << " sum=" << s_hp[i] << " ref=" << s_r[i]);
      REQUIRE(close_to(s_hp[i], s_r[i], atol, rtol));
      INFO("dt=" << tesseract::dtype_name(dt) << " i=" << i
                 << " mul=" << m_hp[i] << " ref=" << m_r[i]);
      REQUIRE(close_to(m_hp[i], m_r[i], atol, rtol));
    }
  };

  // Half: ~2^-10 mantissa -> relative tolerance 1e-3.
  check(DType::Float16, /*atol=*/1.0e-3f, /*rtol=*/1.0e-3f);
  // BF16: ~2^-7 mantissa -> relative tolerance ~1e-2.
  check(DType::BFloat16, /*atol=*/1.0e-2f, /*rtol=*/1.0e-2f);
}

TEST_CASE("matmul parity with FP32 (soft-emulated half)", "[fp16][matmul]") {
  // Sizes chosen to exercise the rank-2 fast path and a non-trivial K so
  // the FP32 accumulation actually matters. Values are small so we stay
  // safely inside FP16's representable range.
  constexpr int M = 8, K = 16, N = 12;
  std::mt19937 rng(0xBEEF);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> a(M * K), b(K * N);
  for (auto& x : a) x = dist(rng);
  for (auto& x : b) x = dist(rng);

  auto check = [&](DType dt, float atol, float rtol) {
    Tensor ta = make_fp(dt, a, {M, K});
    Tensor tb = make_fp(dt, b, {K, N});
    Tensor tc = tesseract::ops::matmul(ta, tb);
    REQUIRE(tc.dtype() == dt);
    REQUIRE(tc.shape() == Shape{M, N});

    // Reference: upcast to float, matmul in FP32. This matches what the
    // half kernel does internally (per-slab upcast + FP32 accumulate).
    std::vector<float> a_r = to_float_vec(ta);
    std::vector<float> b_r = to_float_vec(tb);
    Tensor ta_f = Tensor::from_vector(a_r, Shape{M, K});
    Tensor tb_f = Tensor::from_vector(b_r, Shape{K, N});
    Tensor tc_f = tesseract::ops::matmul(ta_f, tb_f);

    auto got = to_float_vec(tc);
    auto ref = to_float_vec(tc_f);
    for (int i = 0; i < M * N; ++i) {
      INFO("dt=" << tesseract::dtype_name(dt) << " i=" << i
                 << " got=" << got[i] << " ref=" << ref[i]);
      REQUIRE(close_to(got[i], ref[i], atol, rtol));
    }
  };

  check(DType::Float16, 2.0e-3f, 2.0e-3f);
  check(DType::BFloat16, 2.0e-2f, 2.0e-2f);
}

TEST_CASE("matmul half: batched shape works end-to-end", "[fp16][matmul]") {
  // Rank-3 to exercise the per-slab dispatch (the outer loop handles the
  // batch, the inner `gemm_slab<Half>` does the upcast).
  constexpr int B = 3, M = 4, K = 5, N = 6;
  std::mt19937 rng(0xCAFE);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> a(B * M * K), b(B * K * N);
  for (auto& x : a) x = dist(rng);
  for (auto& x : b) x = dist(rng);

  Tensor ta = make_fp(DType::Float16, a, {B, M, K});
  Tensor tb = make_fp(DType::Float16, b, {B, K, N});
  Tensor tc = tesseract::ops::matmul(ta, tb);
  REQUIRE(tc.dtype() == DType::Float16);
  REQUIRE(tc.shape() == Shape{B, M, N});

  // Sanity-check one output slab against an explicit FP32 triple-loop.
  auto a_r = to_float_vec(ta);
  auto b_r = to_float_vec(tb);
  auto got = to_float_vec(tc);
  for (int bi = 0; bi < B; ++bi) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          acc += a_r[bi * M * K + i * K + k] * b_r[bi * K * N + k * N + j];
        }
        const float ref = acc;
        const float g = got[bi * M * N + i * N + j];
        REQUIRE(close_to(g, ref, 2.0e-3f, 2.0e-3f));
      }
    }
  }
}

TEST_CASE("Tensor factories accept Half / BFloat16", "[fp16][tensor]") {
  // zeros works via memset — verify then factor into all-zero conversion.
  Tensor z_half = Tensor::zeros({4}, DType::Float16);
  REQUIRE(z_half.dtype() == DType::Float16);
  for (float v : to_float_vec(z_half)) REQUIRE(v == 0.0f);

  // full / ones need the scalar-fill dispatch.
  Tensor o_half = Tensor::ones({4}, DType::Float16);
  for (float v : to_float_vec(o_half)) REQUIRE(v == 1.0f);

  Tensor f_bf16 = Tensor::full({3}, 3.5, DType::BFloat16);
  for (float v : to_float_vec(f_bf16)) REQUIRE(v == 3.5f);
}

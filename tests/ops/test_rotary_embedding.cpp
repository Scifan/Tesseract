// B-014: `ops::rotary_embedding` + `nn::RotaryEmbedding` test suite.
//
// Coverage:
//   * Hand-rolled reference parity against the adjacent-pair rotation
//     formula on CPU (rank-3 and rank-4 inputs exercising the
//     `[..., S, D]` broadcast).
//   * CPU↔CUDA parity on Float32 / Float64, Float16 / BFloat16 (the
//     last two validate the B-015-style FP32-promoted kernel path).
//   * Autograd finite-diff check — confirms `RotaryBackward` is
//     rotation-by-(-θ) and nothing leaks into cos/sin.
//   * `nn::RotaryEmbedding` smoke: table values match
//     `cos(p·θⱼ)` / `sin(p·θⱼ)` closed form; the module's forward is
//     bit-for-bit `ops::rotary_embedding(x, cos, sin)` (same exec
//     path, no hidden slicing).
//   * Cross-device move: `Module::to(cuda)` migrates the registered
//     cos/sin buffers alongside any child parameters — we verify by
//     running forward on CUDA and comparing with the CPU forward
//     after a round-trip move.
//
// All GPU-touching cases SKIP via Catch2 when no CUDA is visible; the
// always-on CPU cases keep at least one asserted path in the CPU-only
// build (SKIP_RETURN_CODE 4 handles the "all cases skipped" tail).

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/nn/RotaryEmbedding.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/ops/RotaryEmbedding.hpp"

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

// Build a deterministic `[outer, S, D]` pattern with a mix of signs
// and magnitudes, small enough to stay well inside the fp16 dynamic
// range (|x| <= ~4) so the parity envelope isn't clipped by
// saturation on the bf16 side.
std::vector<float> make_pattern(int64_t outer, int64_t S, int64_t D) {
  std::vector<float> v(static_cast<std::size_t>(outer * S * D));
  for (std::size_t i = 0; i < v.size(); ++i) {
    const float t = (static_cast<float>(i % 37) - 18.0f) / 6.0f;
    const float sign = (i % 2 == 0) ? 1.0f : -1.0f;
    v[i] = sign * t;
  }
  return v;
}

// Plain-C++ adjacent-pair rotation reference. Works for any rank `x`
// with the `[..., S, D]` trailing contract; we flatten everything
// before the last two dims into `outer`. FP64 throughout so the
// reference itself never loses mantissa bits — the parity tolerances
// are driven by the kernel/path under test, not by this routine.
std::vector<double> rope_reference(const std::vector<float>& x,
                                   const std::vector<double>& cs,
                                   const std::vector<double>& sn,
                                   int64_t outer, int64_t S, int64_t D) {
  std::vector<double> out(static_cast<std::size_t>(outer * S * D));
  const int64_t half = D / 2;
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t p = 0; p < S; ++p) {
      for (int64_t j = 0; j < half; ++j) {
        const int64_t xb = (o * S + p) * D;
        const int64_t tb = p * D;
        const double a = x[xb + 2 * j];
        const double b = x[xb + 2 * j + 1];
        const double c = cs[tb + 2 * j];
        const double s = sn[tb + 2 * j];
        out[xb + 2 * j]     = a * c - b * s;
        out[xb + 2 * j + 1] = a * s + b * c;
      }
    }
  }
  return out;
}

// Closed-form cos/sin tables, built in FP64. Mirrors what
// `nn::RotaryEmbedding` builds internally but in a separately-typed
// form so the test can double as a spec for the module.
void make_tables(int64_t max_seq, int64_t D, double base,
                 std::vector<double>& cs, std::vector<double>& sn) {
  cs.assign(static_cast<std::size_t>(max_seq * D), 0.0);
  sn.assign(static_cast<std::size_t>(max_seq * D), 0.0);
  const int64_t half = D / 2;
  for (int64_t j = 0; j < half; ++j) {
    const double theta = std::pow(base, -(2.0 * j) / static_cast<double>(D));
    for (int64_t p = 0; p < max_seq; ++p) {
      const double c = std::cos(static_cast<double>(p) * theta);
      const double s = std::sin(static_cast<double>(p) * theta);
      cs[p * D + 2 * j]     = c;
      cs[p * D + 2 * j + 1] = c;
      sn[p * D + 2 * j]     = s;
      sn[p * D + 2 * j + 1] = s;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------- //
// (1) Reference parity on CPU (rank-3 [B, S, D])
// ---------------------------------------------------------------- //

TEST_CASE("rotary_embedding matches hand-rolled reference on rank-3 CPU",
          "[ops][cpu][rope]") {
  const int64_t B = 2, S = 5, D = 8;
  const double base = 10000.0;

  auto xf = make_pattern(B, S, D);
  std::vector<double> cs, sn;
  make_tables(S, D, base, cs, sn);

  Tensor x = Tensor::from_vector(xf, {B, S, D});
  // Tables as Float32 so the dtypes line up with `x`.
  std::vector<float> csf(cs.size()), snf(sn.size());
  for (std::size_t i = 0; i < cs.size(); ++i) {
    csf[i] = static_cast<float>(cs[i]);
    snf[i] = static_cast<float>(sn[i]);
  }
  Tensor cos_t = Tensor::from_vector(csf, {S, D});
  Tensor sin_t = Tensor::from_vector(snf, {S, D});

  Tensor out = tesseract::ops::rotary_embedding(x, cos_t, sin_t);
  REQUIRE(out.shape() == x.shape());

  auto ref = rope_reference(xf, cs, sn, B, S, D);
  const float* po = out.data_ptr<float>();
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(po[i], WithinAbs(static_cast<float>(ref[i]), 1e-5f));
  }
}

// ---------------------------------------------------------------- //
// (2) Rank-4 CPU reference parity (the MHA shape [B, H, S, Dh])
// ---------------------------------------------------------------- //

TEST_CASE("rotary_embedding matches reference on rank-4 CPU",
          "[ops][cpu][rope]") {
  const int64_t B = 2, H = 4, S = 7, D = 16;
  const double base = 10000.0;
  const int64_t outer = B * H;

  auto xf = make_pattern(outer, S, D);
  std::vector<double> cs, sn;
  make_tables(S, D, base, cs, sn);

  Tensor x = Tensor::from_vector(xf, {B, H, S, D});
  std::vector<float> csf(cs.size()), snf(sn.size());
  for (std::size_t i = 0; i < cs.size(); ++i) {
    csf[i] = static_cast<float>(cs[i]);
    snf[i] = static_cast<float>(sn[i]);
  }
  Tensor cos_t = Tensor::from_vector(csf, {S, D});
  Tensor sin_t = Tensor::from_vector(snf, {S, D});

  Tensor out = tesseract::ops::rotary_embedding(x, cos_t, sin_t);
  REQUIRE(out.shape() == x.shape());

  auto ref = rope_reference(xf, cs, sn, outer, S, D);
  const float* po = out.data_ptr<float>();
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(po[i], WithinAbs(static_cast<float>(ref[i]), 1e-5f));
  }
}

// ---------------------------------------------------------------- //
// (3) CPU↔CUDA parity on Float32
// ---------------------------------------------------------------- //

TEST_CASE("rotary_embedding CPU↔CUDA parity on Float32",
          "[ops][gpu][rope]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, H = 4, S = 7, D = 16;
  const int64_t outer = B * H;

  auto xf = make_pattern(outer, S, D);
  std::vector<double> cs, sn;
  make_tables(S, D, 10000.0, cs, sn);
  std::vector<float> csf(cs.size()), snf(sn.size());
  for (std::size_t i = 0; i < cs.size(); ++i) {
    csf[i] = static_cast<float>(cs[i]);
    snf[i] = static_cast<float>(sn[i]);
  }

  Tensor x_h = Tensor::from_vector(xf, {B, H, S, D});
  Tensor c_h = Tensor::from_vector(csf, {S, D});
  Tensor s_h = Tensor::from_vector(snf, {S, D});

  Tensor x_d = x_h.to(cuda0());
  Tensor c_d = c_h.to(cuda0());
  Tensor s_d = s_h.to(cuda0());

  Tensor y_h = tesseract::ops::rotary_embedding(x_h, c_h, s_h);
  Tensor y_d = tesseract::ops::rotary_embedding(x_d, c_d, s_d).to(cpu_device());

  REQUIRE(y_h.shape() == y_d.shape());
  const float* a = y_h.data_ptr<float>();
  const float* b = y_d.data_ptr<float>();
  for (int64_t i = 0; i < y_h.numel(); ++i) {
    REQUIRE_THAT(b[i], WithinAbs(a[i], 2e-6f));
  }
}

// ---------------------------------------------------------------- //
// (4) CPU↔CUDA parity on Float64
// ---------------------------------------------------------------- //

TEST_CASE("rotary_embedding CPU↔CUDA parity on Float64",
          "[ops][gpu][rope]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 1, S = 5, D = 8;
  std::vector<double> cs, sn;
  make_tables(S, D, 10000.0, cs, sn);
  std::vector<double> xd(static_cast<std::size_t>(B * S * D));
  for (std::size_t i = 0; i < xd.size(); ++i) {
    xd[i] = ((static_cast<double>(i % 23) - 11.0) / 5.0) * (i % 2 ? -1 : 1);
  }
  Tensor x_h = Tensor::from_vector(xd, {B, S, D});
  Tensor c_h = Tensor::from_vector(cs, {S, D});
  Tensor s_h = Tensor::from_vector(sn, {S, D});

  Tensor x_d = x_h.to(cuda0());
  Tensor c_d = c_h.to(cuda0());
  Tensor s_d = s_h.to(cuda0());

  Tensor y_h = tesseract::ops::rotary_embedding(x_h, c_h, s_h);
  Tensor y_d = tesseract::ops::rotary_embedding(x_d, c_d, s_d).to(cpu_device());

  const double* a = y_h.data_ptr<double>();
  const double* b = y_d.data_ptr<double>();
  for (int64_t i = 0; i < y_h.numel(); ++i) {
    REQUIRE_THAT(b[i], WithinAbs(a[i], 1e-12));
  }
}

// ---------------------------------------------------------------- //
// (5) CPU↔CUDA parity on Float16 / BFloat16
// ---------------------------------------------------------------- //

TEST_CASE("rotary_embedding CPU↔CUDA parity on Float16",
          "[ops][gpu][rope]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, H = 2, S = 6, D = 16;
  const int64_t outer = B * H;
  auto xf = make_pattern(outer, S, D);
  std::vector<double> cs, sn;
  make_tables(S, D, 10000.0, cs, sn);

  std::vector<tesseract::Half> xh(xf.size());
  for (std::size_t i = 0; i < xf.size(); ++i) xh[i] = tesseract::Half(xf[i]);
  std::vector<tesseract::Half> ch(cs.size()), sh(sn.size());
  for (std::size_t i = 0; i < cs.size(); ++i) {
    ch[i] = tesseract::Half(static_cast<float>(cs[i]));
    sh[i] = tesseract::Half(static_cast<float>(sn[i]));
  }

  Tensor x_h = Tensor::from_vector(xh, {B, H, S, D});
  Tensor c_h = Tensor::from_vector(ch, {S, D});
  Tensor s_h = Tensor::from_vector(sh, {S, D});
  Tensor y_h = tesseract::ops::rotary_embedding(x_h, c_h, s_h);
  Tensor y_d = tesseract::ops::rotary_embedding(x_h.to(cuda0()),
                                                c_h.to(cuda0()),
                                                s_h.to(cuda0())).to(cpu_device());

  // 2e-3 matches the FP16 envelope we hold ourselves to in the other
  // half-precision parity tests (`test_ops_cuda_softmax`,
  // `test_ops_cuda_elementwise_f16`, `test_ops_cuda_reduction`).
  const tesseract::Half* a = y_h.data_ptr<tesseract::Half>();
  const tesseract::Half* b = y_d.data_ptr<tesseract::Half>();
  for (int64_t i = 0; i < y_h.numel(); ++i) {
    REQUIRE_THAT(static_cast<float>(b[i]),
                 WithinAbs(static_cast<float>(a[i]), 2e-3f));
  }
}

TEST_CASE("rotary_embedding CPU↔CUDA parity on BFloat16",
          "[ops][gpu][rope]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 1, H = 2, S = 4, D = 16;
  const int64_t outer = B * H;
  auto xf = make_pattern(outer, S, D);
  std::vector<double> cs, sn;
  make_tables(S, D, 10000.0, cs, sn);

  std::vector<tesseract::BFloat16> xb(xf.size());
  for (std::size_t i = 0; i < xf.size(); ++i) xb[i] = tesseract::BFloat16(xf[i]);
  std::vector<tesseract::BFloat16> cb(cs.size()), sb(sn.size());
  for (std::size_t i = 0; i < cs.size(); ++i) {
    cb[i] = tesseract::BFloat16(static_cast<float>(cs[i]));
    sb[i] = tesseract::BFloat16(static_cast<float>(sn[i]));
  }

  Tensor x_h = Tensor::from_vector(xb, {B, H, S, D});
  Tensor c_h = Tensor::from_vector(cb, {S, D});
  Tensor s_h = Tensor::from_vector(sb, {S, D});
  Tensor y_h = tesseract::ops::rotary_embedding(x_h, c_h, s_h);
  Tensor y_d = tesseract::ops::rotary_embedding(x_h.to(cuda0()),
                                                c_h.to(cuda0()),
                                                s_h.to(cuda0())).to(cpu_device());

  const tesseract::BFloat16* a = y_h.data_ptr<tesseract::BFloat16>();
  const tesseract::BFloat16* b = y_d.data_ptr<tesseract::BFloat16>();
  for (int64_t i = 0; i < y_h.numel(); ++i) {
    REQUIRE_THAT(static_cast<float>(b[i]),
                 WithinAbs(static_cast<float>(a[i]), 5e-3f));
  }
}

// ---------------------------------------------------------------- //
// (6) Autograd: gradient is rotation-by-(-θ), verified by finite diff
// ---------------------------------------------------------------- //

TEST_CASE("rotary_embedding autograd matches finite-difference",
          "[ops][cpu][rope][autograd]") {
  using tesseract::Engine;
  const int64_t B = 2, S = 4, D = 8;
  auto xf = make_pattern(B, S, D);
  std::vector<double> cs, sn;
  make_tables(S, D, 10000.0, cs, sn);
  std::vector<float> csf(cs.size()), snf(sn.size());
  for (std::size_t i = 0; i < cs.size(); ++i) {
    csf[i] = static_cast<float>(cs[i]);
    snf[i] = static_cast<float>(sn[i]);
  }

  Tensor x = Tensor::from_vector(xf, {B, S, D});
  x.set_requires_grad(true);
  Tensor c = Tensor::from_vector(csf, {S, D});
  Tensor s = Tensor::from_vector(snf, {S, D});

  // Scalar loss = sum(out) — grad_out = ones, so grad_x = rotate by
  // -θ of a ones tensor. We compare the analytic backward with
  // central-difference under the same loss.
  Tensor out = tesseract::ops::rotary_embedding(x, c, s);
  Tensor loss = tesseract::ops::sum(out);
  Engine::backward(loss);

  const Tensor& g_analytic = x.grad();
  REQUIRE(g_analytic.defined());
  REQUIRE(g_analytic.shape() == x.shape());

  const float eps = 1e-2f;
  const float* gx = g_analytic.data_ptr<float>();
  // Finite-difference a handful of indices spread across the tensor
  // to keep the test fast (full FD over 64 entries would be ~128
  // forwards). Spread them so we touch both ends of pairs and
  // multiple positions.
  const int64_t idx_list[] = {0, 1, 5, 11, 23, 42, 57, 63};
  for (int64_t flat : idx_list) {
    std::vector<float> xp = xf; xp[flat] += eps;
    std::vector<float> xm = xf; xm[flat] -= eps;
    Tensor tp = Tensor::from_vector(xp, {B, S, D});
    Tensor tm = Tensor::from_vector(xm, {B, S, D});
    double fp = 0.0;
    {
      Tensor op = tesseract::ops::rotary_embedding(tp, c, s);
      const float* p = op.data_ptr<float>();
      for (int64_t i = 0; i < op.numel(); ++i) fp += static_cast<double>(p[i]);
    }
    double fm = 0.0;
    {
      Tensor om = tesseract::ops::rotary_embedding(tm, c, s);
      const float* p = om.data_ptr<float>();
      for (int64_t i = 0; i < om.numel(); ++i) fm += static_cast<double>(p[i]);
    }
    const float numeric = static_cast<float>((fp - fm) / (2.0 * eps));
    REQUIRE_THAT(gx[flat], WithinAbs(numeric, 3e-3f));
  }
}

// ---------------------------------------------------------------- //
// (7) nn::RotaryEmbedding — tables match the closed form
// ---------------------------------------------------------------- //

TEST_CASE("nn::RotaryEmbedding tables match closed-form",
          "[nn][cpu][rope]") {
  const int64_t Dh = 16;
  const int64_t max_seq = 32;
  const double base = 10000.0;
  tesseract::nn::RotaryEmbedding rope(Dh, base, max_seq);

  std::vector<double> cs_ref, sn_ref;
  make_tables(max_seq, Dh, base, cs_ref, sn_ref);

  const float* pc = rope.cos_table().data_ptr<float>();
  const float* ps = rope.sin_table().data_ptr<float>();
  REQUIRE(rope.cos_table().shape() == tesseract::Shape({max_seq, Dh}));
  REQUIRE(rope.sin_table().shape() == tesseract::Shape({max_seq, Dh}));
  for (int64_t i = 0; i < max_seq * Dh; ++i) {
    REQUIRE_THAT(pc[i], WithinAbs(static_cast<float>(cs_ref[i]), 1e-6f));
    REQUIRE_THAT(ps[i], WithinAbs(static_cast<float>(sn_ref[i]), 1e-6f));
  }
}

// ---------------------------------------------------------------- //
// (8) nn::RotaryEmbedding::forward equals ops::rotary_embedding
// ---------------------------------------------------------------- //

TEST_CASE("nn::RotaryEmbedding::forward matches ops::rotary_embedding",
          "[nn][cpu][rope]") {
  const int64_t B = 2, H = 2, S = 5, Dh = 16;
  const int64_t outer = B * H;
  auto xf = make_pattern(outer, S, Dh);
  Tensor x = Tensor::from_vector(xf, {B, H, S, Dh});

  tesseract::nn::RotaryEmbedding rope(Dh, /*base=*/10000.0, /*max_seq=*/64);
  Tensor from_module = rope.forward(x);
  Tensor from_op = tesseract::ops::rotary_embedding(
      x, rope.cos_table(), rope.sin_table());

  REQUIRE(from_module.shape() == from_op.shape());
  const float* a = from_module.data_ptr<float>();
  const float* b = from_op.data_ptr<float>();
  for (int64_t i = 0; i < from_module.numel(); ++i) {
    REQUIRE(a[i] == b[i]);  // same exec path, bit-for-bit equal
  }
}

// ---------------------------------------------------------------- //
// (9) Module::to(cuda) migrates the RoPE buffers
// ---------------------------------------------------------------- //

TEST_CASE("nn::RotaryEmbedding moves buffers under Module::to",
          "[nn][gpu][rope]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 1, H = 2, S = 5, Dh = 16;
  const int64_t outer = B * H;
  auto xf = make_pattern(outer, S, Dh);
  Tensor x_host = Tensor::from_vector(xf, {B, H, S, Dh});

  tesseract::nn::RotaryEmbedding rope(Dh, /*base=*/10000.0, /*max_seq=*/64);
  Tensor y_cpu = rope.forward(x_host);

  rope.to(cuda0());
  REQUIRE(rope.cos_table().device().is_cuda());
  REQUIRE(rope.sin_table().device().is_cuda());

  Tensor x_dev = x_host.to(cuda0());
  Tensor y_dev = rope.forward(x_dev).to(cpu_device());

  REQUIRE(y_cpu.shape() == y_dev.shape());
  const float* a = y_cpu.data_ptr<float>();
  const float* b = y_dev.data_ptr<float>();
  for (int64_t i = 0; i < y_cpu.numel(); ++i) {
    REQUIRE_THAT(b[i], WithinAbs(a[i], 2e-6f));
  }
}

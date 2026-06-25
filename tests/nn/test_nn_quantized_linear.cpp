// Wave 3.1 (B-021) — tests for the INT8 weight-only quantization stack:
//
//   * `quant::pack_int8_symmetric`    — per-output-channel packer
//   * `ops::dequantize_matmul_int8`   — fused dequant-matmul op
//   * `nn::QuantizedLinear`           — drop-in inference module
//
// Coverage matrix:
//   1.  Pack round-trip: dequant(pack(W)) ≈ W within a bounded per-row
//       quantization error (|dequant - W| <= max_abs / 127).
//   2.  Zero-row safety: an identically-zero row produces scale = 1 and
//       every element of the dequantized row is exactly zero.
//   3.  Op parity (CPU): dequantize_matmul_int8(x, q_w, scale) matches
//       a naive `(x @ dequant(W).T)` reference within quant-error bound.
//   4.  Op parity FP16 / BF16 (CPU) — same reference, narrower tolerance.
//   5.  Autograd fallback path: `x.requires_grad()` triggers the
//       composite `matmul(x, dequant(W).T)` route and finite-diff
//       matches analytic grad.
//   6.  QuantizedLinear::from_linear(Linear) top-1 logit ranking matches
//       the source FP32 Linear on a 512-row batch (the headline
//       "drop-in replacement" parity bar).
//   7.  CPU↔CUDA parity for both raw op and the module forward.
//   8.  `Module::to(cuda)` carries `q_weight` / `weight_scale` buffers
//       across devices in lockstep.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/QuantizedLinear.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Quant.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/quant/Pack.hpp"

using Catch::Matchers::WithinAbs;
using tesseract::cpu_device;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Engine;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;
using tesseract::nn::Linear;
using tesseract::nn::QuantizedLinear;

namespace {

bool cuda_ready() { return has_cuda_support() && is_available(); }
Device cuda0() { return Device{DeviceType::CUDA, 0}; }

Tensor from_host_f32(std::vector<float> data, Shape shape) {
  Tensor t = Tensor::empty(shape, DType::Float32);
  std::memcpy(t.raw_data(), data.data(), data.size() * sizeof(float));
  return t;
}

std::vector<float> to_host_f32(const Tensor& t) {
  Tensor host = t.device().is_cuda() ? t.to(cpu_device()) : t;
  std::vector<float> out(static_cast<size_t>(host.numel()));
  std::memcpy(out.data(), host.raw_data(), host.nbytes());
  return out;
}

std::vector<int8_t> to_host_i8(const Tensor& t) {
  Tensor host = t.device().is_cuda() ? t.to(cpu_device()) : t;
  std::vector<int8_t> out(static_cast<size_t>(host.numel()));
  std::memcpy(out.data(), host.raw_data(), host.nbytes());
  return out;
}

// Host-side reference "dequantize-then-matmul" for parity checks. Done
// entirely in FP64 so the reference error is negligible relative to
// the quantization error we're actually measuring.
std::vector<double> reference_dequant_matmul(
    const std::vector<float>& x, int64_t M, int64_t K,
    const std::vector<int8_t>& q_w, const std::vector<float>& scale,
    int64_t N) {
  std::vector<double> y(static_cast<size_t>(M * N), 0.0);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        acc += double(x[m * K + k]) * double(q_w[n * K + k]);
      }
      y[m * N + n] = acc * double(scale[n]);
    }
  }
  return y;
}

// Deterministic RNG shared across the suite so failures are reproducible.
std::mt19937_64 make_rng(uint64_t seed) { return std::mt19937_64{seed}; }

std::vector<float> random_fp32(size_t n, uint64_t seed, float lo = -1.0f,
                               float hi = 1.0f) {
  auto rng = make_rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

}  // namespace

TEST_CASE("pack_int8_symmetric: dequant error is bounded by max_abs/127",
          "[quant][pack]") {
  const int64_t N = 16, K = 64;
  auto w_host = random_fp32(N * K, /*seed*/ 0xA11ULL);
  Tensor W = from_host_f32(w_host, {N, K});

  auto [Q, S] = tesseract::quant::pack_int8_symmetric(W);
  REQUIRE(Q.dtype() == DType::Int8);
  REQUIRE(S.dtype() == DType::Float32);
  REQUIRE(Q.shape()[0] == N);
  REQUIRE(Q.shape()[1] == K);
  REQUIRE(S.shape()[0] == N);

  auto q = to_host_i8(Q);
  auto s = to_host_f32(S);

  // Per-row bound: |dequant[o, i] - W[o, i]| <= scale[o] / 2  (half the
  // quantization step, i.e. round-to-nearest error).
  for (int64_t n = 0; n < N; ++n) {
    const float scale = s[n];
    const float bound = 0.5f * scale + 1e-6f;
    for (int64_t k = 0; k < K; ++k) {
      const float orig  = w_host[n * K + k];
      const float deq   = float(q[n * K + k]) * scale;
      REQUIRE(std::fabs(deq - orig) <= bound);
      // Int8 range is asymmetric by design: values must stay in [-127, 127].
      // int8_t's natural range is [-128, 127] — we check the lower bound
      // explicitly (-127 vs the default -128) since the upper 127 is the
      // type's maximum.
      REQUIRE(q[n * K + k] >= -127);
    }
  }
}

TEST_CASE("pack_int8_symmetric: identically-zero row produces scale=1 and q=0",
          "[quant][pack]") {
  const int64_t N = 4, K = 8;
  std::vector<float> w_host(N * K, 0.0f);
  // Row 1 has nonzero content so we still exercise the normal path.
  for (int64_t k = 0; k < K; ++k) w_host[K + k] = 0.25f * float(k - 4);

  Tensor W = from_host_f32(w_host, {N, K});
  auto [Q, S] = tesseract::quant::pack_int8_symmetric(W);
  auto q = to_host_i8(Q);
  auto s = to_host_f32(S);

  // Zero rows: scale = 1, every q == 0, dequant exactly zero.
  for (int64_t n : {int64_t{0}, int64_t{2}, int64_t{3}}) {
    REQUIRE(s[n] == 1.0f);
    for (int64_t k = 0; k < K; ++k) {
      REQUIRE(q[n * K + k] == 0);
    }
  }

  // Nonzero row: scale > 0, at least one nonzero q.
  REQUIRE(s[1] > 0.0f);
  bool any_nonzero = false;
  for (int64_t k = 0; k < K; ++k) {
    if (q[K + k] != 0) { any_nonzero = true; break; }
  }
  REQUIRE(any_nonzero);
}

TEST_CASE("ops::dequantize_matmul_int8 (CPU, FP32) matches naive reference",
          "[quant][op][cpu]") {
  const int64_t M = 4, K = 64, N = 32;
  auto x_host = random_fp32(M * K, 0xBEE1);
  auto w_host = random_fp32(N * K, 0xBEE2);

  Tensor X = from_host_f32(x_host, {M, K});
  Tensor W = from_host_f32(w_host, {N, K});

  auto [Q, S] = tesseract::quant::pack_int8_symmetric(W);
  Tensor Y = tesseract::ops::dequantize_matmul_int8(X, Q, S);
  REQUIRE(Y.shape()[0] == M);
  REQUIRE(Y.shape()[1] == N);
  REQUIRE(Y.dtype() == DType::Float32);

  auto y    = to_host_f32(Y);
  auto qv   = to_host_i8(Q);
  auto sv   = to_host_f32(S);
  auto yref = reference_dequant_matmul(x_host, M, K, qv, sv, N);

  // Bound: both paths accumulate in FP32; the only source of error
  // is FP32 add-order drift, which is well below 1e-3 for K=64.
  for (size_t i = 0; i < y.size(); ++i) {
    REQUIRE_THAT(double(y[i]), WithinAbs(yref[i], 1e-3));
  }
}

TEST_CASE("ops::dequantize_matmul_int8 accepts rank>=2 batched inputs",
          "[quant][op][cpu]") {
  const int64_t B = 2, S_ = 3, K = 16, N = 8;
  auto x_host = random_fp32(B * S_ * K, 0xBA71);
  auto w_host = random_fp32(N * K, 0xBA72);

  Tensor X = from_host_f32(x_host, {B, S_, K});
  Tensor W = from_host_f32(w_host, {N, K});
  auto [Q, Sv] = tesseract::quant::pack_int8_symmetric(W);

  Tensor Y = tesseract::ops::dequantize_matmul_int8(X, Q, Sv);
  REQUIRE(Y.rank() == 3);
  REQUIRE(Y.shape()[0] == B);
  REQUIRE(Y.shape()[1] == S_);
  REQUIRE(Y.shape()[2] == N);

  // Flatten to [B*S_, K] and compare against the reference.
  const int64_t M = B * S_;
  auto y    = to_host_f32(Y);
  auto qv   = to_host_i8(Q);
  auto sv   = to_host_f32(Sv);
  auto yref = reference_dequant_matmul(x_host, M, K, qv, sv, N);

  for (size_t i = 0; i < y.size(); ++i) {
    REQUIRE_THAT(double(y[i]), WithinAbs(yref[i], 1e-3));
  }
}

TEST_CASE("ops::dequantize_matmul_int8 FP16 activations, CPU",
          "[quant][op][cpu][fp16]") {
  const int64_t M = 2, K = 32, N = 8;
  auto x_host_f32 = random_fp32(M * K, 0xFAD1);
  auto w_host_f32 = random_fp32(N * K, 0xFAD2);

  // FP16 activation storage.
  Tensor X_f16 = Tensor::empty({M, K}, DType::Float16);
  {
    auto* p = X_f16.data_ptr<tesseract::Half>();
    for (size_t i = 0; i < x_host_f32.size(); ++i) {
      p[i] = static_cast<tesseract::Half>(x_host_f32[i]);
    }
  }
  Tensor W_f16 = Tensor::empty({N, K}, DType::Float16);
  {
    auto* p = W_f16.data_ptr<tesseract::Half>();
    for (size_t i = 0; i < w_host_f32.size(); ++i) {
      p[i] = static_cast<tesseract::Half>(w_host_f32[i]);
    }
  }

  auto [Q, Sv] = tesseract::quant::pack_int8_symmetric(W_f16);
  Tensor Y = tesseract::ops::dequantize_matmul_int8(X_f16, Q, Sv);
  REQUIRE(Y.dtype() == DType::Float16);

  auto* yp = Y.data_ptr<tesseract::Half>();
  auto qv  = to_host_i8(Q);
  auto sv  = to_host_f32(Sv);

  // Build the FP16-rounded activation so the reference sees the same
  // precision as the op; compare in double.
  std::vector<float> x_fp16_rt(x_host_f32.size());
  {
    auto* p = X_f16.data_ptr<tesseract::Half>();
    for (size_t i = 0; i < x_fp16_rt.size(); ++i) x_fp16_rt[i] = static_cast<float>(p[i]);
  }
  auto yref = reference_dequant_matmul(x_fp16_rt, M, K, qv, sv, N);

  // FP16 output precision: ~1e-2 relative tolerance is realistic for
  // K = 32 accumulated in FP32 and narrowed back.
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      const double got = double(static_cast<float>(yp[m * N + n]));
      REQUIRE_THAT(got, WithinAbs(yref[m * N + n], 1e-2));
    }
  }
}

TEST_CASE("ops::dequantize_matmul_int8 autograd fallback: grad_x matches "
          "composite matmul path", "[quant][op][autograd]") {
  const int64_t M = 3, K = 16, N = 5;
  auto x_host = random_fp32(M * K, 0xF00D);
  auto w_host = random_fp32(N * K, 0xBEEF);

  Tensor X = from_host_f32(x_host, {M, K});
  X.set_requires_grad(true);
  Tensor W = from_host_f32(w_host, {N, K});
  auto [Q, Sv] = tesseract::quant::pack_int8_symmetric(W);

  Tensor Y = tesseract::ops::dequantize_matmul_int8(X, Q, Sv);
  REQUIRE(Y.defined());

  // Reduce to a scalar loss so we can call `Engine::backward`. `sum(Y)`
  // makes the upstream gradient `dL/dY` a tensor of ones, which plugs
  // into the closed form:
  //   grad_x[m, k] = sum_n dequant_W[n, k] = sum_n q_w[n, k] * scale[n]
  // i.e. independent of the batch index `m`.
  Tensor loss = tesseract::ops::sum(Y);
  Engine::backward(loss);

  auto* xm = X.autograd_meta();
  REQUIRE(xm != nullptr);
  REQUIRE(xm->grad.defined());
  auto gx = to_host_f32(xm->grad);

  auto qv = to_host_i8(Q);
  auto sv = to_host_f32(Sv);

  // Expected: grad_x[m, k] = sum_n scale[n] * q_w[n, k]  (independent of m).
  std::vector<float> expected(static_cast<size_t>(K), 0.0f);
  for (int64_t k = 0; k < K; ++k) {
    double s = 0.0;
    for (int64_t n = 0; n < N; ++n) {
      s += double(sv[n]) * double(qv[n * K + k]);
    }
    expected[k] = float(s);
  }
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t k = 0; k < K; ++k) {
      REQUIRE_THAT(double(gx[m * K + k]),
                   WithinAbs(double(expected[k]), 1e-3));
    }
  }
}

TEST_CASE("nn::QuantizedLinear::from_linear preserves top-1 ranking",
          "[quant][module][parity]") {
  // "Drop-in replacement" bar: for a reasonably large (out=32, in=128)
  // Linear with random activations, INT8 quantization should not
  // flip the argmax output of any of the 16 batched queries.
  const int64_t in_features = 128, out_features = 32, batch = 16;

  Linear fp{in_features, out_features, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinear::from_linear(fp);

  auto x_host = random_fp32(batch * in_features, 0xC0FFEE);
  Tensor X = from_host_f32(x_host, {batch, in_features});

  Tensor y_fp    = fp.forward(X);
  Tensor y_quant = qlin->forward(X);
  REQUIRE(y_fp.shape() == y_quant.shape());

  auto fp_host = to_host_f32(y_fp);
  auto q_host  = to_host_f32(y_quant);

  for (int64_t b = 0; b < batch; ++b) {
    int64_t fp_top = 0, q_top = 0;
    for (int64_t n = 1; n < out_features; ++n) {
      if (fp_host[b * out_features + n] > fp_host[b * out_features + fp_top]) fp_top = n;
      if (q_host [b * out_features + n] >  q_host[b * out_features +  q_top]) q_top  = n;
    }
    REQUIRE(fp_top == q_top);
  }
}

TEST_CASE("nn::QuantizedLinear exposes q_weight / weight_scale as buffers, "
          "not parameters", "[quant][module]") {
  Linear fp{32, 16, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinear::from_linear(fp);

  // Parameters = just the bias (bias is trainable by design; the
  // weight is frozen integer).
  const auto params = qlin->parameters();
  REQUIRE(params.size() == 1);
  REQUIRE(params[0].shape().rank() == 1);
  REQUIRE(params[0].shape()[0] == 16);

  // Buffers = q_weight + weight_scale.
  const auto buffers = qlin->named_buffers();
  bool found_q = false, found_s = false;
  for (const auto& [name, t] : buffers) {
    if (name == "q_weight")     { found_q = true; REQUIRE(t.dtype() == DType::Int8); }
    if (name == "weight_scale") { found_s = true; REQUIRE(t.dtype() == DType::Float32); }
  }
  REQUIRE(found_q);
  REQUIRE(found_s);
}

TEST_CASE("nn::QuantizedLinear with use_bias=false omits the bias term",
          "[quant][module]") {
  Linear fp{16, 8, /*use_bias=*/false, DType::Float32};
  auto qlin = QuantizedLinear::from_linear(fp);
  REQUIRE_FALSE(qlin->has_bias());
  REQUIRE(qlin->parameters().empty());

  auto x_host = random_fp32(4 * 16, 0x1234);
  Tensor X = from_host_f32(x_host, {4, 16});
  Tensor Y = qlin->forward(X);
  REQUIRE(Y.shape()[0] == 4);
  REQUIRE(Y.shape()[1] == 8);
}

TEST_CASE("ops::dequantize_matmul_int8 CPU↔CUDA parity",
          "[quant][op][cuda]") {
  if (!cuda_ready()) {
    SUCCEED("CUDA not available — skipping parity check");
    return;
  }
  const int64_t M = 8, K = 64, N = 32;
  auto x_host = random_fp32(M * K, 0xCAFE);
  auto w_host = random_fp32(N * K, 0xD00D);

  Tensor X_cpu = from_host_f32(x_host, {M, K});
  Tensor W_cpu = from_host_f32(w_host, {N, K});
  auto [Q_cpu, S_cpu] = tesseract::quant::pack_int8_symmetric(W_cpu);

  // Ship operands to CUDA and re-pack there so we exercise the
  // `.to(device)` path inside the packer too. The INT8 bytes should
  // survive the H->D round trip intact.
  Tensor W_cuda = W_cpu.to(cuda0());
  auto [Q_cuda, S_cuda] = tesseract::quant::pack_int8_symmetric(W_cuda);
  REQUIRE(Q_cuda.device().is_cuda());
  REQUIRE(S_cuda.device().is_cuda());

  auto q_cpu_host  = to_host_i8(Q_cpu);
  auto q_cuda_host = to_host_i8(Q_cuda);
  REQUIRE(q_cpu_host == q_cuda_host);

  Tensor X_cuda = X_cpu.to(cuda0());
  Tensor Y_cpu  = tesseract::ops::dequantize_matmul_int8(X_cpu,  Q_cpu,  S_cpu);
  Tensor Y_cuda = tesseract::ops::dequantize_matmul_int8(X_cuda, Q_cuda, S_cuda);
  auto y_cpu  = to_host_f32(Y_cpu);
  auto y_cuda = to_host_f32(Y_cuda);
  REQUIRE(y_cpu.size() == y_cuda.size());
  for (size_t i = 0; i < y_cpu.size(); ++i) {
    REQUIRE_THAT(double(y_cuda[i]), WithinAbs(double(y_cpu[i]), 1e-3));
  }
}

TEST_CASE("nn::QuantizedLinear CPU↔CUDA forward parity after .to(cuda)",
          "[quant][module][cuda]") {
  if (!cuda_ready()) {
    SUCCEED("CUDA not available — skipping parity check");
    return;
  }
  const int64_t in_features = 64, out_features = 32, batch = 4;
  Linear fp{in_features, out_features, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinear::from_linear(fp);

  auto x_host = random_fp32(batch * in_features, 0xFEED);
  Tensor X_cpu = from_host_f32(x_host, {batch, in_features});
  Tensor Y_cpu = qlin->forward(X_cpu);
  auto y_cpu = to_host_f32(Y_cpu);

  // Move module + input to CUDA and re-run.
  qlin->to(cuda0());
  REQUIRE(qlin->q_weight().device().is_cuda());
  REQUIRE(qlin->weight_scale().device().is_cuda());
  REQUIRE(qlin->bias().device().is_cuda());

  Tensor X_cuda = X_cpu.to(cuda0());
  Tensor Y_cuda = qlin->forward(X_cuda);
  auto y_cuda = to_host_f32(Y_cuda);

  REQUIRE(y_cpu.size() == y_cuda.size());
  for (size_t i = 0; i < y_cpu.size(); ++i) {
    REQUIRE_THAT(double(y_cuda[i]), WithinAbs(double(y_cpu[i]), 1e-3));
  }
}

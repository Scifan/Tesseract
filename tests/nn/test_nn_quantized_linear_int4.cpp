// Wave 3.2 (B-021) — INT4 per-group symmetric weight-only quantization
// stack:
//
//   * `quant::pack_int4_group`              — two nibbles / byte packer
//   * `ops::dequantize_matmul_int4_group`   — fused dequant-matmul op
//   * `nn::QuantizedLinearInt4G`            — drop-in inference module
//
// Structure parallels test_nn_quantized_linear.cpp, with the
// INT4-specific cases:
//   1. Pack round-trip: per-group bound, nibble values in [-7, 7].
//   2. Nibble packing convention (low = even-k, high = odd-k).
//   3. Zero-group safety: identically-zero group produces scale=1.
//   4. Op parity (CPU, FP32 + FP16) vs. FP64 hand-rolled reference.
//   5. Autograd fallback: grad_x matches the matmul composite.
//   6. from_linear top-1 ranking matches the source FP Linear.
//   7. CPU↔CUDA parity for op + module.
//   8. Validation failures: K % group_size != 0, odd group_size, ...
//
// Quantization error is larger than INT8 by design (7× coarser
// signed range), so tolerances are scaled appropriately. We do NOT
// aim for exact bitwise agreement between CPU and CUDA — only for
// matching numerical outputs within FP accumulation drift.

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
#include "tesseract/nn/QuantizedLinearInt4G.hpp"
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
using tesseract::nn::QuantizedLinearInt4G;

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

// Sign-extend the low four bits of `nib` to int ∈ [-8, 7]. Matches the
// kernel / op implementations.
int signext4(uint32_t nib) {
  return static_cast<int>((nib ^ 0x8u)) - 8;
}

// Unpack a packed `[N, K/2]` byte array into the flat `N*K` signed
// nibble vector the hand-rolled reference expects. Interpretation
// matches the packer / kernel exactly: low nibble == even-k.
std::vector<int> unpack_nibbles(const std::vector<int8_t>& packed,
                                int64_t N, int64_t K) {
  std::vector<int> out(static_cast<size_t>(N * K), 0);
  const int64_t packed_cols = K / 2;
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      const uint8_t byte = static_cast<uint8_t>(packed[n * packed_cols + (k / 2)]);
      const uint32_t nib =
          static_cast<uint32_t>((byte >> ((k & 1) ? 4 : 0)) & 0xFu);
      out[n * K + k] = signext4(nib);
    }
  }
  return out;
}

// FP64 reference dequantize-matmul for INT4 group symmetric:
//   y[m, n] = sum_k x[m, k] * q4[n, k] * scale[n, k/G]
std::vector<double> reference_dequant_matmul_int4(
    const std::vector<float>& x, int64_t M, int64_t K,
    const std::vector<int>& q4,
    const std::vector<float>& scale, int64_t N, int64_t group_size) {
  std::vector<double> y(static_cast<size_t>(M * N), 0.0);
  const int64_t groups_per_row = K / group_size;
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        const double s = double(scale[n * groups_per_row + (k / group_size)]);
        acc += double(x[m * K + k]) * double(q4[n * K + k]) * s;
      }
      y[m * N + n] = acc;
    }
  }
  return y;
}

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

// -----------------------------------------------------------------------------
// 1. Packer round-trip: per-group bound, nibble range in [-7, 7]
// -----------------------------------------------------------------------------

TEST_CASE("pack_int4_group: dequant error is bounded by scale/2 per group",
          "[quant][pack][int4]") {
  const int64_t N = 8, K = 64, group_size = 16;
  auto w_host = random_fp32(N * K, 0xA14U, -2.0f, 2.0f);
  Tensor W = from_host_f32(w_host, {N, K});

  auto [Q, S] = tesseract::quant::pack_int4_group(W, group_size);
  REQUIRE(Q.dtype() == DType::Int8);
  REQUIRE(S.dtype() == DType::Float32);
  REQUIRE(Q.rank() == 2);
  REQUIRE(Q.shape()[0] == N);
  REQUIRE(Q.shape()[1] == K / 2);
  REQUIRE(S.rank() == 2);
  REQUIRE(S.shape()[0] == N);
  REQUIRE(S.shape()[1] == K / group_size);

  auto q_packed = to_host_i8(Q);
  auto s        = to_host_f32(S);
  auto q4       = unpack_nibbles(q_packed, N, K);

  // Per-group bound: |dequant - W| <= scale[o, g] / 2. Also every
  // reconstructed nibble must be in [-7, 7].
  const int64_t groups_per_row = K / group_size;
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t g = 0; g < groups_per_row; ++g) {
      const float scale = s[n * groups_per_row + g];
      const float bound = 0.5f * scale + 1e-6f;
      for (int64_t j = 0; j < group_size; ++j) {
        const int64_t k = g * group_size + j;
        const int q = q4[n * K + k];
        REQUIRE(q >= -7);
        REQUIRE(q <=  7);
        const float orig = w_host[n * K + k];
        const float deq  = float(q) * scale;
        REQUIRE(std::fabs(deq - orig) <= bound);
      }
    }
  }
}

TEST_CASE("pack_int4_group: nibble ordering low = even-k, high = odd-k",
          "[quant][pack][int4]") {
  // Construct a row where even-k elements are large and odd-k are
  // small — the low nibble of each byte should carry the large value
  // and the high nibble the small one.
  const int64_t N = 1, K = 8, group_size = 8;
  std::vector<float> w_host(N * K);
  for (int64_t k = 0; k < K; ++k) {
    w_host[k] = (k % 2 == 0) ? 7.0f : 0.0f;
  }
  Tensor W = from_host_f32(w_host, {N, K});
  auto [Q, S] = tesseract::quant::pack_int4_group(W, group_size);
  auto packed = to_host_i8(Q);

  // With max_abs = 7, scale = 1. Even-k should encode to 7 (0x7) in
  // the low nibble; odd-k encodes to 0 (0x0) in the high nibble.
  for (int64_t byte_idx = 0; byte_idx < K / 2; ++byte_idx) {
    const uint8_t b = static_cast<uint8_t>(packed[byte_idx]);
    const uint32_t lo = b & 0xFu;
    const uint32_t hi = (b >> 4) & 0xFu;
    REQUIRE(signext4(lo) == 7);
    REQUIRE(signext4(hi) == 0);
  }
}

TEST_CASE("pack_int4_group: identically-zero group uses scale=1, q=0",
          "[quant][pack][int4]") {
  const int64_t N = 2, K = 16, group_size = 8;
  std::vector<float> w_host(N * K, 0.0f);
  // Second row, second group is the only non-zero one.
  for (int64_t j = 0; j < group_size; ++j) w_host[K + group_size + j] = 0.5f * float(j - 3);

  Tensor W = from_host_f32(w_host, {N, K});
  auto [Q, S] = tesseract::quant::pack_int4_group(W, group_size);

  auto s = to_host_f32(S);
  auto q_packed = to_host_i8(Q);
  auto q4 = unpack_nibbles(q_packed, N, K);

  const int64_t groups_per_row = K / group_size;
  // Zero groups → scale == 1, all nibbles zero.
  REQUIRE(s[0 * groups_per_row + 0] == 1.0f);
  REQUIRE(s[0 * groups_per_row + 1] == 1.0f);
  REQUIRE(s[1 * groups_per_row + 0] == 1.0f);
  for (int64_t k = 0; k < K; ++k) REQUIRE(q4[0 * K + k] == 0);
  for (int64_t j = 0; j < group_size; ++j) REQUIRE(q4[1 * K + j] == 0);

  REQUIRE(s[1 * groups_per_row + 1] > 0.0f);
}

// -----------------------------------------------------------------------------
// 2. Op parity (CPU)
// -----------------------------------------------------------------------------

TEST_CASE("ops::dequantize_matmul_int4_group (CPU, FP32) matches reference",
          "[quant][op][int4][cpu]") {
  const int64_t M = 4, K = 64, N = 32, group_size = 16;
  auto x_host = random_fp32(M * K, 0xBE41);
  auto w_host = random_fp32(N * K, 0xBE42, -2.0f, 2.0f);

  Tensor X = from_host_f32(x_host, {M, K});
  Tensor W = from_host_f32(w_host, {N, K});

  auto [Q, S] = tesseract::quant::pack_int4_group(W, group_size);
  Tensor Y = tesseract::ops::dequantize_matmul_int4_group(X, Q, S, group_size);
  REQUIRE(Y.shape()[0] == M);
  REQUIRE(Y.shape()[1] == N);
  REQUIRE(Y.dtype() == DType::Float32);

  auto y    = to_host_f32(Y);
  auto qv   = to_host_i8(Q);
  auto sv   = to_host_f32(S);
  auto q4   = unpack_nibbles(qv, N, K);
  auto yref = reference_dequant_matmul_int4(x_host, M, K, q4, sv, N,
                                            group_size);

  for (size_t i = 0; i < y.size(); ++i) {
    REQUIRE_THAT(double(y[i]), WithinAbs(yref[i], 1e-3));
  }
}

TEST_CASE("ops::dequantize_matmul_int4_group accepts rank>=2 batched inputs",
          "[quant][op][int4][cpu]") {
  const int64_t B = 2, S_ = 3, K = 32, N = 8, group_size = 8;
  auto x_host = random_fp32(B * S_ * K, 0xBA41);
  auto w_host = random_fp32(N * K, 0xBA42);

  Tensor X = from_host_f32(x_host, {B, S_, K});
  Tensor W = from_host_f32(w_host, {N, K});
  auto [Q, Sv] = tesseract::quant::pack_int4_group(W, group_size);

  Tensor Y = tesseract::ops::dequantize_matmul_int4_group(X, Q, Sv, group_size);
  REQUIRE(Y.rank() == 3);
  REQUIRE(Y.shape()[0] == B);
  REQUIRE(Y.shape()[1] == S_);
  REQUIRE(Y.shape()[2] == N);

  const int64_t M = B * S_;
  auto y    = to_host_f32(Y);
  auto qv   = to_host_i8(Q);
  auto sv   = to_host_f32(Sv);
  auto q4   = unpack_nibbles(qv, N, K);
  auto yref = reference_dequant_matmul_int4(x_host, M, K, q4, sv, N,
                                            group_size);
  for (size_t i = 0; i < y.size(); ++i) {
    REQUIRE_THAT(double(y[i]), WithinAbs(yref[i], 1e-3));
  }
}

TEST_CASE("ops::dequantize_matmul_int4_group FP16 activations, CPU",
          "[quant][op][int4][cpu][fp16]") {
  const int64_t M = 2, K = 32, N = 8, group_size = 8;
  auto x_host_f32 = random_fp32(M * K, 0xFAD4);
  auto w_host_f32 = random_fp32(N * K, 0xFAD5);

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

  auto [Q, Sv] = tesseract::quant::pack_int4_group(W_f16, group_size);
  Tensor Y = tesseract::ops::dequantize_matmul_int4_group(X_f16, Q, Sv,
                                                          group_size);
  REQUIRE(Y.dtype() == DType::Float16);

  auto* yp = Y.data_ptr<tesseract::Half>();
  auto qv  = to_host_i8(Q);
  auto sv  = to_host_f32(Sv);
  auto q4  = unpack_nibbles(qv, N, K);

  std::vector<float> x_fp16_rt(x_host_f32.size());
  {
    auto* p = X_f16.data_ptr<tesseract::Half>();
    for (size_t i = 0; i < x_fp16_rt.size(); ++i) {
      x_fp16_rt[i] = static_cast<float>(p[i]);
    }
  }
  auto yref = reference_dequant_matmul_int4(x_fp16_rt, M, K, q4, sv, N,
                                            group_size);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      const double got = double(static_cast<float>(yp[m * N + n]));
      REQUIRE_THAT(got, WithinAbs(yref[m * N + n], 1e-2));
    }
  }
}

// -----------------------------------------------------------------------------
// 3. Autograd fallback
// -----------------------------------------------------------------------------

TEST_CASE("ops::dequantize_matmul_int4_group autograd fallback: grad_x matches "
          "composite matmul", "[quant][op][int4][autograd]") {
  const int64_t M = 3, K = 32, N = 5, group_size = 8;
  auto x_host = random_fp32(M * K, 0xF004);
  auto w_host = random_fp32(N * K, 0xBEE4);

  Tensor X = from_host_f32(x_host, {M, K});
  X.set_requires_grad(true);
  Tensor W = from_host_f32(w_host, {N, K});
  auto [Q, Sv] = tesseract::quant::pack_int4_group(W, group_size);

  Tensor Y = tesseract::ops::dequantize_matmul_int4_group(X, Q, Sv, group_size);
  REQUIRE(Y.defined());

  // sum(Y) → upstream grad all ones → grad_x[m, k] = sum_n
  //   q4[n, k] * scale[n, k/G]  (independent of m).
  Tensor loss = tesseract::ops::sum(Y);
  Engine::backward(loss);

  auto* xm = X.autograd_meta();
  REQUIRE(xm != nullptr);
  REQUIRE(xm->grad.defined());
  auto gx = to_host_f32(xm->grad);

  auto qv = to_host_i8(Q);
  auto sv = to_host_f32(Sv);
  auto q4 = unpack_nibbles(qv, N, K);
  const int64_t groups_per_row = K / group_size;

  std::vector<float> expected(static_cast<size_t>(K), 0.0f);
  for (int64_t k = 0; k < K; ++k) {
    double s = 0.0;
    for (int64_t n = 0; n < N; ++n) {
      s += double(sv[n * groups_per_row + (k / group_size)]) *
           double(q4[n * K + k]);
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

// -----------------------------------------------------------------------------
// 4. Module-level parity
// -----------------------------------------------------------------------------

TEST_CASE("nn::QuantizedLinearInt4G::from_linear top-5 overlap ≥ 4/5",
          "[quant][module][int4][parity]") {
  // INT4 is coarser than INT8 by design — per the B-021 DoD, the
  // parity bar is "top-5 overlap ≥ 4/5" (not strict top-1). This is
  // also the parity bar modern quantized-inference stacks (GPTQ /
  // AWQ / llama.cpp) publish against FP16 baselines. We average over
  // 16 batched queries; the test passes when every query preserves
  // ≥4/5 of the FP16 top-5 output channels in its own INT4 top-5.
  const int64_t in_features = 256, out_features = 64, batch = 16;
  const int64_t group_size = 32;
  const int64_t top_k = 5;
  const int64_t overlap_floor = 4;  // per-query; B-021 DoD

  Linear fp{in_features, out_features, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinearInt4G::from_linear(fp, group_size);
  REQUIRE(qlin->group_size() == group_size);
  REQUIRE(qlin->in_features()  == in_features);
  REQUIRE(qlin->out_features() == out_features);

  auto x_host = random_fp32(batch * in_features, 0xC0FFE4);
  Tensor X = from_host_f32(x_host, {batch, in_features});

  Tensor y_fp    = fp.forward(X);
  Tensor y_quant = qlin->forward(X);
  REQUIRE(y_fp.shape() == y_quant.shape());

  auto fp_host = to_host_f32(y_fp);
  auto q_host  = to_host_f32(y_quant);

  auto top_k_of = [&](const std::vector<float>& y, int64_t b) {
    std::vector<int64_t> idx(static_cast<size_t>(out_features));
    for (int64_t n = 0; n < out_features; ++n) idx[n] = n;
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int64_t a, int64_t c) {
                        return y[b * out_features + a] >
                               y[b * out_features + c];
                      });
    idx.resize(top_k);
    std::sort(idx.begin(), idx.end());
    return idx;
  };

  for (int64_t b = 0; b < batch; ++b) {
    auto fp_top = top_k_of(fp_host, b);
    auto q_top  = top_k_of(q_host,  b);
    std::vector<int64_t> inter;
    std::set_intersection(fp_top.begin(), fp_top.end(),
                          q_top.begin(),  q_top.end(),
                          std::back_inserter(inter));
    INFO("batch=" << b << " fp_top_k vs q_top_k overlap=" << inter.size());
    REQUIRE(static_cast<int64_t>(inter.size()) >= overlap_floor);
  }
}

TEST_CASE("nn::QuantizedLinearInt4G exposes q_weight / weight_scale as buffers, "
          "not parameters", "[quant][module][int4]") {
  Linear fp{64, 16, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinearInt4G::from_linear(fp, 32);

  const auto params = qlin->parameters();
  REQUIRE(params.size() == 1);
  REQUIRE(params[0].shape().rank() == 1);
  REQUIRE(params[0].shape()[0] == 16);

  const auto buffers = qlin->named_buffers();
  bool found_q = false, found_s = false;
  for (const auto& [name, t] : buffers) {
    if (name == "q_weight") {
      found_q = true;
      REQUIRE(t.dtype() == DType::Int8);
      REQUIRE(t.shape()[0] == 16);
      REQUIRE(t.shape()[1] == 64 / 2);
    }
    if (name == "weight_scale") {
      found_s = true;
      REQUIRE(t.dtype() == DType::Float32);
      REQUIRE(t.shape()[0] == 16);
      REQUIRE(t.shape()[1] == 64 / 32);
    }
  }
  REQUIRE(found_q);
  REQUIRE(found_s);
}

TEST_CASE("nn::QuantizedLinearInt4G with use_bias=false omits the bias term",
          "[quant][module][int4]") {
  Linear fp{32, 8, /*use_bias=*/false, DType::Float32};
  auto qlin = QuantizedLinearInt4G::from_linear(fp, 16);
  REQUIRE_FALSE(qlin->has_bias());
  REQUIRE(qlin->parameters().empty());

  auto x_host = random_fp32(4 * 32, 0x1244);
  Tensor X = from_host_f32(x_host, {4, 32});
  Tensor Y = qlin->forward(X);
  REQUIRE(Y.shape()[0] == 4);
  REQUIRE(Y.shape()[1] == 8);
}

// -----------------------------------------------------------------------------
// 5. Validation
// -----------------------------------------------------------------------------

TEST_CASE("pack_int4_group / op reject bad group_size", "[quant][int4][validate]") {
  const int64_t N = 4, K = 16;
  Tensor W = from_host_f32(std::vector<float>(N * K, 0.5f), {N, K});

  REQUIRE_THROWS(tesseract::quant::pack_int4_group(W, 3));   // odd
  REQUIRE_THROWS(tesseract::quant::pack_int4_group(W, 7));   // odd & !div
  REQUIRE_THROWS(tesseract::quant::pack_int4_group(W, 1));   // < 2
  REQUIRE_THROWS(tesseract::quant::pack_int4_group(W, 10));  // !div K

  // Good case — doesn't throw.
  REQUIRE_NOTHROW(tesseract::quant::pack_int4_group(W, 8));
}

// -----------------------------------------------------------------------------
// 6. CPU↔CUDA parity
// -----------------------------------------------------------------------------

TEST_CASE("ops::dequantize_matmul_int4_group CPU↔CUDA parity",
          "[quant][op][int4][cuda]") {
  if (!cuda_ready()) {
    SUCCEED("CUDA not available — skipping parity check");
    return;
  }
  const int64_t M = 8, K = 64, N = 32, group_size = 16;
  auto x_host = random_fp32(M * K, 0xCAF4);
  auto w_host = random_fp32(N * K, 0xD004, -2.0f, 2.0f);

  Tensor X_cpu = from_host_f32(x_host, {M, K});
  Tensor W_cpu = from_host_f32(w_host, {N, K});
  auto [Q_cpu, S_cpu] = tesseract::quant::pack_int4_group(W_cpu, group_size);

  Tensor W_cuda = W_cpu.to(cuda0());
  auto [Q_cuda, S_cuda] =
      tesseract::quant::pack_int4_group(W_cuda, group_size);
  REQUIRE(Q_cuda.device().is_cuda());
  REQUIRE(S_cuda.device().is_cuda());

  // Packed bytes must agree bit-for-bit across the round-trip.
  auto q_cpu_host  = to_host_i8(Q_cpu);
  auto q_cuda_host = to_host_i8(Q_cuda);
  REQUIRE(q_cpu_host == q_cuda_host);

  Tensor X_cuda = X_cpu.to(cuda0());
  Tensor Y_cpu  =
      tesseract::ops::dequantize_matmul_int4_group(X_cpu,  Q_cpu,  S_cpu,
                                                    group_size);
  Tensor Y_cuda =
      tesseract::ops::dequantize_matmul_int4_group(X_cuda, Q_cuda, S_cuda,
                                                    group_size);
  auto y_cpu  = to_host_f32(Y_cpu);
  auto y_cuda = to_host_f32(Y_cuda);
  REQUIRE(y_cpu.size() == y_cuda.size());
  for (size_t i = 0; i < y_cpu.size(); ++i) {
    REQUIRE_THAT(double(y_cuda[i]), WithinAbs(double(y_cpu[i]), 1e-3));
  }
}

TEST_CASE("nn::QuantizedLinearInt4G CPU↔CUDA forward parity after .to(cuda)",
          "[quant][module][int4][cuda]") {
  if (!cuda_ready()) {
    SUCCEED("CUDA not available — skipping parity check");
    return;
  }
  const int64_t in_features = 128, out_features = 32, batch = 4;
  const int64_t group_size = 32;
  Linear fp{in_features, out_features, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinearInt4G::from_linear(fp, group_size);

  auto x_host = random_fp32(batch * in_features, 0xFEE4);
  Tensor X_cpu = from_host_f32(x_host, {batch, in_features});
  Tensor Y_cpu = qlin->forward(X_cpu);
  auto y_cpu = to_host_f32(Y_cpu);

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

// Unit tests for `ops::layer_norm` + `nn::LayerNorm`.
//
// Coverage:
//   * Hand-rolled reference parity on rank-2/3 inputs (FP32).
//   * Bias-affine and bias-free variants (elementwise_affine with/without bias).
//   * CPU↔CUDA parity when CUDA is available (FP32 path; the composite
//     reuses the same primitive kernels validated in B-015).
//   * Autograd finite-diff against analytic grads (weight, bias, input).
//   * `nn::LayerNorm` smoke: registers `weight` + `bias` (or just
//     `weight` when `use_bias=false`), forward is identical to the
//     underlying `ops::layer_norm`, and `Module::to(cuda)` moves both
//     parameters.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/nn/LayerNorm.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Normalization.hpp"
#include "tesseract/ops/Reduction.hpp"

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
using tesseract::nn::LayerNorm;

namespace {

bool cuda_ready() { return has_cuda_support() && is_available(); }
Device cuda0() { return Device{DeviceType::CUDA, 0}; }

Tensor from_host_f32(std::vector<float> data, Shape shape) {
  Tensor t = Tensor::empty(shape, DType::Float32);
  std::memcpy(t.raw_data(), data.data(), data.size() * sizeof(float));
  return t;
}

// Reference: biased variance (PyTorch's `F.layer_norm` / ATen default).
std::vector<double> ln_reference(const std::vector<float>& x,
                                 const std::vector<float>& w,
                                 const std::vector<float>* b,
                                 int64_t outer, int64_t D, double eps) {
  std::vector<double> out(static_cast<std::size_t>(outer * D));
  for (int64_t o = 0; o < outer; ++o) {
    double mean = 0.0;
    for (int64_t j = 0; j < D; ++j) mean += x[o * D + j];
    mean /= static_cast<double>(D);
    double var = 0.0;
    for (int64_t j = 0; j < D; ++j) {
      const double d = x[o * D + j] - mean;
      var += d * d;
    }
    var /= static_cast<double>(D);
    const double denom = std::sqrt(var + eps);
    for (int64_t j = 0; j < D; ++j) {
      const double yhat = (x[o * D + j] - mean) / denom;
      double y = yhat * w[j];
      if (b) y += (*b)[j];
      out[o * D + j] = y;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("ops::layer_norm: forward parity against reference (rank-3, bias)") {
  const int64_t B = 2, S = 3, D = 4;
  std::vector<float> xd(B * S * D);
  for (int i = 0; i < static_cast<int>(xd.size()); ++i) {
    xd[i] = static_cast<float>((i % 7) - 3) * 0.5f;
  }
  std::vector<float> wd = {1.0f, 0.5f, -0.25f, 2.0f};
  std::vector<float> bd = {0.1f, -0.2f, 0.3f, -0.4f};
  Tensor x = from_host_f32(xd, Shape({B, S, D}));
  Tensor w = from_host_f32(wd, Shape({D}));
  Tensor b = from_host_f32(bd, Shape({D}));

  Tensor y = tesseract::ops::layer_norm(x, w, b, 1e-5);
  REQUIRE(y.shape() == Shape({B, S, D}));
  REQUIRE(y.dtype() == DType::Float32);

  const auto ref = ln_reference(xd, wd, &bd, B * S, D, 1e-5);
  const float* yp = y.data_ptr<float>();
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(yp[i]), WithinAbs(ref[i], 1e-5));
  }
}

TEST_CASE("ops::layer_norm: bias-free variant matches reference") {
  const int64_t N = 5, D = 6;
  std::vector<float> xd(N * D);
  for (int i = 0; i < static_cast<int>(xd.size()); ++i) {
    xd[i] = static_cast<float>(((i * 13) % 11) - 5) * 0.3f;
  }
  std::vector<float> wd(D, 1.0f);
  for (int j = 0; j < D; ++j) wd[j] = 0.25f + 0.1f * static_cast<float>(j);
  Tensor x = from_host_f32(xd, Shape({N, D}));
  Tensor w = from_host_f32(wd, Shape({D}));

  Tensor y = tesseract::ops::layer_norm(x, w, Tensor{}, 1e-5);
  const auto ref = ln_reference(xd, wd, nullptr, N, D, 1e-5);
  const float* yp = y.data_ptr<float>();
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(yp[i]), WithinAbs(ref[i], 1e-5));
  }
}

TEST_CASE("ops::layer_norm: autograd — finite-diff vs engine grads") {
  const int64_t N = 3, D = 4;
  std::vector<float> xd(N * D);
  for (int i = 0; i < static_cast<int>(xd.size()); ++i) {
    xd[i] = 0.37f * static_cast<float>(i % 5) - 0.6f;
  }
  std::vector<float> wd = {0.9f, 1.1f, 0.7f, 1.3f};
  std::vector<float> bd = {0.05f, -0.05f, 0.1f, -0.1f};
  Tensor x = from_host_f32(xd, Shape({N, D}));
  Tensor w = from_host_f32(wd, Shape({D}));
  Tensor b = from_host_f32(bd, Shape({D}));
  x.set_requires_grad(true);
  w.set_requires_grad(true);
  b.set_requires_grad(true);

  Tensor y = tesseract::ops::layer_norm(x, w, b, 1e-5);
  Tensor loss = tesseract::ops::sum(y);  // d/dy = 1 everywhere
  Engine::backward(loss);

  const auto* xm = x.autograd_meta();
  const auto* wm = w.autograd_meta();
  const auto* bm = b.autograd_meta();
  REQUIRE(xm != nullptr);
  REQUIRE(xm->grad.defined());
  REQUIRE(wm != nullptr);
  REQUIRE(wm->grad.defined());
  REQUIRE(bm != nullptr);
  REQUIRE(bm->grad.defined());

  // Finite-diff reference at machine-epsilon-aware h; we check that the
  // engine grads match a central-difference estimate to mid-precision.
  auto forward_sum = [&](const std::vector<float>& xv,
                         const std::vector<float>& wv,
                         const std::vector<float>& bv) -> double {
    const auto r = ln_reference(xv, wv, &bv, N, D, 1e-5);
    double s = 0.0;
    for (double v : r) s += v;
    return s;
  };

  const double h = 5e-3;
  const float* xg = xm->grad.data_ptr<float>();
  for (int i = 0; i < N * D; ++i) {
    auto xp = xd; xp[i] += static_cast<float>(h);
    auto xn = xd; xn[i] -= static_cast<float>(h);
    const double fd = (forward_sum(xp, wd, bd) - forward_sum(xn, wd, bd)) / (2 * h);
    REQUIRE_THAT(static_cast<double>(xg[i]), WithinAbs(fd, 1e-2));
  }
  const float* wg = wm->grad.data_ptr<float>();
  for (int j = 0; j < D; ++j) {
    auto wp = wd; wp[j] += static_cast<float>(h);
    auto wn = wd; wn[j] -= static_cast<float>(h);
    const double fd = (forward_sum(xd, wp, bd) - forward_sum(xd, wn, bd)) / (2 * h);
    REQUIRE_THAT(static_cast<double>(wg[j]), WithinAbs(fd, 1e-2));
  }
  const float* bg = bm->grad.data_ptr<float>();
  for (int j = 0; j < D; ++j) {
    // Analytic: dLoss/dbias[j] = sum over batch of 1 == N.
    REQUIRE_THAT(static_cast<double>(bg[j]), WithinAbs(static_cast<double>(N), 1e-5));
  }
}

TEST_CASE("ops::layer_norm: CPU↔CUDA parity (FP32)") {
  if (!cuda_ready()) {
    SKIP("CUDA not available");
  }
  const int64_t B = 2, S = 3, D = 8;
  std::vector<float> xd(B * S * D);
  for (int i = 0; i < static_cast<int>(xd.size()); ++i) {
    xd[i] = 0.11f * static_cast<float>((i * 17) % 23) - 1.3f;
  }
  std::vector<float> wd(D), bd(D);
  for (int j = 0; j < D; ++j) {
    wd[j] = 0.6f + 0.05f * static_cast<float>(j);
    bd[j] = -0.03f * static_cast<float>(j);
  }
  Tensor x_cpu = from_host_f32(xd, Shape({B, S, D}));
  Tensor w_cpu = from_host_f32(wd, Shape({D}));
  Tensor b_cpu = from_host_f32(bd, Shape({D}));
  Tensor x_gpu = x_cpu.to(cuda0());
  Tensor w_gpu = w_cpu.to(cuda0());
  Tensor b_gpu = b_cpu.to(cuda0());

  Tensor y_cpu = tesseract::ops::layer_norm(x_cpu, w_cpu, b_cpu, 1e-5);
  Tensor y_gpu = tesseract::ops::layer_norm(x_gpu, w_gpu, b_gpu, 1e-5);
  Tensor y_gpu_on_cpu = y_gpu.to(cpu_device());

  const float* a = y_cpu.data_ptr<float>();
  const float* b_ = y_gpu_on_cpu.data_ptr<float>();
  for (int i = 0; i < B * S * D; ++i) {
    REQUIRE_THAT(static_cast<double>(a[i]),
                 WithinAbs(static_cast<double>(b_[i]), 1e-5));
  }
}

TEST_CASE("nn::LayerNorm: registers params and applies affine") {
  LayerNorm ln(/*normalized_dim=*/4, /*eps=*/1e-5, /*use_bias=*/true);
  REQUIRE(ln.has_bias());
  auto named = ln.named_parameters();
  REQUIRE(named.size() == 2);
  // Order: weight first (registered first), then bias.
  REQUIRE(named[0].first == "weight");
  REQUIRE(named[1].first == "bias");
  REQUIRE(ln.weight().shape() == Shape({4}));
  REQUIRE(ln.bias().shape() == Shape({4}));

  // Identity init — weight=1, bias=0 → first forward == normalize(x).
  std::vector<float> xd = {1.0f, 2.0f, 3.0f, 4.0f,
                           -1.0f, -2.0f, 0.0f, 1.0f};
  Tensor x = from_host_f32(xd, Shape({2, 4}));
  Tensor y = ln.forward(x);

  // Each row should have ~zero mean / unit variance after init affine.
  const float* yp = y.data_ptr<float>();
  for (int r = 0; r < 2; ++r) {
    double mean = 0.0, var = 0.0;
    for (int j = 0; j < 4; ++j) mean += yp[r * 4 + j];
    mean /= 4.0;
    for (int j = 0; j < 4; ++j) {
      const double d = yp[r * 4 + j] - mean;
      var += d * d;
    }
    var /= 4.0;
    REQUIRE_THAT(mean, WithinAbs(0.0, 1e-5));
    REQUIRE_THAT(var, WithinAbs(1.0, 1e-3));
  }
}

TEST_CASE("nn::LayerNorm: use_bias=false drops bias parameter") {
  LayerNorm ln(/*normalized_dim=*/4, /*eps=*/1e-5, /*use_bias=*/false);
  REQUIRE_FALSE(ln.has_bias());
  auto named = ln.named_parameters();
  REQUIRE(named.size() == 1);
  REQUIRE(named[0].first == "weight");

  std::vector<float> xd = {0.5f, -0.5f, 1.5f, -1.5f};
  Tensor x = from_host_f32(xd, Shape({1, 4}));
  Tensor y = ln.forward(x);
  REQUIRE(y.shape() == Shape({1, 4}));
  // With weight=1, bias absent → should equal plain normalization.
  const float* yp = y.data_ptr<float>();
  double mean = 0.0;
  for (int j = 0; j < 4; ++j) mean += yp[j];
  mean /= 4.0;
  REQUIRE_THAT(mean, WithinAbs(0.0, 1e-5));
}

TEST_CASE("nn::LayerNorm: Module::to(cuda) migrates both params") {
  if (!cuda_ready()) {
    SKIP("CUDA not available");
  }
  LayerNorm ln(8, 1e-5, /*use_bias=*/true);
  std::vector<float> xd(2 * 8);
  for (int i = 0; i < 16; ++i) xd[i] = 0.1f * static_cast<float>(i) - 0.8f;
  Tensor x_cpu = from_host_f32(xd, Shape({2, 8}));
  Tensor y_cpu = ln.forward(x_cpu);

  ln.to(cuda0());
  REQUIRE(ln.weight().device().type == DeviceType::CUDA);
  REQUIRE(ln.bias().device().type == DeviceType::CUDA);
  Tensor x_gpu = x_cpu.to(cuda0());
  Tensor y_gpu_on_cpu = ln.forward(x_gpu).to(cpu_device());

  const float* a = y_cpu.data_ptr<float>();
  const float* b = y_gpu_on_cpu.data_ptr<float>();
  for (int i = 0; i < 16; ++i) {
    REQUIRE_THAT(static_cast<double>(a[i]),
                 WithinAbs(static_cast<double>(b[i]), 1e-5));
  }
}

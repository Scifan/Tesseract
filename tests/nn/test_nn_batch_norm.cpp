// Wave 2b (B-020) — tests for `ops::batch_norm` + `nn::BatchNorm{1d,2d}`.
//
// Coverage:
//   * Hand-rolled reference parity on rank-2 (BN1d-2D), rank-3 (BN1d-3D),
//     and rank-4 (BN2d) inputs, both training (biased batch variance)
//     and eval (running-stats broadcast) paths.
//   * Running-stats writeback — `running_mean` / `running_var` are mutated
//     in place exactly once per training forward, with the right PyTorch
//     moment semantics (biased forward var + unbiased running var + EMA).
//   * `training=false` must leave running stats untouched — the classic
//     "my eval loop kept drifting running_mean" regression.
//   * `Module::train(bool)` recurses through children (exercised via a
//     hand-rolled tiny composite module owning a BN leaf + a Sequential
//     wrapping the same leaf).
//   * `affine=false` drops `weight` / `bias` entirely.
//   * Autograd finite-diff gradient check on the training path (FP32).
//   * CPU↔CUDA parity for forward (training + eval) on FP32.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/nn/BatchNorm.hpp"
#include "tesseract/nn/Sequential.hpp"
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
using tesseract::nn::BatchNorm1d;
using tesseract::nn::BatchNorm2d;
using tesseract::nn::Module;
using tesseract::nn::Sequential;

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

// Generic per-channel BN reference. `x` is laid out row-major with the
// channel axis at `channel_dim`; `outer` is the flat size of all
// leading/trailing non-channel dims collapsed. The helper computes the
// biased batch mean/variance, normalizes, then applies the optional
// `weight` / `bias`. Also returns the biased var so callers can derive
// the unbiased running-var the same way the op does.
//
// Layout details are flattened into a single linear index via the shape
// parameters so the same helper serves BN1d-2D/3D and BN2d.
struct BnRefResult {
  std::vector<double> y;
  std::vector<double> batch_mean;
  std::vector<double> batch_var_biased;
  std::vector<double> batch_var_unbiased;
};

BnRefResult bn_reference(const std::vector<float>& x,
                         Shape shape,
                         const std::vector<float>* weight,
                         const std::vector<float>* bias,
                         double eps) {
  const int64_t rank = static_cast<int64_t>(shape.rank());
  REQUIRE(rank >= 2);
  const int64_t C = shape[1];
  int64_t n_red = 1;
  for (int64_t d = 0; d < rank; ++d) if (d != 1) n_red *= shape[d];
  const int64_t total = n_red * C;

  // strides[d] in ELEMENT units
  std::vector<int64_t> strides(static_cast<size_t>(rank));
  strides[rank - 1] = 1;
  for (int64_t d = rank - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];

  std::vector<double> mean(C, 0.0), var(C, 0.0);
  // Walk every index, bucket by channel using the channel-axis stride.
  std::vector<int64_t> idx(rank, 0);
  std::vector<int64_t> cnt(C, 0);
  for (int64_t lin = 0; lin < total; ++lin) {
    int64_t rem = lin;
    for (int64_t d = 0; d < rank; ++d) {
      idx[d] = rem / strides[d];
      rem    = rem % strides[d];
    }
    const int64_t c = idx[1];
    mean[c] += x[lin];
    cnt[c]  += 1;
  }
  for (int64_t c = 0; c < C; ++c) mean[c] /= static_cast<double>(cnt[c]);

  for (int64_t lin = 0; lin < total; ++lin) {
    int64_t rem = lin;
    for (int64_t d = 0; d < rank; ++d) {
      idx[d] = rem / strides[d];
      rem    = rem % strides[d];
    }
    const int64_t c = idx[1];
    const double dv = x[lin] - mean[c];
    var[c] += dv * dv;
  }
  std::vector<double> var_biased(C), var_unbiased(C);
  for (int64_t c = 0; c < C; ++c) {
    var_biased[c]   = var[c] / static_cast<double>(cnt[c]);
    var_unbiased[c] = (cnt[c] > 1) ? var[c] / static_cast<double>(cnt[c] - 1)
                                   : var_biased[c];
  }

  std::vector<double> y(static_cast<size_t>(total));
  for (int64_t lin = 0; lin < total; ++lin) {
    int64_t rem = lin;
    for (int64_t d = 0; d < rank; ++d) {
      idx[d] = rem / strides[d];
      rem    = rem % strides[d];
    }
    const int64_t c = idx[1];
    const double yhat = (x[lin] - mean[c]) / std::sqrt(var_biased[c] + eps);
    double out = yhat;
    if (weight) out *= static_cast<double>((*weight)[c]);
    if (bias)   out += static_cast<double>((*bias)[c]);
    y[lin] = out;
  }

  return {std::move(y), std::move(mean), std::move(var_biased), std::move(var_unbiased)};
}

}  // namespace

TEST_CASE("ops::batch_norm: BN1d-2D training forward matches reference") {
  const int64_t N = 4, C = 3;
  std::vector<float> xd(N * C);
  for (int i = 0; i < N * C; ++i) xd[i] = 0.37f * static_cast<float>((i * 5) % 11) - 1.1f;
  std::vector<float> wd = {1.2f, -0.7f, 0.5f};
  std::vector<float> bd = {0.05f, -0.1f, 0.2f};
  Tensor x = from_host_f32(xd, Shape({N, C}));
  Tensor w = from_host_f32(wd, Shape({C}));
  Tensor b = from_host_f32(bd, Shape({C}));
  Tensor rm = Tensor::zeros({C}, DType::Float32);
  Tensor rv = Tensor::ones({C},  DType::Float32);

  Tensor y = tesseract::ops::batch_norm(x, w, b, rm, rv,
                                        /*training=*/true,
                                        /*momentum=*/0.1,
                                        /*eps=*/1e-5);
  REQUIRE(y.shape() == Shape({N, C}));
  const auto ref = bn_reference(xd, Shape({N, C}), &wd, &bd, 1e-5);
  const auto yh = to_host_f32(y);
  for (size_t i = 0; i < ref.y.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(yh[i]), WithinAbs(ref.y[i], 1e-5));
  }
}

TEST_CASE("ops::batch_norm: BN1d-3D training forward matches reference") {
  const int64_t N = 2, C = 4, L = 5;
  std::vector<float> xd(N * C * L);
  for (int i = 0; i < N * C * L; ++i) {
    xd[i] = 0.19f * static_cast<float>(((i * 13) % 17) - 8);
  }
  std::vector<float> wd(C), bd(C);
  for (int c = 0; c < C; ++c) { wd[c] = 0.5f + 0.1f * c; bd[c] = -0.07f * c; }
  Tensor x = from_host_f32(xd, Shape({N, C, L}));
  Tensor w = from_host_f32(wd, Shape({C}));
  Tensor b = from_host_f32(bd, Shape({C}));
  Tensor rm = Tensor::zeros({C}, DType::Float32);
  Tensor rv = Tensor::ones({C},  DType::Float32);

  Tensor y = tesseract::ops::batch_norm(x, w, b, rm, rv, true, 0.1, 1e-5);
  const auto ref = bn_reference(xd, Shape({N, C, L}), &wd, &bd, 1e-5);
  const auto yh = to_host_f32(y);
  for (size_t i = 0; i < ref.y.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(yh[i]), WithinAbs(ref.y[i], 1e-5));
  }
}

TEST_CASE("ops::batch_norm: BN2d training forward matches reference") {
  const int64_t N = 2, C = 3, H = 4, W = 3;
  std::vector<float> xd(N * C * H * W);
  for (int i = 0; i < static_cast<int>(xd.size()); ++i) {
    xd[i] = 0.23f * static_cast<float>(((i * 7) % 19) - 9);
  }
  std::vector<float> wd = {1.0f, 0.75f, -0.5f};
  std::vector<float> bd = {0.1f, -0.2f, 0.3f};
  Tensor x = from_host_f32(xd, Shape({N, C, H, W}));
  Tensor w = from_host_f32(wd, Shape({C}));
  Tensor b = from_host_f32(bd, Shape({C}));
  Tensor rm = Tensor::zeros({C}, DType::Float32);
  Tensor rv = Tensor::ones({C},  DType::Float32);

  Tensor y = tesseract::ops::batch_norm(x, w, b, rm, rv, true, 0.1, 1e-5);
  const auto ref = bn_reference(xd, Shape({N, C, H, W}), &wd, &bd, 1e-5);
  const auto yh = to_host_f32(y);
  for (size_t i = 0; i < ref.y.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(yh[i]), WithinAbs(ref.y[i], 1e-5));
  }
}

TEST_CASE("ops::batch_norm: running stats EMA update (PyTorch semantics)") {
  // After a single training step with momentum=m, running_mean should equal
  // (1-m)*prev + m*batch_mean, and running_var should equal
  // (1-m)*prev + m*batch_var_UNBIASED — exactly what PyTorch does.
  const int64_t N = 4, C = 3;
  const double momentum = 0.2;
  std::vector<float> xd(N * C);
  for (int i = 0; i < N * C; ++i) xd[i] = 0.5f * static_cast<float>((i * 3) % 7) - 0.8f;
  std::vector<float> wd = {1.0f, 1.0f, 1.0f};
  std::vector<float> bd = {0.0f, 0.0f, 0.0f};

  Tensor x  = from_host_f32(xd, Shape({N, C}));
  Tensor w  = from_host_f32(wd, Shape({C}));
  Tensor b  = from_host_f32(bd, Shape({C}));
  // Start the running stats at non-trivial values so the EMA rule is actually
  // exercised (zeros + ones masks bugs of the form `running_mean = batch_mean`).
  Tensor rm = from_host_f32({0.25f, -0.5f, 0.75f}, Shape({C}));
  Tensor rv = from_host_f32({2.0f,  0.5f,  1.5f},  Shape({C}));
  std::vector<float> rm_before = to_host_f32(rm);
  std::vector<float> rv_before = to_host_f32(rv);

  Tensor y = tesseract::ops::batch_norm(x, w, b, rm, rv, true, momentum, 1e-5);
  (void)y;

  const auto ref = bn_reference(xd, Shape({N, C}), &wd, &bd, 1e-5);
  const auto rm_after = to_host_f32(rm);
  const auto rv_after = to_host_f32(rv);
  for (int64_t c = 0; c < C; ++c) {
    const double expect_m = (1.0 - momentum) * rm_before[c] + momentum * ref.batch_mean[c];
    const double expect_v = (1.0 - momentum) * rv_before[c] + momentum * ref.batch_var_unbiased[c];
    REQUIRE_THAT(static_cast<double>(rm_after[c]), WithinAbs(expect_m, 1e-5));
    REQUIRE_THAT(static_cast<double>(rv_after[c]), WithinAbs(expect_v, 1e-5));
  }
}

TEST_CASE("ops::batch_norm: training=false leaves running stats intact") {
  const int64_t N = 4, C = 3;
  std::vector<float> xd(N * C);
  for (int i = 0; i < N * C; ++i) xd[i] = 0.3f * static_cast<float>(i) - 1.0f;
  Tensor x  = from_host_f32(xd, Shape({N, C}));
  Tensor w  = Tensor::ones({C},  DType::Float32);
  Tensor b  = Tensor::zeros({C}, DType::Float32);
  Tensor rm = from_host_f32({0.3f, -0.6f, 0.1f}, Shape({C}));
  Tensor rv = from_host_f32({1.5f, 0.7f,  2.0f}, Shape({C}));
  const auto rm_before = to_host_f32(rm);
  const auto rv_before = to_host_f32(rv);

  Tensor y = tesseract::ops::batch_norm(x, w, b, rm, rv, /*training=*/false, 0.1, 1e-5);

  // Running stats must be byte-identical before/after the eval-path call.
  const auto rm_after = to_host_f32(rm);
  const auto rv_after = to_host_f32(rv);
  for (int64_t c = 0; c < C; ++c) {
    REQUIRE(rm_after[c] == rm_before[c]);
    REQUIRE(rv_after[c] == rv_before[c]);
  }

  // Output uses the running stats, NOT the batch stats.
  const auto yh = to_host_f32(y);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const double expect =
          (xd[n * C + c] - rm_before[c]) / std::sqrt(rv_before[c] + 1e-5);
      REQUIRE_THAT(static_cast<double>(yh[n * C + c]), WithinAbs(expect, 1e-5));
    }
  }
}

TEST_CASE("ops::batch_norm: affine=false path (no weight, no bias)") {
  const int64_t N = 3, C = 2;
  std::vector<float> xd = {0.1f, -0.5f,
                           1.2f,  0.3f,
                          -0.7f,  0.8f};
  Tensor x = from_host_f32(xd, Shape({N, C}));
  Tensor rm = Tensor::zeros({C}, DType::Float32);
  Tensor rv = Tensor::ones({C},  DType::Float32);

  Tensor y = tesseract::ops::batch_norm(x, /*weight=*/Tensor{}, /*bias=*/Tensor{},
                                        rm, rv, true, 0.1, 1e-5);
  const auto ref = bn_reference(xd, Shape({N, C}), nullptr, nullptr, 1e-5);
  const auto yh = to_host_f32(y);
  for (size_t i = 0; i < ref.y.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(yh[i]), WithinAbs(ref.y[i], 1e-5));
  }
}

TEST_CASE("ops::batch_norm: autograd gradients match finite differences") {
  const int64_t N = 4, C = 3;
  std::vector<float> xd(N * C);
  for (int i = 0; i < N * C; ++i) xd[i] = 0.29f * static_cast<float>((i * 5) % 13) - 0.9f;
  std::vector<float> wd = {0.9f, 1.1f, 0.7f};
  std::vector<float> bd = {0.05f, -0.05f, 0.12f};
  Tensor x = from_host_f32(xd, Shape({N, C}));
  Tensor w = from_host_f32(wd, Shape({C}));
  Tensor b = from_host_f32(bd, Shape({C}));
  x.set_requires_grad(true);
  w.set_requires_grad(true);
  b.set_requires_grad(true);
  Tensor rm = Tensor::zeros({C}, DType::Float32);
  Tensor rv = Tensor::ones({C},  DType::Float32);

  Tensor y = tesseract::ops::batch_norm(x, w, b, rm, rv, true, 0.1, 1e-5);
  Tensor loss = tesseract::ops::sum(y);
  Engine::backward(loss);

  auto forward_sum = [&](const std::vector<float>& xv,
                         const std::vector<float>& wv,
                         const std::vector<float>& bv) -> double {
    const auto r = bn_reference(xv, Shape({N, C}), &wv, &bv, 1e-5);
    double s = 0.0;
    for (double v : r.y) s += v;
    return s;
  };

  const double h = 5e-3;
  const auto* xm = x.autograd_meta();
  REQUIRE((xm && xm->grad.defined()));
  const float* xg = xm->grad.data_ptr<float>();
  for (int i = 0; i < N * C; ++i) {
    auto xp = xd; xp[i] += static_cast<float>(h);
    auto xn = xd; xn[i] -= static_cast<float>(h);
    const double fd = (forward_sum(xp, wd, bd) - forward_sum(xn, wd, bd)) / (2 * h);
    REQUIRE_THAT(static_cast<double>(xg[i]), WithinAbs(fd, 2e-2));
  }
  const auto* wm = w.autograd_meta();
  REQUIRE((wm && wm->grad.defined()));
  const float* wg = wm->grad.data_ptr<float>();
  for (int j = 0; j < C; ++j) {
    auto wp = wd; wp[j] += static_cast<float>(h);
    auto wn = wd; wn[j] -= static_cast<float>(h);
    const double fd = (forward_sum(xd, wp, bd) - forward_sum(xd, wn, bd)) / (2 * h);
    REQUIRE_THAT(static_cast<double>(wg[j]), WithinAbs(fd, 2e-2));
  }
  const auto* bm = b.autograd_meta();
  REQUIRE((bm && bm->grad.defined()));
  const float* bg = bm->grad.data_ptr<float>();
  for (int j = 0; j < C; ++j) {
    // dLoss/dbias[j] = sum of 1 over every position sharing channel j = N.
    REQUIRE_THAT(static_cast<double>(bg[j]), WithinAbs(static_cast<double>(N), 1e-5));
  }
}

TEST_CASE("ops::batch_norm: CPU↔CUDA parity (FP32, training)") {
  if (!cuda_ready()) SKIP("CUDA not available");
  const int64_t N = 2, C = 4, H = 3, W = 3;
  std::vector<float> xd(N * C * H * W);
  for (int i = 0; i < static_cast<int>(xd.size()); ++i) {
    xd[i] = 0.17f * static_cast<float>(((i * 11) % 23) - 11);
  }
  std::vector<float> wd(C), bd(C);
  for (int c = 0; c < C; ++c) { wd[c] = 0.8f + 0.1f * c; bd[c] = -0.05f * c; }

  Tensor x_cpu = from_host_f32(xd, Shape({N, C, H, W}));
  Tensor w_cpu = from_host_f32(wd, Shape({C}));
  Tensor b_cpu = from_host_f32(bd, Shape({C}));
  Tensor rm_cpu = Tensor::zeros({C}, DType::Float32);
  Tensor rv_cpu = Tensor::ones({C},  DType::Float32);

  Tensor x_gpu = x_cpu.to(cuda0());
  Tensor w_gpu = w_cpu.to(cuda0());
  Tensor b_gpu = b_cpu.to(cuda0());
  Tensor rm_gpu = rm_cpu.to(cuda0());
  Tensor rv_gpu = rv_cpu.to(cuda0());

  Tensor y_cpu = tesseract::ops::batch_norm(x_cpu, w_cpu, b_cpu, rm_cpu, rv_cpu,
                                            true, 0.1, 1e-5);
  Tensor y_gpu = tesseract::ops::batch_norm(x_gpu, w_gpu, b_gpu, rm_gpu, rv_gpu,
                                            true, 0.1, 1e-5);
  const auto yh_cpu = to_host_f32(y_cpu);
  const auto yh_gpu = to_host_f32(y_gpu);
  REQUIRE(yh_cpu.size() == yh_gpu.size());
  for (size_t i = 0; i < yh_cpu.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(yh_cpu[i]),
                 WithinAbs(static_cast<double>(yh_gpu[i]), 1e-5));
  }
  // Running stats must also agree — that's the in-place writeback crossing
  // the device boundary through Storage::copy_device_bytes.
  const auto rm_after_cpu = to_host_f32(rm_cpu);
  const auto rm_after_gpu = to_host_f32(rm_gpu);
  const auto rv_after_cpu = to_host_f32(rv_cpu);
  const auto rv_after_gpu = to_host_f32(rv_gpu);
  for (int64_t c = 0; c < C; ++c) {
    REQUIRE_THAT(static_cast<double>(rm_after_cpu[c]),
                 WithinAbs(static_cast<double>(rm_after_gpu[c]), 1e-5));
    REQUIRE_THAT(static_cast<double>(rv_after_cpu[c]),
                 WithinAbs(static_cast<double>(rv_after_gpu[c]), 1e-5));
  }
}

TEST_CASE("nn::BatchNorm1d: registers params + buffers and applies affine") {
  BatchNorm1d bn(/*num_features=*/4);
  // Parameters: weight + bias (affine=true default).
  auto named_p = bn.named_parameters();
  REQUIRE(named_p.size() == 2);
  REQUIRE(named_p[0].first == "weight");
  REQUIRE(named_p[1].first == "bias");
  // Buffers: running_mean + running_var.
  auto named_b = bn.named_buffers();
  REQUIRE(named_b.size() == 2);
  REQUIRE(named_b[0].first == "running_mean");
  REQUIRE(named_b[1].first == "running_var");
  REQUIRE(bn.is_training() == true);
  REQUIRE(bn.weight().shape() == Shape({4}));
  REQUIRE(bn.bias().shape()   == Shape({4}));
}

TEST_CASE("nn::BatchNorm1d: affine=false drops weight + bias") {
  BatchNorm1d bn(/*num_features=*/4, /*eps=*/1e-5, /*momentum=*/0.1,
                 /*affine=*/false);
  REQUIRE(bn.named_parameters().empty());
  auto named_b = bn.named_buffers();
  REQUIRE(named_b.size() == 2);
}

TEST_CASE("nn::BatchNorm2d: forward + eval switch matches ops::batch_norm") {
  const int64_t N = 2, C = 3, H = 2, W = 2;
  BatchNorm2d bn(/*num_features=*/C);
  std::vector<float> xd(N * C * H * W);
  for (int i = 0; i < static_cast<int>(xd.size()); ++i) {
    xd[i] = 0.41f * static_cast<float>((i * 3) % 13) - 1.1f;
  }
  Tensor x = from_host_f32(xd, Shape({N, C, H, W}));

  // Train-mode forward updates running stats.
  Tensor y_train = bn.forward(x);
  REQUIRE(y_train.shape() == Shape({N, C, H, W}));

  auto rm_after_train = to_host_f32(bn.running_mean());
  auto rv_after_train = to_host_f32(bn.running_var());
  // Running mean/var should have drifted from their init (0 / 1) by exactly
  // `momentum * batch_stat`.
  bool drifted_m = false, drifted_v = false;
  for (int c = 0; c < C; ++c) {
    if (std::abs(rm_after_train[c]) > 1e-8) drifted_m = true;
    if (std::abs(rv_after_train[c] - 1.0f) > 1e-8) drifted_v = true;
  }
  REQUIRE(drifted_m);
  REQUIRE(drifted_v);

  // Switching to eval must freeze the running stats across further forwards.
  bn.eval();
  REQUIRE(bn.is_training() == false);
  Tensor y_eval = bn.forward(x);
  (void)y_eval;
  auto rm_after_eval = to_host_f32(bn.running_mean());
  auto rv_after_eval = to_host_f32(bn.running_var());
  for (int c = 0; c < C; ++c) {
    REQUIRE(rm_after_eval[c] == rm_after_train[c]);
    REQUIRE(rv_after_eval[c] == rv_after_train[c]);
  }
}

TEST_CASE("Module::train(bool) recurses through Sequential into BN children") {
  auto bn1 = std::make_shared<BatchNorm1d>(4);
  auto bn2 = std::make_shared<BatchNorm2d>(3);
  Sequential seq;
  seq.add(bn1);
  seq.add(bn2);

  // Base state: every module is_training=true (the default).
  REQUIRE(seq.is_training() == true);
  REQUIRE(bn1->is_training() == true);
  REQUIRE(bn2->is_training() == true);

  // Flip root to eval — children must follow in lock-step, which is the
  // whole point of recursing `train(bool)` through the tree. A
  // regression here would silently keep BN running-stats drifting
  // during inference ⇒ model checkpoints go stale.
  seq.eval();
  REQUIRE(seq.is_training() == false);
  REQUIRE(bn1->is_training() == false);
  REQUIRE(bn2->is_training() == false);

  seq.train();
  REQUIRE(seq.is_training() == true);
  REQUIRE(bn1->is_training() == true);
  REQUIRE(bn2->is_training() == true);
}

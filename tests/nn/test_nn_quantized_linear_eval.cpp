// Wave 4.4 (B-026) — eval-mode fast path guard for the two quantized
// Linear drop-ins.
//
// The underlying `ops::dequantize_matmul_int8` / `_int4_group` ops
// route through an autograd fallback whenever
// `is_grad_enabled() && x.requires_grad()` that materializes the full
// FP weight `[N, K]` on-device for the backward graph. That is the
// right behavior for genuine training (LoRA + frozen quantized
// backbone, QAT-style probes, …), but it's a silent memory blow-up
// during plain inference when a caller forgets to install a
// `NoGradGuard` around a graph whose inputs happen to carry
// `requires_grad=true`. Wave 4.4 pins down an explicit
// `module.eval()` escape hatch: `is_training() == false` guarantees
// the fused-kernel fast path, independent of `x.requires_grad()` and
// the ambient `NoGradGuard` state.
//
// Coverage:
//   1. INT8 eval mode + `x.requires_grad()==true` → no grad_fn on the
//      output, values match `from_linear(fp).forward(x)` bit-for-bit.
//   2. INT8 train mode + `x.requires_grad()==true` → output carries a
//      grad_fn and `Engine::backward` populates `x.grad` (the
//      fallback path still fires, i.e. backward compatibility).
//   3. INT4G variant of (1) — same semantics, group-scaled nibbles.
//   4. INT4G variant of (2) — same autograd-fallback contract.
//   5. Eval mode with outer `NoGradGuard` is a no-op (nested guards
//      compose, the fast path stays active).
//   6. `module.train()` / `module.eval()` dispatch flips correctly
//      between the two paths on the same module instance.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/QuantizedLinear.hpp"
#include "tesseract/nn/QuantizedLinearInt4G.hpp"
#include "tesseract/ops/Reduction.hpp"

using Catch::Matchers::WithinAbs;
using tesseract::DType;
using tesseract::Engine;
using tesseract::NoGradGuard;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::nn::Linear;
using tesseract::nn::QuantizedLinear;
using tesseract::nn::QuantizedLinearInt4G;

namespace {

Tensor from_host_f32(std::vector<float> data, Shape shape) {
  Tensor t = Tensor::empty(shape, DType::Float32);
  std::memcpy(t.raw_data(), data.data(), data.size() * sizeof(float));
  return t;
}

std::vector<float> to_host_f32(const Tensor& t) {
  std::vector<float> out(static_cast<size_t>(t.numel()));
  std::memcpy(out.data(), t.raw_data(), t.nbytes());
  return out;
}

std::vector<float> random_fp32(size_t n, uint64_t seed,
                               float lo = -1.0f, float hi = 1.0f) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

}  // namespace

// -----------------------------------------------------------------------------
// INT8
// -----------------------------------------------------------------------------

TEST_CASE("QuantizedLinear (INT8) eval() pins fused path even when "
          "x.requires_grad()", "[quant][int8][eval]") {
  const int64_t in_features = 64, out_features = 32, batch = 4;
  Linear fp{in_features, out_features, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinear::from_linear(fp);
  qlin->eval();

  auto x_host = random_fp32(batch * in_features, 0xE7A1);
  Tensor X = from_host_f32(x_host, {batch, in_features});
  // Mimic the silent-regression scenario: user forgot NoGradGuard AND
  // the input tensor carries `requires_grad=true` (common when the
  // activation flowed through an earlier trainable module before
  // reaching the quantized layer).
  X.set_requires_grad(true);
  REQUIRE(tesseract::is_grad_enabled());

  Tensor Y = qlin->forward(X);
  REQUIRE(Y.shape()[0] == batch);
  REQUIRE(Y.shape()[1] == out_features);

  // Fast path is leaf on the tape: `NoGradGuard` inside the module
  // drops both the dequant-fallback edge and the bias-add edge.
  REQUIRE_FALSE(Y.requires_grad());
  REQUIRE(Y.autograd_meta() == nullptr);

  // Parity: identical to running the same module under an outer
  // NoGradGuard (where op-layer also picks the fused kernel).
  Tensor X_ref = from_host_f32(x_host, {batch, in_features});
  Tensor Y_ref;
  {
    NoGradGuard _;
    Y_ref = qlin->forward(X_ref);
  }
  auto y      = to_host_f32(Y);
  auto y_ref  = to_host_f32(Y_ref);
  for (size_t i = 0; i < y.size(); ++i) {
    REQUIRE_THAT(double(y[i]), WithinAbs(double(y_ref[i]), 1e-6));
  }
}

TEST_CASE("QuantizedLinear (INT8) train() still routes grad through the "
          "autograd fallback", "[quant][int8][autograd]") {
  const int64_t in_features = 16, out_features = 8, batch = 3;
  Linear fp{in_features, out_features, /*use_bias=*/false, DType::Float32};
  auto qlin = QuantizedLinear::from_linear(fp);
  qlin->train();  // explicit — default is already true

  auto x_host = random_fp32(batch * in_features, 0xE7A2);
  Tensor X = from_host_f32(x_host, {batch, in_features});
  X.set_requires_grad(true);

  Tensor Y = qlin->forward(X);
  // Fallback active: the output must be differentiable w.r.t. `x`
  // (the op wires `MatMulBackward` against `x` after materializing
  // the FP weight once).
  REQUIRE(Y.requires_grad());
  REQUIRE(Y.autograd_meta() != nullptr);

  // sum(Y) -> scalar loss -> backward; check `x.grad` populated.
  Tensor loss = tesseract::ops::sum(Y);
  Engine::backward(loss);
  auto* xm = X.autograd_meta();
  REQUIRE(xm != nullptr);
  REQUIRE(xm->grad.defined());
  REQUIRE(xm->grad.shape() == X.shape());
}

TEST_CASE("QuantizedLinear (INT8) eval() ↔ train() toggles the dispatch "
          "on the same module instance", "[quant][int8][eval]") {
  const int64_t in = 8, out = 4, batch = 2;
  Linear fp{in, out, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinear::from_linear(fp);

  auto x_host = random_fp32(batch * in, 0xE7A3);

  // Train → fallback → requires_grad=true on output.
  qlin->train();
  {
    Tensor X = from_host_f32(x_host, {batch, in});
    X.set_requires_grad(true);
    Tensor Y = qlin->forward(X);
    REQUIRE(Y.requires_grad());
  }

  // Eval → fast path → requires_grad=false on output, even with the
  // exact same x.requires_grad=true input.
  qlin->eval();
  {
    Tensor X = from_host_f32(x_host, {batch, in});
    X.set_requires_grad(true);
    Tensor Y = qlin->forward(X);
    REQUIRE_FALSE(Y.requires_grad());
  }

  // Back to train — fallback re-engages.
  qlin->train();
  {
    Tensor X = from_host_f32(x_host, {batch, in});
    X.set_requires_grad(true);
    Tensor Y = qlin->forward(X);
    REQUIRE(Y.requires_grad());
  }
}

TEST_CASE("QuantizedLinear (INT8) eval() nested inside outer NoGradGuard "
          "is a no-op", "[quant][int8][eval]") {
  const int64_t in = 32, out = 16, batch = 2;
  Linear fp{in, out, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinear::from_linear(fp);
  qlin->eval();

  auto x_host = random_fp32(batch * in, 0xE7A4);
  Tensor X = from_host_f32(x_host, {batch, in});
  X.set_requires_grad(true);

  // Outer guard already disabled autograd; the module's inner
  // NoGradGuard must nest without corrupting prev_state on scope exit.
  {
    NoGradGuard outer;
    REQUIRE_FALSE(tesseract::is_grad_enabled());
    Tensor Y = qlin->forward(X);
    REQUIRE_FALSE(Y.requires_grad());
  }
  // After the outer guard unwinds, the thread-local grad flag should
  // be back to enabled (not stuck off because of the inner guard).
  REQUIRE(tesseract::is_grad_enabled());
}

// -----------------------------------------------------------------------------
// INT4G
// -----------------------------------------------------------------------------

TEST_CASE("QuantizedLinearInt4G eval() pins fused path even when "
          "x.requires_grad()", "[quant][int4g][eval]") {
  const int64_t in_features = 128, out_features = 32, batch = 4;
  const int64_t group_size = 64;
  Linear fp{in_features, out_features, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinearInt4G::from_linear(fp, group_size);
  qlin->eval();

  auto x_host = random_fp32(batch * in_features, 0xE7B1);
  Tensor X = from_host_f32(x_host, {batch, in_features});
  X.set_requires_grad(true);
  REQUIRE(tesseract::is_grad_enabled());

  Tensor Y = qlin->forward(X);
  REQUIRE(Y.shape()[0] == batch);
  REQUIRE(Y.shape()[1] == out_features);
  REQUIRE_FALSE(Y.requires_grad());
  REQUIRE(Y.autograd_meta() == nullptr);

  // Parity with the same module invoked under an outer NoGradGuard.
  Tensor X_ref = from_host_f32(x_host, {batch, in_features});
  Tensor Y_ref;
  {
    NoGradGuard _;
    Y_ref = qlin->forward(X_ref);
  }
  auto y      = to_host_f32(Y);
  auto y_ref  = to_host_f32(Y_ref);
  for (size_t i = 0; i < y.size(); ++i) {
    REQUIRE_THAT(double(y[i]), WithinAbs(double(y_ref[i]), 1e-6));
  }
}

TEST_CASE("QuantizedLinearInt4G train() still routes grad through the "
          "autograd fallback", "[quant][int4g][autograd]") {
  const int64_t in_features = 64, out_features = 8, batch = 3;
  const int64_t group_size = 32;
  Linear fp{in_features, out_features, /*use_bias=*/false, DType::Float32};
  auto qlin = QuantizedLinearInt4G::from_linear(fp, group_size);
  qlin->train();

  auto x_host = random_fp32(batch * in_features, 0xE7B2);
  Tensor X = from_host_f32(x_host, {batch, in_features});
  X.set_requires_grad(true);

  Tensor Y = qlin->forward(X);
  REQUIRE(Y.requires_grad());
  REQUIRE(Y.autograd_meta() != nullptr);

  Tensor loss = tesseract::ops::sum(Y);
  Engine::backward(loss);
  auto* xm = X.autograd_meta();
  REQUIRE(xm != nullptr);
  REQUIRE(xm->grad.defined());
  REQUIRE(xm->grad.shape() == X.shape());
}

TEST_CASE("QuantizedLinearInt4G eval() ↔ train() toggles the dispatch",
          "[quant][int4g][eval]") {
  const int64_t in = 64, out = 16, batch = 2;
  const int64_t group_size = 32;
  Linear fp{in, out, /*use_bias=*/true, DType::Float32};
  auto qlin = QuantizedLinearInt4G::from_linear(fp, group_size);

  auto x_host = random_fp32(batch * in, 0xE7B3);

  qlin->train();
  {
    Tensor X = from_host_f32(x_host, {batch, in});
    X.set_requires_grad(true);
    Tensor Y = qlin->forward(X);
    REQUIRE(Y.requires_grad());
  }
  qlin->eval();
  {
    Tensor X = from_host_f32(x_host, {batch, in});
    X.set_requires_grad(true);
    Tensor Y = qlin->forward(X);
    REQUIRE_FALSE(Y.requires_grad());
  }
}

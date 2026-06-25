// Wave 4.1 (B-025): unit tests for `ops::swiglu_silu_gate`.
//
// Coverage:
//   * Hand-rolled reference parity (FP32) — `silu(gate) * up`.
//   * Composite-equivalent parity: the fused op must produce the same
//     values (to 1e-5 FP32) as `mul(gate, sigmoid(gate)) * up` on the
//     CPU. That verifies the "inference fast path" doesn't diverge
//     from the training-time autograd path.
//   * CPU↔CUDA parity across all four supported dtypes (FP32 / FP64 /
//     FP16 / BF16), guarded behind a `SKIP` when no GPU is visible.
//   * Autograd finite-diff parity through the composite fallback. The
//     fused CUDA kernel is forward-only by design (same convention as
//     `rms_norm`); when either operand requires grad, the op
//     decomposes into primitives and gradients flow through
//     `SigmoidBackward` + `MulBackward` automatically, so we only
//     have to check that the resulting grads match the analytic
//     finite-difference estimate.
//   * Shape / dtype / device validation — the op-layer boundary
//     rejects mismatched shape, dtype, or device pairs with a
//     readable `TESSERACT_CHECK` message.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Reduction.hpp"

using Catch::Matchers::WithinAbs;
using tesseract::BFloat16;
using tesseract::cpu_device;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Engine;
using tesseract::Half;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

namespace {

bool cuda_ready() { return has_cuda_support() && is_available(); }
Device cuda0() { return Device{DeviceType::CUDA, 0}; }

Tensor from_host_f32(std::vector<float> data, Shape shape) {
  Tensor t = Tensor::empty(shape, DType::Float32);
  std::memcpy(t.raw_data(), data.data(), data.size() * sizeof(float));
  return t;
}

// `silu(g) * u = g * sigmoid(g) * u` evaluated in FP64 for maximum
// headroom; used as the golden reference on every parity path.
std::vector<double> swiglu_reference(const std::vector<float>& gate,
                                     const std::vector<float>& up) {
  std::vector<double> out(gate.size());
  for (std::size_t i = 0; i < gate.size(); ++i) {
    const double g = gate[i];
    const double u = up[i];
    const double sig = 1.0 / (1.0 + std::exp(-g));
    out[i] = (g * sig) * u;
  }
  return out;
}

}  // namespace

TEST_CASE("ops::swiglu_silu_gate: FP32 forward parity vs hand-rolled reference") {
  const int64_t B = 3, S = 4, D = 5;
  std::vector<float> gd(B * S * D), ud(B * S * D);
  for (std::size_t i = 0; i < gd.size(); ++i) {
    gd[i] = 0.25f * static_cast<float>((i % 11) - 5);
    ud[i] = -0.3f + 0.1f * static_cast<float>((i * 7) % 13);
  }
  Tensor g = from_host_f32(gd, Shape({B, S, D}));
  Tensor u = from_host_f32(ud, Shape({B, S, D}));

  Tensor y = tesseract::ops::swiglu_silu_gate(g, u);
  REQUIRE(y.shape() == Shape({B, S, D}));
  REQUIRE(y.dtype() == DType::Float32);
  REQUIRE(y.device() == cpu_device());

  const auto ref = swiglu_reference(gd, ud);
  const float* yp = y.data_ptr<float>();
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(yp[i]), WithinAbs(ref[i], 1e-6));
  }
}

TEST_CASE("ops::swiglu_silu_gate: matches composite `mul(x, sigmoid(x)) * up`") {
  // This is the invariant that guarantees training-time correctness —
  // the fused op must produce numerically identical results to the
  // explicit composite that used to live in `nn::FeedForward::forward`.
  const int64_t N = 64, D = 32;
  std::vector<float> gd(N * D), ud(N * D);
  for (std::size_t i = 0; i < gd.size(); ++i) {
    gd[i] = 0.031f * static_cast<float>(static_cast<int>(i) - 1000);
    ud[i] = 0.19f + 0.047f * static_cast<float>(i % 17);
  }
  Tensor g = from_host_f32(gd, Shape({N, D}));
  Tensor u = from_host_f32(ud, Shape({N, D}));

  Tensor fused = tesseract::ops::swiglu_silu_gate(g, u);
  Tensor composite = tesseract::ops::mul(
      tesseract::ops::mul(g, tesseract::ops::sigmoid(g)), u);

  const float* fp = fused.data_ptr<float>();
  const float* cp = composite.data_ptr<float>();
  for (std::size_t i = 0; i < gd.size(); ++i) {
    REQUIRE_THAT(static_cast<double>(fp[i]),
                 WithinAbs(static_cast<double>(cp[i]), 1e-6));
  }
}

TEST_CASE("ops::swiglu_silu_gate: autograd — finite-diff vs engine grads") {
  // Autograd-active path: fused op decomposes into the composite so
  // gradients flow through the primitive backward nodes. We verify by
  // backward-ing a scalar loss (`sum(out)`) and checking the resulting
  // d/d(gate) and d/d(up) against a central-difference estimate.
  const int64_t N = 3, D = 4;
  std::vector<float> gd(N * D), ud(N * D);
  for (std::size_t i = 0; i < gd.size(); ++i) {
    gd[i] = 0.37f * static_cast<float>(i % 5) - 0.6f;
    ud[i] = 0.21f + 0.13f * static_cast<float>((i * 3) % 7);
  }
  Tensor g = from_host_f32(gd, Shape({N, D}));
  Tensor u = from_host_f32(ud, Shape({N, D}));
  g.set_requires_grad(true);
  u.set_requires_grad(true);

  Tensor y = tesseract::ops::swiglu_silu_gate(g, u);
  Tensor loss = tesseract::ops::sum(y);
  Engine::backward(loss);

  const auto* gm = g.autograd_meta();
  const auto* um = u.autograd_meta();
  REQUIRE(gm != nullptr);
  REQUIRE(gm->grad.defined());
  REQUIRE(um != nullptr);
  REQUIRE(um->grad.defined());

  auto forward_sum = [&](const std::vector<float>& gv,
                         const std::vector<float>& uv) -> double {
    const auto r = swiglu_reference(gv, uv);
    double s = 0.0;
    for (double v : r) s += v;
    return s;
  };

  const double h = 3e-3;
  const float* g_grad = gm->grad.data_ptr<float>();
  const float* u_grad = um->grad.data_ptr<float>();
  for (int i = 0; i < N * D; ++i) {
    auto gp = gd; gp[i] += static_cast<float>(h);
    auto gn = gd; gn[i] -= static_cast<float>(h);
    const double fd_g = (forward_sum(gp, ud) - forward_sum(gn, ud)) / (2 * h);
    REQUIRE_THAT(static_cast<double>(g_grad[i]), WithinAbs(fd_g, 5e-3));

    auto up = ud; up[i] += static_cast<float>(h);
    auto un = ud; un[i] -= static_cast<float>(h);
    const double fd_u = (forward_sum(gd, up) - forward_sum(gd, un)) / (2 * h);
    REQUIRE_THAT(static_cast<double>(u_grad[i]), WithinAbs(fd_u, 5e-3));
  }
}

TEST_CASE("ops::swiglu_silu_gate: validation errors") {
  Tensor g = from_host_f32({1.0f, 2.0f, 3.0f}, Shape({3}));
  Tensor u = from_host_f32({0.5f, -0.5f, 1.5f}, Shape({3}));
  // Shape mismatch.
  Tensor u_bad_shape = from_host_f32({0.5f, -0.5f}, Shape({2}));
  REQUIRE_THROWS(tesseract::ops::swiglu_silu_gate(g, u_bad_shape));
  // Dtype mismatch.
  Tensor u_bad_dtype = Tensor::empty(Shape({3}), DType::Float64);
  std::vector<double> ud64 = {0.5, -0.5, 1.5};
  std::memcpy(u_bad_dtype.raw_data(), ud64.data(), ud64.size() * sizeof(double));
  REQUIRE_THROWS(tesseract::ops::swiglu_silu_gate(g, u_bad_dtype));
  // Integer dtype.
  Tensor gi = Tensor::empty(Shape({3}), DType::Int32);
  Tensor ui = Tensor::empty(Shape({3}), DType::Int32);
  REQUIRE_THROWS(tesseract::ops::swiglu_silu_gate(gi, ui));
}

TEST_CASE("ops::swiglu_silu_gate: CPU↔CUDA parity (FP32)") {
  if (!cuda_ready()) SKIP("CUDA not available");
  const int64_t N = 16, D = 32;
  std::vector<float> gd(N * D), ud(N * D);
  for (std::size_t i = 0; i < gd.size(); ++i) {
    gd[i] = 0.1f * static_cast<float>((i * 17) % 23) - 1.15f;
    ud[i] = 0.08f + 0.03f * static_cast<float>((i * 29) % 31);
  }
  Tensor g_cpu = from_host_f32(gd, Shape({N, D}));
  Tensor u_cpu = from_host_f32(ud, Shape({N, D}));
  Tensor y_cpu = tesseract::ops::swiglu_silu_gate(g_cpu, u_cpu);

  Tensor g_cuda = g_cpu.to(cuda0());
  Tensor u_cuda = u_cpu.to(cuda0());
  Tensor y_cuda = tesseract::ops::swiglu_silu_gate(g_cuda, u_cuda);
  REQUIRE(y_cuda.device() == cuda0());

  Tensor y_back = y_cuda.to(cpu_device());
  const float* cpu_p = y_cpu.data_ptr<float>();
  const float* gpu_p = y_back.data_ptr<float>();
  for (int64_t i = 0; i < N * D; ++i) {
    REQUIRE_THAT(static_cast<double>(gpu_p[i]),
                 WithinAbs(static_cast<double>(cpu_p[i]), 1e-5));
  }
}

TEST_CASE("ops::swiglu_silu_gate: CPU↔CUDA parity (FP64)") {
  if (!cuda_ready()) SKIP("CUDA not available");
  const int64_t N = 8, D = 16;
  std::vector<double> gd(N * D), ud(N * D);
  for (std::size_t i = 0; i < gd.size(); ++i) {
    gd[i] = 0.05 * static_cast<double>((i * 13) % 17) - 0.8;
    ud[i] = 0.2 + 0.03 * static_cast<double>((i * 7) % 11);
  }
  Tensor g_cpu = Tensor::empty(Shape({N, D}), DType::Float64);
  Tensor u_cpu = Tensor::empty(Shape({N, D}), DType::Float64);
  std::memcpy(g_cpu.raw_data(), gd.data(), gd.size() * sizeof(double));
  std::memcpy(u_cpu.raw_data(), ud.data(), ud.size() * sizeof(double));
  Tensor y_cpu = tesseract::ops::swiglu_silu_gate(g_cpu, u_cpu);

  Tensor g_cuda = g_cpu.to(cuda0());
  Tensor u_cuda = u_cpu.to(cuda0());
  Tensor y_cuda = tesseract::ops::swiglu_silu_gate(g_cuda, u_cuda);
  Tensor y_back = y_cuda.to(cpu_device());

  const double* cpu_p = y_cpu.data_ptr<double>();
  const double* gpu_p = y_back.data_ptr<double>();
  for (int64_t i = 0; i < N * D; ++i) {
    REQUIRE_THAT(gpu_p[i], WithinAbs(cpu_p[i], 1e-12));
  }
}

namespace {

template <typename H>
std::vector<H> quantize_half(const std::vector<float>& f) {
  std::vector<H> out(f.size());
  for (std::size_t i = 0; i < f.size(); ++i) out[i] = H(f[i]);
  return out;
}

}  // namespace

TEST_CASE("ops::swiglu_silu_gate: CPU↔CUDA parity (FP16)") {
  if (!cuda_ready()) SKIP("CUDA not available");
  const int64_t N = 8, D = 16;
  std::vector<float> gd(N * D), ud(N * D);
  for (std::size_t i = 0; i < gd.size(); ++i) {
    // Keep FP16 representable (|g| ≤ ~2) so the sigmoid saturation
    // region doesn't dominate and give us artificially easy numerics.
    gd[i] = 0.0625f * static_cast<float>((i * 17) % 19) - 0.75f;
    ud[i] = 0.125f + 0.0625f * static_cast<float>((i * 3) % 7);
  }
  auto gd_h = quantize_half<Half>(gd);
  auto ud_h = quantize_half<Half>(ud);

  Tensor g_cpu = Tensor::empty(Shape({N, D}), DType::Float16);
  Tensor u_cpu = Tensor::empty(Shape({N, D}), DType::Float16);
  std::memcpy(g_cpu.raw_data(), gd_h.data(), gd_h.size() * sizeof(Half));
  std::memcpy(u_cpu.raw_data(), ud_h.data(), ud_h.size() * sizeof(Half));
  Tensor y_cpu = tesseract::ops::swiglu_silu_gate(g_cpu, u_cpu);

  Tensor g_cuda = g_cpu.to(cuda0());
  Tensor u_cuda = u_cpu.to(cuda0());
  Tensor y_cuda = tesseract::ops::swiglu_silu_gate(g_cuda, u_cuda);
  Tensor y_back = y_cuda.to(cpu_device());

  const Half* cpu_p = y_cpu.data_ptr<Half>();
  const Half* gpu_p = y_back.data_ptr<Half>();
  for (int64_t i = 0; i < N * D; ++i) {
    REQUIRE_THAT(static_cast<double>(static_cast<float>(gpu_p[i])),
                 WithinAbs(static_cast<double>(static_cast<float>(cpu_p[i])), 2e-3));
  }
}

TEST_CASE("ops::swiglu_silu_gate: CPU↔CUDA parity (BF16)") {
  if (!cuda_ready()) SKIP("CUDA not available");
  const int64_t N = 8, D = 16;
  std::vector<float> gd(N * D), ud(N * D);
  for (std::size_t i = 0; i < gd.size(); ++i) {
    gd[i] = 0.0625f * static_cast<float>((i * 17) % 19) - 0.75f;
    ud[i] = 0.125f + 0.0625f * static_cast<float>((i * 3) % 7);
  }
  auto gd_b = quantize_half<BFloat16>(gd);
  auto ud_b = quantize_half<BFloat16>(ud);

  Tensor g_cpu = Tensor::empty(Shape({N, D}), DType::BFloat16);
  Tensor u_cpu = Tensor::empty(Shape({N, D}), DType::BFloat16);
  std::memcpy(g_cpu.raw_data(), gd_b.data(), gd_b.size() * sizeof(BFloat16));
  std::memcpy(u_cpu.raw_data(), ud_b.data(), ud_b.size() * sizeof(BFloat16));
  Tensor y_cpu = tesseract::ops::swiglu_silu_gate(g_cpu, u_cpu);

  Tensor g_cuda = g_cpu.to(cuda0());
  Tensor u_cuda = u_cpu.to(cuda0());
  Tensor y_cuda = tesseract::ops::swiglu_silu_gate(g_cuda, u_cuda);
  Tensor y_back = y_cuda.to(cpu_device());

  const BFloat16* cpu_p = y_cpu.data_ptr<BFloat16>();
  const BFloat16* gpu_p = y_back.data_ptr<BFloat16>();
  for (int64_t i = 0; i < N * D; ++i) {
    REQUIRE_THAT(static_cast<double>(static_cast<float>(gpu_p[i])),
                 WithinAbs(static_cast<double>(static_cast<float>(cpu_p[i])), 6e-3));
  }
}

TEST_CASE("ops::swiglu_silu_gate: CUDA vs CPU on realistic FFN shape") {
  if (!cuda_ready()) SKIP("CUDA not available");
  // Real-world Llama shape: B*S = 128 tokens, d_ff = 1024. Confirms
  // the 1-D grid-stride loop behaves on a multi-thousand-block grid.
  const int64_t M = 128, D = 1024;
  std::vector<float> gd(M * D), ud(M * D);
  for (std::size_t i = 0; i < gd.size(); ++i) {
    gd[i] = 0.013f * static_cast<float>((i * 31) % 97) - 0.6f;
    ud[i] = 0.21f + 0.017f * static_cast<float>((i * 19) % 41);
  }
  Tensor g_cpu = from_host_f32(gd, Shape({M, D}));
  Tensor u_cpu = from_host_f32(ud, Shape({M, D}));
  Tensor y_cpu = tesseract::ops::swiglu_silu_gate(g_cpu, u_cpu);

  Tensor g_cuda = g_cpu.to(cuda0());
  Tensor u_cuda = u_cpu.to(cuda0());
  Tensor y_cuda = tesseract::ops::swiglu_silu_gate(g_cuda, u_cuda);
  Tensor y_back = y_cuda.to(cpu_device());

  const float* cpu_p = y_cpu.data_ptr<float>();
  const float* gpu_p = y_back.data_ptr<float>();
  double max_abs = 0.0;
  for (int64_t i = 0; i < M * D; ++i) {
    max_abs = std::max(max_abs, std::abs(static_cast<double>(gpu_p[i]) -
                                         static_cast<double>(cpu_p[i])));
  }
  REQUIRE(max_abs < 1e-5);
}

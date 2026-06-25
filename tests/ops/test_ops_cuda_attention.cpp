// M2J — CPU↔CUDA parity for `ops::attention(q, k, v, mask, causal)`.
// The composite implementation (Q·Kᵀ → softmax → ·V, with Q pre-scaled
// by 1/√d) routes every sub-op through the already-validated CUDA
// bridges landed in M2E/F/G. This test guarantees the composite
// threads through correctly: identical inputs on CPU and CUDA produce
// numerically matching forward + backward outputs under a TF32-aware
// tolerance. The FA3 kernel vendoring + the fused-kernel perf bar land
// in M2L (Hopper-gated) and are not exercised here.

#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Attention.hpp"

using Catch::Matchers::WithinAbs;
using tesseract::cpu_device;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

namespace {

bool cuda_ready() { return has_cuda_support() && is_available(); }
Device cuda0() { return Device{DeviceType::CUDA, 0}; }

std::vector<float> pattern(std::size_t n, float scale, float bias = 0.0f) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = bias + scale * (static_cast<float>(i) - 0.5f * static_cast<float>(n));
  }
  return out;
}

std::vector<float> to_float_vec(const Tensor& t_host) {
  REQUIRE(t_host.device().is_cpu());
  REQUIRE(t_host.dtype() == DType::Float32);
  const float* p = t_host.data_ptr<float>();
  return std::vector<float>(p, p + t_host.numel());
}

void require_close(const std::vector<float>& expected,
                   const std::vector<float>& actual, float tol) {
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE_THAT(actual[i], WithinAbs(expected[i], tol));
  }
}

// Tolerance rationale:
//   cuBLASLt runs FP32 GEMMs on Ada's Tensor Cores in TF32 mode by
//   default, which truncates the mantissa to 10 bits per operand.
//   Errors compound across the two matmuls + softmax, but for the
//   shapes used here (D <= 16, S <= 32) the drift stays < 3e-3
//   absolute on the forward output. Backward gradients share the
//   same envelope since their formula is strict compositions of
//   matmul/softmax_backward.
constexpr float kAttnFwdTol = 3e-3f;
constexpr float kAttnBwdTol = 3e-3f;

}  // namespace

TEST_CASE("CUDA attention rank-4 matches CPU (Float32)",
          "[ops][gpu][attention]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, H = 4, S = 8, D = 16;

  auto q_data = pattern(B * H * S * D, 0.005f);
  auto k_data = pattern(B * H * S * D, 0.008f, 0.1f);
  auto v_data = pattern(B * H * S * D, 0.01f, -0.05f);

  Tensor q_cpu = Tensor::from_vector(q_data, {B, H, S, D});
  Tensor k_cpu = Tensor::from_vector(k_data, {B, H, S, D});
  Tensor v_cpu = Tensor::from_vector(v_data, {B, H, S, D});

  Tensor h_out = tesseract::ops::attention(q_cpu, k_cpu, v_cpu);
  Tensor d_out = tesseract::ops::attention(q_cpu.to(cuda0()),
                                            k_cpu.to(cuda0()),
                                            v_cpu.to(cuda0()));
  REQUIRE(d_out.device() == cuda0());
  REQUIRE(d_out.shape() == Shape({B, H, S, D}));

  require_close(to_float_vec(h_out), to_float_vec(d_out.to(cpu_device())),
                kAttnFwdTol);
}

TEST_CASE("CUDA attention causal mask matches CPU (Float32)",
          "[ops][gpu][attention][causal]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, H = 2, S = 16, D = 8;

  auto q_data = pattern(B * H * S * D, 0.007f);
  auto k_data = pattern(B * H * S * D, -0.003f, 0.2f);
  auto v_data = pattern(B * H * S * D, 0.004f, -0.1f);

  Tensor q_cpu = Tensor::from_vector(q_data, {B, H, S, D});
  Tensor k_cpu = Tensor::from_vector(k_data, {B, H, S, D});
  Tensor v_cpu = Tensor::from_vector(v_data, {B, H, S, D});

  Tensor h_out = tesseract::ops::attention(q_cpu, k_cpu, v_cpu,
                                            /*mask=*/Tensor{},
                                            /*causal=*/true);
  Tensor d_out = tesseract::ops::attention(q_cpu.to(cuda0()),
                                            k_cpu.to(cuda0()),
                                            v_cpu.to(cuda0()),
                                            /*mask=*/Tensor{},
                                            /*causal=*/true);
  REQUIRE(d_out.shape() == Shape({B, H, S, D}));

  require_close(to_float_vec(h_out), to_float_vec(d_out.to(cpu_device())),
                kAttnFwdTol);
}

TEST_CASE("CUDA attention additive mask matches CPU (Float32)",
          "[ops][gpu][attention][mask]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 1, H = 2, S_q = 4, S_k = 6, D = 8;

  auto q_data = pattern(B * H * S_q * D, 0.01f);
  auto k_data = pattern(B * H * S_k * D, 0.02f, 0.1f);
  auto v_data = pattern(B * H * S_k * D, 0.015f, -0.2f);

  // Mask: forbid key position 3 for every query, using a broadcastable
  // [1, 1, 1, S_k] shape (tests that the composite's `ops::add` does
  // the broadcast correctly on CUDA).
  const float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> mask_data(S_k, 0.0f);
  mask_data[3] = neg_inf;

  Tensor q_cpu = Tensor::from_vector(q_data, {B, H, S_q, D});
  Tensor k_cpu = Tensor::from_vector(k_data, {B, H, S_k, D});
  Tensor v_cpu = Tensor::from_vector(v_data, {B, H, S_k, D});
  Tensor m_cpu = Tensor::from_vector(mask_data, {1, 1, 1, S_k});

  Tensor h_out = tesseract::ops::attention(q_cpu, k_cpu, v_cpu, m_cpu);
  Tensor d_out = tesseract::ops::attention(q_cpu.to(cuda0()),
                                            k_cpu.to(cuda0()),
                                            v_cpu.to(cuda0()),
                                            m_cpu.to(cuda0()));
  REQUIRE(d_out.shape() == Shape({B, H, S_q, D}));

  require_close(to_float_vec(h_out), to_float_vec(d_out.to(cpu_device())),
                kAttnFwdTol);
}

TEST_CASE("CUDA attention autograd backward matches CPU (Float32)",
          "[ops][gpu][attention][autograd]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, H = 2, S = 6, D = 8;

  auto q_data = pattern(B * H * S * D, 0.01f);
  auto k_data = pattern(B * H * S * D, 0.02f, 0.05f);
  auto v_data = pattern(B * H * S * D, 0.015f, -0.1f);
  auto g_data = pattern(B * H * S * D, 0.005f);

  // CPU autograd run.
  Tensor hq = Tensor::from_vector(q_data, {B, H, S, D});
  Tensor hk = Tensor::from_vector(k_data, {B, H, S, D});
  Tensor hv = Tensor::from_vector(v_data, {B, H, S, D});
  hq.set_requires_grad(true);
  hk.set_requires_grad(true);
  hv.set_requires_grad(true);
  Tensor hy = tesseract::ops::attention(hq, hk, hv);
  Tensor hg = Tensor::from_vector(g_data, {B, H, S, D});
  tesseract::Engine::backward(hy, hg);

  // CUDA autograd run, identical seed / data.
  Tensor dq = Tensor::from_vector(q_data, {B, H, S, D}).to(cuda0());
  Tensor dk = Tensor::from_vector(k_data, {B, H, S, D}).to(cuda0());
  Tensor dv = Tensor::from_vector(v_data, {B, H, S, D}).to(cuda0());
  dq.set_requires_grad(true);
  dk.set_requires_grad(true);
  dv.set_requires_grad(true);
  Tensor dy = tesseract::ops::attention(dq, dk, dv);
  Tensor dg = Tensor::from_vector(g_data, {B, H, S, D}).to(cuda0());
  tesseract::Engine::backward(dy, dg);

  require_close(to_float_vec(hq.grad()),
                to_float_vec(dq.grad().to(cpu_device())), kAttnBwdTol);
  require_close(to_float_vec(hk.grad()),
                to_float_vec(dk.grad().to(cpu_device())), kAttnBwdTol);
  require_close(to_float_vec(hv.grad()),
                to_float_vec(dv.grad().to(cpu_device())), kAttnBwdTol);
}

TEST_CASE("attention CPU-only smoke (always runs)",
          "[ops][attention]") {
  // Guarantees this TU exercises at least one asserted path in the
  // CPU-only build, matching the pattern used by the M2E/F/G/H CUDA
  // parity TUs.
  const int64_t B = 1, S = 3, D = 4;
  auto q_data = pattern(B * S * D, 0.1f);
  auto k_data = pattern(B * S * D, 0.05f, 0.2f);
  auto v_data = pattern(B * S * D, 0.02f);

  Tensor q = Tensor::from_vector(q_data, {B, S, D});
  Tensor k = Tensor::from_vector(k_data, {B, S, D});
  Tensor v = Tensor::from_vector(v_data, {B, S, D});
  Tensor out = tesseract::ops::attention(q, k, v);
  REQUIRE(out.shape() == Shape({B, S, D}));
  // Each output row is a convex combination of V rows -> bounded by
  // min(V) and max(V) along each feature. Quick sanity bound.
  float vmin = v_data[0], vmax = v_data[0];
  for (float x : v_data) { vmin = std::min(vmin, x); vmax = std::max(vmax, x); }
  const float* p = out.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); ++i) {
    REQUIRE(p[i] >= vmin - 1e-5f);
    REQUIRE(p[i] <= vmax + 1e-5f);
  }
}

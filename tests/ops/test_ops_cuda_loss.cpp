// M2F: CPU↔CUDA parity for cross-entropy (forward + both backward
// entry points — the autograd Node path via `.backward()` and the
// standalone graph-mode `cross_entropy_with_logits_backward`).
//
// Forward tolerance: the CUDA forward uses an `atomicAdd` per row
// into the sum accumulator before the mean-finalize, so the
// summation order is non-deterministic across runs. For N=64 rows
// the observed drift is ≪ 1e-5 on fp32; we use `WithinAbs(..., 1e-5)`
// which gives comfortable headroom.
//
// Backward tolerance: the backward is deterministic (every output
// slot is written by a single thread from a single independent
// read), so we can tighten to `WithinAbs(..., 1e-6)`.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/Engine.hpp"

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Loss.hpp"

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

// Deterministic [N, C] logits pattern: wide magnitude range to
// exercise the stable max-subtract path; both signs. Targets picked
// as `i % C` so every class shows up across the batch.
struct Toy {
  Tensor logits_cpu;
  Tensor targets_cpu;
};

Toy make_toy(int64_t N, int64_t C) {
  std::vector<float> lg(N * C);
  std::vector<int64_t> tg(N);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      lg[n * C + c] = (static_cast<float>(n * C + c)
                       - static_cast<float>(N * C) * 0.5f) * 0.125f;
    }
    tg[n] = n % C;
  }
  Toy t;
  t.logits_cpu  = Tensor::from_vector(lg, {N, C});
  t.targets_cpu = Tensor::from_vector(tg, {N});
  return t;
}

}  // namespace

TEST_CASE("CUDA cross_entropy_with_logits forward matches CPU",
          "[ops][gpu][loss]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 64, C = 10;
  auto t = make_toy(N, C);

  Tensor h_loss = tesseract::ops::cross_entropy_with_logits(
      t.logits_cpu, t.targets_cpu);
  Tensor d_logits  = t.logits_cpu.to(cuda0());
  Tensor d_targets = t.targets_cpu.to(cuda0());
  Tensor d_loss    = tesseract::ops::cross_entropy_with_logits(
      d_logits, d_targets);
  REQUIRE(d_loss.device() == cuda0());
  REQUIRE(d_loss.shape().rank() == 0);

  Tensor back = d_loss.to(cpu_device());
  REQUIRE_THAT(*back.data_ptr<float>(),
               WithinAbs(*h_loss.data_ptr<float>(), 1e-5f));
}

TEST_CASE("CUDA cross_entropy standalone backward matches CPU",
          "[ops][gpu][loss]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 32, C = 7;
  auto t = make_toy(N, C);

  // 0-D grad scalar with magnitude != 1 so the gs scaling actually
  // propagates — catches a forgotten `* gs` in the kernel.
  Tensor g_cpu = Tensor::from_vector(std::vector<float>{0.5f}, {});
  Tensor h_dlogits = tesseract::ops::cross_entropy_with_logits_backward(
      t.logits_cpu, t.targets_cpu, g_cpu);

  Tensor d_logits  = t.logits_cpu.to(cuda0());
  Tensor d_targets = t.targets_cpu.to(cuda0());
  Tensor g_cuda    = g_cpu.to(cuda0());
  Tensor d_dlogits = tesseract::ops::cross_entropy_with_logits_backward(
      d_logits, d_targets, g_cuda);
  REQUIRE(d_dlogits.device() == cuda0());
  REQUIRE(d_dlogits.shape() == h_dlogits.shape());

  Tensor back = d_dlogits.to(cpu_device());
  const float* e = h_dlogits.data_ptr<float>();
  const float* a = back.data_ptr<float>();
  for (int64_t i = 0; i < back.numel(); ++i) {
    REQUIRE_THAT(a[i], WithinAbs(e[i], 1e-6f));
  }
}

TEST_CASE("CUDA cross_entropy autograd backward matches CPU",
          "[ops][gpu][loss]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 16, C = 5;
  auto t = make_toy(N, C);

  // CPU autograd run.
  Tensor h_logits = t.logits_cpu.clone();
  h_logits.set_requires_grad(true);
  Tensor h_loss = tesseract::ops::cross_entropy_with_logits(
      h_logits, t.targets_cpu);
  tesseract::Engine::backward(h_loss);
  Tensor h_g = h_logits.grad();

  // CUDA autograd run.
  Tensor d_logits  = t.logits_cpu.to(cuda0());
  d_logits.set_requires_grad(true);
  Tensor d_targets = t.targets_cpu.to(cuda0());
  Tensor d_loss = tesseract::ops::cross_entropy_with_logits(
      d_logits, d_targets);
  tesseract::Engine::backward(d_loss);
  Tensor d_g = d_logits.grad();
  REQUIRE(d_g.device() == cuda0());
  REQUIRE(d_g.shape() == h_g.shape());

  Tensor back = d_g.to(cpu_device());
  const float* e = h_g.data_ptr<float>();
  const float* a = back.data_ptr<float>();
  for (int64_t i = 0; i < back.numel(); ++i) {
    REQUIRE_THAT(a[i], WithinAbs(e[i], 1e-6f));
  }
}

TEST_CASE("CUDA cross_entropy forward on Float64 matches CPU",
          "[ops][gpu][loss]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t N = 16, C = 8;
  std::vector<double> lg(N * C);
  std::vector<int64_t> tg(N);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      lg[n * C + c] = (static_cast<double>(n * C + c)
                       - static_cast<double>(N * C) * 0.5) * 0.125;
    }
    tg[n] = n % C;
  }
  Tensor h_logits  = Tensor::from_vector(lg, {N, C});
  Tensor h_targets = Tensor::from_vector(tg, {N});
  Tensor h_loss    = tesseract::ops::cross_entropy_with_logits(
      h_logits, h_targets);

  Tensor d_logits  = h_logits.to(cuda0());
  Tensor d_targets = h_targets.to(cuda0());
  Tensor d_loss    = tesseract::ops::cross_entropy_with_logits(
      d_logits, d_targets).to(cpu_device());
  REQUIRE_THAT(*d_loss.data_ptr<double>(),
               WithinAbs(*h_loss.data_ptr<double>(), 1e-12));
}

TEST_CASE("CPU-only cross_entropy smoke", "[ops][cpu][loss]") {
  // Always-running sanity check: argmax-aligned target should give a
  // loss well below the log(C) "random guessing" baseline.
  const int64_t N = 4, C = 3;
  std::vector<float> lg(N * C, 0.0f);
  std::vector<int64_t> tg(N);
  for (int64_t n = 0; n < N; ++n) {
    tg[n] = n % C;
    lg[n * C + tg[n]] = 5.0f;  // big logit on the correct class
  }
  Tensor logits  = Tensor::from_vector(lg, {N, C});
  Tensor targets = Tensor::from_vector(tg, {N});
  Tensor loss    = tesseract::ops::cross_entropy_with_logits(logits, targets);
  REQUIRE(loss.shape().rank() == 0);
  REQUIRE(*loss.data_ptr<float>() < std::log(static_cast<float>(C)));
  REQUIRE(*loss.data_ptr<float>() > 0.0f);
}

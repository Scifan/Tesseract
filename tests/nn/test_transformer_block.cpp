// M2K — single-layer Llama-style transformer block.
//
// Covers three axes of confidence:
//   1. Shape / finiteness / numerical bounds on both forward and
//      autograd backward on the CPU (always-on path).
//   2. CPU↔CUDA parity for forward and backward under a TF32-aware
//      tolerance (matches the M2G / M2J tolerance envelope).
//   3. Building and tearing down the block through `Module::to(...)`
//      reaches every registered sub-module and parameter correctly
//      (regression guard for the M2I migration contract).
//
// The reference is *this same block running on CPU* — we do **not**
// maintain an external PyTorch-captured `.npz` at this milestone.
// Rationale: hermetic C++ tests avoid a second toolchain, and the
// CPU path is itself a composition of already-gradcheck'd primitives
// (rms_norm, attention, Linear, SwiGLU). The external reference bar
// moves to `M2L` alongside the fused attention kernel, where the
// comparison tightens up to bit-parity for FP16/BF16.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/nn/TransformerBlock.hpp"
#include "tesseract/ops/Reduction.hpp"

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

std::vector<float> gaussian(int64_t n, uint64_t seed, float stddev = 1.0f) {
  std::vector<float> v(static_cast<std::size_t>(n));
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.0f, stddev);
  for (auto& x : v) x = nd(rng);
  return v;
}

std::vector<float> host_vec(const Tensor& t_dev) {
  Tensor t = t_dev.device().is_cpu() ? t_dev : t_dev.to(cpu_device());
  REQUIRE(t.dtype() == DType::Float32);
  const float* p = t.data_ptr<float>();
  return std::vector<float>(p, p + t.numel());
}

// Byte-level copy from one float tensor into another, preserving shape
// and dtype. Used to clone the CPU-initialized parameter values into
// the CUDA block so both training runs share identical starting
// weights (same trick as `test_cuda_mnist_parity`). Both tensors live
// on CPU at call time; the CUDA block does a `.to(cuda)` afterwards.
void copy_param_bytes(Tensor& dst_cpu, const Tensor& src_cpu) {
  REQUIRE(dst_cpu.device().is_cpu());
  REQUIRE(src_cpu.device().is_cpu());
  REQUIRE(dst_cpu.dtype() == src_cpu.dtype());
  REQUIRE(dst_cpu.shape() == src_cpu.shape());
  const std::size_t bytes =
      static_cast<std::size_t>(dst_cpu.numel()) * dst_cpu.itemsize();
  std::memcpy(dst_cpu.raw_data(), src_cpu.raw_data(), bytes);
}

// Same tolerance envelope as `test_ops_cuda_attention`: the transformer
// block runs the composite attention + two linear projections + two
// RMSNorms under cuBLASLt TF32 mode on Ada, and errors compound
// monotonically with depth. Empirically the block-level drift with the
// shapes used below stays < 5e-3 on forward, < 6e-3 on backward.
constexpr float kFwdTol = 5e-3f;
constexpr float kBwdTol = 6e-3f;

}  // namespace

TEST_CASE("TransformerBlock CPU forward: shapes + finite",
          "[nn][transformer]") {
  const int64_t B = 2, S = 8, D = 32, H = 4, Dff = 64;
  auto block = std::make_shared<tesseract::nn::TransformerBlock>(
      D, H, Dff, /*norm_eps=*/1e-5, /*causal=*/true, /*use_bias=*/false);

  auto x_data = gaussian(B * S * D, /*seed=*/42);
  Tensor x = Tensor::from_vector(x_data, {B, S, D});

  Tensor out = block->forward(x);
  REQUIRE(out.shape() == Shape({B, S, D}));
  REQUIRE(out.dtype() == DType::Float32);
  REQUIRE(out.device().is_cpu());

  const float* po = out.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); ++i) {
    REQUIRE(std::isfinite(po[i]));
  }
}

TEST_CASE("TransformerBlock CPU backward: shapes + finite grads",
          "[nn][transformer][autograd]") {
  const int64_t B = 2, S = 6, D = 16, H = 4, Dff = 32;
  auto block = std::make_shared<tesseract::nn::TransformerBlock>(
      D, H, Dff, /*norm_eps=*/1e-5, /*causal=*/true, /*use_bias=*/false);

  auto x_data = gaussian(B * S * D, /*seed=*/77);
  Tensor x = Tensor::from_vector(x_data, {B, S, D});
  x.set_requires_grad(true);

  Tensor out = block->forward(x);
  Tensor loss = tesseract::ops::sum(out);
  tesseract::Engine::backward(loss);

  // Input grad shape + finiteness.
  REQUIRE(x.grad().shape() == x.shape());
  const float* pg = x.grad().data_ptr<float>();
  for (int64_t i = 0; i < x.grad().numel(); ++i) {
    REQUIRE(std::isfinite(pg[i]));
  }

  // Every registered parameter must have a gradient of matching shape,
  // and every gradient element must be finite. A module with >= 1
  // parameter must land here since SwiGLU contributes 3 projections,
  // MHA 4 projections, RMSNorm 2 weights — so we expect at least 9
  // parameters in total.
  const auto params = block->parameters();
  REQUIRE(params.size() >= 9);
  int64_t total_param_elems = 0;
  for (const auto& p : params) {
    REQUIRE(p.grad().defined());
    REQUIRE(p.grad().shape() == p.shape());
    const float* pgp = p.grad().data_ptr<float>();
    for (int64_t i = 0; i < p.grad().numel(); ++i) {
      REQUIRE(std::isfinite(pgp[i]));
    }
    total_param_elems += p.grad().numel();
  }
  REQUIRE(total_param_elems > 0);
}

TEST_CASE("TransformerBlock CUDA forward matches CPU (Float32)",
          "[nn][gpu][transformer]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, S = 8, D = 32, H = 4, Dff = 64;

  auto cpu_block = std::make_shared<tesseract::nn::TransformerBlock>(
      D, H, Dff, /*norm_eps=*/1e-5, /*causal=*/true, /*use_bias=*/false);
  auto cuda_block = std::make_shared<tesseract::nn::TransformerBlock>(
      D, H, Dff, /*norm_eps=*/1e-5, /*causal=*/true, /*use_bias=*/false);

  // Copy CPU-initialized params into the CUDA-bound block so both runs
  // start from the same weights. The `parameters()` iteration order is
  // deterministic (insertion order through Module::register_*), so a
  // pairwise copy is sufficient. Copy happens while everything is still
  // on CPU; the `.to(cuda)` call below migrates the CUDA block's
  // leaves into device memory in one pass.
  {
    auto cpu_params = cpu_block->parameters();
    auto cuda_params = cuda_block->parameters();
    REQUIRE(cpu_params.size() == cuda_params.size());
    for (std::size_t i = 0; i < cpu_params.size(); ++i) {
      copy_param_bytes(cuda_params[i], cpu_params[i]);
    }
  }
  cuda_block->to(cuda0());

  auto x_data = gaussian(B * S * D, /*seed=*/123);
  Tensor x_cpu  = Tensor::from_vector(x_data, {B, S, D});
  Tensor x_cuda = Tensor::from_vector(x_data, {B, S, D}).to(cuda0());

  Tensor out_cpu  = cpu_block->forward(x_cpu);
  Tensor out_cuda = cuda_block->forward(x_cuda);
  REQUIRE(out_cuda.device() == cuda0());
  REQUIRE(out_cuda.shape() == Shape({B, S, D}));

  const auto ref = host_vec(out_cpu);
  const auto got = host_vec(out_cuda);
  REQUIRE(ref.size() == got.size());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(got[i], WithinAbs(ref[i], kFwdTol));
  }
}

TEST_CASE("TransformerBlock CUDA backward matches CPU (Float32)",
          "[nn][gpu][transformer][autograd]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, S = 6, D = 16, H = 4, Dff = 32;

  auto cpu_block = std::make_shared<tesseract::nn::TransformerBlock>(
      D, H, Dff, /*norm_eps=*/1e-5, /*causal=*/true, /*use_bias=*/false);
  auto cuda_block = std::make_shared<tesseract::nn::TransformerBlock>(
      D, H, Dff, /*norm_eps=*/1e-5, /*causal=*/true, /*use_bias=*/false);

  {
    auto cpu_params = cpu_block->parameters();
    auto cuda_params = cuda_block->parameters();
    REQUIRE(cpu_params.size() == cuda_params.size());
    for (std::size_t i = 0; i < cpu_params.size(); ++i) {
      copy_param_bytes(cuda_params[i], cpu_params[i]);
    }
  }
  cuda_block->to(cuda0());

  auto x_data = gaussian(B * S * D, /*seed=*/321);
  Tensor x_cpu  = Tensor::from_vector(x_data, {B, S, D});
  Tensor x_cuda = Tensor::from_vector(x_data, {B, S, D}).to(cuda0());
  x_cpu.set_requires_grad(true);
  x_cuda.set_requires_grad(true);

  // Drive both with the same scalar loss (sum). Autograd seeds a
  // 1.0 cotangent against the `sum` output, identically on both
  // devices, so the resulting grads are directly comparable.
  Tensor out_cpu  = cpu_block->forward(x_cpu);
  Tensor out_cuda = cuda_block->forward(x_cuda);
  tesseract::Engine::backward(tesseract::ops::sum(out_cpu));
  tesseract::Engine::backward(tesseract::ops::sum(out_cuda));

  // Input grads first.
  const auto gref = host_vec(x_cpu.grad());
  const auto ggot = host_vec(x_cuda.grad());
  REQUIRE(gref.size() == ggot.size());
  for (std::size_t i = 0; i < gref.size(); ++i) {
    REQUIRE_THAT(ggot[i], WithinAbs(gref[i], kBwdTol));
  }

  // Per-parameter grads pair-wise (insertion order matches since both
  // blocks were constructed with the same config). Each `.grad()`
  // lives on the same device as its parameter, which we bounce back
  // to CPU for the comparison.
  auto cpu_params = cpu_block->parameters();
  auto cuda_params = cuda_block->parameters();
  REQUIRE(cpu_params.size() == cuda_params.size());
  for (std::size_t i = 0; i < cpu_params.size(); ++i) {
    const auto pref = host_vec(cpu_params[i].grad());
    const auto pgot = host_vec(cuda_params[i].grad());
    REQUIRE(pref.size() == pgot.size());
    for (std::size_t j = 0; j < pref.size(); ++j) {
      REQUIRE_THAT(pgot[j], WithinAbs(pref[j], kBwdTol));
    }
  }
}

// B-014: RoPE-on variant of the CPU↔CUDA forward/backward parity
// test. Threads `rope_base=10000` and a generous `rope_max_seq`
// through both blocks, verifies the forward + backward match to the
// same envelope as the no-RoPE variants, and confirms the cached
// cos/sin buffers migrate alongside the Linear weights under
// `Module::to(cuda)`.
TEST_CASE("TransformerBlock with RoPE: CPU↔CUDA parity",
          "[nn][gpu][transformer][rope]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, S = 8, D = 32, H = 4, Dff = 64;
  const double rope_base = 10000.0;
  const int64_t rope_max_seq = 64;

  auto cpu_block = std::make_shared<tesseract::nn::TransformerBlock>(
      D, H, Dff, /*norm_eps=*/1e-5, /*causal=*/true, /*use_bias=*/false,
      DType::Float32, rope_base, rope_max_seq);
  auto cuda_block = std::make_shared<tesseract::nn::TransformerBlock>(
      D, H, Dff, /*norm_eps=*/1e-5, /*causal=*/true, /*use_bias=*/false,
      DType::Float32, rope_base, rope_max_seq);

  {
    auto cpu_params = cpu_block->parameters();
    auto cuda_params = cuda_block->parameters();
    // RoPE adds no trainable parameters — only buffers — so the
    // parameter-count contract is unchanged vs the no-RoPE block.
    REQUIRE(cpu_params.size() == cuda_params.size());
    for (std::size_t i = 0; i < cpu_params.size(); ++i) {
      copy_param_bytes(cuda_params[i], cpu_params[i]);
    }
  }
  cuda_block->to(cuda0());

  auto x_data = gaussian(B * S * D, /*seed=*/555);
  Tensor x_cpu  = Tensor::from_vector(x_data, {B, S, D});
  Tensor x_cuda = Tensor::from_vector(x_data, {B, S, D}).to(cuda0());
  x_cpu.set_requires_grad(true);
  x_cuda.set_requires_grad(true);

  Tensor out_cpu  = cpu_block->forward(x_cpu);
  Tensor out_cuda = cuda_block->forward(x_cuda);
  REQUIRE(out_cuda.shape() == Shape({B, S, D}));

  // Forward parity.
  {
    const auto ref = host_vec(out_cpu);
    const auto got = host_vec(out_cuda);
    REQUIRE(ref.size() == got.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
      REQUIRE_THAT(got[i], WithinAbs(ref[i], kFwdTol));
    }
  }

  // Backward parity on input grads.
  tesseract::Engine::backward(tesseract::ops::sum(out_cpu));
  tesseract::Engine::backward(tesseract::ops::sum(out_cuda));
  {
    const auto gref = host_vec(x_cpu.grad());
    const auto ggot = host_vec(x_cuda.grad());
    REQUIRE(gref.size() == ggot.size());
    for (std::size_t i = 0; i < gref.size(); ++i) {
      REQUIRE_THAT(ggot[i], WithinAbs(gref[i], kBwdTol));
    }
  }

  // Per-parameter grad parity.
  auto cpu_params = cpu_block->parameters();
  auto cuda_params = cuda_block->parameters();
  REQUIRE(cpu_params.size() == cuda_params.size());
  for (std::size_t i = 0; i < cpu_params.size(); ++i) {
    const auto pref = host_vec(cpu_params[i].grad());
    const auto pgot = host_vec(cuda_params[i].grad());
    REQUIRE(pref.size() == pgot.size());
    for (std::size_t j = 0; j < pref.size(); ++j) {
      REQUIRE_THAT(pgot[j], WithinAbs(pref[j], kBwdTol));
    }
  }
}

TEST_CASE("TransformerBlock parameter migration covers every leaf",
          "[nn][transformer]") {
  // Regression guard for the `Module::to(Device)` contract — after
  // the migration, every parameter must report the target device.
  // This catches sub-module-registration bugs (e.g. forgetting to
  // `register_module(...)` a child), which would otherwise silently
  // leave some weights on CPU and cause the CUDA forward to throw
  // with a device-mismatch error.
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t D = 16, H = 4, Dff = 32;
  auto block = std::make_shared<tesseract::nn::TransformerBlock>(
      D, H, Dff, /*norm_eps=*/1e-5, /*causal=*/true, /*use_bias=*/true);
  block->to(cuda0());
  for (const auto& p : block->parameters()) {
    REQUIRE(p.device() == cuda0());
  }
  block->to(cpu_device());
  for (const auto& p : block->parameters()) {
    REQUIRE(p.device() == cpu_device());
  }
}

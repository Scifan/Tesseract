// M4 Track A2 (B-039) — selective state-space scan + Mamba block + model.
//
// The defining correctness property of an SSM is recurrent ≡ parallel: a full-
// sequence (prefill) scan must equal stepping one token at a time while
// threading the hidden state. We pin it at three levels:
//   1. ops::selective_scan: full-L call == per-token loop threading state.
//   2. nn::Mamba block: forward(full) == forward_step loop over an SSMStateCache.
//   3. models::MambaModel: forward(full) last logits == recurrent stepping;
//      plus generate determinism + length.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/models/MambaModel.hpp"
#include "tesseract/nn/Mamba.hpp"
#include "tesseract/nn/SSMStateCache.hpp"
#include "tesseract/ops/Indexing.hpp"
#include "tesseract/ops/SelectiveScan.hpp"

using namespace tesseract;

namespace {

Tensor randf(Shape shape, uint64_t seed, float lo = -1.0f, float hi = 1.0f) {
  Tensor t = Tensor::empty(shape, DType::Float32);
  float* p = t.data_ptr<float>();
  uint64_t s = seed;
  for (int64_t i = 0; i < t.numel(); ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const float u = static_cast<float>((s >> 33) % 100000) / 100000.0f;
    p[i] = lo + u * (hi - lo);
  }
  return t;
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
  REQUIRE(a.numel() == b.numel());
  const float* pa = a.data_ptr<float>();
  const float* pb = b.data_ptr<float>();
  double m = 0.0;
  for (int64_t i = 0; i < a.numel(); ++i)
    m = std::max(m, std::abs(static_cast<double>(pa[i]) - pb[i]));
  return m;
}

Tensor slice_t(const Tensor& x, int64_t t) {
  return x.narrow(1, t, 1).contiguous();
}

}  // namespace

TEST_CASE("selective_scan: recurrent stepping equals full-sequence scan") {
  const int64_t B = 2, L = 6, D = 5, N = 4;
  Tensor u     = randf(Shape({B, L, D}), 0x11);
  Tensor delta = randf(Shape({B, L, D}), 0x22, 0.05f, 1.0f);  // > 0
  Tensor A     = randf(Shape({D, N}), 0x33, -1.5f, -0.1f);    // < 0
  Tensor Bm    = randf(Shape({B, L, N}), 0x44);
  Tensor Cm    = randf(Shape({B, L, N}), 0x55);
  Tensor Dsk   = randf(Shape({D}), 0x66);

  auto full = ops::selective_scan(u, delta, A, Bm, Cm, Dsk);
  REQUIRE(full.y.shape() == Shape({B, L, D}));
  REQUIRE(full.state.shape() == Shape({B, D, N}));

  std::vector<Tensor> ys;
  Tensor state;  // undefined ⇒ zero init for the first step
  for (int64_t t = 0; t < L; ++t) {
    auto step = ops::selective_scan(slice_t(u, t), slice_t(delta, t), A,
                                    slice_t(Bm, t), slice_t(Cm, t), Dsk, state);
    state = step.state;
    ys.push_back(step.y);
  }
  Tensor y_rec = ops::cat(ys, 1);
  REQUIRE(max_abs_diff(full.y, y_rec) < 1e-5);
  REQUIRE(max_abs_diff(full.state, state) < 1e-5);
}

// Phase 5: the CUDA selective-scan kernel must match the CPU reference, both
// for the full prefill scan and for state-threaded decode steps. Sweep a few
// (B,L,D,N) so the per-thread register-state path and the t-recurrence are
// exercised on device.
TEST_CASE("selective_scan: CUDA matches CPU (prefill + decode)",
          "[nn][mamba][ssm][cuda]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  const Device cuda0{DeviceType::CUDA, 0};
  struct Cfg { int64_t B, L, D, N; };
  // L=512/1000 exercise the chunkwise parallel scan (C>=2, chunk=128); the
  // small-L cases stay on the sequential per-thread path.
  for (const Cfg c : {Cfg{2, 16, 8, 4}, Cfg{1, 64, 16, 16}, Cfg{3, 33, 5, 8},
                      Cfg{2, 512, 8, 16}, Cfg{1, 1000, 6, 8}}) {
    Tensor u     = randf(Shape({c.B, c.L, c.D}), 0x11 + c.L);
    Tensor delta = randf(Shape({c.B, c.L, c.D}), 0x22 + c.L, 0.05f, 1.0f);
    Tensor A     = randf(Shape({c.D, c.N}), 0x33 + c.N, -1.5f, -0.1f);
    Tensor Bm    = randf(Shape({c.B, c.L, c.N}), 0x44);
    Tensor Cm    = randf(Shape({c.B, c.L, c.N}), 0x55);
    Tensor Dsk   = randf(Shape({c.D}), 0x66);

    auto cpu = ops::selective_scan(u, delta, A, Bm, Cm, Dsk);
    auto gpu = ops::selective_scan(u.to(cuda0), delta.to(cuda0), A.to(cuda0),
                                   Bm.to(cuda0), Cm.to(cuda0), Dsk.to(cuda0));
    INFO("B=" << c.B << " L=" << c.L << " D=" << c.D << " N=" << c.N);
    REQUIRE(max_abs_diff(cpu.y, gpu.y.to(cpu_device())) < 1e-4);
    REQUIRE(max_abs_diff(cpu.state, gpu.state.to(cpu_device())) < 1e-4);
  }
}

// Phase 5: the full Mamba block (causal conv1d kernel + selective scan +
// projections) on CUDA must match the CPU reference end-to-end.
TEST_CASE("Mamba block: CUDA forward matches CPU", "[nn][mamba][cuda]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  const Device cuda0{DeviceType::CUDA, 0};
  const int64_t B = 2, L = 24, Dm = 32;
  nn::Mamba block(Dm, /*d_state=*/16, /*d_conv=*/4, /*expand=*/2);
  Tensor x = randf(Shape({B, L, Dm}), 0xC0FFEE);

  Tensor y_cpu = block.forward(x).contiguous();
  block.to(cuda0);
  Tensor y_gpu = block.forward(x.to(cuda0)).to(cpu_device()).contiguous();

  REQUIRE(y_gpu.shape() == y_cpu.shape());
  REQUIRE(max_abs_diff(y_cpu, y_gpu) < 1e-4);
}

TEST_CASE("selective_scan: validation rejects shape mismatches") {
  Tensor u     = randf(Shape({1, 3, 4}), 1);
  Tensor delta = randf(Shape({1, 3, 4}), 2, 0.1f, 1.0f);
  Tensor A     = randf(Shape({4, 2}), 3, -1.0f, -0.1f);
  Tensor Bm    = randf(Shape({1, 3, 2}), 4);
  Tensor Cm    = randf(Shape({1, 3, 2}), 5);
  Tensor Dsk   = randf(Shape({4}), 6);
  Tensor bad_A = randf(Shape({3, 2}), 7, -1.0f, -0.1f);  // D mismatch
  REQUIRE_THROWS(ops::selective_scan(u, delta, bad_A, Bm, Cm, Dsk));
}

TEST_CASE("Mamba block: forward_step loop equals full forward") {
  const int64_t L = 7;
  nn::Mamba block(/*d_model=*/8, /*d_state=*/4, /*d_conv=*/3, /*expand=*/2);
  Tensor x = randf(Shape({1, L, 8}), 0xABC);

  Tensor y_full = block.forward(x);
  REQUIRE(y_full.shape() == Shape({1, L, 8}));

  auto cache = block.make_state_cache(1);
  std::vector<Tensor> ys;
  for (int64_t t = 0; t < L; ++t)
    ys.push_back(block.forward_step(slice_t(x, t), cache));
  Tensor y_rec = ops::cat(ys, 1);
  REQUIRE(max_abs_diff(y_full, y_rec) < 1e-4);
}

TEST_CASE("Mamba block: d_conv==1 degenerate path works") {
  const int64_t L = 4;
  nn::Mamba block(/*d_model=*/6, /*d_state=*/3, /*d_conv=*/1, /*expand=*/2);
  Tensor x = randf(Shape({1, L, 6}), 0xDEF);
  Tensor y_full = block.forward(x);
  auto cache = block.make_state_cache(1);
  std::vector<Tensor> ys;
  for (int64_t t = 0; t < L; ++t)
    ys.push_back(block.forward_step(slice_t(x, t), cache));
  REQUIRE(max_abs_diff(y_full, ops::cat(ys, 1)) < 1e-4);
}

TEST_CASE("MambaModel: stepping matches full forward, generate is deterministic") {
  models::MambaConfig cfg;
  cfg.vocab_size = 24;
  cfg.hidden_size = 16;
  cfg.num_hidden_layers = 3;
  cfg.d_state = 8;
  cfg.d_conv = 4;
  cfg.expand = 2;

  models::MambaModel model(cfg);

  std::vector<int32_t> prompt = {1, 4, 9, 2, 7};
  const int64_t S = static_cast<int64_t>(prompt.size());

  // Full forward last-position logits.
  Tensor toks = Tensor::empty(Shape({1, S}), DType::Int64);
  for (int64_t i = 0; i < S; ++i) toks.data_ptr<int64_t>()[i] = prompt[i];
  Tensor full_logits = model.forward(toks);  // [1, S, V]
  Tensor full_last = full_logits.narrow(1, S - 1, 1).contiguous();

  // Recurrent stepping last-position logits.
  auto caches = model.make_state_caches(1);
  Tensor step_logits;
  for (int32_t id : prompt) {
    Tensor t = Tensor::empty(Shape({1, 1}), DType::Int64);
    t.data_ptr<int64_t>()[0] = id;
    step_logits = model.forward_step(t, caches);
  }
  REQUIRE(max_abs_diff(full_last, step_logits) < 1e-4);

  models::MambaModel::GenerateConfig gc;
  gc.max_new_tokens = 8;
  auto a = model.generate(prompt, gc);
  auto b = model.generate(prompt, gc);
  REQUIRE(a == b);
  REQUIRE(a.size() == prompt.size() + 8);
}

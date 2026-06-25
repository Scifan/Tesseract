// Wave 5 (B-027) — end-to-end autoregressive generation tests for
// `tesseract::models::LlamaModel`.
//
// The decode path (`forward_step` + per-layer `KVCache`) must produce the
// *same* logits as the one-shot `forward()` on the same prefix — that is
// the correctness invariant that makes incremental decode sound. We check
// it two ways:
//
//   1. Chunked prefill: `forward_step(tokens[1, S], caches)` over the whole
//      prompt equals `forward(tokens[1, S])` at every position.
//   2. Token-by-token: feeding one token at a time and collecting each
//      step's last-position logits reconstructs `forward()`'s logits row
//      for row.
//
// Plus greedy `generate()` behavior: determinism, prompt prefix, length,
// and early EOS stop. CPU + CUDA (CUDA case skips cleanly without a device).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/utils/Logging.hpp"

using namespace tesseract;
using tesseract::models::LlamaConfig;
using tesseract::models::LlamaModel;

namespace {

LlamaConfig tiny_config() {
  LlamaConfig c;
  c.vocab_size               = 48;
  c.hidden_size              = 32;
  c.num_hidden_layers        = 2;
  c.num_attention_heads      = 4;
  c.intermediate_size        = 64;
  c.max_position_embeddings  = 64;
  c.rope_theta               = 10000.0;
  c.rms_norm_eps             = 1e-5;
  c.tie_word_embeddings      = false;
  c.dtype                    = DType::Float32;
  return c;
}

Tensor make_tokens(const std::vector<int32_t>& ids, Device dev) {
  Tensor t = Tensor::empty({1, static_cast<int64_t>(ids.size())}, DType::Int64,
                           cpu_device());
  int64_t* p = t.data_ptr<int64_t>();
  for (std::size_t i = 0; i < ids.size(); ++i) p[i] = ids[i];
  return dev.is_cpu() ? t : t.to(dev);
}

// Max abs diff between a single logits row (b=0, position `s`) of two
// [1, S, V] tensors (possibly on different S — `a_s` / `b_s` index each).
float row_max_abs_diff(const Tensor& a, int64_t a_s, const Tensor& b, int64_t b_s) {
  const Tensor ah = a.device().is_cpu() ? a.contiguous() : a.to(cpu_device()).contiguous();
  const Tensor bh = b.device().is_cpu() ? b.contiguous() : b.to(cpu_device()).contiguous();
  const int64_t V = ah.shape()[2];
  REQUIRE(bh.shape()[2] == V);
  const float* pa = ah.data_ptr<float>() + a_s * V;
  const float* pb = bh.data_ptr<float>() + b_s * V;
  float m = 0.0f;
  for (int64_t v = 0; v < V; ++v) m = std::max(m, std::abs(pa[v] - pb[v]));
  return m;
}

void run_parity(Device dev) {
  const LlamaConfig cfg = tiny_config();
  auto model = std::make_shared<LlamaModel>(cfg);
  model->to(dev);

  const std::vector<int32_t> prompt = {3, 17, 42, 8, 25, 1, 39};
  const int64_t S = static_cast<int64_t>(prompt.size());
  const Tensor tokens = make_tokens(prompt, dev);

  // One-shot reference.
  const Tensor full = model->forward(tokens);  // [1, S, V]
  REQUIRE(full.shape()[1] == S);

  // 1) Chunked-prefill parity: one forward_step over the whole prompt.
  {
    auto caches = model->make_kv_caches(/*batch=*/1, /*max_len=*/S);
    const Tensor chunk = model->forward_step(tokens, caches);  // [1, S, V]
    REQUIRE(chunk.shape()[1] == S);
    for (int64_t s = 0; s < S; ++s) {
      INFO("chunk parity position " << s);
      REQUIRE(row_max_abs_diff(full, s, chunk, s) < 2e-4f);
    }
  }

  // 2) Token-by-token parity: each step's last logits == forward()'s row.
  {
    auto caches = model->make_kv_caches(/*batch=*/1, /*max_len=*/S);
    for (int64_t s = 0; s < S; ++s) {
      const Tensor step = model->forward_step(
          make_tokens({prompt[static_cast<std::size_t>(s)]}, dev), caches);
      REQUIRE(step.shape()[1] == 1);
      INFO("token-by-token parity position " << s);
      REQUIRE(row_max_abs_diff(full, s, step, 0) < 2e-4f);
    }
  }
}

}  // namespace

TEST_CASE("LlamaModel::forward_step matches one-shot forward (CPU)",
          "[models][llama][generate]") {
  run_parity(cpu_device());
}

TEST_CASE("LlamaModel::forward_step matches one-shot forward (CUDA)",
          "[models][llama][generate][cuda]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  run_parity(Device{DeviceType::CUDA, 0});
}

TEST_CASE("LlamaModel::generate is greedy, deterministic, prompt-prefixed",
          "[models][llama][generate]") {
  const LlamaConfig cfg = tiny_config();
  auto model = std::make_shared<LlamaModel>(cfg);

  const std::vector<int32_t> prompt = {5, 11, 30};
  LlamaModel::GenerateConfig gc;
  gc.max_new_tokens = 6;
  gc.eos_token_id = -1;

  const auto out1 = model->generate(prompt, gc);
  const auto out2 = model->generate(prompt, gc);

  // Full sequence = prompt + max_new_tokens; greedy ⇒ deterministic.
  REQUIRE(out1.size() == prompt.size() + 6);
  REQUIRE(out1 == out2);
  REQUIRE(std::equal(prompt.begin(), prompt.end(), out1.begin()));
  for (int32_t id : out1) {
    REQUIRE(id >= 0);
    REQUIRE(id < cfg.vocab_size);
  }
}

TEST_CASE("LlamaModel::generate stops early at EOS", "[models][llama][generate]") {
  const LlamaConfig cfg = tiny_config();
  auto model = std::make_shared<LlamaModel>(cfg);

  const std::vector<int32_t> prompt = {7, 2, 19, 33};

  // First, learn the greedy first token with no EOS.
  LlamaModel::GenerateConfig no_eos;
  no_eos.max_new_tokens = 1;
  const auto probe = model->generate(prompt, no_eos);
  REQUIRE(probe.size() == prompt.size() + 1);
  const int32_t first_new = probe.back();

  // Now set that exact token as EOS with a generous budget: generation
  // must emit it once and stop immediately.
  LlamaModel::GenerateConfig with_eos;
  with_eos.max_new_tokens = 20;
  with_eos.eos_token_id = first_new;
  const auto stopped = model->generate(prompt, with_eos);
  REQUIRE(stopped.size() == prompt.size() + 1);
  REQUIRE(stopped.back() == first_new);
}

TEST_CASE("LlamaModel::generate rejects empty prompt + out-of-range ids",
          "[models][llama][generate]") {
  const LlamaConfig cfg = tiny_config();
  auto model = std::make_shared<LlamaModel>(cfg);
  LlamaModel::GenerateConfig gc;
  gc.max_new_tokens = 1;
  REQUIRE_THROWS(model->generate({}, gc));
  // id == vocab_size is out of range [0, vocab_size).
  REQUIRE_THROWS(model->generate({static_cast<int32_t>(cfg.vocab_size)}, gc));
}

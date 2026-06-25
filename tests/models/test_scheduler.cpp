// Wave 7 (B-029) — continuous-batching scheduler.
//
// The scheduler multiplexes many requests over a shared paged KV pool
// with dynamic admission, per-request sampling + EOS, and block
// reclamation. Its correctness contract is strong and exact: each
// request's output must be **identical** to running
// `LlamaModel::generate` for that request standalone — regardless of how
// requests are interleaved or how tight the block budget is. Paged and
// contiguous caches store the same bytes and the gather is an exact
// copy, so on CPU the logits (hence greedy argmax and seeded samples)
// match bit-for-bit.
//
// What this pins down:
//   * greedy parity vs standalone generate across mixed prompt lengths;
//   * dynamic admission: a batch cap smaller than the request count
//     forces staggered admission + block recycling, output unchanged;
//   * sampling parity with per-request seeds;
//   * EOS early-stop parity;
//   * a tiny pool serves more requests than fit at once via recycling,
//     and all blocks are reclaimed when the queue drains;
//   * input validation + EngineConfig sizing checks.

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/models/Llama.hpp"
#include "tesseract/models/Scheduler.hpp"

using tesseract::models::ContinuousBatchingScheduler;
using tesseract::models::EngineConfig;
using tesseract::models::LlamaConfig;
using tesseract::models::LlamaModel;
using tesseract::models::RequestId;

namespace {

std::shared_ptr<LlamaModel> tiny_model() {
  LlamaConfig cfg;
  cfg.vocab_size = 64;
  cfg.hidden_size = 32;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.intermediate_size = 64;
  cfg.max_position_embeddings = 128;
  cfg.rope_theta = 10000.0;
  cfg.tie_word_embeddings = false;
  return std::make_shared<LlamaModel>(cfg);
}

LlamaModel::GenerateConfig greedy(int64_t max_new, int32_t eos = -1) {
  LlamaModel::GenerateConfig g;
  g.max_new_tokens = max_new;
  g.eos_token_id = eos;
  return g;
}

}  // namespace

TEST_CASE("Scheduler: greedy output matches standalone generate", "[models][scheduler]") {
  auto model = tiny_model();

  const std::vector<std::vector<int32_t>> prompts = {
      {1, 2, 3}, {10, 20}, {5}, {7, 8, 9, 11, 13}, {2, 4, 6, 8},
  };
  const std::vector<int64_t> max_new = {12, 8, 16, 5, 10};

  // References from standalone generate (contiguous caches).
  std::vector<std::vector<int32_t>> ref;
  for (std::size_t i = 0; i < prompts.size(); ++i)
    ref.push_back(model->generate(prompts[i], greedy(max_new[i])));

  EngineConfig ec;
  ec.block_size = 8;
  ec.num_blocks = 256;
  ec.max_seq_len = 64;
  ec.max_batch_size = 16;  // all admitted at once
  ContinuousBatchingScheduler sched(model, ec);

  std::vector<RequestId> ids;
  for (std::size_t i = 0; i < prompts.size(); ++i)
    ids.push_back(sched.add_request(prompts[i], greedy(max_new[i])));

  sched.run();

  for (std::size_t i = 0; i < ids.size(); ++i) {
    REQUIRE(sched.is_finished(ids[i]));
    REQUIRE(sched.result(ids[i]) == ref[i]);
  }
  // Queue drained ⇒ every block reclaimed.
  REQUIRE(sched.allocated_blocks() == 0);
  REQUIRE(!sched.has_pending());
}

TEST_CASE("Scheduler: staggered admission + recycling preserves output", "[models][scheduler]") {
  auto model = tiny_model();
  const std::vector<std::vector<int32_t>> prompts = {
      {1, 2, 3, 4}, {9, 8, 7}, {15, 16}, {3, 1, 4, 1, 5}, {20}, {6, 6, 6},
  };
  std::vector<std::vector<int32_t>> ref;
  for (const auto& p : prompts) ref.push_back(model->generate(p, greedy(10)));

  EngineConfig ec;
  ec.block_size = 8;
  ec.num_blocks = 64;
  ec.max_seq_len = 32;
  ec.max_batch_size = 2;  // only 2 of 6 run concurrently → staggered
  ContinuousBatchingScheduler sched(model, ec);

  std::vector<RequestId> ids;
  for (const auto& p : prompts) ids.push_back(sched.add_request(p, greedy(10)));

  // Drive tick-by-tick; batch cap must hold every step.
  int guard = 0;
  while (sched.has_pending()) {
    REQUIRE(sched.num_running() <= ec.max_batch_size);
    sched.step();
    REQUIRE(++guard < 10000);
  }
  for (std::size_t i = 0; i < ids.size(); ++i)
    REQUIRE(sched.result(ids[i]) == ref[i]);
  REQUIRE(sched.allocated_blocks() == 0);
}

TEST_CASE("Scheduler: sampling parity with per-request seeds", "[models][scheduler]") {
  auto model = tiny_model();

  auto sample_cfg = [](int64_t max_new, uint64_t seed, double temp) {
    LlamaModel::GenerateConfig g;
    g.max_new_tokens = max_new;
    g.do_sample = true;
    g.seed = seed;
    g.sampling.temperature = temp;
    g.sampling.top_k = 20;
    g.sampling.top_p = 0.95;
    return g;
  };

  const std::vector<LlamaModel::GenerateConfig> cfgs = {
      sample_cfg(10, 1, 0.8), sample_cfg(12, 2, 1.0), sample_cfg(8, 3, 0.5),
  };
  const std::vector<std::vector<int32_t>> prompts = {{1, 2}, {5, 6, 7}, {9}};

  std::vector<std::vector<int32_t>> ref;
  for (std::size_t i = 0; i < prompts.size(); ++i)
    ref.push_back(model->generate(prompts[i], cfgs[i]));

  EngineConfig ec;
  ec.block_size = 8;
  ec.num_blocks = 128;
  ec.max_seq_len = 64;
  ec.max_batch_size = 3;
  ContinuousBatchingScheduler sched(model, ec);

  std::vector<RequestId> ids;
  for (std::size_t i = 0; i < prompts.size(); ++i)
    ids.push_back(sched.add_request(prompts[i], cfgs[i]));
  sched.run();

  for (std::size_t i = 0; i < ids.size(); ++i)
    REQUIRE(sched.result(ids[i]) == ref[i]);
}

TEST_CASE("Scheduler: EOS early-stop matches generate", "[models][scheduler]") {
  auto model = tiny_model();
  const std::vector<int32_t> prompt = {3, 14, 15, 9};

  // Discover the token greedy emits first, then use it as EOS so both
  // paths must stop after exactly one generated token.
  const auto full = model->generate(prompt, greedy(8));
  const int32_t first_gen = full[prompt.size()];

  const auto ref = model->generate(prompt, greedy(8, /*eos=*/first_gen));
  REQUIRE(static_cast<int64_t>(ref.size()) == static_cast<int64_t>(prompt.size()) + 1);

  EngineConfig ec;
  ec.max_seq_len = 32;
  ec.num_blocks = 64;
  ec.block_size = 8;
  ContinuousBatchingScheduler sched(model, ec);
  const RequestId id = sched.add_request(prompt, greedy(8, first_gen));
  sched.run();
  REQUIRE(sched.result(id) == ref);
}

TEST_CASE("Scheduler: tiny pool serves many requests via recycling", "[models][scheduler]") {
  auto model = tiny_model();
  const std::vector<std::vector<int32_t>> prompts = {
      {1, 2}, {3, 4}, {5, 6}, {7, 8},
  };
  std::vector<std::vector<int32_t>> ref;
  for (const auto& p : prompts) ref.push_back(model->generate(p, greedy(6)));

  // block_size 8, max_seq_len 16 ⇒ 2 blocks per request; pool holds 2 ⇒
  // exactly ONE request resident at a time. Recycling must let all four
  // through.
  EngineConfig ec;
  ec.block_size = 8;
  ec.num_blocks = 2;
  ec.max_seq_len = 16;
  ec.max_batch_size = 4;
  ContinuousBatchingScheduler sched(model, ec);

  std::vector<RequestId> ids;
  for (const auto& p : prompts) ids.push_back(sched.add_request(p, greedy(6)));
  sched.run();

  for (std::size_t i = 0; i < ids.size(); ++i)
    REQUIRE(sched.result(ids[i]) == ref[i]);
  REQUIRE(sched.allocated_blocks() == 0);
}

TEST_CASE("Scheduler: INT8 paged KV matches generate(kv_int8)", "[models][scheduler]") {
  auto model = tiny_model();

  const std::vector<std::vector<int32_t>> prompts = {
      {1, 2, 3}, {10, 20}, {5}, {7, 8, 9, 11, 13},
  };
  const std::vector<int64_t> max_new = {10, 8, 12, 6};

  // Reference: standalone generate with INT8 (contiguous) KV. The paged
  // INT8 pool stores byte-identical quantized K/V (Wave-13 proves the
  // paged vs contiguous dequant view is bit-exact), and on CPU
  // forward_step_batched falls back to the same per-sequence dequant +
  // attention as forward_step — so the scheduler's INT8-paged output must
  // match generate(kv_int8) bit-for-bit.
  std::vector<std::vector<int32_t>> ref;
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    auto g = greedy(max_new[i]);
    g.kv_int8 = true;
    ref.push_back(model->generate(prompts[i], g));
  }

  EngineConfig ec;
  ec.block_size = 8;
  ec.num_blocks = 256;
  ec.max_seq_len = 64;
  ec.max_batch_size = 16;
  ec.kv_int8 = true;  // INT8 paged pools
  ContinuousBatchingScheduler sched(model, ec);

  std::vector<RequestId> ids;
  for (std::size_t i = 0; i < prompts.size(); ++i)
    ids.push_back(sched.add_request(prompts[i], greedy(max_new[i])));
  sched.run();

  for (std::size_t i = 0; i < ids.size(); ++i) {
    REQUIRE(sched.is_finished(ids[i]));
    REQUIRE(sched.result(ids[i]) == ref[i]);
  }
  // INT8 blocks recycle just like FP ones.
  REQUIRE(sched.allocated_blocks() == 0);
  REQUIRE(!sched.has_pending());
}

TEST_CASE("Scheduler: input validation + config sizing", "[models][scheduler]") {
  auto model = tiny_model();
  EngineConfig ec;
  ec.block_size = 8;
  ec.num_blocks = 64;
  ec.max_seq_len = 32;
  ContinuousBatchingScheduler sched(model, ec);

  REQUIRE_THROWS(sched.add_request({}, greedy(4)));                  // empty
  REQUIRE_THROWS(sched.add_request({999}, greedy(4)));              // OOB id
  REQUIRE_THROWS(sched.add_request({1, 2}, greedy(100)));           // > max_seq_len
  REQUIRE_THROWS(sched.result(424242));                             // unknown id

  // A pool too small for even one max-length request is rejected at ctor.
  EngineConfig bad;
  bad.block_size = 8;
  bad.num_blocks = 1;
  bad.max_seq_len = 64;  // needs 8 blocks
  REQUIRE_THROWS(ContinuousBatchingScheduler(model, bad));
}

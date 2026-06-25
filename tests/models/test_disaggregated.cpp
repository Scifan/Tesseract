// Wave 19 (B-036) — disaggregated prefill / decode.
//
// The contract is exactness: splitting inference into a prefill role (which
// computes the prompt's KV and exports it) and a decode role (which imports
// that KV and generates) must produce the SAME tokens as monolithic
// `model->generate`. The KV blob is copied byte-for-byte and the decode
// path is identical, so on CPU the output matches bit-for-bit.

#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/models/Disaggregated.hpp"
#include "tesseract/models/Llama.hpp"

using tesseract::models::DisaggregatedEngine;
using tesseract::models::KvTransfer;
using tesseract::models::LlamaConfig;
using tesseract::models::LlamaModel;

namespace {

std::shared_ptr<LlamaModel> tiny_model() {
  LlamaConfig c;
  c.vocab_size = 48;
  c.hidden_size = 32;
  c.num_hidden_layers = 2;
  c.num_attention_heads = 4;
  c.intermediate_size = 64;
  c.max_position_embeddings = 256;
  c.rope_theta = 10000.0;
  c.tie_word_embeddings = false;
  return std::make_shared<LlamaModel>(c);
}

LlamaModel::GenerateConfig greedy(int64_t max_new, int32_t eos = -1) {
  LlamaModel::GenerateConfig g;
  g.max_new_tokens = max_new;
  g.eos_token_id = eos;
  return g;
}

}  // namespace

TEST_CASE("Disaggregated: generate matches monolithic generate", "[models][disagg]") {
  auto model = tiny_model();
  DisaggregatedEngine engine(model);

  const std::vector<std::vector<int32_t>> prompts = {
      {1, 2, 3}, {10, 20}, {5}, {7, 8, 9, 11, 13},
  };
  const int64_t max_new = 24;

  for (const auto& prompt : prompts) {
    const auto ref = model->generate(prompt, greedy(max_new));
    const auto out = engine.generate(prompt, max_new);
    REQUIRE(out.tokens == ref);
    REQUIRE(out.prompt_len == static_cast<int64_t>(prompt.size()));
  }
}

TEST_CASE("Disaggregated: explicit prefill→decode across separate roles", "[models][disagg]") {
  auto model = tiny_model();
  // Two engine instances over the same (replicated) weights — a prefill
  // worker and a decode worker connected only by the KV transfer blob.
  DisaggregatedEngine prefill_worker(model);
  DisaggregatedEngine decode_worker(model);

  const std::vector<int32_t> prompt = {3, 1, 4, 1, 5, 9};
  const int64_t max_new = 20;
  const auto ref = model->generate(prompt, greedy(max_new));

  KvTransfer blob = prefill_worker.prefill(prompt);
  REQUIRE(static_cast<int64_t>(blob.layer_kv.size()) == model->num_layers());
  REQUIRE(blob.prompt_len == static_cast<int64_t>(prompt.size()));
  // Each layer's exported K/V is [1, H, P, D].
  for (const auto& [k, v] : blob.layer_kv) {
    REQUIRE(k.shape()[2] == static_cast<int64_t>(prompt.size()));
    REQUIRE(v.shape()[2] == static_cast<int64_t>(prompt.size()));
  }

  const auto out = decode_worker.decode(blob, prompt, max_new);
  REQUIRE(out.tokens == ref);
}

TEST_CASE("Disaggregated: EOS early stop matches generate", "[models][disagg]") {
  auto model = tiny_model();
  DisaggregatedEngine engine(model);

  const std::vector<int32_t> prompt = {2, 4, 6, 8};
  const auto full = model->generate(prompt, greedy(8));
  const int32_t first_gen = full[prompt.size()];

  const auto ref = model->generate(prompt, greedy(8, /*eos=*/first_gen));
  const auto out = engine.generate(prompt, 8, /*eos=*/first_gen);
  REQUIRE(out.tokens == ref);
  REQUIRE(static_cast<int64_t>(out.tokens.size()) ==
          static_cast<int64_t>(prompt.size()) + 1);
}

TEST_CASE("Disaggregated: input validation", "[models][disagg]") {
  auto model = tiny_model();
  DisaggregatedEngine engine(model);

  REQUIRE_THROWS(engine.prefill({}));  // empty prompt

  const std::vector<int32_t> prompt = {1, 2, 3};
  KvTransfer blob = engine.prefill(prompt);
  // Wrong prompt metadata length is rejected.
  REQUIRE_THROWS(engine.decode(blob, {1, 2}, 4));
  // Corrupt the layer count.
  KvTransfer bad = blob;
  bad.layer_kv.pop_back();
  REQUIRE_THROWS(engine.decode(bad, prompt, 4));
}

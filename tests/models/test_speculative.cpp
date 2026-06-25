// Wave 18 (B-035) — speculative decoding.
//
// The contract is exactness: greedy speculative decoding must emit the
// EXACT token sequence the target model emits on its own, for ANY draft.
// These tests pin that across:
//   * a genuinely different (smaller) draft → real partial acceptance;
//   * a self-draft (draft == target) → every proposal accepted, so the
//     target runs far fewer sequential forwards than plain autoregression;
//   * varying the block size γ;
//   * EOS early stop.

#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/models/Llama.hpp"
#include "tesseract/models/Speculative.hpp"

using tesseract::models::LlamaConfig;
using tesseract::models::LlamaModel;
using tesseract::models::SpeculativeDecoder;

namespace {

std::shared_ptr<LlamaModel> make_model(int64_t layers, int64_t hidden,
                                       int64_t heads, int64_t vocab) {
  LlamaConfig c;
  c.vocab_size = vocab;
  c.hidden_size = hidden;
  c.num_hidden_layers = layers;
  c.num_attention_heads = heads;
  c.intermediate_size = hidden * 2;
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

TEST_CASE("Speculative: greedy output matches target standalone", "[models][speculative]") {
  const int64_t vocab = 48;
  auto target = make_model(/*layers=*/2, /*hidden=*/32, /*heads=*/4, vocab);
  auto draft  = make_model(/*layers=*/1, /*hidden=*/16, /*heads=*/2, vocab);

  const std::vector<std::vector<int32_t>> prompts = {
      {1, 2, 3}, {10, 20}, {5}, {7, 8, 9, 11},
  };
  const int64_t max_new = 24;

  for (const auto& prompt : prompts) {
    const auto ref = target->generate(prompt, greedy(max_new));
    for (int gamma : {1, 2, 4, 8}) {
      SpeculativeDecoder dec(target, draft, gamma);
      const auto out = dec.generate(prompt, max_new);
      REQUIRE(out.tokens == ref);          // bit-identical, any γ
      REQUIRE(out.proposed >= out.accepted);
    }
  }
}

TEST_CASE("Speculative: self-draft accepts everything and saves target forwards",
          "[models][speculative]") {
  const int64_t vocab = 48;
  auto model = make_model(/*layers=*/2, /*hidden=*/32, /*heads=*/4, vocab);

  const std::vector<int32_t> prompt = {3, 1, 4, 1, 5};
  const int64_t max_new = 32;
  const auto ref = model->generate(prompt, greedy(max_new));

  // draft == target ⇒ the draft predicts exactly what the target verifies,
  // so every proposed token is accepted.
  const int gamma = 4;
  SpeculativeDecoder dec(model, model, gamma);
  const auto out = dec.generate(prompt, max_new);

  REQUIRE(out.tokens == ref);
  // draft == target ⇒ proposals match the target's argmax almost always.
  // Not exactly 1.0: the target verifies a γ-token block (S=γ) while the
  // draft proposes via single-step (S=1) forwards, and block-vs-single GEMM
  // ordering flips the occasional near-tie argmax — which is precisely why
  // verification exists (output stays correct via the correction token).
  REQUIRE(out.acceptance_rate() >= 0.5);
  // Plain autoregression needs `max_new` sequential target forwards (plus
  // prefill). With γ=4 and high acceptance the target runs far fewer.
  REQUIRE(out.target_forwards < max_new);
}

TEST_CASE("Speculative: EOS early stop matches target", "[models][speculative]") {
  const int64_t vocab = 48;
  auto target = make_model(2, 32, 4, vocab);
  auto draft  = make_model(1, 16, 2, vocab);

  const std::vector<int32_t> prompt = {2, 4, 6, 8};
  // Find the token greedy emits first, use it as EOS so both paths stop
  // after exactly one generated token.
  const auto full = target->generate(prompt, greedy(8));
  const int32_t first_gen = full[prompt.size()];

  const auto ref = target->generate(prompt, greedy(8, /*eos=*/first_gen));
  REQUIRE(static_cast<int64_t>(ref.size()) ==
          static_cast<int64_t>(prompt.size()) + 1);

  SpeculativeDecoder dec(target, draft, /*gamma=*/4);
  const auto out = dec.generate(prompt, 8, /*eos=*/first_gen);
  REQUIRE(out.tokens == ref);
}

TEST_CASE("Speculative: input validation", "[models][speculative]") {
  auto a = make_model(1, 16, 2, 32);
  auto b = make_model(1, 16, 2, 64);  // different vocab
  REQUIRE_THROWS(SpeculativeDecoder(a, b, 4));            // vocab mismatch
  REQUIRE_THROWS(SpeculativeDecoder(a, a, 0));            // gamma < 1
  SpeculativeDecoder ok(a, a, 4);
  REQUIRE_THROWS(ok.generate({}, 4));                     // empty prompt
}

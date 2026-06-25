// Wave 6 (B-028) — token sampling tests.
//
// Validates the logits-processing pipeline (repetition penalty →
// temperature → top-k → top-p → multinomial) on hand-constructed logits
// where the correct behavior is analytically clear, plus the seeded
// determinism contract and the `LlamaModel::generate(do_sample=true)`
// integration.

#include <array>
#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/models/Llama.hpp"
#include "tesseract/models/Sampler.hpp"

using tesseract::models::LlamaConfig;
using tesseract::models::LlamaModel;
using tesseract::models::SamplingParams;
using tesseract::models::Sampler;
using tesseract::models::sample_from_logits;

namespace {

int32_t draw(std::span<const float> logits, const SamplingParams& p,
             std::mt19937_64& rng,
             std::span<const int32_t> prev = {}) {
  return sample_from_logits(logits, p, prev, rng);
}

}  // namespace

TEST_CASE("sample_from_logits: temperature<=0 is greedy argmax", "[models][sampler]") {
  const std::array<float, 5> logits = {0.1f, 2.5f, -1.0f, 2.4f, 0.0f};
  SamplingParams p;
  p.temperature = 0.0;  // greedy
  std::mt19937_64 rng(123);
  for (int i = 0; i < 32; ++i) {
    REQUIRE(draw(logits, p, rng) == 1);  // index of max (2.5)
  }
}

TEST_CASE("sample_from_logits: top_k restricts support to k highest", "[models][sampler]") {
  // logits favor indices {3, 1} as the two largest.
  const std::array<float, 6> logits = {0.0f, 5.0f, 1.0f, 6.0f, 0.5f, 2.0f};
  SamplingParams p;
  p.temperature = 1.0;
  p.top_k = 2;
  std::mt19937_64 rng(7);
  std::map<int32_t, int> hist;
  for (int i = 0; i < 4000; ++i) hist[draw(logits, p, rng)]++;
  // Only the top-2 indices (3 and 1) may ever be drawn.
  for (const auto& [id, count] : hist) {
    INFO("drew id " << id << " count " << count);
    REQUIRE((id == 3 || id == 1));
  }
  REQUIRE(hist.count(3) == 1);
  REQUIRE(hist.count(1) == 1);
  // Higher logit (3) should dominate.
  REQUIRE(hist[3] > hist[1]);
}

TEST_CASE("sample_from_logits: top_p keeps only the nucleus", "[models][sampler]") {
  // One token dominates the softmax mass; nucleus p=0.5 should keep just it.
  const std::array<float, 4> logits = {10.0f, 0.0f, -1.0f, 0.5f};
  SamplingParams p;
  p.temperature = 1.0;
  p.top_p = 0.5;
  std::mt19937_64 rng(99);
  for (int i = 0; i < 2000; ++i) {
    REQUIRE(draw(logits, p, rng) == 0);  // dominant token only
  }
}

TEST_CASE("sample_from_logits: top_p always keeps at least one token", "[models][sampler]") {
  // Even a tiny p must not produce an empty nucleus.
  const std::array<float, 3> logits = {1.0f, 1.0f, 1.0f};
  SamplingParams p;
  p.temperature = 1.0;
  p.top_p = 0.01;
  std::mt19937_64 rng(5);
  const int32_t id = draw(logits, p, rng);
  REQUIRE(id >= 0);
  REQUIRE(id < 3);
}

TEST_CASE("sample_from_logits: same seed reproduces the same draws", "[models][sampler]") {
  const std::array<float, 8> logits = {0.5f, 1.0f, -0.3f, 2.0f, 0.1f, 1.5f, -1.0f, 0.7f};
  SamplingParams p;
  p.temperature = 1.0;
  std::mt19937_64 a(2026), b(2026);
  for (int i = 0; i < 100; ++i) {
    REQUIRE(draw(logits, p, a) == draw(logits, p, b));
  }
  // A different seed should (with overwhelming probability) diverge.
  std::mt19937_64 c(2026), d(99);
  std::vector<int32_t> sc, sd;
  for (int i = 0; i < 100; ++i) { sc.push_back(draw(logits, p, c)); sd.push_back(draw(logits, p, d)); }
  REQUIRE(sc != sd);
}

TEST_CASE("sample_from_logits: repetition penalty suppresses prior tokens", "[models][sampler]") {
  // Two near-equal favorites (0 and 1). Penalizing token 0 should flip the
  // majority to token 1.
  const std::array<float, 4> logits = {3.0f, 3.0f, -2.0f, -2.0f};
  std::array<int32_t, 1> prev = {0};

  SamplingParams base;
  base.temperature = 1.0;
  std::mt19937_64 r0(11);
  int count0_no_pen = 0;
  for (int i = 0; i < 4000; ++i) if (draw(logits, base, r0) == 0) ++count0_no_pen;

  SamplingParams pen = base;
  pen.repetition_penalty = 2.0;  // halve positive logit of token 0
  std::mt19937_64 r1(11);
  int count0_pen = 0;
  for (int i = 0; i < 4000; ++i) {
    if (draw(logits, pen, r1, std::span<const int32_t>(prev.data(), prev.size())) == 0) ++count0_pen;
  }
  INFO("count0 no-pen=" << count0_no_pen << " pen=" << count0_pen);
  REQUIRE(count0_pen < count0_no_pen);  // penalty reduced token 0's share
}

TEST_CASE("sample_from_logits: temperature spreads the distribution", "[models][sampler]") {
  const std::array<float, 3> logits = {2.0f, 1.0f, 0.0f};
  std::mt19937_64 lo_rng(3), hi_rng(3);

  SamplingParams lo; lo.temperature = 0.25;  // sharpen
  SamplingParams hi; hi.temperature = 4.0;   // flatten

  int top_lo = 0, top_hi = 0;
  for (int i = 0; i < 4000; ++i) {
    if (draw(logits, lo, lo_rng) == 0) ++top_lo;
    if (draw(logits, hi, hi_rng) == 0) ++top_hi;
  }
  // Low temperature concentrates mass on the argmax far more than high T.
  REQUIRE(top_lo > top_hi);
}

TEST_CASE("LlamaModel::generate sampling is seed-deterministic and greedy-default",
          "[models][llama][sampler]") {
  LlamaConfig cfg;
  cfg.vocab_size = 48;
  cfg.hidden_size = 32;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.intermediate_size = 64;
  cfg.max_position_embeddings = 64;
  cfg.rope_theta = 10000.0;
  cfg.tie_word_embeddings = false;
  auto model = std::make_shared<LlamaModel>(cfg);

  const std::vector<int32_t> prompt = {5, 11, 30};

  // do_sample with a fixed seed reproduces; a different seed diverges.
  LlamaModel::GenerateConfig s1;
  s1.max_new_tokens = 16;
  s1.do_sample = true;
  s1.sampling.temperature = 1.0;
  s1.seed = 1234;
  const auto a = model->generate(prompt, s1);
  const auto b = model->generate(prompt, s1);
  REQUIRE(a == b);

  LlamaModel::GenerateConfig s2 = s1;
  s2.seed = 9999;
  const auto c = model->generate(prompt, s2);
  REQUIRE(a != c);  // different seed ⇒ different continuation

  // Greedy default path is unaffected by seed.
  LlamaModel::GenerateConfig g1;
  g1.max_new_tokens = 16;
  g1.seed = 1;
  LlamaModel::GenerateConfig g2 = g1;
  g2.seed = 2;
  REQUIRE(model->generate(prompt, g1) == model->generate(prompt, g2));

  // Temperature -> 0 sampling matches greedy.
  LlamaModel::GenerateConfig greedy_via_sample;
  greedy_via_sample.max_new_tokens = 16;
  greedy_via_sample.do_sample = true;
  greedy_via_sample.sampling.temperature = 0.0;
  greedy_via_sample.seed = 42;
  REQUIRE(model->generate(prompt, greedy_via_sample) == model->generate(prompt, g1));
}

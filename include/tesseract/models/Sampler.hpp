#pragma once

// Token sampling for autoregressive generation — Wave 6 (B-028).
//
// Operates on a single position's logits row (host FP32) and returns the
// next token id. Reproduces the HuggingFace `transformers` /
// vLLM `SamplingParams` logits-processing order so a caller that already
// has generation configs from those ecosystems gets matching behavior:
//
//   1. repetition penalty (CTRL-style: divide positive logits, multiply
//      negative logits, for every previously-seen token)
//   2. temperature scaling (logits /= T). T <= 0 ⇒ greedy (argmax),
//      which also short-circuits every downstream filter.
//   3. top-k filter (keep the k highest logits, rest → -inf)
//   4. top-p / nucleus filter (keep the smallest high-prob prefix whose
//      cumulative probability ≥ p, rest → -inf)
//   5. softmax → multinomial draw from a seeded RNG.
//
// All filters are no-ops at their identity values (penalty 1.0, T 1.0,
// top_k 0, top_p 1.0), so the default `SamplingParams` is plain
// temperature-1 sampling from the full distribution; `temperature <= 0`
// is the greedy path.

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace tesseract::models {

struct SamplingParams {
  double temperature        = 1.0;  // <= 0 ⇒ greedy (argmax)
  int    top_k              = 0;    // 0 ⇒ disabled
  double top_p              = 1.0;  // 1.0 ⇒ disabled (nucleus)
  double repetition_penalty = 1.0;  // 1.0 ⇒ disabled
};

// Pure function: pick the next token id from `logits` (length = vocab).
// `prev_tokens` feeds the repetition penalty (ids outside [0, vocab) are
// ignored). `rng` is advanced for any stochastic draw; greedy draws leave
// it untouched. Throws if `logits` is empty.
int32_t sample_from_logits(std::span<const float> logits,
                           const SamplingParams& params,
                           std::span<const int32_t> prev_tokens,
                           std::mt19937_64& rng);

// Stateful convenience wrapper owning a seeded RNG.
class Sampler {
 public:
  explicit Sampler(SamplingParams params, uint64_t seed = 0)
      : params_(params), rng_(seed) {}

  int32_t sample(std::span<const float> logits,
                 std::span<const int32_t> prev_tokens) {
    return sample_from_logits(logits, params_, prev_tokens, rng_);
  }

  const SamplingParams& params() const noexcept { return params_; }
  void reseed(uint64_t seed) { rng_.seed(seed); }

 private:
  SamplingParams params_;
  std::mt19937_64 rng_;
};

}  // namespace tesseract::models

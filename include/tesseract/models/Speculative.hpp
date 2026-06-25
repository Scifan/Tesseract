#pragma once

// Wave 18 (B-035) — speculative decoding.
//
// A small, cheap *draft* model proposes γ tokens autoregressively; the
// large *target* model then verifies all γ in a SINGLE forward pass over
// the proposed block. Under greedy verification we accept the longest
// proposed prefix that matches the target's own argmax at each position,
// then append the target's argmax at the first mismatch (the "correction"
// / bonus token). Each round commits between 1 and γ+1 tokens for one
// target forward (plus one tiny resync forward), so when the draft is good
// the target runs far fewer sequential steps than plain autoregression.
//
// Correctness contract (the test pins it): greedy speculative decoding is
// **token-for-token identical** to running the target model greedily on its
// own — the draft only affects *speed*, never the output. This holds for
// ANY draft model (even a random one): a bad draft simply means few tokens
// are accepted per round.
//
// KV management: both models keep contiguous `nn::KVCache`s. The target
// speculatively appends all γ proposals; on partial acceptance the caches
// are rewound with `set_current_len` to the accepted length and the
// correction token is re-fed, so the cached KV always matches the
// committed sequence exactly.

#include <cstdint>
#include <memory>
#include <vector>

#include "tesseract/models/Llama.hpp"

namespace tesseract::models {

class SpeculativeDecoder {
 public:
  // `target` is the model whose output is reproduced; `draft` proposes
  // candidates. Both must share a vocabulary. `gamma` is the proposal
  // block size per round (≥ 1).
  SpeculativeDecoder(std::shared_ptr<LlamaModel> target,
                     std::shared_ptr<LlamaModel> draft, int gamma = 4);

  struct Result {
    std::vector<int32_t> tokens;  // prompt + generated (HF convention)
    int64_t proposed = 0;         // total draft tokens proposed
    int64_t accepted = 0;         // total draft tokens accepted (excl. corrections)
    int64_t rounds = 0;           // verification rounds
    int64_t target_forwards = 0;  // target forward_step calls (incl. prefill + resync)
    // accepted / proposed — higher ⇒ the draft tracks the target better.
    double acceptance_rate() const {
      return proposed > 0 ? static_cast<double>(accepted) /
                                static_cast<double>(proposed)
                          : 0.0;
    }
  };

  // Greedy speculative generation. Output is identical to
  // `target->generate(prompt, greedy(max_new_tokens, eos))`.
  Result generate(const std::vector<int32_t>& prompt_ids,
                  int64_t max_new_tokens, int32_t eos_token_id = -1);

  int gamma() const noexcept { return gamma_; }

 private:
  std::shared_ptr<LlamaModel> target_;
  std::shared_ptr<LlamaModel> draft_;
  int gamma_;
};

}  // namespace tesseract::models

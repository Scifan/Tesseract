#pragma once

// Wave 19 (B-036) — disaggregated prefill / decode.
//
// Production LLM serving (vLLM-P/D, NVIDIA Dynamo, DistServe) splits the two
// phases of inference onto separate workers: PREFILL is compute-bound (one
// big parallel pass over the whole prompt) while DECODE is memory-bandwidth
// bound (many tiny sequential steps). Running them on the same worker makes
// them interfere — a long prefill stalls every in-flight decode. Disaggre-
// gation runs them as distinct roles and MIGRATES the KV cache the prefill
// worker computed over to the decode worker, which then drives generation.
//
// This is the single-process model of that architecture. The
// machinery that matters is the explicit KV handoff:
//
//   * `prefill(prompt)` runs the prompt through the model into its OWN KV
//     caches, then exports each layer's `[1, H, P, D]` K/V (and the first
//     token) as a `KvTransfer` — the blob a real prefill worker would ship
//     across the wire. Its caches are released immediately afterward.
//
//   * `decode(transfer, prompt, …)` imports that K/V into a SEPARATE set of
//     decode caches and continues autoregressively from the first token.
//
// The decode role never re-runs the prompt — it only ingests KV and steps.
// Correctness contract: the disaggregated output is token-for-token
// identical to monolithic `model->generate(prompt, greedy)`, because the KV
// bytes are copied exactly and the decode path is the same code.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/models/Llama.hpp"

namespace tesseract::models {

// Serialized KV state handed from the prefill role to the decode role.
struct KvTransfer {
  // Per-layer contiguous K and V over the prompt prefix: each `[1, H, P, D]`.
  std::vector<std::pair<Tensor, Tensor>> layer_kv;
  int32_t first_token = -1;   // target's argmax at the last prompt position
  int64_t prompt_len  = 0;
};

class DisaggregatedEngine {
 public:
  explicit DisaggregatedEngine(std::shared_ptr<LlamaModel> model);

  struct Result {
    std::vector<int32_t> tokens;   // prompt + generated (HF convention)
    int64_t prompt_len  = 0;
    int64_t decode_steps = 0;      // sequential decode forwards run
  };

  // PREFILL role: compute the prompt's KV + first token, export as a blob.
  KvTransfer prefill(const std::vector<int32_t>& prompt_ids);

  // DECODE role: import `transfer`'s KV into fresh caches and generate.
  // `prompt_ids` is metadata only (to assemble the full output sequence) —
  // the prompt is NOT re-run; generation resumes purely from the migrated
  // KV and the carried first token.
  Result decode(const KvTransfer& transfer,
                const std::vector<int32_t>& prompt_ids, int64_t max_new_tokens,
                int32_t eos_token_id = -1);

  // Convenience: prefill then decode in one call.
  Result generate(const std::vector<int32_t>& prompt_ids,
                  int64_t max_new_tokens, int32_t eos_token_id = -1);

 private:
  std::shared_ptr<LlamaModel> model_;
};

}  // namespace tesseract::models

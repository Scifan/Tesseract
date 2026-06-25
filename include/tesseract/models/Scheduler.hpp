#pragma once

// Wave 7 (B-029): continuous-batching inference scheduler.
//
// Wave 5's `LlamaModel::generate` runs ONE sequence start-to-finish.
// A server multiplexes many requests that arrive and finish at different
// times and have wildly different lengths. The scheduler is the control
// plane that makes that efficient:
//
//   * a SHARED `PagedKVPool` per layer (Wave 7) so a finished request's
//     blocks recycle immediately into the budget the next request draws
//     from — resident KV memory tracks the live token count across ALL
//     requests, not `num_requests · max_len`;
//   * dynamic admission: waiting requests are admitted up to a batch cap
//     and only when the pool can hold their prompt, so the engine never
//     over-commits memory;
//   * per-request sampling (each request carries its own `GenerateConfig`
//     + seeded RNG) and per-request EOS / length stopping;
//   * block reclamation on completion.
//
// MVP scope (honest about what's deferred). Decode is driven per request
// (one `forward_step` per active request per tick) rather than as one
// fused ragged batch. That delivers the full continuous-batching
// *semantics* — dynamic admit/evict, shared paged memory, heterogeneous
// per-request sampling — and makes each request's output **bit-identical
// to running `generate` standalone** (the parity contract the tests
// enforce). Fusing the active set into a single padded/ragged
// `forward_step` (the throughput win) needs the ragged paged-attention
// kernel deferred as B-019b+, and lands without changing this API.

#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/models/Sampler.hpp"
#include "tesseract/nn/KVCacheBase.hpp"
#include "tesseract/nn/PagedKVPool.hpp"
#include "tesseract/nn/QuantizedPagedKVPool.hpp"

namespace tesseract::models {

using RequestId = int64_t;

struct EngineConfig {
  int64_t block_size     = 16;    // tokens per physical block
  int64_t num_blocks     = 2048;  // blocks in EACH per-layer shared pool
  int64_t max_seq_len    = 2048;  // per-request cap on prompt + generated
  int64_t max_batch_size = 16;    // max requests decoding concurrently
  // Wave 14 (B-032++++): when true, the per-layer shared pools store KV in
  // INT8 (`nn::QuantizedPagedKVPool` + `QuantizedPagedKVCache`) instead of
  // full precision — paging's no-padding residency × quant's ~4× (vs FP32)
  // / ~2× (vs FP16) per-token shrink, feeding the Wave-12 fused INT8 paged
  // attention on the CUDA decode path. Lossy: a request's output is
  // bounded-error vs the FP path, not bit-identical (same contract as
  // `GenerateConfig::kv_int8`).
  bool    kv_int8        = false;
};

class ContinuousBatchingScheduler {
 public:
  ContinuousBatchingScheduler(std::shared_ptr<LlamaModel> model,
                              EngineConfig cfg);

  // Enqueue a request. It joins the waiting queue and is admitted by a
  // later `step()` when there is batch room + enough free blocks for its
  // prompt. Returns the assigned id. Throws on an empty prompt, an
  // out-of-range token, or a prompt that cannot fit the pool even when
  // empty.
  RequestId add_request(std::vector<int32_t> prompt_ids,
                        LlamaModel::GenerateConfig gen);

  // Advance the engine one tick: admit waiting requests up to capacity
  // (prefilling each), emit one token for every running request,
  // and retire + reclaim blocks for any that hit EOS / their length cap.
  // Returns true while any request is still waiting or running.
  bool step();

  // Drive `step()` until every request has finished.
  void run();

  bool has_pending() const noexcept {
    return !waiting_.empty() || !running_.empty();
  }
  int64_t num_waiting() const noexcept {
    return static_cast<int64_t>(waiting_.size());
  }
  int64_t num_running() const noexcept {
    return static_cast<int64_t>(running_.size());
  }
  bool is_finished(RequestId id) const;

  // Full token sequence (prompt followed by generated ids — HF
  // convention) for a request. Valid once the request is finished;
  // throws for an unknown id.
  const std::vector<int32_t>& result(RequestId id) const;

  // Live blocks held across one per-layer pool — the residency metric.
  // Reads whichever pool flavor (FP or INT8) backs this engine.
  int64_t allocated_blocks() const noexcept {
    if (!qpools_.empty()) return qpools_.front()->num_allocated();
    return pools_.empty() ? 0 : pools_.front()->num_allocated();
  }
  int64_t free_blocks() const noexcept {
    if (!qpools_.empty()) return qpools_.front()->num_free();
    return pools_.empty() ? 0 : pools_.front()->num_free();
  }

 private:
  struct Request {
    RequestId id;
    LlamaModel::GenerateConfig gen;
    std::vector<int32_t> tokens;   // prompt + generated so far
    int64_t prompt_len = 0;
    int64_t generated  = 0;
    bool finished      = false;
    std::vector<std::shared_ptr<nn::KVCacheBase>> caches;  // per-layer paged
    std::unique_ptr<Sampler> sampler;  // null ⇒ greedy
    Tensor last_logits;                // logits feeding the next emit
  };

  // Blocks one request needs to hold `len` tokens in a single pool.
  int64_t blocks_for(int64_t len) const {
    return (len + cfg_.block_size - 1) / cfg_.block_size;
  }

  Tensor make_tokens(const int32_t* ids, int64_t n) const;
  int32_t pick_next(Request& r) const;     // argmax or sampled
  void admit_one(Request&& r);             // prefill + move to running
  void retire(Request& r);                 // free blocks, mark finished

  std::shared_ptr<LlamaModel> model_;
  EngineConfig cfg_;
  Device dev_;
  // Exactly one of these is populated per engine (FP vs INT8 KV), one
  // pool per layer, shared across every request's per-layer cache.
  std::vector<std::shared_ptr<nn::PagedKVPool>> pools_;
  std::vector<std::shared_ptr<nn::QuantizedPagedKVPool>> qpools_;

  std::deque<Request> waiting_;
  std::vector<RequestId> running_;                       // ids in requests_
  std::unordered_map<RequestId, Request> requests_;      // running + finished
  RequestId next_id_ = 0;
};

}  // namespace tesseract::models

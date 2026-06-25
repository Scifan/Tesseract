#include "tesseract/models/Scheduler.hpp"

#include <limits>
#include <span>
#include <utility>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/nn/PagedKVCache.hpp"
#include "tesseract/nn/QuantizedPagedKVCache.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::models {

namespace {

// Argmax over the vocab axis of the final position of a [1, S, V] logits
// tensor (b=0, s=S-1). Mirrors the greedy path in LlamaModel::generate so
// scheduler output is bit-identical to standalone generation.
int32_t argmax_last(const Tensor& logits) {
  const Tensor host = logits.device().is_cpu() ? logits : logits.to(cpu_device());
  const int64_t S = host.shape()[1];
  const int64_t V = host.shape()[2];
  int64_t best = 0;
  dispatch_float_with_half(host.dtype(), [&]<typename T>() {
    const T* p = host.data_ptr<T>() + (S - 1) * V;
    double best_val = -std::numeric_limits<double>::infinity();
    for (int64_t v = 0; v < V; ++v) {
      const double val = static_cast<double>(p[v]);
      if (val > best_val) { best_val = val; best = v; }
    }
  });
  return static_cast<int32_t>(best);
}

std::vector<float> last_row(const Tensor& logits) {
  const Tensor host = logits.device().is_cpu() ? logits : logits.to(cpu_device());
  const int64_t S = host.shape()[1];
  const int64_t V = host.shape()[2];
  std::vector<float> row(static_cast<std::size_t>(V));
  dispatch_float_with_half(host.dtype(), [&]<typename T>() {
    const T* p = host.data_ptr<T>() + (S - 1) * V;
    for (int64_t v = 0; v < V; ++v)
      row[static_cast<std::size_t>(v)] = static_cast<float>(p[v]);
  });
  return row;
}

}  // namespace

ContinuousBatchingScheduler::ContinuousBatchingScheduler(
    std::shared_ptr<LlamaModel> model, EngineConfig cfg)
    : model_(std::move(model)),
      cfg_(cfg),
      dev_(model_->embed_tokens()->weight().device()) {
  TESSERACT_CHECK(cfg_.block_size > 0 && cfg_.num_blocks > 0 &&
                      cfg_.max_seq_len > 0 && cfg_.max_batch_size > 0,
                  "EngineConfig fields must be positive");
  TESSERACT_CHECK(blocks_for(cfg_.max_seq_len) <= cfg_.num_blocks,
                  "EngineConfig: a single max-length request needs {} blocks "
                  "but the pool only has {} — raise num_blocks or lower "
                  "max_seq_len/block_size",
                  blocks_for(cfg_.max_seq_len), cfg_.num_blocks);
  if (cfg_.kv_int8) {
    qpools_ = model_->make_quantized_layer_pools(cfg_.num_blocks, cfg_.block_size);
  } else {
    pools_ = model_->make_layer_pools(cfg_.num_blocks, cfg_.block_size);
  }
}

Tensor ContinuousBatchingScheduler::make_tokens(const int32_t* ids,
                                                int64_t n) const {
  Tensor t = Tensor::empty({1, n}, DType::Int64, cpu_device());
  int64_t* p = t.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) p[i] = static_cast<int64_t>(ids[i]);
  return dev_.is_cpu() ? t : t.to(dev_);
}

RequestId ContinuousBatchingScheduler::add_request(
    std::vector<int32_t> prompt_ids, LlamaModel::GenerateConfig gen) {
  TESSERACT_CHECK(!prompt_ids.empty(),
                  "scheduler::add_request: prompt must be non-empty");
  TESSERACT_CHECK(gen.max_new_tokens >= 0,
                  "scheduler::add_request: max_new_tokens must be >= 0");
  const int64_t vocab = model_->config().vocab_size;
  for (int32_t id : prompt_ids) {
    TESSERACT_CHECK(id >= 0 && id < vocab,
                    "scheduler::add_request: prompt id {} out of range [0, {})",
                    id, vocab);
  }
  const int64_t prompt_len = static_cast<int64_t>(prompt_ids.size());
  TESSERACT_CHECK(prompt_len + gen.max_new_tokens <= cfg_.max_seq_len,
                  "scheduler::add_request: prompt({}) + max_new_tokens({}) > "
                  "max_seq_len({})",
                  prompt_len, gen.max_new_tokens, cfg_.max_seq_len);

  Request r;
  r.id = next_id_++;
  r.gen = gen;
  r.tokens = std::move(prompt_ids);
  r.prompt_len = prompt_len;
  if (gen.do_sample) r.sampler = std::make_unique<Sampler>(gen.sampling, gen.seed);
  waiting_.push_back(std::move(r));
  return waiting_.back().id;
}

int32_t ContinuousBatchingScheduler::pick_next(Request& r) const {
  if (!r.sampler) return argmax_last(r.last_logits);
  const std::vector<float> row = last_row(r.last_logits);
  return r.sampler->sample(
      std::span<const float>(row.data(), row.size()),
      std::span<const int32_t>(r.tokens.data(), r.tokens.size()));
}

void ContinuousBatchingScheduler::admit_one(Request&& r_in) {
  const RequestId id = r_in.id;
  r_in.caches = cfg_.kv_int8
                    ? model_->make_quantized_paged_kv_caches(qpools_,
                                                             cfg_.max_seq_len)
                    : model_->make_paged_kv_caches(pools_, cfg_.max_seq_len);
  // Prefill the whole prompt in one chunked-decode step.
  Tensor toks = make_tokens(r_in.tokens.data(), r_in.prompt_len);
  r_in.last_logits = model_->forward_step(toks, r_in.caches);
  r_in.generated = 0;

  auto [it, ok] = requests_.emplace(id, std::move(r_in));
  Request& r = it->second;
  if (r.gen.max_new_tokens <= 0) {
    retire(r);  // nothing to generate; free the prompt blocks
  } else {
    running_.push_back(id);
  }
}

void ContinuousBatchingScheduler::retire(Request& r) {
  for (auto& cache : r.caches) {
    if (auto* paged = dynamic_cast<nn::PagedKVCache*>(cache.get())) {
      paged->reset();  // return this request's blocks to the shared pool
    } else if (auto* qpaged =
                   dynamic_cast<nn::QuantizedPagedKVCache*>(cache.get())) {
      qpaged->reset();
    }
  }
  r.caches.clear();
  r.last_logits = Tensor();
  r.finished = true;
}

bool ContinuousBatchingScheduler::step() {
  // 1) Admission: fill batch room with waiting requests whose prompt fits.
  while (!waiting_.empty() &&
         static_cast<int64_t>(running_.size()) < cfg_.max_batch_size) {
    const int64_t need = blocks_for(waiting_.front().prompt_len);
    if (free_blocks() < need) {
      // Not enough blocks right now. If nothing is running to reclaim
      // from, this is a permanent stall — surface it loudly.
      TESSERACT_CHECK(!running_.empty(),
                      "scheduler: request needs {} blocks, {} free, and no "
                      "running requests to reclaim from (pool too small)",
                      need, free_blocks());
      break;
    }
    Request r = std::move(waiting_.front());
    waiting_.pop_front();
    admit_one(std::move(r));
  }

  // 2) Decode. First sample the next token for every running request
  //    (from the logits the previous step produced), then fold the still-
  //    active sequences into a single batched forward: embedding, all
  //    projections, the FFN, and the LM head run once over the whole
  //    active set (Wave 10 / B-032 compute-batched decode), while each
  //    sequence's attention threads through its own paged cache. On CPU
  //    this is bit-identical to per-request decode (each GEMM row is
  //    independent); on CUDA it matches within float tolerance.
  std::vector<RequestId> still;
  std::vector<int32_t> next_ids;     // one fed token per still-active request
  still.reserve(running_.size());
  next_ids.reserve(running_.size());
  for (RequestId id : running_) {
    Request& r = requests_.at(id);
    const int32_t next = pick_next(r);
    r.tokens.push_back(next);
    ++r.generated;
    const bool stop =
        (r.gen.eos_token_id >= 0 && next == r.gen.eos_token_id) ||
        (r.generated >= r.gen.max_new_tokens);
    if (stop) {
      retire(r);
    } else {
      still.push_back(id);
      next_ids.push_back(next);
    }
  }

  if (!still.empty()) {
    const int64_t A = static_cast<int64_t>(still.size());
    Tensor toks = make_tokens(next_ids.data(), A);     // [A, 1]
    toks = ops::reshape(toks, Shape({A, 1}));
    std::vector<std::vector<std::shared_ptr<nn::KVCacheBase>>> batched;
    batched.reserve(still.size());
    for (RequestId id : still) batched.push_back(requests_.at(id).caches);

    Tensor logits = model_->forward_step_batched(toks, batched);  // [A,1,V]
    for (int64_t i = 0; i < A; ++i) {
      requests_.at(still[static_cast<std::size_t>(i)]).last_logits =
          logits.narrow(/*dim=*/0, i, 1);              // [1, 1, V]
    }
  }
  running_.swap(still);

  return has_pending();
}

void ContinuousBatchingScheduler::run() {
  while (has_pending()) step();
}

bool ContinuousBatchingScheduler::is_finished(RequestId id) const {
  auto it = requests_.find(id);
  return it != requests_.end() && it->second.finished;
}

const std::vector<int32_t>& ContinuousBatchingScheduler::result(
    RequestId id) const {
  auto it = requests_.find(id);
  TESSERACT_CHECK(it != requests_.end(),
                  "scheduler::result: unknown request id {}", id);
  return it->second.tokens;
}

}  // namespace tesseract::models

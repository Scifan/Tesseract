#include "tesseract/models/Disaggregated.hpp"

#include <limits>
#include <utility>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::models {

namespace {

// Argmax over the vocab axis at the last position of a [1, S, V] logits
// tensor — matches LlamaModel::generate so output stays bit-identical.
int32_t argmax_last(const Tensor& logits) {
  const Tensor host = logits.device().is_cpu() ? logits : logits.to(cpu_device());
  const int64_t S = host.shape()[1];
  const int64_t V = host.shape()[2];
  int32_t best = 0;
  dispatch_float_with_half(host.dtype(), [&]<typename T>() {
    const T* p = host.data_ptr<T>() + (S - 1) * V;
    double best_val = -std::numeric_limits<double>::infinity();
    for (int64_t v = 0; v < V; ++v) {
      const double val = static_cast<double>(p[v]);
      if (val > best_val) { best_val = val; best = static_cast<int32_t>(v); }
    }
  });
  return best;
}

}  // namespace

DisaggregatedEngine::DisaggregatedEngine(std::shared_ptr<LlamaModel> model)
    : model_(std::move(model)) {
  TESSERACT_CHECK(model_, "DisaggregatedEngine: null model");
}

KvTransfer DisaggregatedEngine::prefill(const std::vector<int32_t>& prompt_ids) {
  TESSERACT_CHECK(!prompt_ids.empty(), "DisaggregatedEngine::prefill: empty prompt");
  NoGradGuard no_grad;
  const Device dev = model_->embed_tokens()->weight().device();
  const int64_t P = static_cast<int64_t>(prompt_ids.size());

  Tensor toks = Tensor::empty({1, P}, DType::Int64, cpu_device());
  int64_t* tp = toks.data_ptr<int64_t>();
  for (int64_t i = 0; i < P; ++i) tp[i] = static_cast<int64_t>(prompt_ids[i]);
  if (!dev.is_cpu()) toks = toks.to(dev);

  // Prefill into the prefill role's own caches (sized to the prompt only).
  auto caches = model_->make_kv_caches(/*batch=*/1, P);
  const Tensor logits = model_->forward_step(toks, caches);

  KvTransfer t;
  t.prompt_len = P;
  t.first_token = argmax_last(logits);
  t.layer_kv.reserve(caches.size());
  for (const auto& c : caches) {
    // Export an owning, contiguous copy — this is the blob that crosses to
    // the decode worker; the prefill caches are dropped on return.
    t.layer_kv.emplace_back(c->keys_view().contiguous(),
                            c->values_view().contiguous());
  }
  return t;
}

DisaggregatedEngine::Result DisaggregatedEngine::decode(
    const KvTransfer& transfer, const std::vector<int32_t>& prompt_ids,
    int64_t max_new_tokens, int32_t eos_token_id) {
  TESSERACT_CHECK(static_cast<int64_t>(transfer.layer_kv.size()) ==
                      model_->num_layers(),
                  "DisaggregatedEngine::decode: transfer has {} layers, model "
                  "has {}", transfer.layer_kv.size(), model_->num_layers());
  TESSERACT_CHECK(static_cast<int64_t>(prompt_ids.size()) == transfer.prompt_len,
                  "DisaggregatedEngine::decode: prompt_ids length {} != "
                  "transfer.prompt_len {}", prompt_ids.size(), transfer.prompt_len);
  NoGradGuard no_grad;
  const Device dev = model_->embed_tokens()->weight().device();
  const int64_t P = transfer.prompt_len;

  auto make_tokens = [&](int32_t id) -> Tensor {
    Tensor t = Tensor::empty({1, 1}, DType::Int64, cpu_device());
    t.data_ptr<int64_t>()[0] = static_cast<int64_t>(id);
    return dev.is_cpu() ? t : t.to(dev);
  };

  // Import the migrated KV into the decode role's separate caches.
  auto caches = model_->make_kv_caches(/*batch=*/1, P + max_new_tokens);
  for (std::size_t l = 0; l < caches.size(); ++l) {
    caches[l]->append(transfer.layer_kv[l].first, transfer.layer_kv[l].second);
  }

  Result r;
  r.prompt_len = P;
  r.tokens = prompt_ids;

  if (max_new_tokens <= 0) return r;

  // The first generated token came from the prefill pass; commit it.
  r.tokens.push_back(transfer.first_token);
  int64_t generated = 1;
  if (eos_token_id >= 0 && transfer.first_token == eos_token_id) return r;

  // Continue: feed the just-emitted token through the migrated KV.
  int32_t last = transfer.first_token;
  while (generated < max_new_tokens) {
    const Tensor logits = model_->forward_step(make_tokens(last), caches);
    ++r.decode_steps;
    const int32_t next = argmax_last(logits);
    r.tokens.push_back(next);
    ++generated;
    last = next;
    if (eos_token_id >= 0 && next == eos_token_id) break;
  }
  return r;
}

DisaggregatedEngine::Result DisaggregatedEngine::generate(
    const std::vector<int32_t>& prompt_ids, int64_t max_new_tokens,
    int32_t eos_token_id) {
  const KvTransfer t = prefill(prompt_ids);
  return decode(t, prompt_ids, max_new_tokens, eos_token_id);
}

}  // namespace tesseract::models

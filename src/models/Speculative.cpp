#include "tesseract/models/Speculative.hpp"

#include <limits>
#include <utility>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::models {

namespace {

// Argmax over the vocab axis at sequence position `s` of a [1, S, V] logits
// tensor. Reads on host; handles FP32/FP64/FP16/BF16 — mirrors the greedy
// path in LlamaModel::generate so the speculative output stays bit-identical.
int32_t argmax_at(const Tensor& logits, int64_t s) {
  const Tensor host = logits.device().is_cpu() ? logits : logits.to(cpu_device());
  const int64_t V = host.shape()[2];
  int32_t best = 0;
  dispatch_float_with_half(host.dtype(), [&]<typename T>() {
    const T* p = host.data_ptr<T>() + s * V;
    double best_val = -std::numeric_limits<double>::infinity();
    for (int64_t v = 0; v < V; ++v) {
      const double val = static_cast<double>(p[v]);
      if (val > best_val) { best_val = val; best = static_cast<int32_t>(v); }
    }
  });
  return best;
}

}  // namespace

SpeculativeDecoder::SpeculativeDecoder(std::shared_ptr<LlamaModel> target,
                                       std::shared_ptr<LlamaModel> draft,
                                       int gamma)
    : target_(std::move(target)), draft_(std::move(draft)), gamma_(gamma) {
  TESSERACT_CHECK(target_ && draft_, "SpeculativeDecoder: null model");
  TESSERACT_CHECK(gamma_ >= 1, "SpeculativeDecoder: gamma must be >= 1");
  TESSERACT_CHECK(target_->config().vocab_size == draft_->config().vocab_size,
                  "SpeculativeDecoder: target/draft vocab mismatch ({} vs {})",
                  target_->config().vocab_size, draft_->config().vocab_size);
}

SpeculativeDecoder::Result SpeculativeDecoder::generate(
    const std::vector<int32_t>& prompt_ids, int64_t max_new_tokens,
    int32_t eos_token_id) {
  TESSERACT_CHECK(!prompt_ids.empty(), "SpeculativeDecoder: empty prompt");
  NoGradGuard no_grad;

  const Device dev = target_->embed_tokens()->weight().device();
  const int64_t P = static_cast<int64_t>(prompt_ids.size());
  // Headroom: prompt + all requested tokens + one over-shoot block.
  const int64_t max_len = P + max_new_tokens + gamma_ + 2;

  auto t_caches = target_->make_kv_caches(/*batch=*/1, max_len);
  auto d_caches = draft_->make_kv_caches(/*batch=*/1, max_len);

  auto make_tokens = [&](const int32_t* p, int64_t n) -> Tensor {
    Tensor t = Tensor::empty({1, n}, DType::Int64, cpu_device());
    int64_t* q = t.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) q[i] = static_cast<int64_t>(p[i]);
    return dev.is_cpu() ? t : t.to(dev);
  };
  auto set_len = [](auto& caches, int64_t len) {
    for (auto& c : caches) c->set_current_len(len);
  };

  Result r;
  r.tokens = prompt_ids;

  // Prefill both models over the prompt; carry the logits that predict the
  // next position (index P) for each.
  Tensor t_logits = target_->forward_step(make_tokens(prompt_ids.data(), P), t_caches);
  Tensor d_logits = draft_->forward_step(make_tokens(prompt_ids.data(), P), d_caches);
  ++r.target_forwards;
  int32_t t_next = argmax_at(t_logits, P - 1);  // target argmax for position P
  int32_t d_next = argmax_at(d_logits, P - 1);  // draft argmax for position P

  int64_t generated = 0;
  bool stop = false;

  while (generated < max_new_tokens && !stop) {
    const int64_t len = static_cast<int64_t>(r.tokens.size());

    // ---- Draft phase: propose gamma tokens ----------------------------
    std::vector<int32_t> q;
    q.reserve(static_cast<std::size_t>(gamma_));
    int32_t dl = d_next;
    for (int k = 0; k < gamma_; ++k) {
      q.push_back(dl);
      if (k + 1 < gamma_) {
        Tensor o = draft_->forward_step(make_tokens(&dl, 1), d_caches);
        dl = argmax_at(o, 0);
      }
    }
    // Draft cache now holds len + (gamma-1) entries (q[0..gamma-2] appended).
    r.proposed += gamma_;

    // ---- Verify phase: one target forward over all gamma proposals ----
    Tensor tv = target_->forward_step(
        make_tokens(q.data(), static_cast<int64_t>(q.size())), t_caches);
    ++r.target_forwards;
    ++r.rounds;
    // Expected (target greedy) tokens at positions len..len+gamma:
    //   e[0] = t_next (predicts pos len, from before this block)
    //   e[i] = argmax(tv[:, i-1, :]) (predicts pos len+i)
    // Accept q[i] while q[i] == e[i]; n = accepted count in [0, gamma].
    int n = 0;
    while (n < gamma_) {
      const int32_t e = (n == 0) ? t_next : argmax_at(tv, n - 1);
      if (q[static_cast<std::size_t>(n)] != e) break;
      ++n;
    }
    const int32_t correction = (n == 0) ? t_next : argmax_at(tv, n - 1);
    r.accepted += n;

    // Committed this round: q[0..n-1] (accepted) + correction. The committed
    // tokens equal the target's own greedy argmaxes for positions len..len+n,
    // which is exactly what target-only greedy decoding would emit.
    for (int i = 0; i < n && generated < max_new_tokens && !stop; ++i) {
      r.tokens.push_back(q[static_cast<std::size_t>(i)]);
      ++generated;
      if (eos_token_id >= 0 && q[static_cast<std::size_t>(i)] == eos_token_id)
        stop = true;
    }
    bool correction_committed = false;
    if (!stop && generated < max_new_tokens) {
      r.tokens.push_back(correction);
      ++generated;
      correction_committed = true;
      if (eos_token_id >= 0 && correction == eos_token_id) stop = true;
    }

    if (stop || generated >= max_new_tokens) break;

    // ---- Resync caches to the committed length ------------------------
    // Accepted prefix q[0..n-1] is already correctly cached in both models
    // (positions len..len+n-1). Drop everything past it, then re-feed the
    // correction so both caches + the carried next-logits match the
    // committed sequence exactly.
    set_len(t_caches, len + n);
    set_len(d_caches, len + n);
    if (correction_committed) {
      Tensor to = target_->forward_step(make_tokens(&correction, 1), t_caches);
      Tensor doo = draft_->forward_step(make_tokens(&correction, 1), d_caches);
      ++r.target_forwards;
      t_next = argmax_at(to, 0);
      d_next = argmax_at(doo, 0);
    }
  }

  return r;
}

}  // namespace tesseract::models

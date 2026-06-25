#include "tesseract/models/MambaModel.hpp"

#include <limits>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/nn/ModuleList.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::models {

MambaModel::MambaModel(const MambaConfig& config) : config_(config) {
  TESSERACT_CHECK(config.vocab_size > 0 && config.hidden_size > 0 &&
                  config.num_hidden_layers > 0,
                  "MambaModel: vocab/hidden/num_layers must be positive");

  embed_tokens_ = std::make_shared<nn::Embedding>(config.vocab_size,
                                                  config.hidden_size,
                                                  config.dtype);
  register_module("embed_tokens", embed_tokens_);

  auto layers_holder = std::make_shared<nn::ModuleList>();
  auto norms_holder  = std::make_shared<nn::ModuleList>();
  layers_.reserve(static_cast<std::size_t>(config.num_hidden_layers));
  norms_.reserve(static_cast<std::size_t>(config.num_hidden_layers));
  for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
    auto blk = std::make_shared<nn::Mamba>(config.hidden_size, config.d_state,
                                           config.d_conv, config.expand,
                                           config.dt_rank, config.dtype);
    auto nrm = std::make_shared<nn::RMSNorm>(config.hidden_size,
                                             config.rms_norm_eps, config.dtype);
    layers_holder->append(blk);
    norms_holder->append(nrm);
    layers_.push_back(std::move(blk));
    norms_.push_back(std::move(nrm));
  }
  register_module("layers", layers_holder);
  register_module("norms", norms_holder);

  norm_ = std::make_shared<nn::RMSNorm>(config.hidden_size, config.rms_norm_eps,
                                        config.dtype);
  register_module("norm", norm_);

  lm_head_ = std::make_shared<nn::Linear>(config.hidden_size, config.vocab_size,
                                          /*use_bias=*/false, config.dtype);
  register_module("lm_head", lm_head_);
}

Tensor MambaModel::forward(const Tensor& tokens) {
  TESSERACT_CHECK(tokens.defined() && tokens.dtype() == DType::Int64 &&
                  tokens.rank() == 2,
                  "MambaModel::forward: tokens must be Int64 [B, S]");
  NoGradGuard nogg;
  Tensor x = embed_tokens_->forward(tokens);  // [B, S, D]
  for (std::size_t l = 0; l < layers_.size(); ++l) {
    Tensor h = norms_[l]->forward(x);
    x = ops::add(x, layers_[l]->forward(h));
  }
  x = norm_->forward(x);
  return lm_head_->forward(x);  // [B, S, V]
}

Tensor MambaModel::forward_step(const Tensor& tokens,
                                std::vector<nn::SSMStateCache>& caches) {
  TESSERACT_CHECK(tokens.defined() && tokens.dtype() == DType::Int64 &&
                  tokens.rank() == 2 && tokens.shape()[1] == 1,
                  "MambaModel::forward_step: tokens must be Int64 [B, 1]");
  TESSERACT_CHECK(static_cast<int64_t>(caches.size()) == num_layers(),
                  "MambaModel::forward_step: expected {} caches, got {}",
                  num_layers(), caches.size());
  NoGradGuard nogg;
  Tensor x = embed_tokens_->forward(tokens);  // [B, 1, D]
  for (std::size_t l = 0; l < layers_.size(); ++l) {
    Tensor h = norms_[l]->forward(x);
    x = ops::add(x, layers_[l]->forward_step(h, caches[l]));
  }
  x = norm_->forward(x);
  return lm_head_->forward(x);  // [B, 1, V]
}

std::vector<nn::SSMStateCache> MambaModel::make_state_caches(
    int64_t batch) const {
  std::vector<nn::SSMStateCache> caches;
  caches.reserve(layers_.size());
  for (const auto& blk : layers_) caches.push_back(blk->make_state_cache(batch));
  return caches;
}

namespace {

int32_t argmax_last_position(const Tensor& logits) {
  const Tensor host = logits.device().is_cpu() ? logits
                                               : logits.to(cpu_device());
  const int64_t S = host.shape()[1];
  const int64_t V = host.shape()[2];
  int64_t best = 0;
  dispatch_float(host.dtype(), [&]<typename T>() {
    const T* p = host.data_ptr<T>() + (S - 1) * V;  // b=0, s=S-1
    double best_val = -std::numeric_limits<double>::infinity();
    for (int64_t v = 0; v < V; ++v) {
      const double val = static_cast<double>(p[v]);
      if (val > best_val) { best_val = val; best = v; }
    }
  });
  return static_cast<int32_t>(best);
}

std::vector<float> last_logits_row(const Tensor& logits) {
  const Tensor host = logits.device().is_cpu() ? logits
                                               : logits.to(cpu_device());
  const int64_t S = host.shape()[1];
  const int64_t V = host.shape()[2];
  std::vector<float> row(static_cast<std::size_t>(V));
  dispatch_float(host.dtype(), [&]<typename T>() {
    const T* p = host.data_ptr<T>() + (S - 1) * V;
    for (int64_t v = 0; v < V; ++v)
      row[static_cast<std::size_t>(v)] = static_cast<float>(p[v]);
  });
  return row;
}

}  // namespace

std::vector<int32_t> MambaModel::generate(const std::vector<int32_t>& prompt_ids,
                                          const GenerateConfig& cfg) {
  TESSERACT_CHECK(!prompt_ids.empty(),
                  "MambaModel::generate: prompt_ids must be non-empty");
  TESSERACT_CHECK(cfg.max_new_tokens >= 0,
                  "MambaModel::generate: max_new_tokens must be >= 0");
  for (int32_t id : prompt_ids)
    TESSERACT_CHECK(id >= 0 && id < config_.vocab_size,
                    "MambaModel::generate: prompt id {} out of range [0, {})",
                    id, config_.vocab_size);

  NoGradGuard nogg;
  const Device dev = embed_tokens_->weight().device();
  auto caches = make_state_caches(/*batch=*/1);

  auto make_token = [&](int32_t id) -> Tensor {
    Tensor t = Tensor::empty(Shape({1, 1}), DType::Int64, cpu_device());
    t.data_ptr<int64_t>()[0] = static_cast<int64_t>(id);
    return dev.is_cpu() ? t : t.to(dev);
  };

  Sampler sampler(cfg.sampling, cfg.seed);
  std::vector<int32_t> result = prompt_ids;

  // Recurrent prefill: stream the prompt one token at a time so each layer's
  // SSM/conv state is populated; the logits after the last prompt token drive
  // the first generated token.
  Tensor logits;
  for (int32_t id : prompt_ids) logits = forward_step(make_token(id), caches);

  auto pick_next = [&](const Tensor& lg) -> int32_t {
    if (!cfg.do_sample) return argmax_last_position(lg);
    const std::vector<float> row = last_logits_row(lg);
    return sampler.sample(std::span<const float>(row.data(), row.size()),
                          std::span<const int32_t>(result.data(),
                                                   result.size()));
  };

  for (int64_t i = 0; i < cfg.max_new_tokens; ++i) {
    const int32_t next = pick_next(logits);
    result.push_back(next);
    if (cfg.eos_token_id >= 0 && next == cfg.eos_token_id) break;
    if (i + 1 < cfg.max_new_tokens) logits = forward_step(make_token(next),
                                                          caches);
  }
  return result;
}

}  // namespace tesseract::models

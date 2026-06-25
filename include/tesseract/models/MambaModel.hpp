#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/models/Sampler.hpp"
#include "tesseract/nn/Embedding.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Mamba.hpp"
#include "tesseract/nn/Module.hpp"
#include "tesseract/nn/RMSNorm.hpp"
#include "tesseract/nn/SSMStateCache.hpp"

namespace tesseract::models {

// Minimal Mamba-1 language model (M4 Track A2 / B-039) — the end-to-end
// vehicle proving the SSM stack: embed → N × (RMSNorm → Mamba → residual) →
// final RMSNorm → lm_head. Mirrors `LlamaModel`'s shape contract so the
// Wave-6 `Sampler` (and, later, the scheduler) drives it unchanged.
struct MambaConfig {
  int64_t vocab_size        = 32000;
  int64_t hidden_size       = 256;   // d_model
  int64_t num_hidden_layers = 4;
  int64_t d_state           = 16;    // SSM state width N
  int64_t d_conv            = 4;     // causal conv kernel
  int64_t expand            = 2;     // d_inner = expand * d_model
  int64_t dt_rank           = 0;     // 0 ⇒ ceil(d_model / 16)
  double  rms_norm_eps      = 1e-5;
  DType   dtype             = DType::Float32;
  int32_t eos_token_id      = -1;
};

class MambaModel : public nn::Module {
 public:
  explicit MambaModel(const MambaConfig& config);

  // Full-sequence forward. `tokens` is Int64 [B, S]; returns [B, S, vocab].
  Tensor forward(const Tensor& tokens) override;

  // Single-token incremental forward threading one `SSMStateCache` per layer.
  // `tokens` is Int64 [B, 1]; returns [B, 1, vocab].
  Tensor forward_step(const Tensor& tokens,
                      std::vector<nn::SSMStateCache>& caches);

  // One zero-initialized decode cache per layer.
  std::vector<nn::SSMStateCache> make_state_caches(int64_t batch) const;

  struct GenerateConfig {
    int64_t        max_new_tokens = 32;
    int32_t        eos_token_id   = -1;
    bool           do_sample      = false;
    SamplingParams sampling{};
    uint64_t       seed           = 0;
  };
  std::vector<int32_t> generate(const std::vector<int32_t>& prompt_ids,
                                const GenerateConfig& cfg);

  const MambaConfig& config() const noexcept { return config_; }
  int64_t num_layers() const noexcept {
    return static_cast<int64_t>(layers_.size());
  }
  const std::vector<std::shared_ptr<nn::Mamba>>& layers() const { return layers_; }

 private:
  MambaConfig config_;
  std::shared_ptr<nn::Embedding>             embed_tokens_;
  std::vector<std::shared_ptr<nn::Mamba>>    layers_;
  std::vector<std::shared_ptr<nn::RMSNorm>>  norms_;
  std::shared_ptr<nn::RMSNorm>               norm_;
  std::shared_ptr<nn::Linear>                lm_head_;
};

}  // namespace tesseract::models

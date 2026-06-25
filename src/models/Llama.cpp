#include "tesseract/models/Llama.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Storage.hpp"
#include "tesseract/io/SafeTensors.hpp"
#include "tesseract/nn/FeedForward.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/ModuleList.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"
#include "tesseract/nn/PagedKVCache.hpp"
#include "tesseract/nn/QuantizedKVCache.hpp"
#include "tesseract/nn/QuantizedPagedKVCache.hpp"
#include "tesseract/quant/Scheme.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::models {

// ---------------------------------------------------------------------------
// LlamaConfig presets
// ---------------------------------------------------------------------------

LlamaConfig LlamaConfig::llama_3_2_1b() {
  LlamaConfig c;
  c.vocab_size               = 128256;
  c.hidden_size              = 2048;
  c.num_hidden_layers        = 16;
  c.num_attention_heads      = 32;     // published 32 Q, 8 KV (GQA)
  c.num_key_value_heads      = 8;
  c.intermediate_size        = 8192;
  c.max_position_embeddings  = 131072;
  c.rope_theta               = 500000.0;
  c.rms_norm_eps             = 1e-5;
  c.tie_word_embeddings      = true;
  c.dtype                    = DType::Float32;
  return c;
}

// ---------------------------------------------------------------------------
// HF config.json → LlamaConfig (Wave 16 / B-033)
//
// A focused scalar extractor for the flat top-level fields of an HF
// `config.json`. We deliberately avoid pulling in a full JSON dependency:
// config.json's architecture fields are all top-level `"key": <scalar>`
// pairs (numbers, booleans, or short strings), so a key-seek + scalar-read
// is sufficient and dependency-free. Nested objects (e.g. `rope_scaling`)
// are simply ignored — their keys won't match the ones we look for.
// ---------------------------------------------------------------------------
namespace {

// Locate the value token following a top-level `"key"`: returns the index
// just past the colon, or npos if the key is absent. Skips matches that are
// not immediately followed (modulo whitespace) by ':' so a key appearing as
// a *value* string can't be mistaken for a field.
std::size_t find_value_pos(const std::string& s, const std::string& key) {
  const std::string quoted = "\"" + key + "\"";
  std::size_t from = 0;
  while (true) {
    const std::size_t k = s.find(quoted, from);
    if (k == std::string::npos) return std::string::npos;
    std::size_t i = k + quoted.size();
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i < s.size() && s[i] == ':') {
      ++i;
      while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
      return i;
    }
    from = k + quoted.size();
  }
}

std::optional<double> read_number(const std::string& s, const std::string& key) {
  const std::size_t i = find_value_pos(s, key);
  if (i == std::string::npos) return std::nullopt;
  std::size_t j = i;
  // A JSON number: optional sign, digits, '.', exponent.
  while (j < s.size() &&
         (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '-' ||
          s[j] == '+' || s[j] == '.' || s[j] == 'e' || s[j] == 'E'))
    ++j;
  if (j == i) return std::nullopt;
  try {
    return std::stod(s.substr(i, j - i));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<bool> read_bool(const std::string& s, const std::string& key) {
  const std::size_t i = find_value_pos(s, key);
  if (i == std::string::npos) return std::nullopt;
  if (s.compare(i, 4, "true") == 0) return true;
  if (s.compare(i, 5, "false") == 0) return false;
  return std::nullopt;
}

std::optional<std::string> read_string(const std::string& s,
                                       const std::string& key) {
  const std::size_t i = find_value_pos(s, key);
  if (i == std::string::npos || i >= s.size() || s[i] != '"') return std::nullopt;
  const std::size_t start = i + 1;
  const std::size_t end = s.find('"', start);
  if (end == std::string::npos) return std::nullopt;
  return s.substr(start, end - start);
}

}  // namespace

LlamaConfig LlamaConfig::from_json(const std::string& json_text) {
  const bool looks_like_object =
      json_text.find('{') != std::string::npos;
  TESSERACT_CHECK(looks_like_object,
                  "LlamaConfig::from_json: not a JSON object");
  LlamaConfig c;
  // The five architecture fields are required — a config.json missing any
  // of them isn't a usable model spec, and silently defaulting (e.g.
  // vocab_size → 32000) would only surface as a confusing shape mismatch
  // deep inside the safetensors loader.
  const auto vocab  = read_number(json_text, "vocab_size");
  const auto hidden = read_number(json_text, "hidden_size");
  const auto nlay   = read_number(json_text, "num_hidden_layers");
  const auto nhead  = read_number(json_text, "num_attention_heads");
  const auto ff     = read_number(json_text, "intermediate_size");
  TESSERACT_CHECK(vocab && hidden && nlay && nhead && ff,
                  "LlamaConfig::from_json: missing one of the required fields "
                  "vocab_size / hidden_size / num_hidden_layers / "
                  "num_attention_heads / intermediate_size");
  c.vocab_size        = static_cast<int64_t>(*vocab);
  c.hidden_size       = static_cast<int64_t>(*hidden);
  c.num_hidden_layers = static_cast<int64_t>(*nlay);
  c.num_attention_heads = static_cast<int64_t>(*nhead);
  c.intermediate_size = static_cast<int64_t>(*ff);
  if (auto v = read_number(json_text, "num_key_value_heads"))
    c.num_key_value_heads = static_cast<int64_t>(*v);
  // Mixtral-style MoE fields (B-038). HF uses `num_local_experts`; accept the
  // generic `num_experts` alias too.
  if (auto v = read_number(json_text, "num_local_experts"))
    c.num_experts = static_cast<int64_t>(*v);
  else if (auto v2 = read_number(json_text, "num_experts"))
    c.num_experts = static_cast<int64_t>(*v2);
  if (auto v = read_number(json_text, "num_experts_per_tok"))
    c.num_experts_per_tok = static_cast<int64_t>(*v);
  if (auto v = read_number(json_text, "max_position_embeddings"))
    c.max_position_embeddings = static_cast<int64_t>(*v);
  if (auto v = read_number(json_text, "rope_theta")) c.rope_theta = *v;
  if (auto v = read_number(json_text, "rms_norm_eps")) c.rms_norm_eps = *v;
  if (auto v = read_bool(json_text, "tie_word_embeddings"))
    c.tie_word_embeddings = *v;
  if (auto v = read_number(json_text, "bos_token_id"))
    c.bos_token_id = static_cast<int32_t>(*v);
  if (auto v = read_number(json_text, "eos_token_id"))
    c.eos_token_id = static_cast<int32_t>(*v);
  if (auto v = read_string(json_text, "torch_dtype")) {
    if (*v == "float32" || *v == "float")        c.dtype = DType::Float32;
    else if (*v == "float16" || *v == "half")    c.dtype = DType::Float16;
    else if (*v == "bfloat16")                   c.dtype = DType::BFloat16;
    else if (*v == "float64" || *v == "double")  c.dtype = DType::Float64;
    // unknown → leave default
  }
  TESSERACT_CHECK(c.vocab_size > 0 && c.hidden_size > 0 &&
                      c.num_hidden_layers > 0 && c.num_attention_heads > 0 &&
                      c.intermediate_size > 0,
                  "LlamaConfig::from_json: required fields must be positive");
  return c;
}

LlamaConfig LlamaConfig::from_json_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  TESSERACT_CHECK(in.good(), "LlamaConfig::from_json_file: cannot open {}", path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return from_json(ss.str());
}

// ---------------------------------------------------------------------------
// Name translation: local (Tesseract module tree) → Hugging Face safetensors
// ---------------------------------------------------------------------------

namespace {

void replace_first(std::string& s, std::string_view from, std::string_view to) {
  const auto pos = s.find(from);
  if (pos != std::string::npos) {
    s.replace(pos, from.size(), to);
  }
}

}  // namespace

std::string llama_local_to_hf_name(std::string_view local) {
  // Special cases first — lm_head keeps its top-level name in HF.
  if (local == "lm_head.weight") return "lm_head.weight";

  std::string s(local);

  // Everything else lives under "model." in HF.
  s = "model." + s;

  // Rename children that differ between HF's naming and ours.
  //   our "norm_1" (first block RMSNorm) == HF "input_layernorm"
  //   our "norm_2" (second block RMSNorm) == HF "post_attention_layernorm"
  //   our "attn"                           == HF "self_attn"
  //   our "ffn"                            == HF "mlp"
  // The block-index segment between `.layers.` and the child name is preserved
  // verbatim since we use identical numeric indexing.
  replace_first(s, ".attn.",   ".self_attn.");
  replace_first(s, ".ffn.",    ".mlp.");
  replace_first(s, ".norm_1.", ".input_layernorm.");
  replace_first(s, ".norm_2.", ".post_attention_layernorm.");
  return s;
}

// ---------------------------------------------------------------------------
// LlamaModel construction
// ---------------------------------------------------------------------------

LlamaModel::LlamaModel(const LlamaConfig& config) : config_(config) {
  TESSERACT_CHECK(config.vocab_size > 0 && config.hidden_size > 0 &&
                      config.num_hidden_layers > 0 &&
                      config.num_attention_heads > 0 &&
                      config.intermediate_size > 0 &&
                      config.max_position_embeddings > 0,
                  "LlamaConfig: all dimensions must be positive");
  TESSERACT_CHECK(config.hidden_size % config.num_attention_heads == 0,
                  "LlamaConfig: hidden_size ({}) must be divisible by num_attention_heads ({})",
                  config.hidden_size, config.num_attention_heads);
  TESSERACT_CHECK(config.num_attention_heads % config.kv_heads() == 0,
                  "LlamaConfig: num_attention_heads ({}) must be divisible by "
                  "num_key_value_heads ({})",
                  config.num_attention_heads, config.kv_heads());

  embed_tokens_ = std::make_shared<nn::Embedding>(config.vocab_size,
                                                  config.hidden_size,
                                                  config.dtype);
  register_module("embed_tokens", embed_tokens_);

  // Build all blocks first, then wrap them in a plain `nn::Module` holder
  // registered under the name `layers`. That gives each block a qualified
  // name of `layers.{i}.…`, which is exactly HF's convention (and what
  // `llama_local_to_hf_name` expects).
  layers_.reserve(static_cast<std::size_t>(config.num_hidden_layers));
  auto layers_holder = std::make_shared<nn::ModuleList>();
  for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
    auto block = std::make_shared<nn::TransformerBlock>(
        /*d_model=*/config.hidden_size,
        /*num_heads=*/config.num_attention_heads,
        /*d_ff=*/config.intermediate_size,
        /*norm_eps=*/config.rms_norm_eps,
        /*causal=*/true,
        /*use_bias=*/false,       // Llama has no biases anywhere
        /*dtype=*/config.dtype,
        /*rope_base=*/config.rope_theta,
        /*rope_max_seq=*/config.max_position_embeddings,
        /*num_kv_heads=*/config.kv_heads(),
        /*num_experts=*/config.num_experts,
        /*num_experts_per_tok=*/config.num_experts_per_tok);
    layers_holder->append(block);
    layers_.push_back(std::move(block));
  }
  register_module("layers", layers_holder);

  norm_ = std::make_shared<nn::RMSNorm>(config.hidden_size,
                                        config.rms_norm_eps,
                                        config.dtype);
  register_module("norm", norm_);

  lm_head_ = std::make_shared<nn::Linear>(config.hidden_size,
                                          config.vocab_size,
                                          /*use_bias=*/false,
                                          config.dtype);
  register_module("lm_head", lm_head_);
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

Tensor LlamaModel::forward(const Tensor& tokens) {
  TESSERACT_CHECK(tokens.defined(), "LlamaModel::forward: tokens undefined");
  TESSERACT_CHECK(tokens.dtype() == DType::Int64,
                  "LlamaModel::forward: tokens must be Int64 (got {})",
                  dtype_name(tokens.dtype()));
  TESSERACT_CHECK(tokens.rank() == 2,
                  "LlamaModel::forward: expected rank-2 tokens [B, S], got {}",
                  tokens.shape().to_string());

  Tensor x = embed_tokens_->forward(tokens);     // [B, S, D]
  for (auto& block : layers_) {
    x = block->forward(x);                       // [B, S, D]
  }
  x = norm_->forward(x);                          // [B, S, D]
  return lm_head_->forward(x);                    // [B, S, V]
}

// ---------------------------------------------------------------------------
// Incremental decode + greedy generation (Wave 5 / B-027)
// ---------------------------------------------------------------------------

std::vector<std::shared_ptr<nn::KVCache>> LlamaModel::make_kv_caches(
    int64_t batch, int64_t max_len) const {
  const Tensor& w = embed_tokens_->weight();
  // KV cache stores KV heads (GQA): Hkv ≤ num_attention_heads. Head dim is
  // still hidden / num_attention_heads.
  const int64_t Hkv = config_.kv_heads();
  const int64_t Dh  = config_.hidden_size / config_.num_attention_heads;
  std::vector<std::shared_ptr<nn::KVCache>> caches;
  caches.reserve(static_cast<std::size_t>(num_layers()));
  for (int64_t l = 0; l < num_layers(); ++l) {
    caches.push_back(std::make_shared<nn::KVCache>(
        batch, Hkv, Dh, max_len, config_.dtype, w.device()));
  }
  return caches;
}

std::vector<std::shared_ptr<nn::KVCacheBase>>
LlamaModel::make_quantized_kv_caches(int64_t batch, int64_t max_len) const {
  const Tensor& w = embed_tokens_->weight();
  const int64_t Hkv = config_.kv_heads();
  const int64_t Dh  = config_.hidden_size / config_.num_attention_heads;
  std::vector<std::shared_ptr<nn::KVCacheBase>> caches;
  caches.reserve(static_cast<std::size_t>(num_layers()));
  for (int64_t l = 0; l < num_layers(); ++l) {
    caches.push_back(std::make_shared<nn::QuantizedKVCache>(
        batch, Hkv, Dh, max_len, config_.dtype, w.device()));
  }
  return caches;
}

Tensor LlamaModel::forward_step(const Tensor& tokens,
                                std::vector<std::shared_ptr<nn::KVCache>>& caches) {
  // Upcast each contiguous cache to the base interface and delegate.
  std::vector<std::shared_ptr<nn::KVCacheBase>> base(caches.begin(),
                                                     caches.end());
  return forward_step(tokens, base);
}

Tensor LlamaModel::forward_step(
    const Tensor& tokens,
    std::vector<std::shared_ptr<nn::KVCacheBase>>& caches) {
  TESSERACT_CHECK(tokens.defined(), "LlamaModel::forward_step: tokens undefined");
  TESSERACT_CHECK(tokens.dtype() == DType::Int64,
                  "LlamaModel::forward_step: tokens must be Int64 (got {})",
                  dtype_name(tokens.dtype()));
  TESSERACT_CHECK(tokens.rank() == 2,
                  "LlamaModel::forward_step: expected rank-2 tokens [B, S_new], got {}",
                  tokens.shape().to_string());
  TESSERACT_CHECK(static_cast<int64_t>(caches.size()) == num_layers(),
                  "LlamaModel::forward_step: expected {} caches (one per layer), got {}",
                  num_layers(), caches.size());

  NoGradGuard nogg;
  Tensor x = embed_tokens_->forward(tokens);       // [B, S_new, D]
  for (std::size_t l = 0; l < layers_.size(); ++l) {
    TESSERACT_CHECK(caches[l] != nullptr,
                    "LlamaModel::forward_step: cache for layer {} is null", l);
    x = layers_[l]->forward_step(x, *caches[l]);    // [B, S_new, D]
  }
  x = norm_->forward(x);                            // [B, S_new, D]
  return lm_head_->forward(x);                      // [B, S_new, V]
}

Tensor LlamaModel::forward_step_batched(
    const Tensor& tokens,
    std::vector<std::vector<std::shared_ptr<nn::KVCacheBase>>>& caches) {
  TESSERACT_CHECK(tokens.defined(),
                  "LlamaModel::forward_step_batched: tokens undefined");
  TESSERACT_CHECK(tokens.dtype() == DType::Int64,
                  "LlamaModel::forward_step_batched: tokens must be Int64 "
                  "(got {})", dtype_name(tokens.dtype()));
  TESSERACT_CHECK(tokens.rank() == 2,
                  "LlamaModel::forward_step_batched: expected rank-2 tokens "
                  "[A, S_new], got {}", tokens.shape().to_string());
  const int64_t A = tokens.shape()[0];
  TESSERACT_CHECK(static_cast<int64_t>(caches.size()) == A,
                  "LlamaModel::forward_step_batched: {} cache sets for a batch "
                  "of {}", caches.size(), A);
  const int64_t L = num_layers();
  for (int64_t r = 0; r < A; ++r) {
    TESSERACT_CHECK(static_cast<int64_t>(caches[r].size()) == L,
                    "LlamaModel::forward_step_batched: sequence {} has {} "
                    "caches, expected {} (one per layer)", r, caches[r].size(),
                    L);
  }

  NoGradGuard nogg;
  Tensor x = embed_tokens_->forward(tokens);        // [A, S_new, D]
  std::vector<nn::KVCacheBase*> per_layer(static_cast<std::size_t>(A));
  for (std::size_t l = 0; l < layers_.size(); ++l) {
    for (int64_t r = 0; r < A; ++r) {
      nn::KVCacheBase* c = caches[static_cast<std::size_t>(r)][l].get();
      TESSERACT_CHECK(c != nullptr,
                      "LlamaModel::forward_step_batched: cache [{}][{}] null",
                      r, l);
      per_layer[static_cast<std::size_t>(r)] = c;
    }
    x = layers_[l]->forward_step_batched(x, per_layer);  // [A, S_new, D]
  }
  x = norm_->forward(x);                            // [A, S_new, D]
  return lm_head_->forward(x);                      // [A, S_new, V]
}

std::vector<std::shared_ptr<nn::PagedKVPool>> LlamaModel::make_layer_pools(
    int64_t num_blocks, int64_t block_size) const {
  const Tensor& w = embed_tokens_->weight();
  const int64_t Hkv = config_.kv_heads();
  const int64_t Dh  = config_.hidden_size / config_.num_attention_heads;
  std::vector<std::shared_ptr<nn::PagedKVPool>> pools;
  pools.reserve(static_cast<std::size_t>(num_layers()));
  for (int64_t l = 0; l < num_layers(); ++l) {
    pools.push_back(std::make_shared<nn::PagedKVPool>(
        Hkv, Dh, block_size, num_blocks, config_.dtype, w.device()));
  }
  return pools;
}

std::vector<std::shared_ptr<nn::KVCacheBase>> LlamaModel::make_paged_kv_caches(
    const std::vector<std::shared_ptr<nn::PagedKVPool>>& pools,
    int64_t max_len) const {
  TESSERACT_CHECK(static_cast<int64_t>(pools.size()) == num_layers(),
                  "LlamaModel::make_paged_kv_caches: expected {} pools, got {}",
                  num_layers(), pools.size());
  std::vector<std::shared_ptr<nn::KVCacheBase>> caches;
  caches.reserve(pools.size());
  for (const auto& pool : pools) {
    caches.push_back(
        std::make_shared<nn::PagedKVCache>(pool, /*batch=*/1, max_len));
  }
  return caches;
}

std::vector<std::shared_ptr<nn::QuantizedPagedKVPool>>
LlamaModel::make_quantized_layer_pools(int64_t num_blocks,
                                       int64_t block_size) const {
  const Tensor& w = embed_tokens_->weight();
  const int64_t Hkv = config_.kv_heads();
  const int64_t Dh  = config_.hidden_size / config_.num_attention_heads;
  std::vector<std::shared_ptr<nn::QuantizedPagedKVPool>> pools;
  pools.reserve(static_cast<std::size_t>(num_layers()));
  for (int64_t l = 0; l < num_layers(); ++l) {
    pools.push_back(std::make_shared<nn::QuantizedPagedKVPool>(
        Hkv, Dh, block_size, num_blocks, config_.dtype, w.device()));
  }
  return pools;
}

std::vector<std::shared_ptr<nn::KVCacheBase>>
LlamaModel::make_quantized_paged_kv_caches(
    const std::vector<std::shared_ptr<nn::QuantizedPagedKVPool>>& pools,
    int64_t max_len) const {
  TESSERACT_CHECK(static_cast<int64_t>(pools.size()) == num_layers(),
                  "LlamaModel::make_quantized_paged_kv_caches: expected {} "
                  "pools, got {}", num_layers(), pools.size());
  std::vector<std::shared_ptr<nn::KVCacheBase>> caches;
  caches.reserve(pools.size());
  for (const auto& pool : pools) {
    caches.push_back(
        std::make_shared<nn::QuantizedPagedKVCache>(pool, /*batch=*/1, max_len));
  }
  return caches;
}

namespace {

// Argmax over the vocab axis of the final position of a [B, S, V] logits
// tensor (b=0, s=S-1). Reads on host; handles FP32/FP64/FP16/BF16.
int32_t argmax_last_position(const Tensor& logits) {
  const Tensor host = logits.device().is_cpu() ? logits : logits.to(cpu_device());
  const int64_t S = host.shape()[1];
  const int64_t V = host.shape()[2];
  int64_t best = 0;
  dispatch_float_with_half(host.dtype(), [&]<typename T>() {
    const T* p = host.data_ptr<T>() + (S - 1) * V;  // b=0, s=S-1
    double best_val = -std::numeric_limits<double>::infinity();
    for (int64_t v = 0; v < V; ++v) {
      const double val = static_cast<double>(p[v]);
      if (val > best_val) { best_val = val; best = v; }
    }
  });
  return static_cast<int32_t>(best);
}

// Copy the final position's logits row (b=0, s=S-1) of a [B, S, V] tensor
// into a host FP32 vector for the sampler. Handles FP32/FP64/FP16/BF16.
std::vector<float> last_logits_row(const Tensor& logits) {
  const Tensor host = logits.device().is_cpu() ? logits : logits.to(cpu_device());
  const int64_t S = host.shape()[1];
  const int64_t V = host.shape()[2];
  std::vector<float> row(static_cast<std::size_t>(V));
  dispatch_float_with_half(host.dtype(), [&]<typename T>() {
    const T* p = host.data_ptr<T>() + (S - 1) * V;  // b=0, s=S-1
    for (int64_t v = 0; v < V; ++v) row[static_cast<std::size_t>(v)] = static_cast<float>(p[v]);
  });
  return row;
}

}  // namespace

std::vector<int32_t> LlamaModel::generate(const std::vector<int32_t>& prompt_ids,
                                          const GenerateConfig& cfg) {
  TESSERACT_CHECK(!prompt_ids.empty(),
                  "LlamaModel::generate: prompt_ids must be non-empty");
  TESSERACT_CHECK(cfg.max_new_tokens >= 0,
                  "LlamaModel::generate: max_new_tokens must be >= 0");

  NoGradGuard nogg;
  const Device dev = embed_tokens_->weight().device();
  const int64_t prompt_len = static_cast<int64_t>(prompt_ids.size());
  const int64_t max_len = prompt_len + cfg.max_new_tokens;

  for (int32_t id : prompt_ids) {
    TESSERACT_CHECK(id >= 0 && id < config_.vocab_size,
                    "LlamaModel::generate: prompt id {} out of range [0, {})",
                    id, config_.vocab_size);
  }

  // Full-precision KV caches by default; INT8-quantized when requested.
  // Both satisfy `KVCacheBase`, so the decode loop drives them through
  // the generic `forward_step` overload either way.
  std::vector<std::shared_ptr<nn::KVCacheBase>> caches;
  if (cfg.kv_int8) {
    caches = make_quantized_kv_caches(/*batch=*/1, std::max<int64_t>(max_len, 1));
  } else {
    auto fp = make_kv_caches(/*batch=*/1, std::max<int64_t>(max_len, 1));
    caches.assign(fp.begin(), fp.end());
  }

  // Build an Int64 [1, n] token tensor on the model's device.
  auto make_tokens = [&](const int32_t* ids, int64_t n) -> Tensor {
    Tensor t = Tensor::empty({1, n}, DType::Int64, cpu_device());
    int64_t* p = t.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) p[i] = static_cast<int64_t>(ids[i]);
    return dev.is_cpu() ? t : t.to(dev);
  };

  std::vector<int32_t> result = prompt_ids;

  // Greedy by default; stochastic sampling (seeded, reproducible) when
  // `cfg.do_sample`. The sampler reads the full logits row + the sequence
  // so far (for repetition penalty).
  Sampler sampler(cfg.sampling, cfg.seed);
  auto pick_next = [&](const Tensor& logits) -> int32_t {
    if (!cfg.do_sample) return argmax_last_position(logits);
    const std::vector<float> row = last_logits_row(logits);
    return sampler.sample(std::span<const float>(row.data(), row.size()),
                          std::span<const int32_t>(result.data(), result.size()));
  };

  // Prefill the whole prompt in one chunked-decode step.
  Tensor logits = forward_step(make_tokens(prompt_ids.data(), prompt_len), caches);

  for (int64_t i = 0; i < cfg.max_new_tokens; ++i) {
    const int32_t next = pick_next(logits);
    result.push_back(next);
    if (cfg.eos_token_id >= 0 && next == cfg.eos_token_id) break;
    if (i + 1 < cfg.max_new_tokens) {
      logits = forward_step(make_tokens(&next, 1), caches);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

namespace {

// Host-side floating-point dtype conversion (B-022): rematerialize `src`
// in `target` dtype on the CPU, routing every element through FP32. Used
// by the loader so one on-disk checkpoint (e.g. fp32 or bf16) can seed a
// model declared in a different precision (e.g. fp16) without an external
// offline conversion pass — the same upcast/downcast vLLM does at load.
Tensor cast_floating_cpu(const Tensor& src_cpu, DType target) {
  Tensor out = Tensor::empty(src_cpu.shape(), target, cpu_device());
  const int64_t n = src_cpu.numel();
  dispatch_float_with_half(src_cpu.dtype(), [&]<typename S>() {
    const S* sp = src_cpu.data_ptr<S>();
    dispatch_float_with_half(target, [&]<typename D>() {
      D* dp = out.data_ptr<D>();
      for (int64_t i = 0; i < n; ++i)
        dp[i] = static_cast<D>(static_cast<float>(sp[i]));
    });
  });
  return out;
}

void copy_into_param(Tensor& dst, const Tensor& src, const std::string& name) {
  TESSERACT_CHECK(dst.defined(), "load_safetensors: dst '{}' undefined", name);
  TESSERACT_CHECK(src.defined(), "load_safetensors: src '{}' undefined", name);
  TESSERACT_CHECK(dst.shape() == src.shape(),
                  "load_safetensors: shape mismatch for '{}': dst={}, src={}",
                  name, dst.shape().to_string(), src.shape().to_string());
  TESSERACT_CHECK(dst.is_contiguous(),
                  "load_safetensors: dst '{}' must be contiguous", name);
  TESSERACT_CHECK(src.is_contiguous(),
                  "load_safetensors: src '{}' must be contiguous", name);
  if (dst.nbytes() == 0) return;

  // Fast path: identical dtype is a straight byte copy.
  if (dst.dtype() == src.dtype()) {
    Storage::copy_device_bytes(dst.raw_data(), dst.device(),
                               src.raw_data(), src.device(),
                               dst.nbytes());
    return;
  }

  // Conversion path: only float↔float is meaningful (int params, e.g.
  // none here today, would be a real mismatch). Convert on the CPU then
  // ship the result to the destination device.
  TESSERACT_CHECK(dtype_is_floating(dst.dtype()) && dtype_is_floating(src.dtype()),
                  "load_safetensors: cannot convert '{}' from {} to {} "
                  "(only float↔float conversion is supported)",
                  name, dtype_name(src.dtype()), dtype_name(dst.dtype()));
  Tensor src_cpu = src.device().is_cpu() ? src : src.to(cpu_device());
  Tensor casted = cast_floating_cpu(src_cpu, dst.dtype());
  Storage::copy_device_bytes(dst.raw_data(), dst.device(),
                             casted.raw_data(), casted.device(),
                             dst.nbytes());
}

}  // namespace

std::vector<std::string> LlamaModel::load_safetensors(const std::string& path) {
  io::SafeTensors st = io::SafeTensors::open(path);

  // Track which HF keys we consume so we can report the remainder.
  std::unordered_set<std::string> consumed;
  consumed.reserve(st.keys().size());

  // First pass: walk our module's parameters and pull each one from
  // the safetensors file (or, for tied lm_head, from embed_tokens).
  bool saw_embed = false;
  for (auto& [local, param_handle] : named_parameters()) {
    // `param_handle` is a Tensor handle sharing the parameter's impl,
    // so memcpy into its storage is what we want.
    Tensor param = param_handle;

    const std::string hf = llama_local_to_hf_name(local);

    if (local == "lm_head.weight" && config_.tie_word_embeddings &&
        !st.contains(hf)) {
      // Tied: seed lm_head from embed_tokens.weight. Both are stored as
      // [vocab, hidden] so the byte layout matches exactly.
      TESSERACT_CHECK(saw_embed,
                      "load_safetensors: tie_word_embeddings=true but "
                      "embed_tokens.weight was not loaded first");
      const Tensor& src = embed_tokens_->weight();
      copy_into_param(param, src, local);
      continue;
    }

    if (!st.contains(hf)) {
      TESSERACT_THROW(
          "load_safetensors: required tensor '{}' (local '{}') missing from '{}'",
          hf, local, path);
    }

    Tensor loaded = st.load(hf, param.device());
    copy_into_param(param, loaded, local);
    consumed.insert(hf);
    if (local == "embed_tokens.weight") saw_embed = true;
  }

  // Second pass: surface which HF keys went unused. Callers typically
  // log or ignore these. Common extras:
  //   - `model.rotary_emb.inv_freq` (HF emits precomputed RoPE; we
  //     recompute ours lazily in `nn::RotaryEmbedding`, so this is
  //     redundant for us).
  //   - `lm_head.weight` when `tie_word_embeddings=true` and the
  //     checkpoint still stored a separate copy.
  std::vector<std::string> leftovers;
  leftovers.reserve(st.keys().size() - consumed.size());
  for (const auto& k : st.keys()) {
    if (consumed.find(k) == consumed.end()) leftovers.push_back(k);
  }
  return leftovers;
}

std::shared_ptr<LlamaModel> LlamaModel::from_pretrained(
    const std::string& path, const LlamaConfig& config) {
  auto model = std::make_shared<LlamaModel>(config);
  (void)model->load_safetensors(path);
  return model;
}

// ---------------------------------------------------------------------------
// Quantization walker (Wave 3.3 / B-021 final)
// ---------------------------------------------------------------------------

void LlamaModel::quantize_(const quant::Scheme& scheme) {
  // 1) Every TransformerBlock's attention + FFN. The block itself
  //    owns no FP `Linear` directly — its RMSNorms are purposely
  //    left FP (tiny, numerically sensitive), the `.attn` and `.ffn`
  //    children are where the real weights live.
  for (auto& block : layers_) {
    block->attn()->quantize_(scheme);
    // MoE blocks (B-038) expose a null dense `ffn()`; their expert
    // quantization is a B-038+ follow-up, so skip rather than crash.
    if (block->ffn()) block->ffn()->quantize_(scheme);
  }

  // 2) `lm_head_`. This is the single biggest Linear in a vocab-
  //    large Llama (hidden_size × vocab_size), so quantizing it is
  //    where most of the memory win comes from. Walker is idempotent
  //    — if a caller already swapped this slot (tests, perhaps)
  //    we leave it alone.
  if (auto fp_head = std::dynamic_pointer_cast<nn::Linear>(lm_head_)) {
    auto replacement = quant::quantize_linear(*fp_head, scheme);
    lm_head_ = replacement;
    replace_module("lm_head", replacement);
  }

  // NOTE: `embed_tokens_` and `norm_` intentionally skipped. See the
  // rationale on `LlamaModel::quantize_`'s header comment.
}

}  // namespace tesseract::models

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/Embedding.hpp"
#include "tesseract/nn/KVCache.hpp"
#include "tesseract/nn/KVCacheBase.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/PagedKVPool.hpp"
#include "tesseract/nn/QuantizedPagedKVPool.hpp"
#include "tesseract/nn/Module.hpp"
#include "tesseract/models/Sampler.hpp"
#include "tesseract/nn/RMSNorm.hpp"
#include "tesseract/nn/TransformerBlock.hpp"
#include "tesseract/quant/Scheme.hpp"

namespace tesseract::models {

// Hyperparameter set for a Llama-family model, mirroring Hugging Face's
// `LlamaConfig` field names so a caller that already has a `config.json`
// from the HF hub can populate this struct one-to-one.
struct LlamaConfig {
  int64_t vocab_size            = 32000;
  int64_t hidden_size           = 4096;    // d_model
  int64_t num_hidden_layers     = 32;
  int64_t num_attention_heads   = 32;      // query heads
  // Grouped-query attention (Wave 8 / B-030). 0 ⇒ equals
  // num_attention_heads (plain MHA). A smaller positive value (must
  // divide num_attention_heads) selects GQA — every Llama-3 / Qwen2 /
  // Mistral checkpoint sets this.
  int64_t num_key_value_heads   = 0;
  int64_t intermediate_size     = 11008;   // d_ff (SwiGLU hidden width)

  // Mixture-of-experts (M4 Track A1 / B-038). `num_experts == 0` ⇒ dense
  // SwiGLU FFN (every Llama-1/2/3 checkpoint). A positive value selects a
  // Mixtral-style sparse MoE FFN with `num_experts_per_tok` active experts
  // per token (HF `num_local_experts` / `num_experts_per_tok`).
  int64_t num_experts           = 0;
  int64_t num_experts_per_tok   = 0;

  int64_t max_position_embeddings = 2048;
  double  rope_theta            = 10000.0;
  double  rms_norm_eps          = 1e-5;

  // When true, skip loading `lm_head.weight` from safetensors and instead
  // seed it from `embed_tokens.weight` (copy; no storage aliasing yet —
  // see B-021). HF sets this for small Llama variants (e.g. Llama-3.2-1B,
  // SmolLM) and clears it for 7B+.
  bool    tie_word_embeddings   = false;

  // Parameter dtype; all learnable tensors are allocated with this dtype.
  // The loader can convert from a different on-disk dtype only when
  // `strict_dtype=false`; currently we require an exact match (future
  // FP16 ↔ FP32 loader conversion is tracked as B-022).
  DType   dtype                 = DType::Float32;

  // Special token ids (HF `config.json` `bos_token_id` / `eos_token_id`).
  // `-1` ⇒ unset. Populated by `from_json`; the generation loop uses
  // `eos_token_id` for early stop when the caller doesn't override it.
  int32_t bos_token_id          = -1;
  int32_t eos_token_id          = -1;

  // Resolved KV-head count: `num_key_value_heads` if positive, else
  // `num_attention_heads` (plain MHA).
  int64_t kv_heads() const noexcept {
    return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
  }

  // Returns a config that matches the published Llama-3.2-1B architecture
  // (32 query heads, 8 KV heads — GQA).
  static LlamaConfig llama_3_2_1b();

  // Wave 16 (B-033): parse a Hugging Face `config.json` into a LlamaConfig.
  // Reads the standard top-level scalar fields — `vocab_size`,
  // `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
  // `num_key_value_heads`, `intermediate_size`, `max_position_embeddings`,
  // `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`, `torch_dtype`
  // (float32/float16/bfloat16 → `dtype`), `bos_token_id`, `eos_token_id`.
  // Missing fields keep their struct defaults. `from_json_file` reads the
  // file then delegates. Throws on unreadable file / malformed JSON object.
  static LlamaConfig from_json(const std::string& json_text);
  static LlamaConfig from_json_file(const std::string& path);
};

// Llama-family decoder-only transformer:
//
//   tokens: Int64 [B, S]
//        -> embed_tokens
//        -> N × TransformerBlock (RMSNorm → RoPE-MHA → Residual → RMSNorm → SwiGLU-FFN → Residual)
//        -> final RMSNorm
//        -> lm_head (Linear with no bias)
//        -> logits: [B, S, vocab_size]
//
// The block stack reuses `nn::TransformerBlock`, so RoPE + RMSNorm + SwiGLU
// all share the same paths that M2K/B-014 already benchmark. This class
// is intentionally lightweight: it composes existing modules and adds
// only the outer forward and the HF weight loader.
class LlamaModel : public nn::Module {
 public:
  explicit LlamaModel(const LlamaConfig& config);

  Tensor forward(const Tensor& tokens) override;

  // Wave 5 (B-027) incremental decode through the block stack.
  // `tokens` is Int64 `[B, S_new]` (S_new == 1 for token-by-token
  // decode, larger for one-shot prefill). `caches` must hold exactly
  // `num_layers()` caches, each sized/typed/deviced to match this
  // model (batch B, H = num_attention_heads, D_head = hidden/H). Each
  // layer's K/V is appended into its cache and attention runs against
  // the full accumulated prefix. Returns logits `[B, S_new, vocab]`.
  // Inference-only (NoGradGuard); the caller owns cache lifetime so the
  // same caches can be reused across prefill + every decode step.
  Tensor forward_step(const Tensor& tokens,
                      std::vector<std::shared_ptr<nn::KVCache>>& caches);

  // Wave 7 (B-029) generic overload: drive incremental decode through
  // any `KVCacheBase` (contiguous `nn::KVCache` or paged
  // `nn::PagedKVCache`). The continuous-batching scheduler uses this
  // with per-request paged caches sharing per-layer pools.
  Tensor forward_step(const Tensor& tokens,
                      std::vector<std::shared_ptr<nn::KVCacheBase>>& caches);

  // Wave 10 (B-032) compute-batched decode. `tokens` is `[A, S_new]` —
  // `A` independent sequences stacked on the batch axis (typically
  // `S_new == 1` during steady-state decode). `caches` is indexed
  // `[sequence][layer]`: each of the `A` sequences brings its own
  // per-layer cache vector (each cache `batch()==1`, possibly at a
  // different length). Embedding, all projections, the FFN, and the LM
  // head run **once** over the whole `A`-batch (the throughput win),
  // while attention threads each sequence through its own cache. Emits
  // `[A, S_new, V]`. Per-sequence output is identical to looping
  // `forward_step` over each sequence — bit-identical on CPU, within
  // float tolerance on CUDA (batched vs single GEMM). Inference-only.
  Tensor forward_step_batched(
      const Tensor& tokens,
      std::vector<std::vector<std::shared_ptr<nn::KVCacheBase>>>& caches);

  // Allocate one contiguous KV cache per layer, sized for up to
  // `max_len` tokens, on the model's current device/dtype. Convenience
  // for callers driving `forward_step` / `generate`.
  std::vector<std::shared_ptr<nn::KVCache>> make_kv_caches(
      int64_t batch, int64_t max_len) const;

  // Wave 9 (B-031): allocate one INT8-quantized KV cache per layer
  // (`nn::QuantizedKVCache`), sized for up to `max_len` tokens, on the
  // model's device/dtype. Returned as `KVCacheBase` so they feed the
  // generic `forward_step` directly. ~4× (vs FP32) / ~2× (vs FP16)
  // smaller persistent KV footprint at a bounded-error decode cost.
  std::vector<std::shared_ptr<nn::KVCacheBase>> make_quantized_kv_caches(
      int64_t batch, int64_t max_len) const;

  // Wave 7 (B-029): allocate one shared `PagedKVPool` per layer on the
  // model's device/dtype, carved into `num_blocks` blocks of
  // `block_size` tokens. Many requests' per-layer `PagedKVCache`s bind
  // to these pools so finished requests' blocks recycle into the shared
  // budget — the core continuous-batching memory win.
  std::vector<std::shared_ptr<nn::PagedKVPool>> make_layer_pools(
      int64_t num_blocks, int64_t block_size) const;

  // Allocate one `PagedKVCache` per layer (batch=1) bound to the given
  // per-layer `pools`, sized for up to `max_len` tokens. Returned as
  // `KVCacheBase` so they feed the generic `forward_step` directly.
  std::vector<std::shared_ptr<nn::KVCacheBase>> make_paged_kv_caches(
      const std::vector<std::shared_ptr<nn::PagedKVPool>>& pools,
      int64_t max_len) const;

  // Wave 14 (B-032++++): the INT8 siblings of the two methods above —
  // allocate one shared `QuantizedPagedKVPool` per layer, and one
  // `QuantizedPagedKVCache` per layer bound to them. This is what lets the
  // continuous-batching scheduler hold paged KV in INT8 (paging's
  // no-padding residency × quant's ~4×/2× per-token shrink), feeding the
  // Wave-12 fused `paged_decode_attention_int8` op on the CUDA decode path.
  std::vector<std::shared_ptr<nn::QuantizedPagedKVPool>>
  make_quantized_layer_pools(int64_t num_blocks, int64_t block_size) const;

  std::vector<std::shared_ptr<nn::KVCacheBase>> make_quantized_paged_kv_caches(
      const std::vector<std::shared_ptr<nn::QuantizedPagedKVPool>>& pools,
      int64_t max_len) const;

  // Greedy (argmax) autoregressive generation for a single sequence.
  // Prefills `prompt_ids`, then decodes one token at a time until
  // `max_new_tokens` is reached or `eos_token_id` is produced. Returns
  // the full sequence (prompt followed by generated ids — the HF
  // `generate` convention). Runs on the model's current device.
  struct GenerateConfig {
    int64_t max_new_tokens = 32;
    int32_t eos_token_id   = -1;  // -1 ⇒ no early stop
    // When false (default), decode greedily (argmax) — deterministic and
    // independent of `seed`. When true, draw each token with `sampling`
    // (temperature / top-k / top-p / repetition penalty) from a `seed`-ed
    // RNG so a fixed seed reproduces the sequence.
    bool           do_sample = false;
    SamplingParams sampling{};
    uint64_t       seed = 0;
    // Wave 9 (B-031): when true, decode through INT8-quantized KV caches
    // (`nn::QuantizedKVCache`) instead of full-precision ones — a ~4×/2×
    // cut in KV memory. Lossy: output is bounded-error, not bit-identical
    // to the FP path.
    bool           kv_int8 = false;
  };
  std::vector<int32_t> generate(const std::vector<int32_t>& prompt_ids,
                                const GenerateConfig& cfg);

  const LlamaConfig& config() const noexcept { return config_; }
  int64_t num_layers() const noexcept {
    return static_cast<int64_t>(layers_.size());
  }

  // Direct accessors — useful for tests that want to poke one specific
  // parameter without walking `named_parameters()`. Not part of the
  // "stable" API; prefer `named_parameters()` for general tooling.
  const std::shared_ptr<nn::Embedding>& embed_tokens() const { return embed_tokens_; }
  const std::vector<std::shared_ptr<nn::TransformerBlock>>& layers() const { return layers_; }
  const std::shared_ptr<nn::RMSNorm>& norm() const { return norm_; }

  // Held as `shared_ptr<Module>` post-Wave-3.3 so the lm_head slot
  // can switch between an FP `nn::Linear` and a quantized
  // `nn::QuantizedLinear*` drop-in after `quantize_`. Tests that
  // want the FP-only view (e.g. `.weight()`) should
  // `dynamic_pointer_cast<nn::Linear>(model->lm_head())` and assert
  // the cast is non-null before quantization.
  const std::shared_ptr<nn::Module>& lm_head() const { return lm_head_; }

  // Load weights from a Hugging Face-format safetensors file into the
  // existing parameter tensors (in-place `memcpy`, no new storage
  // allocated). Returns the list of HF keys that were present in the
  // file but had no match in our model — typical for extras like
  // `model.rotary_emb.inv_freq` or `lm_head.weight` when
  // tie_word_embeddings=true. Throws if a required parameter is
  // missing from the file.
  std::vector<std::string> load_safetensors(const std::string& path);

  // One-shot factory: construct a LlamaModel from `config` and load
  // `path` into it. Returns a shared_ptr so callers can use the model
  // polymorphically through `nn::Module`.
  static std::shared_ptr<LlamaModel> from_pretrained(
      const std::string& path, const LlamaConfig& config);

  // Wave 3.3 (B-021) in-place model-wide quantization. Walks every
  // transformer block's `MultiHeadAttention` + `FeedForward` and
  // calls their `quantize_` (swaps the four attention projections
  // and the three SwiGLU projections to `scheme`'s variant). Also
  // swaps `lm_head_` itself, since it's the single biggest Linear
  // in a vocab-large Llama and contributes most of the memory win.
  //
  // Embedding and RMSNorm weights are deliberately left in FP: the
  // embedding is accessed as a lookup (quantization would hurt
  // rarely-seen tokens and the storage is already well-bounded by
  // vocab), and the RMSNorm scale is a per-channel FP multiplier
  // whose quantization error compounds through every subsequent
  // layer without saving meaningful memory.
  //
  // Idempotent per-child: re-invoking with the same scheme is a
  // no-op; re-invoking with a different scheme is **not** supported
  // (the FP weights needed to re-pack are gone). Callers who need
  // a different scheme should rebuild the model from weights.
  void quantize_(const quant::Scheme& scheme);

 private:
  LlamaConfig config_;
  std::shared_ptr<nn::Embedding>                          embed_tokens_;
  std::vector<std::shared_ptr<nn::TransformerBlock>>      layers_;
  std::shared_ptr<nn::RMSNorm>                            norm_;
  // Held as `Module` so the quantizer can drop-in a
  // `QuantizedLinear*`. Pre-quantize this is always `nn::Linear`;
  // the `lm_head()` accessor documents that invariant for callers.
  std::shared_ptr<nn::Module>                             lm_head_;
};

// Translate a local parameter name (as produced by
// `Module::named_parameters()` on a `LlamaModel`) to its canonical HF
// Llama safetensors key. Examples:
//
//   "embed_tokens.weight"              -> "model.embed_tokens.weight"
//   "layers.0.norm_1.weight"           -> "model.layers.0.input_layernorm.weight"
//   "layers.7.attn.q_proj.weight"      -> "model.layers.7.self_attn.q_proj.weight"
//   "layers.7.ffn.gate_proj.weight"    -> "model.layers.7.mlp.gate_proj.weight"
//   "layers.31.norm_2.weight"          -> "model.layers.31.post_attention_layernorm.weight"
//   "norm.weight"                      -> "model.norm.weight"
//   "lm_head.weight"                   -> "lm_head.weight"
//
// This is exposed in the header so tests and alternative loaders can
// reuse the translation without duplicating the rules.
std::string llama_local_to_hf_name(std::string_view local);

}  // namespace tesseract::models

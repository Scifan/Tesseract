#pragma once

#include <memory>
#include <vector>

#include "tesseract/nn/KVCacheBase.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"
#include "tesseract/nn/RotaryEmbedding.hpp"
#include "tesseract/quant/Scheme.hpp"

namespace tesseract::nn {

// Multi-head self-attention, Llama-style.
//
// With `rope_base == 0.0` (or `max_seq <= 0`) the attention runs
// without any position encoding — identical to the M2K behavior, the
// module just projects, attends, and merges. Set a positive
// `rope_base` (10000.0 is the Llama default) along with a non-zero
// `max_seq` and the module constructs an `nn::RotaryEmbedding` child
// that applies RoPE to Q and K right before `ops::attention`. The
// child is moved alongside the Linear projections by `Module::to()`,
// so `.to(cuda)` migrates the cached cos/sin tables with the rest
// of the block and the forward path stays single-device.
//
// Forward (x: [B, S, D_model]):
//
//   q = q_proj(x).reshape([B, S, H, D_head]).permute([0, 2, 1, 3])  // [B, H, S, D_head]
//   k, v = same
//   out_heads = ops::attention(q, k, v, causal=true)                // [B, H, S, D_head]
//   out = out_heads.permute([0, 2, 1, 3]).reshape([B, S, D_model])
//   return o_proj(out)
//
// Grouped-query attention (GQA), Wave 8 (B-030). When
// `num_kv_heads < num_heads` the K/V projections emit only
// `num_kv_heads · D_head` features (a smaller KV cache — the whole point
// of GQA) and each KV head is shared by `num_heads / num_kv_heads` query
// heads. K/V are repeat-expanded back to `num_heads` right before
// attention (the PyTorch `repeat_kv` recipe). `num_kv_heads == 0` (the
// default) or `== num_heads` is plain multi-head attention. Every modern
// Llama-3 / Qwen2 / Mistral checkpoint is GQA, so this is the gate to
// loading them.
//
// The module owns the four projection `Linear`s as registered children,
// so `.to(cuda)` / `.parameters()` recurse correctly. Biases are
// optional and follow the `use_bias` flag (Llama disables them, but we
// leave the default on for general-purpose use; the M2K test sets
// `use_bias=false` for Llama parity).
class MultiHeadAttention : public Module {
 public:
  // `num_kv_heads == 0` ⇒ defaults to `num_heads` (plain MHA). Otherwise
  // it must divide `num_heads`.
  MultiHeadAttention(int64_t d_model, int64_t num_heads,
                     bool use_bias = true,
                     bool causal = true,
                     DType dtype = DType::Float32,
                     double rope_base = 0.0,
                     int64_t rope_max_seq = 0,
                     int64_t num_kv_heads = 0);

  Tensor forward(const Tensor& x) override;

  // Wave 2.1 decode-step variant. `x` is `[B, S_new, D_model]` where
  // `S_new` is typically 1 (one decoded token) but may be larger at
  // prefill time. Projects Q/K/V, applies RoPE at positions
  // `[cache.current_len(), cache.current_len() + S_new)`, appends
  // K/V into the cache, runs attention against the **full** prefix
  // (cache.keys_view() / values_view(), shape
  // `[B, H, cache.current_len() + S_new, D_head]`), and emits
  // `[B, S_new, D_model]`. Causal flag has no effect here — query
  // positions come chronologically last so every cache position is
  // a valid key.
  //
  // No autograd: this path is inference-only, the returned tensor
  // has no grad_fn. That matches the M2 `ops::attention` composite's
  // "NoGradGuard" contract at decode time and keeps the cache from
  // accidentally capturing stale prefix graphs.
  // Accepts any `KVCacheBase` — the contiguous `KVCache` (Wave 2.1) or
  // the paged `PagedKVCache` (Wave 4.5) — through the same call.
  Tensor forward_step(const Tensor& x, KVCacheBase& cache);

  // Wave 10 (B-032) compute-batched decode. `x` is `[A, S_new, D_model]`
  // — `A` independent sequences stacked on the batch axis, each with its
  // own single-sequence (`batch()==1`) cache in `caches` (size `A`). The
  // four projections run **once** over all `A·S_new` rows (the batching
  // win), while RoPE / cache append / gather / attention loop per
  // sequence because each cache is at a different length (`current_len`).
  // Emits `[A, S_new, D_model]`. Per-row math is identical to calling
  // `forward_step` on each sequence separately — bit-identical on CPU,
  // within float tolerance on CUDA (batched vs single GEMM). Inference-
  // only (NoGradGuard). Used by the continuous-batching scheduler to fold
  // the active set's decode into batched matmuls.
  Tensor forward_step_batched(const Tensor& x,
                              const std::vector<KVCacheBase*>& caches);

  int64_t d_model() const noexcept { return d_model_; }
  int64_t num_heads() const noexcept { return num_heads_; }
  int64_t num_kv_heads() const noexcept { return num_kv_heads_; }
  int64_t head_dim() const noexcept { return head_dim_; }
  bool causal() const noexcept { return causal_; }
  bool has_rope() const noexcept { return rope_ != nullptr; }

  // Wave 3.3 (B-021) in-place quantization. Walks the four
  // projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and replaces
  // any that are still FP `nn::Linear` with the quantized drop-in
  // module implied by `scheme`. Already-quantized projections are
  // left alone — the operation is idempotent per-projection so
  // calling `quantize_` twice is safe. Projections are the only
  // things quantized; `rope_` keeps its FP cos/sin tables because
  // they're *not* a Linear and live on the pre-attention rotation
  // path where quantization would corrupt numerics.
  //
  // Mutates `this` in place: the old FP `Linear` children are
  // replaced under the same `"q_proj" / "k_proj" / "v_proj" /
  // "o_proj"` slots via `Module::replace_module`, so
  // `named_parameters()` walk order is preserved and any external
  // `shared_ptr` to the old `Linear` remains valid (but
  // disconnected) for any caller who captured it.
  void quantize_(const quant::Scheme& scheme);

  // Accessors for testing. Return the current occupant of each
  // slot. Pre-quantization these point to `Linear`; post-quantization
  // they point to `QuantizedLinear` / `QuantizedLinearInt4G`.
  const std::shared_ptr<Module>& q_proj() const { return q_proj_; }
  const std::shared_ptr<Module>& k_proj() const { return k_proj_; }
  const std::shared_ptr<Module>& v_proj() const { return v_proj_; }
  const std::shared_ptr<Module>& o_proj() const { return o_proj_; }

 private:
  int64_t d_model_;
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t head_dim_;
  int64_t kv_dim_;       // num_kv_heads_ * head_dim_
  bool causal_;

  // Held as `std::shared_ptr<Module>` rather than `Linear` so
  // `quantize_` can swap in a `QuantizedLinear` / `QuantizedLinearInt4G`
  // without rebuilding the module. `forward()` dispatches through
  // `Module::forward` so the call site is the same either way.
  std::shared_ptr<Module> q_proj_;
  std::shared_ptr<Module> k_proj_;
  std::shared_ptr<Module> v_proj_;
  std::shared_ptr<Module> o_proj_;
  std::shared_ptr<RotaryEmbedding> rope_;  // optional
};

}  // namespace tesseract::nn

#pragma once

// Wave 2.1 (B-019): KV cache for decode-phase incremental attention.
//
// The M2K `nn::MultiHeadAttention::forward(x)` re-runs the full
// `[B, S, D]` self-attention on every forward, which makes autoregressive
// decode O(S³) in compute and O(S²) in memory per step. This header
// introduces a minimal, correctness-first KV cache that:
//
//   * Pre-allocates a single contiguous `[B, H, max_len, D_head]` buffer
//     per (keys, values) — so the cache lives on the same device as
//     the rest of the attention block and `Module::to(cuda)` doesn't
//     need to know about it.
//   * Appends new `[B, H, S_new, D_head]` K/V slabs at `current_len_`
//     via device-aware byte copies — no composite ops, no new kernels,
//     no grad plumbing.
//   * Exposes `keys_view()` / `values_view()` returning read-only
//     `[B, H, current_len_, D_head]` **views** (no copy) built on top
//     of the `Tensor::narrow` view primitive.
//
// Why contiguous-backed, not paged-yet:
//
//   * Paged allocation (vLLM-style block table) is strictly a *memory-
//     efficiency* win for *concurrent* requests. Single-request decode
//     — which is what `llama_infer` does today and what the Wave 2.1
//     bench measures — gets full compute-reuse benefit from a plain
//     pre-allocated cache. We ship the paged variant (B-019b) once
//     continuous batching (Wave 4) actually creates the pressure it's
//     designed to relieve.
//   * All callers access K/V only through `keys_view()` /
//     `values_view()`. When we swap in the paged storage, those two
//     helpers gain indirection but the public API is unchanged.
//
// Lifetime: `KVCache` owns its storage; `reset()` rewinds `current_len_`
// to zero without reallocating so the same cache can be reused across
// prompts. No autograd: every tensor stored in the cache is detached
// and carries no grad_fn — the cache is a pure inference structure.

#include <cstdint>
#include <utility>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/KVCacheBase.hpp"

namespace tesseract::nn {

class KVCache : public KVCacheBase {
 public:
  // Allocate an empty cache sized for up to `max_len` tokens. `dtype`
  // and `device` must match the attention block that will write into
  // it (verified by `append()` on every call).
  KVCache(int64_t batch, int64_t num_heads, int64_t head_dim,
          int64_t max_len, DType dtype = DType::Float32,
          Device device = cpu_device());

  // Copy `k_new` and `v_new` into positions `[current_len_,
  // current_len_ + S_new)` along the seq dim and advance
  // `current_len_`. Inputs are expected to be
  // `[B, H, S_new, D_head]` contiguous and on the cache's device /
  // dtype. Returns the updated `current_len_` after the append.
  //
  // S_new is inferred from `k_new.shape()[2]`; `v_new` must agree.
  int64_t append(const Tensor& k_new, const Tensor& v_new) override;

  // Views over the valid prefix. Both are `[B, H, current_len_, D_head]`
  // and share storage with the underlying slabs — no copy, safe to
  // feed directly into `ops::attention` as the K/V operands.
  Tensor keys_view() const override;
  Tensor values_view() const override;

  // Reset current_len_ to zero. Storage is retained so the next
  // prompt can reuse the buffer — this is the fast path for the
  // "same model, many prompts" serving pattern.
  void reset() noexcept { current_len_ = 0; }

  // Forcibly override the position counter used by the next `append()`
  // and by `{keys,values}_view()`. **Capture-only escape hatch.**
  //
  // The CUDA Graph capture path (Wave 4.3 / B-023b) invokes the
  // user-supplied decode-step closure three times in a row
  // (two warmup passes + one capture pass) against the same
  // `KVCache` instance, so a plain `cache.append(k,v)` inside the
  // closure would advance `current_len_` by `Sn` each pass and land
  // the actual capture at `target_pos + 2·Sn` instead of
  // `target_pos`. The usual fix would be to store `target_pos`
  // elsewhere and rebuild the cache in-closure, but that's a lot of
  // ceremony for what is really "re-seed the position counter to the
  // value this captured graph was designed for."
  //
  // Semantics:
  //   * `len` must be in `[0, max_len_ - S_new]` where `S_new` is
  //     the chunk size the caller is about to append; we only
  //     validate `len <= max_len_` here because the chunk size is
  //     unknown at this call site — the subsequent `append()` still
  //     enforces the full bound.
  //   * Does *not* touch the slab contents. If `len > 0`, the caller
  //     is expected to have already populated `keys_/values_` at
  //     positions `[0, len)` (either via prior `append()` calls or
  //     by captured warmup writes that target those positions).
  //
  // Outside of graph-capture usage, prefer `reset()` + `append()`.
  //
  // Throws if `len < 0` or `len > max_len_`.
  void set_current_len(int64_t len);

  int64_t batch()       const noexcept override { return batch_; }
  int64_t num_heads()   const noexcept override { return num_heads_; }
  int64_t head_dim()    const noexcept override { return head_dim_; }
  int64_t max_len()     const noexcept { return max_len_; }
  int64_t current_len() const noexcept override { return current_len_; }
  DType   dtype()       const noexcept { return dtype_; }
  Device  device()      const noexcept { return device_; }

 private:
  int64_t batch_;
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t max_len_;
  DType   dtype_;
  Device  device_;
  int64_t current_len_{0};
  Tensor  keys_;    // [B, H, max_len, D_head] contiguous
  Tensor  values_;  // same
};

}  // namespace tesseract::nn

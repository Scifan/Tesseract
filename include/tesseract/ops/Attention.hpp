#pragma once

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// Scaled dot-product attention (SDPA).
//
//     out = softmax( Q · Kᵀ / √d + mask [+ causal] ) · V
//
// This is the public API contract that will later sit in front of a fused
// FlashAttention-3 kernel on Hopper (see M2L). The M2J implementation is a
// composite that reuses the already-CUDA-resident `matmul` (M2G) + `softmax`
// (M2F) + `add`/`mul` (M2E) kernels, so forward and backward run entirely
// on-device for both CPU and CUDA tensors.
//
// Shapes:
//   q: [..., S_q, D]
//   k: [..., S_k, D]
//   v: [..., S_k, D_v]
//   mask (optional, may be undefined): broadcast-compatible with
//     [..., S_q, S_k]. Values are *added* to the pre-softmax logits, so
//     use `-inf` (or a very negative scalar) to mask a position and `0`
//     to keep it. Must share Q/K/V's floating-point dtype when defined.
//   causal: if true, an upper-triangular `-inf` mask is applied on top
//     of `mask`, forbidding query position `i` from attending to key
//     position `j > i`. Requires S_q == S_k.
//   dropout_p: M2J accepts only `0.0`. Stochastic dropout lands with the
//     RNG HAL in M3 and the full FlashAttention-3 kernel in M2L.
//
// Returns: a contiguous tensor of shape [..., S_q, D_v] on the same
// device/dtype as the inputs.
//
// Broadcasting rules for the leading batch dims follow NumPy/PyTorch
// semantics — `q.shape[:-2]`, `k.shape[:-2]`, and `v.shape[:-2]` are
// broadcast to a common batch shape (so a rank-2 tensor can be used
// against a rank-4 batch, etc.). K and V's trailing sequence dim must
// agree (S_k); the head / batch dims may broadcast.
Tensor attention(const Tensor& q,
                 const Tensor& k,
                 const Tensor& v,
                 const Tensor& mask = Tensor{},
                 bool causal = false,
                 double dropout_p = 0.0);

// GQA-native single-query decode attention (inference fast path).
//
//   q: [B, H,   1,   D]   (one new query row per sequence)
//   k: [B, Hkv, S_k, D]   (full KV prefix, NOT expanded to H heads)
//   v: [B, Hkv, S_k, D]
//
// Query head `h` attends to KV head `h / (H / Hkv)`, so the caller never
// materializes the `repeat_kv` expansion (which copies the entire KV cache
// H/Hkv× every layer/step — ~25% of decode GPU time at GQA ratios like
// 32/4). Routes to the fused split-K decode kernel on CUDA; on CPU (or any
// shape the fused kernel can't take) it falls back to expanding K/V and
// calling the composite `attention`. Forward / inference only (no autograd
// edge); `H % Hkv == 0` required. Returns [B, H, 1, D] contiguous.
Tensor decode_attention_gqa(const Tensor& q,
                            const Tensor& k,
                            const Tensor& v,
                            bool causal = false);

// GQA-native multi-query (prefill) attention.
//
//   q: [B, H,   S_q, D]
//   k: [B, Hkv, S_k, D]   (NOT expanded to H heads)
//   v: [B, Hkv, S_k, D]
//
// The full-prompt prefill case (`S_q == S_k`, `causal == true`) where the
// new queries attend causally over the whole prefix. Like
// `decode_attention_gqa`, query head `h` reads KV head `h / (H / Hkv)`
// inside the fused kernel, so no `repeat_kv` materialization is needed
// (that 8× KV copy is ~1.5 ms / TinyLlama prefill at a 32/4 GQA ratio).
// Routes to the fused FlashAttention kernel on CUDA; falls back to
// expanding K/V + composite `attention` otherwise. Forward / inference
// only; `H % Hkv == 0` required. Returns [B, H, S_q, D] contiguous.
Tensor prefill_attention_gqa(const Tensor& q,
                             const Tensor& k,
                             const Tensor& v,
                             bool causal = true);

// B-024c fast path for `MultiHeadAttention::forward_step` prefill: identical
// math to `prefill_attention_gqa`, but reads Q (contiguous permuted view) and
// K/V (the KV-cache `[B,Hkv,S,D]` narrows) *in place* via element-strides and
// writes its output in **BSHD** `[B, S_q, H, D]` layout. This removes the
// per-layer `contiguous()` copies of the KV narrows and the `[B,H,S,D] →
// [B,S,H,D]` output transpose — the largest non-GEMM prefill cost. Returns an
// **undefined Tensor** when the fused CUDA path is not eligible (CPU, FP64,
// non-unit head-dim stride, autograd on, …); the caller must fall back to the
// contiguous `prefill_attention_gqa` + transpose. Inference only.
Tensor prefill_attention_gqa_bshd(const Tensor& q,
                                  const Tensor& k,
                                  const Tensor& v,
                                  bool causal = true);

}  // namespace tesseract::ops

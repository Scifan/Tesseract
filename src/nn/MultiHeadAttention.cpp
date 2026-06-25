#include "tesseract/nn/MultiHeadAttention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Attention.hpp"
#include "tesseract/ops/Indexing.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/nn/PagedAttention.hpp"
#include "tesseract/nn/PagedKVCache.hpp"
#include "tesseract/nn/QuantizedPagedKVCache.hpp"
#include "tesseract/nn/RotaryEmbedding.hpp"
#include "tesseract/quant/Scheme.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

namespace {

// GQA repeat: expand `[B, Hkv, S, Dh]` to `[B, Hkv*G, S, Dh]` by sharing
// each KV head across `G` consecutive query heads (the PyTorch
// `repeat_kv` interleave: head order is kv0,kv0,…,kv1,kv1,…). `G == 1` is
// a no-op so plain MHA pays nothing. Implemented with autograd-aware view
// ops so gradients flow back (and accumulate) onto the KV heads.
Tensor repeat_kv(const Tensor& t, int64_t group) {
  if (group == 1) return t;
  const int64_t B = t.shape()[0];
  const int64_t Hkv = t.shape()[1];
  const int64_t S = t.shape()[2];
  const int64_t Dh = t.shape()[3];
  Tensor r = ops::reshape(t, Shape({B, Hkv, 1, S, Dh}));
  r = ops::broadcast_to(r, Shape({B, Hkv, group, S, Dh}));
  return ops::reshape(r, Shape({B, Hkv * group, S, Dh}));
}

}  // namespace

MultiHeadAttention::MultiHeadAttention(int64_t d_model, int64_t num_heads,
                                       bool use_bias, bool causal, DType dtype,
                                       double rope_base, int64_t rope_max_seq,
                                       int64_t num_kv_heads)
    : d_model_(d_model),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads > 0 ? num_kv_heads : num_heads),
      head_dim_(d_model / num_heads),
      kv_dim_((num_kv_heads > 0 ? num_kv_heads : num_heads) * (d_model / num_heads)),
      causal_(causal) {
  TESSERACT_CHECK(d_model > 0 && num_heads > 0,
                  "MultiHeadAttention: d_model ({}) and num_heads ({}) must be positive",
                  d_model, num_heads);
  TESSERACT_CHECK(d_model % num_heads == 0,
                  "MultiHeadAttention: d_model ({}) must be divisible by num_heads ({})",
                  d_model, num_heads);
  TESSERACT_CHECK(num_heads_ % num_kv_heads_ == 0,
                  "MultiHeadAttention: num_heads ({}) must be divisible by "
                  "num_kv_heads ({})",
                  num_heads_, num_kv_heads_);

  // Standard four-projection layout; Llama collapses Q/K/V into a single
  // "packed" projection for throughput, but keeping them split here
  // matches the public API (`Linear`) 1:1 and lets a future M2L fusion
  // pass pattern-match three concurrent projections without needing a
  // separate `PackedLinear` type first. Tradeoff: three extra matmul
  // launches on the forward path — negligible vs the attention cost
  // for transformer-sized shapes.
  //
  // GQA: Q stays `d_model → d_model` (num_heads heads), but K/V shrink to
  // `d_model → kv_dim_` (num_kv_heads heads). For plain MHA kv_dim_ ==
  // d_model so this is unchanged.
  q_proj_ = std::make_shared<Linear>(d_model, d_model, use_bias, dtype);
  k_proj_ = std::make_shared<Linear>(d_model, kv_dim_, use_bias, dtype);
  v_proj_ = std::make_shared<Linear>(d_model, kv_dim_, use_bias, dtype);
  o_proj_ = std::make_shared<Linear>(d_model, d_model, use_bias, dtype);

  register_module("q_proj", q_proj_);
  register_module("k_proj", k_proj_);
  register_module("v_proj", v_proj_);
  register_module("o_proj", o_proj_);

  // Optional rotary position embedding. Llama defaults are base=10000
  // and max_seq=model-context-length (4096 for Llama-2, 8192 for
  // Llama-3). Passing `rope_base <= 0` or `rope_max_seq <= 0` skips
  // the RoPE path entirely, which preserves the M2K pre-B-014
  // behavior for any caller that wants a bare attention block.
  if (rope_base > 0.0 && rope_max_seq > 0) {
    rope_ = std::make_shared<RotaryEmbedding>(head_dim_, rope_base,
                                              rope_max_seq, dtype);
    register_module("rope", rope_);
  }
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
  TESSERACT_CHECK(x.rank() == 3,
                  "MultiHeadAttention::forward: expected rank-3 input [B, S, D], got {}",
                  x.shape().to_string());
  TESSERACT_CHECK(x.shape()[2] == d_model_,
                  "MultiHeadAttention::forward: input feature dim {} != d_model {}",
                  x.shape()[2], d_model_);

  const int64_t B = x.shape()[0];
  const int64_t S = x.shape()[1];
  const int64_t H = num_heads_;
  const int64_t Hkv = num_kv_heads_;
  const int64_t Dh = head_dim_;

  // 1) Project. Linear accepts any rank >= 2 (M2K-relaxed). Q comes out
  //    [B, S, d_model] (H heads); K/V come out [B, S, kv_dim] (Hkv heads).
  Tensor q = q_proj_->forward(x);
  Tensor k = k_proj_->forward(x);
  Tensor v = v_proj_->forward(x);

  // 2) Split heads: [B, S, n*Dh] → [B, S, n, Dh] → [B, n, S, Dh].
  //    We use `ops::reshape` (autograd-aware, takes the
  //    contiguous->strided path transparently) and `ops::permute` which
  //    both have backward nodes so grads flow back through the
  //    restructure.
  auto split_heads = [&](Tensor t, int64_t n) -> Tensor {
    Tensor r = ops::reshape(t, Shape({B, S, n, Dh}));
    return ops::permute(r, {0, 2, 1, 3});  // [B, n, S, Dh]
  };
  q = split_heads(q, H);
  k = split_heads(k, Hkv);
  v = split_heads(v, Hkv);

  // 2b) Rotary position embedding on Q and K (V is left untouched —
  //     this matches the GPT-NeoX / Llama convention). The rope child
  //     module handles the `[B, n, S, Dh]` → `[B, n, S, Dh]` shape
  //     contract directly; its forward treats the last two dims as
  //     `(S, d_head)` and broadcasts the cos/sin table across the leading
  //     dims. Skipped entirely when no RoPE was configured.
  if (rope_) {
    q = rope_->forward(q);
    k = rope_->forward(k);
  }

  // 2c) GQA: share each KV head across H/Hkv query heads (no-op for MHA).
  k = repeat_kv(k, H / Hkv);
  v = repeat_kv(v, H / Hkv);

  // 3) Attention. Causal flag flows straight through; no additive mask
  //    for the self-attention path at this milestone — padding masks
  //    will be caller-supplied in future examples.
  Tensor out_heads = ops::attention(q, k, v, /*mask=*/Tensor{}, causal_, /*dropout_p=*/0.0);

  // 4) Merge heads: [B, H, S, Dh] → [B, S, H, Dh] → [B, S, d_model].
  //    The `reshape` here is **not** a plain view because the preceding
  //    `permute` made the memory non-contiguous; `ops::reshape` falls
  //    back to `ops::contiguous` + view internally, so this is always
  //    safe (at the cost of one extra copy — measured cheap relative to
  //    the attention cost, and eliminated once M2L fuses the chain).
  Tensor out_bsh = ops::permute(out_heads, {0, 2, 1, 3});
  Tensor out_flat = ops::reshape(out_bsh, Shape({B, S, d_model_}));

  // 5) Output projection.
  return o_proj_->forward(out_flat);
}

namespace {

// Rectangular causal mask for chunked-prefill decode steps.
//
// Query positions are `[pos, pos + S_new)`, keys span `[0, pos + S_new)`.
// Query `i` (global position `pos + i`) must see only keys `j <= pos + i`.
// The returned additive mask has shape `[S_new, pos + S_new]` with `0` on
// allowed slots and `-inf` on forbidden ones; broadcasting handles the
// leading `[B, H]` dims inside `ops::attention`'s add(scores, mask).
//
// Built on CPU in q's dtype then migrated to q's device. `-inf`
// representation: IEEE +/-infinity for FP32/FP64 and the half-precision
// `Inf` encoding for FP16/BF16 (all four dtypes have a representable
// infinity so `-inf + x` correctly flushes softmax rows to zero).
Tensor make_decode_mask(int64_t S_new, int64_t pos, DType dt, Device dev) {
  TESSERACT_CHECK(S_new > 0 && pos >= 0,
                  "make_decode_mask: invalid S_new={}, pos={}", S_new, pos);
  const int64_t S_k = pos + S_new;
  Tensor mask = Tensor::zeros({S_new, S_k}, dt, cpu_device());

  dispatch_float_with_half(dt, [&]<typename T>() {
    T* p = mask.data_ptr<T>();
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < S_new; ++i) {
      const int64_t first_forbidden = pos + i + 1;
      for (int64_t j = first_forbidden; j < S_k; ++j) {
        p[i * S_k + j] = static_cast<T>(neg_inf);
      }
    }
  });

  if (dev.is_cpu()) return mask;
  return mask.to(dev);
}

// Single-sequence attention core shared by the batched decode path.
// Inputs are already head-split: `q` is `[1, H, Sn, Dh]`, `k_new`/`v_new`
// are `[1, Hkv, Sn, Dh]` (pre-RoPE). Applies RoPE@pos to q and k_new,
// appends the new K/V into `cache`, gathers the full prefix, repeat-
// expands the Hkv KV heads to H query heads (GQA), runs attention, and
// merges heads back to `[1, Sn, D_model]`. Identical to the inner half of
// `forward_step` but factored so `forward_step_batched` can drive it per
// sequence after a single batched projection.
Tensor attend_single(Tensor q, Tensor k_new, Tensor v_new,
                     RotaryEmbedding* rope, KVCacheBase& cache,
                     int64_t H, int64_t Hkv, int64_t Sn, int64_t Dh,
                     int64_t d_model) {
  const int64_t pos = cache.current_len();
  if (rope != nullptr) {
    q     = rope->forward_offset(q,     pos);
    k_new = rope->forward_offset(k_new, pos);
  }
  k_new = ops::contiguous(k_new);
  v_new = ops::contiguous(v_new);
  cache.append(k_new, v_new);
  Tensor k_all = repeat_kv(cache.keys_view(),   H / Hkv);
  Tensor v_all = repeat_kv(cache.values_view(), H / Hkv);
  Tensor mask;
  if (Sn > 1) mask = make_decode_mask(Sn, pos, q.dtype(), q.device());
  Tensor out_heads = ops::attention(q, k_all, v_all, mask,
                                    /*causal=*/false, /*dropout_p=*/0.0);
  Tensor out_bsh = ops::permute(out_heads, {0, 2, 1, 3});
  return ops::reshape(out_bsh, Shape({1, Sn, d_model}));
}

}  // namespace

Tensor MultiHeadAttention::forward_step(const Tensor& x, KVCacheBase& cache) {
  TESSERACT_CHECK(x.rank() == 3,
                  "MultiHeadAttention::forward_step: expected rank-3 input "
                  "[B, S_new, D_model], got {}", x.shape().to_string());
  TESSERACT_CHECK(x.shape()[2] == d_model_,
                  "MultiHeadAttention::forward_step: input feature dim {} "
                  "!= d_model {}", x.shape()[2], d_model_);
  TESSERACT_CHECK(x.shape()[0] == cache.batch(),
                  "MultiHeadAttention::forward_step: batch {} disagrees with "
                  "cache batch {}", x.shape()[0], cache.batch());
  TESSERACT_CHECK(num_kv_heads_ == cache.num_heads() &&
                  head_dim_ == cache.head_dim(),
                  "MultiHeadAttention::forward_step: KV head shape disagreement "
                  "(module: Hkv={}, Dh={}; cache: H={}, Dh={})",
                  num_kv_heads_, head_dim_, cache.num_heads(), cache.head_dim());

  const int64_t B   = x.shape()[0];
  const int64_t Sn  = x.shape()[1];
  const int64_t H   = num_heads_;
  const int64_t Hkv = num_kv_heads_;
  const int64_t Dh  = head_dim_;
  const int64_t pos = cache.current_len();

  // Inference-only: every step runs under NoGradGuard so the
  // projections don't accumulate autograd edges into the cache's
  // detached storage. This also lets the Wave 2.2 fused RMSNorm
  // path take over inside the surrounding `TransformerBlock` —
  // both require grad mode to be off.
  NoGradGuard nogg;

  // 1) Project the new-tokens slab. `Linear` accepts any rank >= 2
  //    so we get Q/K/V as [B, Sn, D_model] directly.
  Tensor q = q_proj_->forward(x);
  Tensor k_new = k_proj_->forward(x);
  Tensor v_new = v_proj_->forward(x);

  // 2) Split heads — Q into H heads, K/V into Hkv heads (GQA). `reshape`
  //    + `permute` mirror the full `forward()` recipe.
  auto split_heads = [&](Tensor t, int64_t n) -> Tensor {
    Tensor r = ops::reshape(t, Shape({B, Sn, n, Dh}));
    return ops::permute(r, {0, 2, 1, 3});
  };
  q     = split_heads(q, H);
  k_new = split_heads(k_new, Hkv);
  v_new = split_heads(v_new, Hkv);

  // 3) RoPE at positions [pos, pos + Sn). Applied to Q and to the
  //    new K only — V is never rotated (GPT-NeoX / Llama convention).
  //    Rotating K *before* appending into the cache means the cached
  //    K already has its positional rotation baked in, which is
  //    critical: the attention step below treats the cache as a
  //    flat key slab and cannot re-apply RoPE per-entry.
  if (rope_) {
    q     = rope_->forward_offset(q,     pos);
    k_new = rope_->forward_offset(k_new, pos);
  }

  // 4) `ops::attention` requires contiguous operands when paired
  //    with strided views (the kernel drops into the composite
  //    softmax / matmul path). `k_new` / `v_new` come out of
  //    permute → non-contiguous; the cache's `append` requires
  //    contiguous sources, so we realize them here.
  k_new = ops::contiguous(k_new);
  v_new = ops::contiguous(v_new);

  // 5) Append into the cache (this advances current_len_ by Sn),
  //    then read back the `[B, H, pos + Sn, Dh]` prefix view for
  //    attention. `keys_view` / `values_view` are zero-copy narrows.
  cache.append(k_new, v_new);
  Tensor k_all = cache.keys_view();    // [B, Hkv, pos+Sn, Dh]
  Tensor v_all = cache.values_view();

  // 6) Attention. Two regimes:
  //      * `Sn == 1`: pure single-query decode. Route to the GQA-native
  //        fused split-K kernel via `decode_attention_gqa`, which maps each
  //        query head to its KV head *inside* the kernel — so we skip the
  //        `repeat_kv` materialization that otherwise copies the entire KV
  //        cache H/Hkv× every layer/step (~25% of decode GPU time at a
  //        32/4 GQA ratio). The KV cache views may be non-contiguous
  //        narrows, so realize them first (cheap: contiguous already on the
  //        common append-then-read path).
  //      * `Sn > 1` (chunked prefill): query positions `[pos, pos + Sn)`
  //        attend to keys `[0, pos + Sn)`. A `Sn == Sk` causal flag can't
  //        express this, so we expand KV heads and materialize a
  //        rectangular `[Sn, pos+Sn]` additive mask for the composite path.
  Tensor out_heads;
  if (Sn == 1) {
    Tensor k_c = k_all.is_contiguous() ? k_all : ops::contiguous(k_all);
    Tensor v_c = v_all.is_contiguous() ? v_all : ops::contiguous(v_all);
    out_heads = ops::decode_attention_gqa(q, k_c, v_c, /*causal=*/false);
  } else if (pos == 0) {
    // Full-prompt prefill is a *square* causal attention (S_q == S_k).
    // Route to the GQA-native fused FlashAttention kernel: a single
    // launch with the causal mask generated on-chip and KV heads mapped
    // inside the kernel — so we skip both the composite path's 6 kernel
    // launches + `[S_q, S_k]` score-matrix HBM round trip AND the
    // `repeat_kv` 8× KV materialization (together ~3.2 ms / TinyLlama
    // prefill at a 32/4 GQA ratio, measured).
    //
    // B-024c: prefer the BSHD fast path — it reads the KV-cache narrows in
    // place (no `contiguous()` copy) and writes output in [B, Sn, H, Dh]
    // so the head-merge below is a free reshape instead of a transpose.
    // These layout copies were the largest remaining non-GEMM prefill cost.
    Tensor out_bshd =
        ops::prefill_attention_gqa_bshd(q, k_all, v_all, /*causal=*/true);
    if (out_bshd.defined()) {
      Tensor out_flat = ops::reshape(out_bshd, Shape({B, Sn, d_model_}));
      return o_proj_->forward(out_flat);
    }
    // Fallback (CPU / FP64 / non-unit head stride): realize the narrows
    // contiguous and take the BHSD path + transpose.
    Tensor k_c = k_all.is_contiguous() ? k_all : ops::contiguous(k_all);
    Tensor v_c = v_all.is_contiguous() ? v_all : ops::contiguous(v_all);
    out_heads = ops::prefill_attention_gqa(q, k_c, v_c, /*causal=*/true);
  } else {
    // Chunked prefill (pos > 0): rectangular [S_q, pos+S_q] attention
    // that the square `causal` flag can't express — expand KV heads and
    // keep the additive mask + composite path.
    k_all = repeat_kv(k_all, H / Hkv);
    v_all = repeat_kv(v_all, H / Hkv);
    Tensor mask = make_decode_mask(Sn, pos, q.dtype(), q.device());
    out_heads = ops::attention(q, k_all, v_all, mask,
                               /*causal=*/false, /*dropout_p=*/0.0);
  }

  // 7) Merge heads and project back to D_model.
  Tensor out_bsh  = ops::permute(out_heads, {0, 2, 1, 3});
  Tensor out_flat = ops::reshape(out_bsh, Shape({B, Sn, d_model_}));
  return o_proj_->forward(out_flat);
}

Tensor MultiHeadAttention::forward_step_batched(
    const Tensor& x, const std::vector<KVCacheBase*>& caches) {
  TESSERACT_CHECK(x.rank() == 3,
                  "MultiHeadAttention::forward_step_batched: expected rank-3 "
                  "input [A, S_new, D_model], got {}", x.shape().to_string());
  TESSERACT_CHECK(x.shape()[2] == d_model_,
                  "MultiHeadAttention::forward_step_batched: feature dim {} != "
                  "d_model {}", x.shape()[2], d_model_);
  const int64_t A  = x.shape()[0];
  const int64_t Sn = x.shape()[1];
  const int64_t H  = num_heads_;
  const int64_t Hkv = num_kv_heads_;
  const int64_t Dh = head_dim_;
  TESSERACT_CHECK(static_cast<int64_t>(caches.size()) == A,
                  "MultiHeadAttention::forward_step_batched: {} caches for a "
                  "batch of {}", caches.size(), A);

  NoGradGuard nogg;

  // Batched projections — one matmul each over all A·Sn rows. This is the
  // compute-batching win; the per-sequence work below is only RoPE +
  // cache append/gather + the (memory-bound) attention.
  Tensor q = q_proj_->forward(x);      // [A, Sn, H*Dh]
  Tensor k = k_proj_->forward(x);      // [A, Sn, Hkv*Dh]
  Tensor v = v_proj_->forward(x);      // [A, Sn, Hkv*Dh]

  // Fused ragged paged decode fast path. For a CUDA single-token decode
  // (Sn == 1) where every cache is paged and shares one per-layer pool,
  // collapse the per-sequence gather + repeat_kv + attention loop into ONE
  // launch over the whole active set: K/V read in place from the pool via
  // each request's block table, GQA head mapping in-kernel, online softmax.
  // RoPE + cache append stay per-request (cheap memory ops). Two pool
  // flavors take this path:
  //   * FP paged pools (Wave 11, B-032+)        → `paged_decode_attention`;
  //   * INT8 paged pools (Wave 13, B-032+++)    → `paged_decode_attention_int8`
  //     (dequant fused into the kernel — no FP-prefix gather).
  // CPU / contiguous caches / chunked-prefill (Sn > 1) keep the exact
  // per-sequence path below, so the scheduler's CPU bit-exact parity holds.
  if (Sn >= 1 && x.device().is_cuda() && A > 0) {
    std::vector<PagedKVCache*> pc(static_cast<std::size_t>(A));
    std::vector<QuantizedPagedKVCache*> qc(static_cast<std::size_t>(A));
    bool all_paged = true, all_qpaged = true;
    for (int64_t r = 0; r < A; ++r) {
      pc[static_cast<std::size_t>(r)] = dynamic_cast<PagedKVCache*>(caches[r]);
      qc[static_cast<std::size_t>(r)] = dynamic_cast<QuantizedPagedKVCache*>(caches[r]);
      all_paged  &= pc[static_cast<std::size_t>(r)] != nullptr;
      all_qpaged &= qc[static_cast<std::size_t>(r)] != nullptr;
    }

   if (Sn > 1 && (all_paged || all_qpaged)) {
    // Wave 14 (B-032++++): fused paged PREFILL. Chunked-prefill (Sn > 1)
    // over the active set collapses into one launch with a causal mask —
    // the per-sequence `attend_single` loop below is bypassed. Per-request
    // RoPE@pos + append stay (cheap), then `paged_prefill_attention[_int8]`
    // reads K/V in place and applies the `kv_len - S + s` causal bound.
    const Device dev = x.device();
    auto run_prefill =
        [&](int64_t block_size, auto&& block_table_of, auto&& attn_op) -> Tensor {
      std::vector<Tensor> q_list;
      q_list.reserve(static_cast<std::size_t>(A));
      std::vector<int32_t> kvlens_host(static_cast<std::size_t>(A));
      int64_t max_logical = 1;
      for (int64_t r = 0; r < A; ++r) {
        KVCacheBase* c = caches[r];
        TESSERACT_CHECK(c->batch() == 1,
                        "forward_step_batched(paged-prefill): cache {} batch "
                        "{} != 1", r, c->batch());
        auto split = [&](const Tensor& t, int64_t n) -> Tensor {
          Tensor row = t.narrow(/*dim=*/0, r, 1);          // [1, Sn, n*Dh]
          Tensor rr  = ops::reshape(row, Shape({1, Sn, n, Dh}));
          return ops::permute(rr, {0, 2, 1, 3});           // [1, n, Sn, Dh]
        };
        const int64_t pos = c->current_len();
        Tensor qr = split(q, H);                           // [1, H, Sn, Dh]
        Tensor kr = split(k, Hkv);
        Tensor vr = split(v, Hkv);
        if (rope_) {
          qr = rope_->forward_offset(qr, pos);
          kr = rope_->forward_offset(kr, pos);
        }
        kr = ops::contiguous(kr);
        vr = ops::contiguous(vr);
        c->append(kr, vr);
        // [1, H, Sn, Dh] → [1, Sn, H, Dh] so the kernel's [A,S,H,D] layout
        // is a plain stack over the active set.
        Tensor q_shd = ops::contiguous(ops::permute(qr, {0, 2, 1, 3}));
        q_list.push_back(q_shd);
        const int64_t len = c->current_len();
        kvlens_host[static_cast<std::size_t>(r)] = static_cast<int32_t>(len);
        max_logical = std::max(max_logical, (len + block_size - 1) / block_size);
      }

      Tensor q4 = (A == 1) ? ops::contiguous(q_list[0]) : ops::cat(q_list, 0);
      Tensor q_heads = ops::reshape(q4, Shape({A, Sn, H, Dh}));  // [A,Sn,H,Dh]

      Tensor bt_host = Tensor::empty(Shape({A, max_logical}), DType::Int32,
                                     cpu_device());
      int32_t* btp = bt_host.data_ptr<int32_t>();
      for (int64_t i = 0; i < A * max_logical; ++i) btp[i] = 0;
      for (int64_t r = 0; r < A; ++r) {
        const std::vector<int32_t>& bt = block_table_of(r);
        for (std::size_t i = 0; i < bt.size(); ++i)
          btp[r * max_logical + static_cast<int64_t>(i)] = bt[i];
      }
      Tensor kvl_host = Tensor::empty(Shape({A}), DType::Int32, cpu_device());
      int32_t* lp = kvl_host.data_ptr<int32_t>();
      for (int64_t r = 0; r < A; ++r) lp[r] = kvlens_host[static_cast<std::size_t>(r)];

      Tensor block_tables = bt_host.to(dev);
      Tensor kv_lens      = kvl_host.to(dev);

      const double scale = 1.0 / std::sqrt(static_cast<double>(Dh));
      Tensor o = attn_op(q_heads, block_tables, kv_lens, scale);  // [A,Sn,H,Dh]
      Tensor merged = ops::reshape(o, Shape({A, Sn, d_model_}));  // [A,Sn,D]
      return o_proj_->forward(merged);
    };

    if (all_paged) {
      const std::shared_ptr<PagedKVPool>& pool = pc[0]->pool();
      for (int64_t r = 0; r < A; ++r)
        TESSERACT_CHECK(pc[static_cast<std::size_t>(r)]->pool() == pool,
                        "forward_step_batched(paged-prefill): cache {} must "
                        "share the layer pool", r);
      return run_prefill(
          pool->block_size(),
          [&](int64_t r) -> const std::vector<int32_t>& {
            return pc[static_cast<std::size_t>(r)]->block_table(0);
          },
          [&](const Tensor& qh, const Tensor& bt, const Tensor& kvl,
              double scale) -> Tensor {
            return paged_prefill_attention(qh, pool->keys(), pool->values(), bt,
                                           kvl, scale, H / Hkv);
          });
    }
    {
      const std::shared_ptr<QuantizedPagedKVPool>& pool = qc[0]->pool();
      for (int64_t r = 0; r < A; ++r)
        TESSERACT_CHECK(qc[static_cast<std::size_t>(r)]->pool() == pool,
                        "forward_step_batched(qpaged-prefill): cache {} must "
                        "share the layer pool", r);
      return run_prefill(
          pool->block_size(),
          [&](int64_t r) -> const std::vector<int32_t>& {
            return qc[static_cast<std::size_t>(r)]->block_table(0);
          },
          [&](const Tensor& qh, const Tensor& bt, const Tensor& kvl,
              double scale) -> Tensor {
            return paged_prefill_attention_int8(
                qh, pool->keys(), pool->key_scale(), pool->values(),
                pool->value_scale(), bt, kvl, scale, H / Hkv);
          });
    }
   }

   if (Sn == 1) {
    // Shared driver: per-request RoPE + append, then assemble the
    // [A, max_logical] block table + [A] lens and dispatch to `attn_op`,
    // which closes over the pool tensors and picks the FP / INT8 kernel.
    const Device dev = x.device();
    auto run_fused =
        [&](int64_t block_size, auto&& block_table_of, auto&& attn_op) -> Tensor {
      std::vector<Tensor> q_list;
      q_list.reserve(static_cast<std::size_t>(A));
      std::vector<int32_t> lens_host(static_cast<std::size_t>(A));
      int64_t max_logical = 1;
      for (int64_t r = 0; r < A; ++r) {
        KVCacheBase* c = caches[r];
        TESSERACT_CHECK(c->batch() == 1,
                        "forward_step_batched(paged): cache {} batch {} != 1",
                        r, c->batch());
        auto split = [&](const Tensor& t, int64_t n) -> Tensor {
          Tensor row = t.narrow(/*dim=*/0, r, 1);          // [1, 1, n*Dh]
          Tensor rr  = ops::reshape(row, Shape({1, 1, n, Dh}));
          return ops::permute(rr, {0, 2, 1, 3});           // [1, n, 1, Dh]
        };
        const int64_t pos = c->current_len();
        Tensor qr = split(q, H);
        Tensor kr = split(k, Hkv);
        Tensor vr = split(v, Hkv);
        if (rope_) {
          qr = rope_->forward_offset(qr, pos);
          kr = rope_->forward_offset(kr, pos);
        }
        kr = ops::contiguous(kr);
        vr = ops::contiguous(vr);
        c->append(kr, vr);
        q_list.push_back(qr);                              // [1, H, 1, Dh]
        const int64_t len = c->current_len();
        lens_host[static_cast<std::size_t>(r)] = static_cast<int32_t>(len);
        max_logical = std::max(max_logical, (len + block_size - 1) / block_size);
      }

      Tensor q4 = (A == 1) ? ops::contiguous(q_list[0]) : ops::cat(q_list, 0);
      Tensor q_heads = ops::reshape(q4, Shape({A, H, Dh}));  // [A, H, Dh]

      Tensor bt_host = Tensor::empty(Shape({A, max_logical}), DType::Int32,
                                     cpu_device());
      int32_t* btp = bt_host.data_ptr<int32_t>();
      for (int64_t i = 0; i < A * max_logical; ++i) btp[i] = 0;
      for (int64_t r = 0; r < A; ++r) {
        const std::vector<int32_t>& bt = block_table_of(r);
        for (std::size_t i = 0; i < bt.size(); ++i)
          btp[r * max_logical + static_cast<int64_t>(i)] = bt[i];
      }
      Tensor lens_host_t = Tensor::empty(Shape({A}), DType::Int32, cpu_device());
      int32_t* lp = lens_host_t.data_ptr<int32_t>();
      for (int64_t r = 0; r < A; ++r) lp[r] = lens_host[static_cast<std::size_t>(r)];

      Tensor block_tables = bt_host.to(dev);
      Tensor lens_t       = lens_host_t.to(dev);

      const double scale = 1.0 / std::sqrt(static_cast<double>(Dh));
      Tensor o = attn_op(q_heads, block_tables, lens_t, scale);  // [A, H, Dh]
      Tensor merged = ops::reshape(o, Shape({A, 1, d_model_}));  // [A, 1, D]
      return o_proj_->forward(merged);
    };

    if (all_paged) {
      const std::shared_ptr<PagedKVPool>& pool = pc[0]->pool();
      for (int64_t r = 0; r < A; ++r)
        TESSERACT_CHECK(pc[static_cast<std::size_t>(r)]->pool() == pool,
                        "forward_step_batched(paged): cache {} must share the "
                        "layer pool", r);
      return run_fused(
          pool->block_size(),
          [&](int64_t r) -> const std::vector<int32_t>& {
            return pc[static_cast<std::size_t>(r)]->block_table(0);
          },
          [&](const Tensor& qh, const Tensor& bt, const Tensor& ln,
              double scale) -> Tensor {
            return paged_decode_attention(qh, pool->keys(), pool->values(), bt,
                                          ln, scale, H / Hkv);
          });
    }
    if (all_qpaged) {
      const std::shared_ptr<QuantizedPagedKVPool>& pool = qc[0]->pool();
      for (int64_t r = 0; r < A; ++r)
        TESSERACT_CHECK(qc[static_cast<std::size_t>(r)]->pool() == pool,
                        "forward_step_batched(qpaged): cache {} must share the "
                        "layer pool", r);
      return run_fused(
          pool->block_size(),
          [&](int64_t r) -> const std::vector<int32_t>& {
            return qc[static_cast<std::size_t>(r)]->block_table(0);
          },
          [&](const Tensor& qh, const Tensor& bt, const Tensor& ln,
              double scale) -> Tensor {
            return paged_decode_attention_int8(
                qh, pool->keys(), pool->key_scale(), pool->values(),
                pool->value_scale(), bt, ln, scale, H / Hkv);
          });
    }
   }
  }

  std::vector<Tensor> outs;
  outs.reserve(static_cast<std::size_t>(A));
  for (int64_t r = 0; r < A; ++r) {
    TESSERACT_CHECK(caches[r] != nullptr,
                    "MultiHeadAttention::forward_step_batched: cache {} null", r);
    TESSERACT_CHECK(caches[r]->batch() == 1,
                    "MultiHeadAttention::forward_step_batched: cache {} has "
                    "batch {} (each batched-decode sequence needs its own "
                    "batch-1 cache)", r, caches[r]->batch());
    TESSERACT_CHECK(num_kv_heads_ == caches[r]->num_heads() &&
                    head_dim_ == caches[r]->head_dim(),
                    "MultiHeadAttention::forward_step_batched: cache {} KV "
                    "head shape disagreement", r);

    auto split = [&](const Tensor& t, int64_t n) -> Tensor {
      Tensor row = t.narrow(/*dim=*/0, r, 1);          // [1, Sn, n*Dh]
      Tensor rr  = ops::reshape(row, Shape({1, Sn, n, Dh}));
      return ops::permute(rr, {0, 2, 1, 3});           // [1, n, Sn, Dh]
    };
    Tensor qr = split(q, H);
    Tensor kr = split(k, Hkv);
    Tensor vr = split(v, Hkv);
    outs.push_back(attend_single(qr, kr, vr, rope_.get(), *caches[r],
                                 H, Hkv, Sn, Dh, d_model_));  // [1, Sn, D]
  }

  // Restack the per-sequence outputs and project once more (batched).
  Tensor merged = (A == 1) ? outs[0] : ops::cat(outs, /*dim=*/0);  // [A, Sn, D]
  return o_proj_->forward(merged);
}

void MultiHeadAttention::quantize_(const quant::Scheme& scheme) {
  // Walk the four projection slots and swap any FP `Linear` to the
  // scheme-specific quantized drop-in. `dynamic_pointer_cast` on
  // `Linear` returns null for anything already quantized, so the
  // idempotence guarantee documented in the header falls out for
  // free. The slot name passed to `replace_module` must match the
  // original `register_module` name exactly — otherwise
  // `named_parameters()` would diverge from the pre-quantize layout
  // and every checkpoint loader / logger that keyed on the old
  // names would break.
  auto try_swap = [&](const std::string& slot,
                      std::shared_ptr<Module>& holder) {
    auto as_fp = std::dynamic_pointer_cast<Linear>(holder);
    if (!as_fp) return;  // already quantized (or a different type)
    auto replacement = quant::quantize_linear(*as_fp, scheme);
    holder = replacement;
    replace_module(slot, replacement);
  };
  try_swap("q_proj", q_proj_);
  try_swap("k_proj", k_proj_);
  try_swap("v_proj", v_proj_);
  try_swap("o_proj", o_proj_);
}

}  // namespace tesseract::nn

#pragma once

// Wave 11 (B-032+): fused ragged paged decode-attention.
//
// Wave 10 batched the dense decode layers (projections / FFN / LM head)
// across the active set but still looped attention per sequence — each
// loop iteration does `keys_view()` (a full gather of the request's KV
// prefix out of the scattered paged blocks into a fresh contiguous slab),
// a `repeat_kv` (GQA head replication), and a separate `ops::attention`
// launch. For A active requests that is A gathers + A·G× KV materializes +
// A attention launches per layer per tick — all of it memory-bound and
// launch-bound on the decode hot path.
//
// `paged_decode_attention` collapses that into a SINGLE kernel over the
// whole active set, reading K/V straight from the shared physical pool via
// each request's block table (no gather), mapping query head `h` to KV
// head `h / group` on the fly (no `repeat_kv` materialization), and
// running an online (FlashAttention-style) softmax so the `[1, S_k]` score
// row is never written to HBM. One query token per request (decode).
//
//   q            : [A, H, D]                          RoPE already applied
//   k_pool/v_pool: [num_blocks, Hkv, block_size, D]   shared per-layer pool
//   block_tables : [A, max_logical]  Int32            logical → physical id
//   lens         : [A]               Int32            each request's KV length
//   scale        : softmax scale (1/sqrt(D))
//   group        : H / Hkv (GQA; group == 1 ⇒ plain MHA)
//   returns O    : [A, H, D]
//
// All interior math (scores, exp, weighted-V, normalize) is FP32
// regardless of storage dtype (FP32/FP16/BF16), matching the
// `ops::attention` / Wave 4.2 fused-attention numerical contract — so the
// result tracks the gather+`repeat_kv`+`ops::attention` reference to the
// usual softmax tolerance. A request with `lens[r] == 0` yields a zero
// output row (same fallback as a fully-masked attention row).

#include <cstdint>

#include "tesseract/core/Tensor.hpp"

namespace tesseract::nn {

Tensor paged_decode_attention(const Tensor& q, const Tensor& k_pool,
                              const Tensor& v_pool, const Tensor& block_tables,
                              const Tensor& lens, double scale, int64_t group);

// Wave 12 (B-032++): INT8-direct variant. Same single-launch ragged decode
// as `paged_decode_attention`, but K/V are stored INT8 with a per-(block,
// head, slot) FP32 scale (the Wave-9 per-token, per-head symmetric scheme:
// one scale per `D`-vector) and dequantized **inside the kernel** — the
// FP-prefix transient `paged_decode_attention` would read is never built.
// This unifies the GQA (B-030) + KV-quant (B-031) + paged-attention
// (B-032+) fast paths: an INT8 paged pool gives ~4× (vs FP32) / ~2× (vs
// FP16) smaller resident KV, read straight into attention.
//
//   q            : [A, H, D]                          RoPE already applied (FP)
//   k_pool/v_pool: [num_blocks, Hkv, block_size, D]   Int8
//   k_scale/v_scale: [num_blocks, Hkv, block_size]    Float32 (one per D-vec)
//   block_tables : [A, max_logical]  Int32
//   lens         : [A]               Int32
//   scale, group : as above
//   returns O    : [A, H, D]  (q's dtype)
//
// Numerically equals `paged_decode_attention` run on the dequantized pool
// (`dequantize_kv_per_token(k_pool, k_scale)`), bit-identical on CPU and
// within float tolerance on CUDA.
Tensor paged_decode_attention_int8(const Tensor& q, const Tensor& k_pool,
                                   const Tensor& k_scale, const Tensor& v_pool,
                                   const Tensor& v_scale,
                                   const Tensor& block_tables,
                                   const Tensor& lens, double scale,
                                   int64_t group);

// Wave 14 (B-032++++): fused paged PREFILL attention (S_new > 1).
//
// `paged_decode_attention` handles one new query token per request; chunked
// prefill (and batched prompt processing) feed S_new > 1 new tokens at once,
// which previously fell back to the per-sequence `attend_single` loop. This
// op collapses that into a SINGLE launch over the whole active set with a
// causal mask: request r's query token s sits at global position
// `kv_lens[r] - S + s`, so it attends to keys `[0, kv_lens[r] - S + s]`.
// K/V are read in place from the paged pool (no gather / no repeat_kv),
// online softmax, FP32 interior math — same numerical contract as
// `paged_decode_attention`.
//
//   q            : [A, S, H, D]                       RoPE already applied
//   k_pool/v_pool: [num_blocks, Hkv, block_size, D]   shared per-layer pool
//   block_tables : [A, max_logical]  Int32
//   kv_lens      : [A]               Int32  total ctx length (incl. the S new)
//   scale, group : as above (group == H/Hkv)
//   returns O    : [A, S, H, D]
//
// Requires `kv_lens[r] >= S` for every request. A request whose causal bound
// is empty yields a zero row (cannot happen for S >= 1 since position s ≥ 0).
Tensor paged_prefill_attention(const Tensor& q, const Tensor& k_pool,
                               const Tensor& v_pool, const Tensor& block_tables,
                               const Tensor& kv_lens, double scale,
                               int64_t group);

// INT8-direct prefill: K/V int8 + per-(block,head,slot) FP32 scale,
// dequantized inside the kernel. Numerically equals `paged_prefill_attention`
// on the dequantized pool.
Tensor paged_prefill_attention_int8(const Tensor& q, const Tensor& k_pool,
                                    const Tensor& k_scale, const Tensor& v_pool,
                                    const Tensor& v_scale,
                                    const Tensor& block_tables,
                                    const Tensor& kv_lens, double scale,
                                    int64_t group);

}  // namespace tesseract::nn

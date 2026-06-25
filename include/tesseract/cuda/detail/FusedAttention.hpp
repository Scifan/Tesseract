#pragma once

// Internal CUDA bridge for the fused SDPA / FlashAttention-2 kernel
// (Wave 4.2 / B-024).
//
// Scope & layering match the rest of the `detail/` bridges
// (`SwiGLU.hpp`, `RMSNorm.hpp`, `RotaryEmbedding.hpp`): the header
// stays C++17-parseable (no `std::span`, no CUDA types), and the
// entry point takes `const void*` / `void*` with an explicit `DType`
// selector so it can be included from plain `.cpp` translation
// units sitting under `tesseract_ops`.
//
// Algorithm (online softmax + incremental O accumulator, the FA2
// recipe restricted to the shapes `ops::attention` sees today):
//
//     scores[j] = <Q[q, :], K[j, :]> / sqrt(D)      (j ∈ [0, S_k))
//     if causal && j > q:  scores[j] = -inf
//     probs[q, j] = exp(scores[j] - max_j scores[j]) / sum exp(...)
//     O[q, :]    = sum_j probs[q, j] * V[j, :]
//
// Online update (per tile of size BLOCK_K):
//     m_new = max(m_old, max(tile_scores))
//     α     = exp(m_old - m_new)
//     s_j   = exp(tile_scores[j] - m_new)
//     l_new = l_old * α + sum_j s_j
//     O_new = O_old * α + sum_j s_j * V[j, :]
//
// Shape contract (enforced at the op-layer boundary):
//   * Q: [B, H, S_q, D]
//   * K: [B, H, S_k, D]
//   * V: [B, H, S_k, D]       (D_v == D required in this MVP)
//   * O: [B, H, S_q, D]       (allocated by the caller)
//   All four tensors are row-major contiguous with matching dtype.
//   MHA only (no GQA broadcast across H): the caller falls back to
//   the composite path when K/V's H differs from Q's H.
//
// Feature gates (the caller routes to the composite otherwise):
//   * No external mask tensor. Arbitrary additive masks defeat the
//     memory-bound premise of FA2 — the composite still covers
//     them via `matmul + add + softmax + matmul` on-device.
//   * `dropout_p == 0`. Stochastic dropout arrives with the RNG HAL
//     and is stitched in as a separate wave.
//   * `D <= 128`. The only head dims that ship in Llama-family
//     architectures; the kernel's shared-memory layout is sized
//     for D=128 and threads-per-block pick up slack at D=64.
//
// Dtype policy (matches B-015 / B-016 / RMSNorm / SwiGLU):
//   * Float32 / Float64 — accumulator is the storage type.
//   * Float16 / BFloat16 — load through FP32, run the entire
//     attention (scores, exp, rescale, weighted V sum, divide) in
//     FP32, narrow back on store. The `expf` never sees half
//     precision, so saturation tails behave identically across
//     storage widths.
//
// Forward-only. Backward flows through the autograd-composite path
// in `src/ops/cpu/Attention.cpp` (Q·Kᵀ → softmax → ·V), same
// convention as `launch_rms_norm` / `launch_swiglu_silu_gate`. The
// fused path only fires when `GradMode::is_enabled() == false` or
// none of Q/K/V `requires_grad`.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// Fused FA2 forward. One CUDA block per `(b, h, q)` triple; each
// block streams over `S_k` in tiles of `BLOCK_K = 64` keys, keeping
// `O[q, :]`, running softmax max `m`, and running softmax sum `l`
// in registers / shared memory across tiles.
//
// Returns after launching on `stream`; the caller is responsible for
// ordering any subsequent reads through the same stream (all of
// `ops::attention`'s downstream consumers already do). `q`, `k`, `v`,
// `o` must be on `device_index`; the op layer validates this.
//
// `scale_inv_sqrt_d` is passed in pre-computed rather than derived
// from `D` inside the kernel so the exact FP32 value matches the
// composite path's `ops::mul(Q, 1/sqrt(D))` scalar to the bit. The
// parity tests require this — cuBLASLt and our reference both fold
// the scale into the pre-softmax logits, and any drift in the
// scalar shows up as a systematic bias across every row.
// `H_kv` is the number of KV heads (GQA): query head `h` reads KV head
// `h / (H / H_kv)`. Pass `H_kv == H` (or 0) for standard MHA, in which
// case K/V must carry H heads. GQA mapping is currently honored on the
// `S_q == 1` decode split-K path only; for `S_q > 1` the caller must
// pass K/V already expanded to H heads (H_kv == H).
// `strides` (B-024c): optional pointer to 9 element-strides that let the
// prefill kernels read Q / K / V and write O in place from non-contiguous
// layouts — the `[B,H,S,D]` permuted views and `[B,Hkv,S,D]` KV-cache
// narrows that `MultiHeadAttention::forward_step` produces — instead of
// forcing a `contiguous()` copy per tensor per layer (the largest non-GEMM
// prefill cost; see backlog B-024c). Order:
//   { q_batch, q_head, q_seq,  k_batch, k_head, k_seq,  o_batch, o_head, o_seq }
// V shares K's layout. The innermost (head-dim) stride is always 1. Pass
// `nullptr` for the standard row-major contiguous layout, in which case the
// kernels derive the strides from the shape. Only honored on the prefill
// (`S_q > 1`) kernels; the decode split-K path keeps the contiguous contract.
void launch_fused_attention(DType dtype, int device_index,
                            int64_t B, int64_t H, int64_t H_kv,
                            int64_t S_q, int64_t S_k, int64_t D,
                            float scale_inv_sqrt_d, bool causal,
                            const void* q, const void* k, const void* v,
                            void* o, void* stream,
                            const int64_t* strides = nullptr);

}  // namespace tesseract::cuda::detail

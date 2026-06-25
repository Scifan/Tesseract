#pragma once

#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// Rotary position embedding (RoPE) with cached cos/sin tables.
//
// Constructs per-position-per-feature-pair rotation angles
//
//     θⱼ = base^(-2j/d_head)        for j ∈ [0, d_head/2)
//     cos[p, 2j]   = cos[p, 2j+1]   = cos(p · θⱼ)       p ∈ [0, max_seq)
//     sin[p, 2j]   = sin[p, 2j+1]   = sin(p · θⱼ)
//
// and stores the two `[max_seq, d_head]` tables as registered
// **buffers** (not parameters — no grad, no optimizer step). Moving
// the module across devices via `Module::to(Device)` moves the
// tables along with the Linear weights in the enclosing attention
// block, keeping the whole graph device-consistent.
//
// Forward contract: `x` is `[..., S, d_head]` (in practice
// `[B, H, S, D_head]` from the `MultiHeadAttention` split-heads).
// `S` must be ≤ `max_seq`; the forward slices the first `S` rows
// of the cached tables and dispatches `ops::rotary_embedding`.
//
// Convention: adjacent-pair rotation (GPT-NeoX / Llama-1 style),
// matching the public `ops::rotary_embedding` contract. Doubling
// the entries across the pair (`cos[p, 2j] == cos[p, 2j+1]`) lets
// the kernel stay a pure elementwise multiply-add per output slot.
class RotaryEmbedding : public Module {
 public:
  RotaryEmbedding(int64_t d_head,
                  double base = 10000.0,
                  int64_t max_seq = 4096,
                  DType dtype = DType::Float32);

  Tensor forward(const Tensor& x) override;

  // Decode-time variant: rotate `x` using positions
  // `[pos_offset, pos_offset + S)` from the cached tables. Used by
  // `MultiHeadAttention::forward_step` (Wave 2.1 KV cache) so the
  // RoPE angles for a newly-appended token match the full-sequence
  // prefill path. `pos_offset + S <= max_seq_` is enforced.
  Tensor forward_offset(const Tensor& x, int64_t pos_offset);

  int64_t d_head() const noexcept { return d_head_; }
  int64_t max_seq() const noexcept { return max_seq_; }
  double base() const noexcept { return base_; }

  const Tensor& cos_table() const { return cos_; }
  const Tensor& sin_table() const { return sin_; }

 private:
  int64_t d_head_;
  int64_t max_seq_;
  double base_;
  Tensor cos_;  // [max_seq_, d_head_]
  Tensor sin_;  // [max_seq_, d_head_]
};

}  // namespace tesseract::nn

#pragma once

// Diffusion Transformer (DiT) block — M4 Track A3 (B-040), DESIGN PLACEHOLDER.
//
// This header captures the *interface* for a DiT block; the runtime (forward,
// diffusion scheduler, VAE/UNet wiring) is deferred to M5 — see
// `docs/m4-plan.md` Track A3 and ADR-0006. The modality (non-autoregressive
// iterative denoising) is far enough from the current LLM stack that shipping a
// runtime now would be premature; the design is recorded so the eventual
// implementation slots into the existing `nn::Module` / `TransformerBlock`
// machinery without surprises.
//
// Design (adaLN-Zero, the Peebles & Xie 2023 recipe):
//
//   A DiT block is a standard pre-norm transformer block whose LayerNorms are
//   replaced by *adaptive* LayerNorm conditioned on a `[B, d_cond]` vector
//   `c = timestep_embedding(t) + class_embedding(y)`:
//
//     (shift1, scale1, gate1, shift2, scale2, gate2) = MLP(SiLU(c))   # 6·D
//     h = x + gate1 · attention(modulate(LN(x), shift1, scale1))
//     out = h + gate2 · mlp(modulate(LN(h), shift2, scale2))
//
//   where `modulate(z, shift, scale) = z · (1 + scale) + shift` (broadcast over
//   the sequence axis) and the gates are zero-initialized ("adaLN-Zero") so each
//   block starts as the identity — the property that makes DiT train stably.
//   Everything except the conditioning MLP + modulate is the existing
//   `TransformerBlock`; the delta is purely the norm/conditioning path.
//
// IR representation note (for Track C alignment): `modulate` is two fused
// affine ops over `[B, S, D]` with `[B, 1, D]` broadcast operands — expressible
// as the existing `mul`/`add` lowerings plus a `dit.modulate` marker, so a DiT
// block becomes round-trippable through the `tesseract` dialect on the same
// footing as a Llama block (B-044) once A3's runtime lands.
//
// The diffusion sampling loop (host-side scheduler over t = T..0 calling the
// model with timestep conditioning) is orthogonal to the autoregressive
// KV-cache loop and will live in a `models::DiT` driver in M5.

#include "tesseract/nn/Module.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

// Placeholder interface. Construction is allowed (so downstream design code can
// reference the shape contract), but `forward` throws: the runtime is M5 scope.
class DiTBlock : public Module {
 public:
  DiTBlock(int64_t d_model, int64_t num_heads, int64_t d_ff, int64_t d_cond)
      : d_model_(d_model), num_heads_(num_heads), d_ff_(d_ff),
        d_cond_(d_cond) {}

  // x: [B, S, d_model], cond: [B, d_cond]. Not implemented at A3.
  Tensor forward_cond(const Tensor& /*x*/, const Tensor& /*cond*/) {
    TESSERACT_THROW(
        "nn::DiTBlock is an M4 Track A3 design placeholder; the DiT runtime is "
        "deferred to M5 (see docs/m4-plan.md Track A3 / ADR-0006).");
  }

  int64_t d_model() const noexcept { return d_model_; }
  int64_t num_heads() const noexcept { return num_heads_; }
  int64_t d_ff() const noexcept { return d_ff_; }
  int64_t d_cond() const noexcept { return d_cond_; }

 private:
  int64_t d_model_;
  int64_t num_heads_;
  int64_t d_ff_;
  int64_t d_cond_;
};

}  // namespace tesseract::nn

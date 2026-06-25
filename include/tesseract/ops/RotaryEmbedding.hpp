#pragma once

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// Rotary position embedding (RoPE), as introduced by Su et al. 2021
// ("RoFormer") and used by every Llama-family model since Llama-1.
//
// Conceptually, RoPE treats each consecutive pair `(x[2j], x[2j+1])`
// of the last dim as a 2D vector and rotates it by an angle that is
// a function of both the feature index `j` and the position along
// the second-to-last dim. With `cos[p,2j] = cos[p,2j+1] = cos(p·θⱼ)`
// and likewise for sin, the forward pass is
//
//     out[..., p, 2j]   = x[..., p, 2j]   * cos[p,2j]
//                       - x[..., p, 2j+1] * sin[p,2j]
//     out[..., p, 2j+1] = x[..., p, 2j]   * sin[p,2j+1]
//                       + x[..., p, 2j+1] * cos[p,2j+1]
//
// Shapes:
//
//   * `x`      — `[..., S, D]`, rank ≥ 2. `D` must be even (we rotate
//                adjacent pairs so the last dim has to split cleanly).
//   * `cos`    — `[S_table, D]` with `S_table ≥ S`. Only the first
//                `S` rows are read; this lets a caller keep a single
//                max-length `[max_seq, D]` table and hand it in
//                unsliced for every forward, which is the path
//                `nn::RotaryEmbedding` exercises.
//   * `sin`    — same shape rule as `cos`.
//
// Dtype. `x`, `cos`, `sin` must share the same floating-point dtype
// (Float32 / Float64 / Float16 / BFloat16). Half-precision paths
// promote to FP32 internally and narrow on store, matching the
// convention established in B-015 / B-016.
//
// Autograd. The Jacobian of a 2D rotation is orthogonal, so the
// backward is the same kernel with `sin` negated — rotation by -θ.
// `cos` / `sin` are treated as constants (they come from a cached
// table in `nn::RotaryEmbedding`); no grad flows back to them.
//
// The op is a leaf on the CUDA backend via
// `cuda::detail::launch_rotary_embedding` and a plain loop on the
// CPU backend (same layering as softmax / rms_norm).
Tensor rotary_embedding(const Tensor& x,
                        const Tensor& cos,
                        const Tensor& sin);

}  // namespace tesseract::ops

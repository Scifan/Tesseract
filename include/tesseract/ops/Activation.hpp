#pragma once

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// Element-wise unary floating-point ops. Input must be Float32 or Float64.

Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor exp(const Tensor& x);
Tensor log(const Tensor& x);
// `sqrt(x)`. Added in M2K so that `ops::rms_norm` can stay a composite
// of already-CUDA-resident primitives without a dedicated normalization
// kernel. Backward: `dx = 0.5 * g / sqrt(x) = 0.5 * g / y`. Undefined
// for `x < 0` (returns NaN on IEEE-754 hardware); the op layer does
// **not** clamp, since our only caller pre-adds a positive `eps` so
// the argument is strictly positive.
Tensor sqrt(const Tensor& x);

// Wave 4.1 (B-025): fused SwiGLU activation used by Llama-style FFNs.
//
//     out[i] = silu(gate[i]) * up[i]
//            = (gate[i] * sigmoid(gate[i])) * up[i]
//
// `gate` and `up` must be defined tensors of identical shape, dtype,
// and device. Floating-point dtypes only (Float32 / Float64 / Float16 /
// BFloat16); integer tensors are rejected at the op-layer boundary.
//
// Forward fast path: when `gate` and `up` live on CUDA, are contiguous,
// and no autograd is active, this dispatches into the single-pass
// `launch_swiglu_silu_gate` kernel. That collapses the composite
// `sigmoid` + `mul` + `mul` chain (3 reads + 3 writes of the `[..., d_ff]`
// intermediate) into 2 reads + 1 write, giving a ≥2× bandwidth win
// on the memory-bound tail of every Llama FFN.
//
// Autograd / fallback path: when either operand requires gradients or
// the tensors are on the CPU / non-contiguous, the op decomposes into
// the equivalent `mul(gate, sigmoid(gate)) * up` composite. Gradients
// then flow through the already-wired `SigmoidBackward` / `MulBackward`
// nodes — the fused CUDA path is forward-only by design, matching the
// B-022 `rms_norm` convention.
//
// Returns a contiguous tensor of `gate.shape()` on the same device and
// dtype as the inputs. Emits a single `swiglu_silu_gate` marker into
// the active `GraphScope` (including on the composite fallback) so a
// future MLIR pattern-matcher sees the fusion atomically.
Tensor swiglu_silu_gate(const Tensor& gate, const Tensor& up);

// Fused relu backward: `grad_out * (x > 0)`. Used by the graph-level
// interpreter (it drives forward inference without an autograd tape,
// so it materializes the backward kernels through these explicit
// entry points). Does NOT record anything on the tape or the active
// GraphScope. Shapes and dtype must match; floating-point only.
Tensor relu_backward(const Tensor& x, const Tensor& grad_out);

}  // namespace tesseract::ops

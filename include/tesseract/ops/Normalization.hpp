#pragma once

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// Root-mean-square layer normalization (Zhang & Sennrich, 2019), the
// non-bias, non-centered variant used throughout the Llama family:
//
//     rms  = sqrt(mean(x² , dim=-1, keepdim=True) + eps)
//     y    = (x / rms) * weight                            // [..., D]
//
// M2K landed this as a *composite* over the already-CUDA-resident
// primitives (`mul`, `mean`, `add`, `sqrt`, `div`, `broadcast`). That
// keeps both forward and backward on-device on every backend (CPU,
// CUDA) without a dedicated fused kernel — autograd flows through the
// inner ops automatically. A single `rms_norm` marker op is emitted
// into the active GraphScope at the tail so a future MLIR pattern-
// matcher can collapse the primitive chain into a fused
// `tesseract.rms_norm` before codegen.
//
// Shapes:
//   x:      [..., D]
//   weight: [D]   (per-channel affine scale, typically initialized to 1)
//   eps:    numerical stabilizer added *before* the sqrt, always > 0.
//           Defaults to 1e-5 (Llama's convention).
//
// Returns: a contiguous tensor of shape `x.shape()` on the same device /
// dtype as `x`. `weight` must share the dtype and device of `x`.
Tensor rms_norm(const Tensor& x, const Tensor& weight, double eps = 1e-5);

// Standard LayerNorm (Ba et al., 2016), computed across the last `D` dims
// of `x` with per-channel learnable affine parameters:
//
//     mean  = mean(x, dim=-1, keepdim=True)
//     var   = mean((x - mean)², dim=-1, keepdim=True)    // biased (PyTorch default)
//     yhat  = (x - mean) / sqrt(var + eps)
//     y     = yhat * weight + bias                       // per-channel scale + shift
//
// Like `rms_norm`, this is implemented as a composite over already-CUDA-
// resident primitives so the whole path runs on-device on either backend
// and autograd threads through the inner ops' backward nodes.
//
// Shapes:
//   x:      [..., D]
//   weight: [D]              (scale; typically initialized to 1)
//   bias:   [D] or undefined (shift; typically initialized to 0). Passing
//                              an undefined tensor skips the bias term
//                              entirely — the LN convention used by
//                              RoBERTa-family models that omit bias.
//   eps:    numerical stabilizer, > 0; defaults to 1e-5.
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  double eps = 1e-5);

// Wave 2b (B-020) — standard BatchNorm used in CV models, computed per-channel
// across every non-channel dimension (N, and any spatial dims):
//
//     // training=true (default):
//     mu_B  = mean(x, dims=all_but_channel)
//     var_B = mean((x - mu_B)²,   dims=all_but_channel)   // biased (1/N_reduce)
//     y     = (x - mu_B) / sqrt(var_B + eps) * weight + bias
//
//     // running stats (updated in-place during training only):
//     running_mean ← (1-momentum) · running_mean + momentum · mu_B
//     running_var  ← (1-momentum) · running_var  + momentum · var_B_unbiased
//                       where var_B_unbiased = var_B · N / (N - 1)
//
//     // training=false:
//     y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
//
// Shapes:
//   x:             [N, C]         (BN1d-2D)
//                | [N, C, L]      (BN1d-3D)
//                | [N, C, H, W]   (BN2d)
//   weight:       [C] or undefined (defined → per-channel scale)
//   bias:         [C] or undefined (defined → per-channel shift)
//   running_mean: [C] (required buffer; mutated in place when training=true)
//   running_var:  [C] (required buffer; mutated in place when training=true)
//
// Biased / unbiased matches PyTorch verbatim: the forward path normalizes
// with biased variance (1/N) while the running-var buffer tracks the
// unbiased estimator (1/(N-1)). Implemented as a composite over existing
// primitives (mean, sub, mul, sqrt, div, add, view/reshape) so both the
// forward and the gradient path run on CPU and CUDA without any dedicated
// kernel — exactly the pattern RMSNorm / LayerNorm use.
//
// The running-stats writeback goes through `Storage::copy_device_bytes`
// under a `NoGradGuard`, so the buffers stay detached from the autograd
// graph and the mutation is observable to every Tensor handle that
// shares impl with the supplied `running_mean` / `running_var` — which
// is how `nn::BatchNorm{1d,2d}` gets the update for free via
// `register_buffer`.
Tensor batch_norm(const Tensor& x,
                  const Tensor& weight,         // [C] or undefined
                  const Tensor& bias,           // [C] or undefined
                  const Tensor& running_mean,   // [C]
                  const Tensor& running_var,    // [C]
                  bool training,
                  double momentum = 0.1,
                  double eps = 1e-5);

}  // namespace tesseract::ops

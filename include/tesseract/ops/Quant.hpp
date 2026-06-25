#pragma once

// Wave 3 (B-021): weight-only quantization inference ops.
//
// These ops are the runtime counterpart of the packers in
// `tesseract/quant/Pack.hpp`. They take a float activation and the
// packed integer weight + scales and produce a full-precision output
// in one pass — no materialized dequantized weight tensor ever lands
// in memory.
//
// Wave 3.1 ships `dequantize_matmul_int8` (per-output-channel
// symmetric INT8); Wave 3.2 will add `dequantize_matmul_int4_group`
// (per-group symmetric INT4, group size 128 default) using the same
// op-layer shape.

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// INT8 symmetric weight-only dequantize-matmul.
//
//   y[..., o] = sum_k x[..., k] * (scale[o] * q_w[o, k])
//             = scale[o] * sum_k x[..., k] * q_w[o, k]
//
// Input constraints:
//   * `x` is a floating-point tensor of rank >= 2, last dim `K`.
//     Dtype must be Float32 / Float16 / BFloat16 (same dtype policy
//     as `ops::rms_norm`).
//   * `q_w` is a rank-2 `[N, K]` Int8 tensor, contiguous. Produced
//     by `quant::pack_int8_symmetric`.
//   * `scale` is a rank-1 `[N]` Float32 tensor, contiguous. Same
//     producer. FP32 regardless of `x.dtype` — the scale is kept in
//     FP32 so dequantization error is purely from the quantized
//     weight, not the scale.
//
// Output: same dtype as `x`, shape `[..., N]` (trailing dim replaced
// by `N`). Rank >= 2 inputs are supported via leading-dim flatten: we
// treat the input as `[M, K]` with `M = prod(x.shape[:-1])`, run the
// dequant-matmul, and reshape back to `[..., N]`.
//
// Autograd: `q_w` and `scale` are frozen inference tensors and are
// *not* differentiated through. The gradient path for `x` is
// equivalent to `grad_x = matmul(grad_y, dequant(q_w) * scale)`,
// implemented via a small composite so we never materialize the
// full FP32 weight at forward time; the backward pass does allow one
// transient FP32 weight tensor because backward is already off the
// decode hot path.
//
// CUDA: dispatches to `cuda::detail::launch_dequant_matmul_int8`
// (one fused pass over `q_w`). CPU: a blocked FP32-accumulator
// reference, correct for any legal shape but not perf-tuned — the
// CPU path exists primarily so packers can be tested without a GPU.
Tensor dequantize_matmul_int8(const Tensor& x,
                              const Tensor& q_w,
                              const Tensor& scale);

// INT4 per-group symmetric weight-only dequantize-matmul (Wave 3.2).
//
//   group_idx(k) = k / group_size
//   q_w[o, k]    = sign_extend(low 4 bits of q_packed[o, k/2], nibble
//                              selected by (k & 1))               ∈ [-7, 7]
//   y[..., o]    = sum_k x[..., k] * q_w[o, k] * scale[o, group_idx(k)]
//
// Input constraints:
//   * `x` is a floating-point tensor of rank >= 2, last dim `K`.
//     Dtype must be Float32 / Float16 / BFloat16.
//   * `q_packed` is a rank-2 `[N, K / 2]` Int8 tensor, contiguous.
//     Produced by `quant::pack_int4_group`. Low nibble of each byte
//     is the even-k element, high nibble is the odd-k element; both
//     nibbles represent a signed `[-7, 7]` value (two's-complement in
//     the low four bits of the unsigned byte).
//   * `scale` is a rank-2 `[N, K / group_size]` Float32 tensor,
//     contiguous. FP32 regardless of `x.dtype`.
//   * `group_size` must be even, >= 2, and evenly divide `K`.
//
// Output: same dtype as `x`, shape `[..., N]`.
//
// Autograd mirrors the INT8 path exactly: `q_packed` and `scale` are
// frozen, `grad_x` is routed through a composite that dequantizes
// weights to FP once under `NoGradGuard` and reuses `ops::matmul` for
// the backward.
//
// CUDA: `cuda::detail::launch_dequant_matmul_int4_group`.
// CPU: reference implementation (nibble unpack + FP32 accumulator).
Tensor dequantize_matmul_int4_group(const Tensor& x,
                                    const Tensor& q_packed,
                                    const Tensor& scale,
                                    int64_t group_size);

}  // namespace tesseract::ops

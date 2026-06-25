#pragma once

// Wave 9 (B-031): device-resident KV-cache quantization (INT8).
//
// Unlike the weight packers in `Pack.hpp` — one-shot, host-side, run at
// model-load time — these two ops run on the *decode hot path*: every
// `forward_step` quantizes the freshly-projected K/V slab on append and
// dequantizes the cached prefix on read. So they must stay on whatever
// device the cache lives on (no host round trip per token). CPU and CUDA
// implementations are numerically identical (FP32 absmax + scale, banker's
// round, clamp to [-127, 127]).
//
// Granularity: **per-token, per-head** symmetric INT8. The last dim is the
// head-dim `D_head`; every `[D_head]` vector gets its own FP32 scale
// (`absmax / 127`). This is the accuracy sweet spot for KV quant — a single
// scalar per (b, h, token) is cheap to store (B·H·L floats) yet keeps the
// per-vector dynamic range tight, far better than a single tensor-wide
// scale and within rounding noise of per-channel for attention.
//
//   scale[row] = max_d |x[row, d]| / 127          (1.0 for an all-zero row)
//   q[row, d]  = round_nearest_even(x[row, d] / scale[row])  ∈ [-127, 127]
//   x'[row, d] = q[row, d] * scale[row]            (dequantize)
//
// where `row` ranges over all-but-the-last dim (flattened).

#include <utility>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Tensor.hpp"

namespace tesseract::quant {

// Quantize a floating tensor to per-row (per-last-dim-vector) symmetric
// INT8. `x` must be rank >= 2, contiguous, and Float32 / Float16 /
// BFloat16. Returns `(q, scale)` where:
//   * `q.dtype == Int8`, `q.shape == x.shape`, on `x.device()`.
//   * `scale.dtype == Float32`, `scale.shape == x.shape` minus the last
//     dim (rank-1 lower), on `x.device()`.
std::pair<Tensor, Tensor> quantize_kv_per_token(const Tensor& x);

// Inverse of `quantize_kv_per_token`. `q` is Int8 rank >= 2 contiguous;
// `scale` is Float32 with rank `q.rank() - 1` (one scale per row),
// contiguous; both on the same device. Returns a tensor of `out_dtype`
// (Float32 / Float16 / BFloat16), shape == `q.shape`, on `q.device()`,
// with `out[row, d] = q[row, d] * scale[row]`.
Tensor dequantize_kv_per_token(const Tensor& q, const Tensor& scale,
                               DType out_dtype);

}  // namespace tesseract::quant

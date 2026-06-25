#pragma once

// Wave 3 (B-021): weight-only quantization packers.
//
// The public entry points here take a floating-point weight tensor
// (FP32 / FP16 / BF16, always rank >= 1 with the last dim being the
// "in features" dim) and return (a) a packed integer replacement and
// (b) the per-row / per-group FP32 scales used to dequantize it at
// runtime. The returned tensors are ready to hand to
// `ops::dequantize_matmul_int8` (Wave 3.1) or
// `ops::dequantize_matmul_int4_group` (Wave 3.2, not yet shipped).
//
// Design choices:
//   * Symmetric quantization (no zero-point). All scales are stored
//     in FP32 regardless of the source dtype. The extra four bytes
//     per output channel are rounding-error-free and remove the one
//     source of FP16-accumulated drift we would otherwise have.
//   * "Weight-only" — activations are kept in their original dtype.
//     This is the quant flavor modern LLM inference engines (GPTQ,
//     AWQ, llama.cpp Q8/Q4, ...) are built around: the memory win
//     comes from compressing the weight, not the activation, and
//     the compute win comes from saturating more MAC/sec at the
//     same memory bandwidth.
//   * Computed on the CPU, on the source-device tensor. The packer
//     pulls weights back to host memory, quantizes, and ships the
//     result back to whichever device the caller wants. This costs
//     one extra `.to(cpu)` + `.to(device)` round trip per packed
//     layer — small relative to the quality win, and matches the
//     normal model-loading idiom of "pack once, serve many times."
//
// See `src/quant/Pack.cpp` for the numerical contract.

#include <utility>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Tensor.hpp"

namespace tesseract::quant {

// -----------------------------------------------------------------------------
// INT8 per-output-channel symmetric quantization.
//
//   q_w[o, i] = round(w[o, i] / scale[o])             ∈ [-127, 127]
//   scale[o]  = max_i |w[o, i]| / 127                 (FP32)
//
// The range `[-127, 127]` is asymmetric by one (no `-128`) so
// dequantization is a pure `q_w * scale` multiply with no zero-point
// bookkeeping — same convention as GPTQ / AWQ / llama.cpp Q8_0.
//
// Input constraints:
//   * `weight` must be rank >= 2. The last dim is treated as the
//     "in features" reduction dim; every other dim is flattened into
//     an "out channels" dim. For a rank-2 `[out, in]` Linear weight
//     this is the identity; for higher-rank tensors (e.g. Conv-style
//     filters) we quantize per-outer-slice.
//   * `weight.dtype` must be Float32 / Float16 / BFloat16. FP64 is
//     rejected — no realistic quantized-inference stack uses it.
//
// Returns `(q_w, scale)` where:
//   * `q_w.dtype == DType::Int8`, shape == `weight.shape()`,
//     device == `weight.device()` (or CPU if `out_device` overrides).
//   * `scale.dtype == DType::Float32`, shape == `[out]` (the flat
//     out-channel dim), device matches `q_w`.
//
// Rows that are identically zero get a scale of 1.0 (not 0) so the
// dequantized result is still exactly zero. Saturation to `[-127,
// 127]` is deterministic round-to-nearest-even; no random rounding.
std::pair<Tensor, Tensor> pack_int8_symmetric(const Tensor& weight);

// -----------------------------------------------------------------------------
// INT4 per-group symmetric quantization (Wave 3.2).
//
//   group_idx(k)    = k / group_size
//   scale[o, g]     = max_{k in group g} |w[o, k]| / 7          (FP32)
//   q_w_nibble[o,k] = round(w[o, k] / scale[o, group_idx(k)])   ∈ [-7, 7]
//
// Each group shares a single scale, so a group of 128 elements
// contributes 128*4 + 32 = 544 bits (1.063 bits per parameter of
// overhead on top of the 4-bit payload). That's the same trade-off
// GPTQ / AWQ / llama.cpp Q4_0 have converged on.
//
// Storage layout:
//   * `q_packed.dtype == DType::Int8`. Each byte packs **two** 4-bit
//     nibbles, even-k in the **low** nibble, odd-k in the **high**
//     nibble. Shape is `[out, in_features / 2]`. The `-7..+7` signed
//     nibble is stored as two's-complement (so `-7 -> 0x9`, `-1 ->
//     0xF`, `0 -> 0x0`, `7 -> 0x7`); call sites unpack via
//     sign-extending the low 4 bits.
//   * `scale.dtype == DType::Float32`, shape `[out, in_features /
//     group_size]`.
//
// Input constraints:
//   * `weight` must be rank >= 2; the last dim is "in features."
//   * `weight.dtype` must be Float32 / Float16 / BFloat16.
//   * `in_features % group_size == 0` and `group_size % 2 == 0`
//     (nibble pairs stay within a single group, so a group boundary
//     never cuts a byte in half).
//   * `group_size` must be >= 2 and <= in_features.
//
// Groups that are identically zero get a scale of 1.0 so the
// dequantized result is still exactly zero.
std::pair<Tensor, Tensor> pack_int4_group(const Tensor& weight,
                                          int64_t group_size = 128);

}  // namespace tesseract::quant

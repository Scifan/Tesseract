#pragma once

// Wave 3.1 (B-021): CUDA bridge for INT8 symmetric weight-only
// dequantize-matmul. Same layering convention as `detail/RMSNorm.hpp`
// and `detail/RotaryEmbedding.hpp` — this header stays C++17-parseable
// (no `std::span`, no CUDA types), and the entry point takes raw
// `const void*` / `void*` with an explicit `DType` selector so it can
// be included from plain `.cpp` op-layer TUs.
//
// Scope (Wave 3.1 — INT8 symmetric, per-output-channel scale):
//   * `x` is a row-major contiguous tensor whose last dim is the
//     in-feature dim `K`. All leading dims are collapsed into `M`
//     at the op-layer boundary.
//   * `q_w` is a row-major contiguous INT8 tensor `[N, K]` — the
//     quantized replacement for the classical FP weight
//     `[out_features, in_features]`. Values are integers in
//     `[-127, 127]`; the off-by-one asymmetry (no -128) keeps
//     dequantization a pure scale multiply without a zero-point.
//   * `scale` is a row-major contiguous FP32 tensor `[N]`.
//     FP32 regardless of `x.dtype` — at the accuracy bar we care
//     about for weight-only quant, keeping the scale in FP32 costs
//     four bytes per output channel and eliminates one source of
//     dequantization error.
//   * `y` is a row-major contiguous `[M, N]` tensor, same dtype as
//     `x`. Caller allocates it.
//
// Dtype policy (matches B-022 / B-015 / B-016 / RoPE):
//   * Float32 / Float16 / BFloat16 — the inner matmul-reduction always
//     accumulates in FP32. Half-precision storage is loaded and stored
//     through FP32 conversion; the `scale * acc` multiply also runs in
//     FP32 before narrowing to the storage dtype.
//   * Float64 — not supported in Wave 3.1. A future variant can add a
//     double-scale path if a downstream model wants quantized FP64
//     weights, but it's never a realistic target (weights are always
//     fp32/fp16/bf16 in the wild).
//
// Forward-only — Wave 3.1 ships inference quantization. Training-time
// quant-aware paths (QAT) and INT4-group quant share this kernel's
// storage layout in spirit (Wave 3.2 will add a second `_int4_group`
// entry point); backward is not wired here because the rule for weight-
// only quantization during inference is "don't differentiate through
// the frozen integer weights." Upstream autograd on `x` flows through
// the op-layer wrapper's `DequantMatMulInt8Backward` node, implemented
// purely via existing primitives.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// Launch config (documented in `src/cuda/DequantMatMul.cu`):
//   grid  = (N, M, 1)           — one block per output pair
//   block = (kBlockSize, 1, 1)  — 256 threads, block-reduce over K
//
// Each block streams one row of `x` and one row of `q_w`, accumulates
// their FP32 dot product, and writes `y[m, n] = (acc * scale[n])` in
// the storage dtype. Read of `scale[n]` happens on thread 0 post-
// reduction so it's a single broadcast load per block.
//
// `x` / `q_w` / `scale` / `y` must all be contiguous on the target
// device; callers enforce that at the op-layer boundary. The kernel
// never dereferences out-of-bound addresses — the `m >= M || n >= N`
// short-circuit at the top of the kernel is defensive in case the
// launcher is ever called with a trimmed grid.
void launch_dequant_matmul_int8(DType dtype, int device_index,
                                int64_t M, int64_t N, int64_t K,
                                const void* x,
                                const void* q_w,
                                const void* scale,
                                void* y,
                                void* stream);

// -----------------------------------------------------------------------------
// Wave 3.2: INT4 per-group symmetric weight-only dequantize-matmul.
//
// Same overall grid/block layout as the INT8 launcher — one block per
// `(m, n)` output pair, block-reduction over `K` — but the weight is
// packed as two 4-bit signed nibbles per byte. Convention:
//   * `q_packed` is a row-major contiguous INT8 tensor of shape
//     `[N, K / 2]`. Every byte encodes two `[-7, 7]` signed values;
//     the low nibble is the even-k slot, the high nibble is the odd-k
//     slot. Signed nibbles are stored two's-complement (`-7 -> 0x9`,
//     `-1 -> 0xF`, ..., `7 -> 0x7`); the kernel recovers the value by
//     sign-extending the low four bits.
//   * `scale` is a row-major contiguous FP32 tensor of shape
//     `[N, K / group_size]`. Every k index in group
//     `g = k / group_size` shares `scale[n, g]`.
//   * `group_size` is a compile-time friendly runtime value. The
//     launcher asserts `group_size % 2 == 0` and `K % group_size == 0`
//     so a byte never straddles a group boundary.
//
// Dtype policy / accuracy rules match the INT8 launcher exactly —
// FP32 accumulator, FP32 scale, FP16/BF16 storage goes through
// FP32 promotion on load and narrowing on store.
//
// Forward-only. The op-layer wrapper (ops::dequantize_matmul_int4_group)
// routes `x.requires_grad()` activations through a composite
// `matmul(dequantize_to_fp(...))` fallback so autograd on `x`
// still works. Weights are frozen.
void launch_dequant_matmul_int4_group(DType dtype, int device_index,
                                      int64_t M, int64_t N, int64_t K,
                                      int64_t group_size,
                                      const void* x,
                                      const void* q_packed,
                                      const void* scale,
                                      void* y,
                                      void* stream);

}  // namespace tesseract::cuda::detail

#pragma once

#include <cstdint>

// M4 Phase 6 — batch-1 (decode) W8A8 GEMV for CPU inference, the kernel
// that closes the gap to llama.cpp's hand-tuned ggml path.
//
// Decode is a batch-1 matrix-vector product `y[N] = W[N,K] @ x[K]`. At
// real model sizes it is *memory-bandwidth bound* on reading `W`: storing
// `W` as INT8 (1 byte/weight) instead of FP32 (4 bytes) cuts the traffic
// 4x, and AVX-512-VNNI `vpdpbusd` does 64 int8 multiply-accumulates per
// instruction per 512-bit lane. Quantizing the activation to INT8 too
// (W8A8, exactly like ggml's Q8 path) lets the whole reduction stay in
// the VNNI integer pipeline.
//
// Numerics (symmetric, zero-point-free):
//   x is quantized per-vector:  xq[k] = round(x[k] / xscale),
//                               xscale = max_k|x[k]| / 127.
//   W is quantized per-row:     W[n,k] = round(w[n,k] / wscale[n]).
//   y[n] = xscale * wscale[n] * sum_k xq[k] * W[n,k].
//
// VNNI requires an *unsigned* activation operand, so we map the signed
// xq into u8 via `xor 0x80` (== +128) and correct with the precomputed
// per-row weight sum:  sum_k (xq[k]+128) * W[n,k]
//                      = sum_k xq[k]*W[n,k] + 128 * wrowsum[n].
namespace tesseract::ops::detail {

// `true` iff the runtime CPU has AVX-512 F+BW+VNNI. On machines without
// it the GEMV falls back to a portable scalar reduction (still INT8, so
// still bandwidth-optimal — just without the VNNI throughput).
bool gemv_vnni_supported() noexcept;

// Dynamic per-vector symmetric INT8 quantization of `x[K]` into `xq[K]`.
// Returns the scale (`max|x| / 127`, or 0 if x is all-zero). `xq` must
// have room for `K` bytes.
float quantize_row_int8(const float* x, std::int64_t K,
                        std::int8_t* xq) noexcept;

// Precompute the per-row weight sum `wrowsum[n] = sum_k W[n,k]` needed by
// the VNNI u8 offset correction. `W` is row-major `[N, K]` INT8.
void compute_row_sums(const std::int8_t* W, std::int64_t N, std::int64_t K,
                      std::int32_t* wrowsum) noexcept;

// W8A8 GEMV: `y[n] = xscale * wscale[n] * sum_k xq[k] * W[n,k]`.
//   * `W`        row-major `[N, K]` INT8, contiguous.
//   * `wscale`   `[N]` FP32 per-row dequant scale.
//   * `wrowsum`  `[N]` INT32 from `compute_row_sums`.
//   * `xq`       `[K]` INT8 activation from `quantize_row_int8`.
//   * `xscale`   activation scale from `quantize_row_int8`.
//   * `y`        `[N]` FP32 output.
// Parallelizes over `N` (capped OpenMP team, like the FP32 GEMM kernel).
void gemv_w8a8(const std::int8_t* W, const float* wscale,
               const std::int32_t* wrowsum, const std::int8_t* xq,
               float xscale, float* y, std::int64_t N,
               std::int64_t K) noexcept;

// Single-thread worksharing variant: computes only output rows `[n0, n1)`,
// no OpenMP inside. Call from within one caller-owned parallel region so a
// whole fused decode block forks the thread team *once* (per token) instead
// of once per GEMV — this is what lets the W8A8 path scale to many cores
// without paying OpenMP fork/barrier overhead on every small matvec.
void gemv_w8a8_range(const std::int8_t* W, const float* wscale,
                     const std::int32_t* wrowsum, const std::int8_t* xq,
                     float xscale, float* y, std::int64_t n0, std::int64_t n1,
                     std::int64_t K) noexcept;

}  // namespace tesseract::ops::detail

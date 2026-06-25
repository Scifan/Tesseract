#pragma once

// M4 perf-closeout Phase 3 — FP8 (E4M3) tensor-core GEMM bridge.
//
// Ada (SM 8.9) and Hopper expose FP8 tensor cores delivering ~2× the
// FP16 tensor-core throughput. cuBLASLt's FP8 path has a hard "TN"
// constraint: the A operand must be transposed (op=T) and B
// non-transposed (op=N), both with the K dimension contiguous. The
// bridge maps a standard Linear `Y[M,N] = X[M,K] · W[N,K]^T` onto that
// constraint so:
//   * X  — activations, row-major [M,K]  (E4M3)
//   * W  — weights,     row-major [N,K]  (E4M3, the nn.Linear layout)
//   * Y  — output,      row-major [M,N]  (BF16)
// per-tensor FP32 scales `x_scale` / `w_scale` are folded in by cuBLASLt
// (effective product `(x_scale·X)·(w_scale·W)^T`). Header stays
// CUDA-type-free per the `detail/` layering convention.

#include <cstdint>

namespace tesseract::cuda::detail {

// Returns true iff the device supports FP8 tensor-core matmul (SM >= 89).
bool fp8_gemm_supported(int device_index);

// Convert a contiguous FP32 device buffer to E4M3 (1 byte/elem). `dst`
// must hold `n` bytes. Runs on `stream`.
void quantize_to_fp8_e4m3(const float* src, void* dst, int64_t n,
                          void* stream);

// Y[M,N] (BF16) = (x_scale·X[M,K]) · (w_scale·W[N,K])^T  via FP8 TC.
// X, W are E4M3 device buffers (row-major as described above); Y is a
// BF16 device buffer [M,N] row-major. `x_scale`/`w_scale` are host
// FP32 per-tensor scales. M, N, K should be multiples of 16.
void launch_fp8_linear(int device_index, int64_t M, int64_t N, int64_t K,
                       const void* X, const void* W, void* Y,
                       float x_scale, float w_scale, void* stream);

}  // namespace tesseract::cuda::detail

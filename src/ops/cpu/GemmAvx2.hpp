#pragma once

#include <cstdint>

// Hand-written 4x8 FP32 GEMM microkernel using AVX2 + FMA. Gated by runtime
// CPU-feature detection: the `_supported()` helper returns `false` on
// machines without AVX2 or on non-x86 builds, and `gemm_avx2_f32` returns
// `false` in that case so the caller can fall back to the scalar path.
//
// Preconditions when `gemm_avx2_f32` returns `true`:
//   * `A`, `B`, `C` are distinct (no aliasing). `A` is `[M, K]` row-major
//     (stride K), `B` is `[K, N]` row-major (stride N), `C` is `[M, N]`
//     row-major (stride N).
//   * `C` need not be pre-zeroed — every output element is fully
//     overwritten by the kernel (either by the microkernel or the scalar
//     edge handler).
//   * All three pointers are aligned on at least `sizeof(float)`; the
//     kernel issues unaligned AVX loads/stores, so higher alignment is
//     not required.
namespace tesseract::ops::detail {

// `true` iff the current runtime CPU supports the instruction set this
// microkernel requires (AVX2 + FMA). Result is cheap to compute but is
// typically called once and cached by the dispatcher.
bool gemm_avx2_f32_supported() noexcept;

// Compute C = A * B with the AVX2 4x8 microkernel. Returns `false` on
// unsupported CPUs or builds; the caller must use a fallback path in
// that case. `M`, `N`, `K` may take any non-negative values — the
// kernel handles non-multiples of the tile size via a scalar edge
// handler.
bool gemm_avx2_f32(const float* A, const float* B, float* C,
                   std::int64_t M, std::int64_t N, std::int64_t K);

// Adaptive OpenMP thread count for a GEMM of total multiply-add `work`
// (= M*N*K). Returns 1 below the parallel threshold. Above it, scales
// threads with work but **caps** them: large thread teams on many-CCD
// boxes (e.g. EPYC) pay a multi-millisecond barrier/fork cost per
// parallel region that dwarfs the compute and produces a catastrophic
// throughput cliff (measured: 96 threads → 0.65 GFLOP/s at 128^3 vs 32
// threads → 650 GFLOP/s at 512^3). The cap defaults to
// min(omp_max_threads, 32) and is overridable via the
// `TESSERACT_GEMM_MAX_THREADS` environment variable.
int gemm_num_threads(std::int64_t work) noexcept;

}  // namespace tesseract::ops::detail

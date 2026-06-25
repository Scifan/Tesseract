#include "GemmAvx2.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Compile this TU unconditionally. On non-x86 targets the AVX paths are
// `#ifdef`-gated out and the public API returns `false`, letting the
// caller fall back to the scalar `gemm_naive` without any linker tricks.
#if defined(__x86_64__) || defined(_M_X64)
  #define TESSERACT_AVX2_KERNEL_AVAILABLE 1
  #include <immintrin.h>
#else
  #define TESSERACT_AVX2_KERNEL_AVAILABLE 0
#endif

#if defined(TESSERACT_HAS_OPENMP)
  #include <omp.h>
#endif

namespace tesseract::ops::detail {

int gemm_num_threads(std::int64_t work) noexcept {
#if defined(TESSERACT_HAS_OPENMP)
  // Below the parallel threshold the fork/join overhead dominates.
  constexpr std::int64_t kParallelThreshold = std::int64_t{64} * 64 * 128;
  if (work < kParallelThreshold) return 1;

  // Resolve the cap once: min(omp_max, 32) unless overridden by env.
  static const int cap = [] {
    int c = 32;
#if defined(TESSERACT_HAS_OPENMP)
    const int omp_max = omp_get_max_threads();
    if (omp_max > 0) c = std::min(c, omp_max);
#endif
    if (const char* e = std::getenv("TESSERACT_GEMM_MAX_THREADS")) {
      const int v = std::atoi(e);
      if (v > 0) c = v;
    }
    return c < 1 ? 1 : c;
  }();

  // One thread per ~1M multiply-adds, clamped to [1, cap].
  constexpr std::int64_t kWorkPerThread = std::int64_t{1} << 20;
  std::int64_t want = work / kWorkPerThread;
  if (want < 1) want = 1;
  if (want > cap) want = cap;
  return static_cast<int>(want);
#else
  (void)work;
  return 1;
#endif
}

#if TESSERACT_AVX2_KERNEL_AVAILABLE

// Use per-function target attributes so the rest of the library can be
// compiled with a conservative baseline (e.g. sse4.2) and only this TU's
// microkernels emit AVX2 + FMA instructions. Runtime dispatch guarantees
// we never enter one of these functions on a CPU that lacks the ISA.
#if defined(__GNUC__) || defined(__clang__)
  #define TESSERACT_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
  #define TESSERACT_AVX2_TARGET
#endif

// 4x8 FP32 FMA microkernel. Rank of the register file is:
//   4 ymm accumulators, one per output row (8 floats each).
// Each K-step loads B[k, 0..8) once and broadcasts four A[i+r, k] values
// across the lanes, giving 4 FMAs = 32 flops per K step. For a
// row-major B with row length `ldb`, this is stride-1 access on B and
// stride-K on A — both stay in L1 for the 512^3 target.
TESSERACT_AVX2_TARGET
static inline void mker_4x8(const float* a, const float* b, float* c,
                            std::int64_t K, std::int64_t lda,
                            std::int64_t ldb, std::int64_t ldc) {
  __m256 c0 = _mm256_setzero_ps();
  __m256 c1 = _mm256_setzero_ps();
  __m256 c2 = _mm256_setzero_ps();
  __m256 c3 = _mm256_setzero_ps();
  const float* a0 = a + 0 * lda;
  const float* a1 = a + 1 * lda;
  const float* a2 = a + 2 * lda;
  const float* a3 = a + 3 * lda;
  for (std::int64_t k = 0; k < K; ++k) {
    const __m256 bv = _mm256_loadu_ps(b + k * ldb);
    const __m256 av0 = _mm256_broadcast_ss(a0 + k);
    const __m256 av1 = _mm256_broadcast_ss(a1 + k);
    const __m256 av2 = _mm256_broadcast_ss(a2 + k);
    const __m256 av3 = _mm256_broadcast_ss(a3 + k);
    c0 = _mm256_fmadd_ps(av0, bv, c0);
    c1 = _mm256_fmadd_ps(av1, bv, c1);
    c2 = _mm256_fmadd_ps(av2, bv, c2);
    c3 = _mm256_fmadd_ps(av3, bv, c3);
  }
  _mm256_storeu_ps(c + 0 * ldc, c0);
  _mm256_storeu_ps(c + 1 * ldc, c1);
  _mm256_storeu_ps(c + 2 * ldc, c2);
  _mm256_storeu_ps(c + 3 * ldc, c3);
}

// Scalar edge tile for output rows / cols that don't fit the 4x8 tile.
// Writes (overwrites) `c[i*ldc + j]` for every `(i, j)` in the given
// ranges. Small enough that auto-vectorization + OpenMP outer
// parallelism handle it adequately — the fast path keeps the AVX
// kernel hot.
static inline void scalar_edge(const float* A, const float* B, float* C,
                               std::int64_t row_start, std::int64_t row_end,
                               std::int64_t col_start, std::int64_t col_end,
                               std::int64_t K, std::int64_t N) {
  for (std::int64_t i = row_start; i < row_end; ++i) {
    for (std::int64_t j = col_start; j < col_end; ++j) {
      float acc = 0.0f;
      const float* ar = A + i * K;
      const float* bc = B + j;
      for (std::int64_t k = 0; k < K; ++k) acc += ar[k] * bc[k * N];
      C[i * N + j] = acc;
    }
  }
}

TESSERACT_AVX2_TARGET
static void gemm_avx2_f32_tiled(const float* A, const float* B, float* C,
                                std::int64_t M, std::int64_t N, std::int64_t K) {
  const std::int64_t M4 = (M / 4) * 4;
  const std::int64_t N8 = (N / 8) * 8;
  // Parallelize over the outer i-block: each 4-row band writes a
  // disjoint slice of C so no locking is needed. Small problems skip
  // the parallel region so OMP fork/join doesn't dominate.
  const std::int64_t work = M4 * N8 * K;
  const int nthreads = gemm_num_threads(work);
  (void)nthreads;
#if defined(TESSERACT_HAS_OPENMP)
  #pragma omp parallel for schedule(static) if(nthreads > 1) num_threads(nthreads)
#endif
  for (std::int64_t i = 0; i < M4; i += 4) {
    for (std::int64_t j = 0; j < N8; j += 8) {
      mker_4x8(A + i * K, B + j, C + i * N + j, K, K, N, N);
    }
  }
  // Right edge: rows [0..M4), cols [N8..N).
  if (N8 < N) {
    scalar_edge(A, B, C, 0, M4, N8, N, K, N);
  }
  // Bottom edge: rows [M4..M), full column range.
  if (M4 < M) {
    scalar_edge(A, B, C, M4, M, 0, N, K, N);
  }
}

bool gemm_avx2_f32_supported() noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

bool gemm_avx2_f32(const float* A, const float* B, float* C,
                   std::int64_t M, std::int64_t N, std::int64_t K) {
  if (M <= 0 || N <= 0 || K <= 0) {
    // Match the naive kernel's behaviour on degenerate shapes: a tile
    // with a zero dim produces no output — nothing to write.
    return true;
  }
  gemm_avx2_f32_tiled(A, B, C, M, N, K);
  return true;
}

#else  // !TESSERACT_AVX2_KERNEL_AVAILABLE

bool gemm_avx2_f32_supported() noexcept { return false; }
bool gemm_avx2_f32(const float*, const float*, float*,
                   std::int64_t, std::int64_t, std::int64_t) {
  return false;
}

#endif

}  // namespace tesseract::ops::detail

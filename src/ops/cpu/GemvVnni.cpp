#include "GemvVnni.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "GemmAvx2.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #include <immintrin.h>
  #define TESSERACT_X86 1
#else
  #define TESSERACT_X86 0
#endif

#if defined(TESSERACT_HAS_OPENMP)
  #include <omp.h>
#endif

namespace tesseract::ops::detail {

namespace {

#if TESSERACT_X86
// Horizontal sum of a 512-bit int32 vector.
__attribute__((target("avx512f"))) inline std::int32_t hsum_i32x16(
    __m512i v) noexcept {
  return _mm512_reduce_add_epi32(v);
}
#endif

// --- scalar reference (portable fallback + non-x86), row range [n0, n1) ----
void gemv_w8a8_scalar_range(const std::int8_t* W, const float* wscale,
                            const std::int8_t* xq, float xscale, float* y,
                            std::int64_t n0, std::int64_t n1,
                            std::int64_t K) noexcept {
  for (std::int64_t n = n0; n < n1; ++n) {
    const std::int8_t* w = W + n * K;
    std::int32_t acc = 0;
    for (std::int64_t k = 0; k < K; ++k)
      acc += static_cast<std::int32_t>(xq[k]) * static_cast<std::int32_t>(w[k]);
    y[n] = xscale * wscale[n] * static_cast<float>(acc);
  }
}

#if TESSERACT_X86
// --- AVX-512-VNNI W8A8 GEMV, row range [n0, n1) ---------------------------
// Activation is pre-offset to u8 (xor 0x80); we subtract 128*wrowsum[n] to
// recover the true signed dot product. Four independent accumulators hide
// the vpdpbusd latency. No OpenMP inside — the caller owns parallelism, so
// this is safe to call from within a single parallel region (one fork per
// decode token instead of one per GEMV).
__attribute__((target("avx512f,avx512bw,avx512vnni"))) void gemv_w8a8_vnni_range(
    const std::int8_t* W, const float* wscale, const std::int32_t* wrowsum,
    const std::int8_t* xq, float xscale, float* y, std::int64_t n0,
    std::int64_t n1, std::int64_t K) noexcept {
  const __m512i sign_flip = _mm512_set1_epi8(static_cast<char>(0x80));

  for (std::int64_t n = n0; n < n1; ++n) {
    const std::int8_t* w = W + n * K;
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();

    std::int64_t k = 0;
    const std::int64_t k256 = K & ~static_cast<std::int64_t>(255);
    for (; k < k256; k += 256) {
      // Load activations, map signed->unsigned via xor 0x80.
      __m512i a0 = _mm512_xor_si512(
          _mm512_loadu_si512(reinterpret_cast<const void*>(xq + k + 0)), sign_flip);
      __m512i a1 = _mm512_xor_si512(
          _mm512_loadu_si512(reinterpret_cast<const void*>(xq + k + 64)), sign_flip);
      __m512i a2 = _mm512_xor_si512(
          _mm512_loadu_si512(reinterpret_cast<const void*>(xq + k + 128)), sign_flip);
      __m512i a3 = _mm512_xor_si512(
          _mm512_loadu_si512(reinterpret_cast<const void*>(xq + k + 192)), sign_flip);
      __m512i b0 = _mm512_loadu_si512(reinterpret_cast<const void*>(w + k + 0));
      __m512i b1 = _mm512_loadu_si512(reinterpret_cast<const void*>(w + k + 64));
      __m512i b2 = _mm512_loadu_si512(reinterpret_cast<const void*>(w + k + 128));
      __m512i b3 = _mm512_loadu_si512(reinterpret_cast<const void*>(w + k + 192));
      acc0 = _mm512_dpbusd_epi32(acc0, a0, b0);
      acc1 = _mm512_dpbusd_epi32(acc1, a1, b1);
      acc2 = _mm512_dpbusd_epi32(acc2, a2, b2);
      acc3 = _mm512_dpbusd_epi32(acc3, a3, b3);
    }
    for (; k + 64 <= K; k += 64) {
      __m512i a0 = _mm512_xor_si512(
          _mm512_loadu_si512(reinterpret_cast<const void*>(xq + k)), sign_flip);
      __m512i b0 = _mm512_loadu_si512(reinterpret_cast<const void*>(w + k));
      acc0 = _mm512_dpbusd_epi32(acc0, a0, b0);
    }
    acc0 = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1),
                            _mm512_add_epi32(acc2, acc3));
    std::int32_t dp = hsum_i32x16(acc0);
    // Scalar tail (K not a multiple of 64). The activation here is the raw
    // signed xq (no +128 offset), so it contributes to the true dot product
    // directly and must NOT be corrected below.
    std::int32_t tail = 0;
    for (; k < K; ++k)
      tail += static_cast<std::int32_t>(xq[k]) * static_cast<std::int32_t>(w[k]);
    // Undo the +128 offset only over the VNNI-processed span [0, k_vnni).
    // wrowsum is the full-row sum; subtract the tail's weight contribution
    // so the 128-correction covers exactly the offset elements.
    std::int32_t tail_wsum = 0;
    for (std::int64_t kk = (K & ~static_cast<std::int64_t>(63)); kk < K; ++kk)
      tail_wsum += static_cast<std::int32_t>(w[kk]);
    const std::int32_t real = dp - 128 * (wrowsum[n] - tail_wsum) + tail;
    y[n] = xscale * wscale[n] * static_cast<float>(real);
  }
}
#endif  // TESSERACT_X86

}  // namespace

bool gemv_vnni_supported() noexcept {
#if TESSERACT_X86
  return __builtin_cpu_supports("avx512f") &&
         __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vnni");
#else
  return false;
#endif
}

float quantize_row_int8(const float* x, std::int64_t K,
                        std::int8_t* xq) noexcept {
  float amax = 0.0f;
  for (std::int64_t k = 0; k < K; ++k) amax = std::max(amax, std::fabs(x[k]));
  if (amax == 0.0f) {
    for (std::int64_t k = 0; k < K; ++k) xq[k] = 0;
    return 0.0f;
  }
  const float scale = amax / 127.0f;
  const float inv = 127.0f / amax;
  for (std::int64_t k = 0; k < K; ++k) {
    const float r = std::nearbyint(x[k] * inv);
    const float c = std::min(127.0f, std::max(-127.0f, r));
    xq[k] = static_cast<std::int8_t>(c);
  }
  return scale;
}

void compute_row_sums(const std::int8_t* W, std::int64_t N, std::int64_t K,
                      std::int32_t* wrowsum) noexcept {
#if defined(TESSERACT_HAS_OPENMP)
  const int nthreads = gemm_num_threads(N * K);
  #pragma omp parallel for schedule(static) num_threads(nthreads) if(nthreads > 1)
#endif
  for (std::int64_t n = 0; n < N; ++n) {
    const std::int8_t* w = W + n * K;
    std::int32_t s = 0;
    for (std::int64_t k = 0; k < K; ++k) s += static_cast<std::int32_t>(w[k]);
    wrowsum[n] = s;
  }
}

void gemv_w8a8_range(const std::int8_t* W, const float* wscale,
                     const std::int32_t* wrowsum, const std::int8_t* xq,
                     float xscale, float* y, std::int64_t n0, std::int64_t n1,
                     std::int64_t K) noexcept {
#if TESSERACT_X86
  if (gemv_vnni_supported()) {
    gemv_w8a8_vnni_range(W, wscale, wrowsum, xq, xscale, y, n0, n1, K);
    return;
  }
#endif
  gemv_w8a8_scalar_range(W, wscale, xq, xscale, y, n0, n1, K);
}

void gemv_w8a8(const std::int8_t* W, const float* wscale,
               const std::int32_t* wrowsum, const std::int8_t* xq,
               float xscale, float* y, std::int64_t N,
               std::int64_t K) noexcept {
#if defined(TESSERACT_HAS_OPENMP)
  const int nthreads = gemm_num_threads(N * K);
  if (nthreads > 1) {
    #pragma omp parallel num_threads(nthreads)
    {
      const int tid = omp_get_thread_num();
      const int nt = omp_get_num_threads();
      const std::int64_t chunk = (N + nt - 1) / nt;
      const std::int64_t n0 = std::min(N, static_cast<std::int64_t>(tid) * chunk);
      const std::int64_t n1 = std::min(N, n0 + chunk);
      gemv_w8a8_range(W, wscale, wrowsum, xq, xscale, y, n0, n1, K);
    }
    return;
  }
#endif
  gemv_w8a8_range(W, wscale, wrowsum, xq, xscale, y, 0, N, K);
}

}  // namespace tesseract::ops::detail

// Micro-benchmark: tesseract::ops::matmul vs a naive triple-loop baseline.
// Prints GFLOP/s for a handful of sizes so we can regress-check the CPU kernel
// during M0 -> M1 work. Run as: ./benchmarks/bench_matmul

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/MatMul.hpp"

#if defined(TESSERACT_HAS_EIGEN)
#define EIGEN_DONT_ALIGN_STATICALLY 1
#include <Eigen/Core>
#endif

// The benchmark can poke the 4x8 AVX2 microkernel directly even when the
// library itself chose a different dispatch tier — useful to measure the
// kernel's raw throughput against Eigen on the same host.
namespace tesseract::ops::detail {
bool gemm_avx2_f32_supported() noexcept;
bool gemm_avx2_f32(const float* A, const float* B, float* C,
                   std::int64_t M, std::int64_t N, std::int64_t K);
}

namespace {

using Clock = std::chrono::steady_clock;

void fill_random(tesseract::Tensor& t, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  float* p = t.data_ptr<float>();
  const int64_t n = t.numel();
  for (int64_t i = 0; i < n; ++i) p[i] = dist(rng);
}

void naive_gemm(const float* A, const float* B, float* C,
                int64_t M, int64_t N, int64_t K) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
}

double elapsed_sec(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

struct Case {
  int64_t M, N, K;
  int iters;
};

void run_case(const Case& c) {
  using namespace tesseract;
  auto A = Tensor::empty({c.M, c.K}, DType::Float32);
  auto B = Tensor::empty({c.K, c.N}, DType::Float32);
  fill_random(A, 1);
  fill_random(B, 2);

  // Tesseract ops path.
  {
    auto t0 = Clock::now();
    for (int i = 0; i < c.iters; ++i) {
      auto C = ops::matmul(A, B);
      (void)C;
    }
    const double secs = elapsed_sec(t0, Clock::now()) / c.iters;
    const double flops = 2.0 * c.M * c.N * c.K;
    std::printf("[tesseract]  M=%ld N=%ld K=%ld  %8.3f ms  %7.2f GFLOP/s\n",
                long(c.M), long(c.N), long(c.K), secs * 1000.0, flops / secs / 1e9);
  }

  // Naive baseline.
  {
    std::vector<float> Cbuf(static_cast<std::size_t>(c.M * c.N));
    auto t0 = Clock::now();
    for (int i = 0; i < c.iters; ++i) {
      naive_gemm(A.data_ptr<float>(), B.data_ptr<float>(), Cbuf.data(), c.M, c.N, c.K);
    }
    const double secs = elapsed_sec(t0, Clock::now()) / c.iters;
    const double flops = 2.0 * c.M * c.N * c.K;
    std::printf("[naive    ]  M=%ld N=%ld K=%ld  %8.3f ms  %7.2f GFLOP/s\n",
                long(c.M), long(c.N), long(c.K), secs * 1000.0, flops / secs / 1e9);
  }

  // Direct AVX2 microkernel. Bypasses the whole `ops::matmul` dispatch so
  // the row reflects the kernel itself (no autograd record, no Tensor
  // allocation overhead). Only runs on hosts where the CPU probe says
  // AVX2+FMA is available.
  if (tesseract::ops::detail::gemm_avx2_f32_supported()) {
    std::vector<float> Cbuf(static_cast<std::size_t>(c.M * c.N));
    auto t0 = Clock::now();
    for (int i = 0; i < c.iters; ++i) {
      tesseract::ops::detail::gemm_avx2_f32(
          A.data_ptr<float>(), B.data_ptr<float>(), Cbuf.data(),
          c.M, c.N, c.K);
    }
    const double secs = elapsed_sec(t0, Clock::now()) / c.iters;
    const double flops = 2.0 * c.M * c.N * c.K;
    std::printf("[avx2     ]  M=%ld N=%ld K=%ld  %8.3f ms  %7.2f GFLOP/s  (checksum=%g)\n",
                long(c.M), long(c.N), long(c.K), secs * 1000.0,
                flops / secs / 1e9, static_cast<double>(Cbuf[0]));
  }

#if defined(TESSERACT_HAS_EIGEN)
  // Eigen direct map + `*`. This is what `ops::matmul` dispatches to
  // internally when the library was built with `TESSERACT_USE_EIGEN=ON`,
  // so the [tesseract] and [eigen] rows should track to within ~2% noise
  // in that config — this row's role is to make the ceiling explicit even
  // when the library was built without Eigen (e.g. for regression
  // comparisons).
  {
    using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> Am(A.data_ptr<float>(), c.M, c.K);
    Eigen::Map<const RowMat> Bm(B.data_ptr<float>(), c.K, c.N);
    RowMat Cm(c.M, c.N);
    auto t0 = Clock::now();
    for (int i = 0; i < c.iters; ++i) {
      Cm.noalias() = Am * Bm;
    }
    // Touch Cm so the optimizer doesn't drop the product entirely.
    const double secs = elapsed_sec(t0, Clock::now()) / c.iters;
    const double flops = 2.0 * c.M * c.N * c.K;
    std::printf("[eigen    ]  M=%ld N=%ld K=%ld  %8.3f ms  %7.2f GFLOP/s  (checksum=%g)\n",
                long(c.M), long(c.N), long(c.K), secs * 1000.0, flops / secs / 1e9,
                static_cast<double>(Cm(0, 0)));
  }
#endif
  std::printf("\n");
}

struct BatchedCase {
  int64_t Batch, M, N, K;
  int iters;
};

// Batched matmul bench. Measures two things:
//   1. The single batched `ops::matmul([B,M,K] @ [B,K,N])` call.
//   2. A for-loop that calls `ops::matmul` on each (M,K)/(K,N) slab.
// The first should approach the throughput of the second, modulo the
// per-call bookkeeping that the batched path amortizes by looping inside
// the kernel. We want (1) to be no worse than (2).
void run_batched_case(const BatchedCase& c) {
  using namespace tesseract;
  auto A = Tensor::empty({c.Batch, c.M, c.K}, DType::Float32);
  auto B = Tensor::empty({c.Batch, c.K, c.N}, DType::Float32);
  fill_random(A, 11);
  fill_random(B, 22);

  // Batched kernel.
  {
    auto t0 = Clock::now();
    for (int i = 0; i < c.iters; ++i) {
      auto C = ops::matmul(A, B);
      (void)C;
    }
    const double secs = elapsed_sec(t0, Clock::now()) / c.iters;
    const double flops = 2.0 * c.Batch * c.M * c.N * c.K;
    std::printf("[batched  ]  B=%ld M=%ld N=%ld K=%ld  %8.3f ms  %7.2f GFLOP/s\n",
                long(c.Batch), long(c.M), long(c.N), long(c.K),
                secs * 1000.0, flops / secs / 1e9);
  }

  // Per-slab reference (loop of rank-2 matmuls). We reshape the batched
  // buffers to rank-2 views on each iteration by indexing into the raw
  // contiguous memory — simulates user code that would call `matmul` in
  // a Python loop.
  {
    std::vector<float> Cbuf(static_cast<std::size_t>(c.Batch * c.M * c.N));
    auto t0 = Clock::now();
    for (int i = 0; i < c.iters; ++i) {
      for (int64_t b = 0; b < c.Batch; ++b) {
        auto As = Tensor::from_blob(A.data_ptr<float>() + b * c.M * c.K,
                                    {c.M, c.K}, DType::Float32);
        auto Bs = Tensor::from_blob(B.data_ptr<float>() + b * c.K * c.N,
                                    {c.K, c.N}, DType::Float32);
        auto Cs = ops::matmul(As, Bs);
        (void)Cs;
      }
    }
    const double secs = elapsed_sec(t0, Clock::now()) / c.iters;
    const double flops = 2.0 * c.Batch * c.M * c.N * c.K;
    std::printf("[per-slab ]  B=%ld M=%ld N=%ld K=%ld  %8.3f ms  %7.2f GFLOP/s\n",
                long(c.Batch), long(c.M), long(c.N), long(c.K),
                secs * 1000.0, flops / secs / 1e9);
  }

  // AVX2 microkernel, one call per batch slab — the direct analogue of
  // `[per-slab]` but skipping all the Tesseract Tensor scaffolding.
  if (tesseract::ops::detail::gemm_avx2_f32_supported()) {
    std::vector<float> Cbuf(static_cast<std::size_t>(c.M * c.N));
    double checksum = 0.0;
    auto t0 = Clock::now();
    for (int i = 0; i < c.iters; ++i) {
      for (int64_t b = 0; b < c.Batch; ++b) {
        tesseract::ops::detail::gemm_avx2_f32(
            A.data_ptr<float>() + b * c.M * c.K,
            B.data_ptr<float>() + b * c.K * c.N,
            Cbuf.data(), c.M, c.N, c.K);
      }
    }
    checksum = static_cast<double>(Cbuf[0]);
    const double secs = elapsed_sec(t0, Clock::now()) / c.iters;
    const double flops = 2.0 * c.Batch * c.M * c.N * c.K;
    std::printf("[avx2     ]  B=%ld M=%ld N=%ld K=%ld  %8.3f ms  %7.2f GFLOP/s  (checksum=%g)\n",
                long(c.Batch), long(c.M), long(c.N), long(c.K),
                secs * 1000.0, flops / secs / 1e9, checksum);
  }

#if defined(TESSERACT_HAS_EIGEN)
  // Eigen loop-over-slabs. Same role as the rank-2 [eigen] row: an
  // explicit ceiling so we can see how close the batched `ops::matmul`
  // gets to what Eigen+slab-loop can do.
  {
    using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMat Cm(c.M, c.N);
    double checksum = 0.0;
    auto t0 = Clock::now();
    for (int i = 0; i < c.iters; ++i) {
      for (int64_t b = 0; b < c.Batch; ++b) {
        Eigen::Map<const RowMat> Am(A.data_ptr<float>() + b * c.M * c.K, c.M, c.K);
        Eigen::Map<const RowMat> Bm(B.data_ptr<float>() + b * c.K * c.N, c.K, c.N);
        Cm.noalias() = Am * Bm;
      }
    }
    checksum = static_cast<double>(Cm(0, 0));
    const double secs = elapsed_sec(t0, Clock::now()) / c.iters;
    const double flops = 2.0 * c.Batch * c.M * c.N * c.K;
    std::printf("[eigen    ]  B=%ld M=%ld N=%ld K=%ld  %8.3f ms  %7.2f GFLOP/s  (checksum=%g)\n",
                long(c.Batch), long(c.M), long(c.N), long(c.K),
                secs * 1000.0, flops / secs / 1e9, checksum);
  }
#endif
  std::printf("\n");
}

}  // namespace

int main() {
  const Case cases[] = {
      { 64,  64,  64, 20},
      {128, 128, 128, 10},
      {256, 256, 256,  5},
      {512, 512, 512,  3},
  };
  for (const auto& c : cases) run_case(c);

  // B-004 DoD sizes: attention-style shapes where `cat`/`split` feeds a
  // batched matmul. The 8x512x512 point is the canonical headline case.
  const BatchedCase bcases[] = {
      {  4, 128, 128, 128, 10},
      {  8, 512, 512, 512,  3},
      { 16, 256, 256, 256,  3},
  };
  std::printf("-- batched matmul --\n");
  for (const auto& bc : bcases) run_batched_case(bc);
  return 0;
}

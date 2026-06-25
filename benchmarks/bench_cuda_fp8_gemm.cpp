// M4 perf-closeout Phase 3 — bench: FP8 (E4M3) tensor-core GEMM vs FP16
// cuBLASLt GEMM (the exact path PyTorch eager dispatches to on Ada).
//
// Both Tesseract's FP16 GEMM and PyTorch's FP16 GEMM call cuBLAS, so the
// "FP16 cuBLASLt" column here is a faithful stand-in for PyTorch FP16 —
// the comparison is apples-to-apples on the same vendor library. The win
// is structural: Ada's FP8 tensor cores run at ~2× the FP16 TC rate, so
// the FP8 path beats the FP16 path (and therefore PyTorch FP16) on the
// dense linear-layer line.
//
// Hard bar: FP8 TFLOPS >= 1.30× FP16 TFLOPS at the largest square shape.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/Fp8MatMul.hpp"
#include "tesseract/cuda/detail/MatMul.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using tesseract::DType;
namespace det = tesseract::cuda::detail;

namespace {

// Numeric sanity: FP8 E4M3 GEMM vs an FP32 host reference on a small
// shape. E4M3 has ~2 decimal digits, so we accept a few-percent mean
// relative error — enough to catch a layout/transpose bug.
bool fp8_numeric_ok(cudaStream_t stream) {
  const int64_t M = 256, N = 256, K = 256;
  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0.0f, 0.20f);
  std::vector<float> hX(M * K), hW(N * K);
  for (auto& v : hX) v = dist(rng);
  for (auto& v : hW) v = dist(rng);

  std::vector<float> ref(M * N, 0.0f);
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += hX[m * K + k] * hW[n * K + k];
      ref[m * N + n] = acc;
    }

  float *dXf = nullptr, *dWf = nullptr;
  void *dX = nullptr, *dW = nullptr, *dY = nullptr;
  bench::check_cuda(cudaMalloc(&dXf, M * K * sizeof(float)), "Xf");
  bench::check_cuda(cudaMalloc(&dWf, N * K * sizeof(float)), "Wf");
  bench::check_cuda(cudaMalloc(&dX, M * K), "X");
  bench::check_cuda(cudaMalloc(&dW, N * K), "W");
  bench::check_cuda(cudaMalloc(&dY, M * N * sizeof(__nv_bfloat16)), "Y");
  bench::check_cuda(cudaMemcpy(dXf, hX.data(), M * K * sizeof(float),
                               cudaMemcpyHostToDevice), "cpX");
  bench::check_cuda(cudaMemcpy(dWf, hW.data(), N * K * sizeof(float),
                               cudaMemcpyHostToDevice), "cpW");

  det::quantize_to_fp8_e4m3(dXf, dX, M * K, stream);
  det::quantize_to_fp8_e4m3(dWf, dW, N * K, stream);
  det::launch_fp8_linear(0, M, N, K, dX, dW, dY, 1.0f, 1.0f, stream);
  bench::check_cuda(cudaStreamSynchronize(stream), "sync");

  std::vector<__nv_bfloat16> hY(M * N);
  bench::check_cuda(cudaMemcpy(hY.data(), dY, M * N * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost), "cpY");
  double num = 0.0, den = 0.0;
  for (int64_t i = 0; i < M * N; ++i) {
    const double got = static_cast<double>(__bfloat162float(hY[i]));
    num += std::abs(got - ref[i]);
    den += std::abs(ref[i]);
  }
  const double rel = (den > 0) ? num / den : 0.0;
  std::printf("  numeric sanity   : mean rel err = %.4f (E4M3, want < 0.06)\n", rel);
  cudaFree(dXf); cudaFree(dWf); cudaFree(dX); cudaFree(dW); cudaFree(dY);
  return rel < 0.06;
}

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_fp8_gemm] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  if (!det::fp8_gemm_supported(0)) {
    std::printf("[bench_cuda_fp8_gemm] device < SM 8.9, no FP8 — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_fp8_gemm (FP8 E4M3 vs FP16 cuBLASLt)");

  bench::BenchStream bs;
  cudaStream_t stream = bs.native();

  const bool num_ok = fp8_numeric_ok(stream);

  std::printf("\n%-8s | %12s %12s | %10s %10s | %8s\n",
              "N (sq)", "fp8_us", "fp16_us", "fp8_TFLOP", "fp16_TFLOP", "speedup");
  std::printf("%s\n", "----------------------------------------------"
                      "-------------------------------");

  double min_speedup = 1e9;
  int64_t hard_n = 0;
  for (int64_t n : {1024, 2048, 4096, 8192}) {
    const int64_t M = n, N = n, K = n;
    // FP8 buffers.
    void *dX = nullptr, *dW = nullptr, *dY = nullptr;
    bench::check_cuda(cudaMalloc(&dX, M * K), "X");
    bench::check_cuda(cudaMalloc(&dW, N * K), "W");
    bench::check_cuda(cudaMalloc(&dY, M * N * sizeof(__nv_bfloat16)), "Y");
    cudaMemsetAsync(dX, 0, M * K, stream);
    cudaMemsetAsync(dW, 0, N * K, stream);
    // FP16 buffers.
    void *hA = nullptr, *hB = nullptr, *hC = nullptr;
    bench::check_cuda(cudaMalloc(&hA, M * K * sizeof(__half)), "A16");
    bench::check_cuda(cudaMalloc(&hB, K * N * sizeof(__half)), "B16");
    bench::check_cuda(cudaMalloc(&hC, M * N * sizeof(__half)), "C16");
    cudaMemsetAsync(hA, 0, M * K * sizeof(__half), stream);
    cudaMemsetAsync(hB, 0, K * N * sizeof(__half), stream);
    bench::check_cuda(cudaStreamSynchronize(stream), "init");

    auto fp8 = [&](cudaStream_t s) {
      det::launch_fp8_linear(0, M, N, K, dX, dW, dY, 1.0f, 1.0f, s);
    };
    auto fp16 = [&](cudaStream_t s) {
      det::launch_matmul(DType::Float16, 0, M, N, K,
                         hA, det::MmOp::None, K,
                         hB, det::MmOp::None, N,
                         hC, N, s);
    };
    fp8(stream); fp16(stream);
    bench::check_cuda(cudaStreamSynchronize(stream), "warm");

    auto s8 = bench::steady_state_time(fp8, stream);
    auto s16 = bench::steady_state_time(fp16, stream);
    const double flop = 2.0 * double(M) * double(N) * double(K);
    const double t8 = flop / (s8.mean_us * 1e-6) / 1e12;
    const double t16 = flop / (s16.mean_us * 1e-6) / 1e12;
    const double sp = s16.mean_us / s8.mean_us;
    std::printf("%-8ld | %12.2f %12.2f | %10.1f %10.1f | %7.2fx\n",
                long(n), s8.mean_us, s16.mean_us, t8, t16, sp);
    if (n >= 4096) { min_speedup = std::min(min_speedup, sp); hard_n = n; }

    cudaFree(dX); cudaFree(dW); cudaFree(dY);
    cudaFree(hA); cudaFree(hB); cudaFree(hC);
  }
  std::printf("\n");

  bool ok = bench::hard_check(num_ok, "FP8 numeric sanity (rel err < 0.06)",
                              num_ok ? 1.0 : 0.0, 1.0);
  if (hard_n > 0) {
    ok = bench::hard_check(min_speedup >= 1.30,
                           "FP8 GEMM speedup over FP16 cuBLASLt @ N>=4096 >= 1.30",
                           min_speedup, 1.30) && ok;
  }
  return ok ? bench::kExitOk : bench::kExitPerfMiss;
}

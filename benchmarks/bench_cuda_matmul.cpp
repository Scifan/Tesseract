// M2L.1 — bench: `ops::matmul` on CUDA vs our own `launch_matmul`
// bridge. Measures exactly the overhead the op layer adds on top of
// the cuBLASLt call we already issue — this is the number we can
// actually move with code changes.
//
// Hard bars (exit 1 if missed; see `cuda_bench_util.hpp` for the
// shared harness):
//
//   * MEDIAN additive overhead (min_ours_µs − min_bridge_µs) ≤ 5 µs
//     across all shapes. This is the number the op layer can actually
//     move: tensor/view setup, `detect_cuda_mat_layout`, batch
//     bookkeeping, autograd-disabled fast path, and the tensor
//     allocate/deallocate pair. We use MEDIAN (not max) because
//     cuBLASLt's `AlgoGetHeuristic` occasionally picks a slightly
//     different kernel variant between the bridge and ours path —
//     the picks are shape-keyed but not call-keyed, so one path can
//     land on a marginally slower algo. Median is robust to one or
//     two such flips across a 10-shape sweep.
//   * PER-SHAPE ratio floor (min_bridge / min_ours) ≥ 0.95 — guards
//     against catastrophic regression (e.g., cache miss, accidentally
//     falling off the fast path). We don't try to hit 0.99 per-shape
//     because cuBLASLt's intrinsic kernel variance at 72-500 µs kernel
//     times is ~5-10% run-to-run, even with a shared handle and
//     best-of-5 sampling.
//   * Dispatch overhead at 4096² FP32 ≤ 20 µs. A single anchor shape
//     where the kernel is ~4 ms and cuBLASLt variance is below 1%, so
//     this one DOES pin the op-layer cost precisely.
//
// We also emit an informational "vs. raw cuBLASLt" row using a
// SECOND set of hand-built descriptors + heuristic pick on the same
// handle. That number is noisy — cuBLASLt's `AlgoGetHeuristic` can
// return different algo variants between two calls with identical
// inputs, so the bench-local descriptor may land on a slower (or
// faster!) algo than our cache did. It's useful as a sanity check
// but NOT suitable for a hard gate.
//
// Shapes (square M=N=K): 512, 1024, 2048, 4096, 8192.
// Dtypes: Float32 (TF32 on Ada/Hopper via CUBLAS_COMPUTE_32F),
//         Float16 (Tensor Cores at FP32 accumulation).
//
// Design notes:
//   * Both timed paths run on the BenchStream (installed via
//     `StreamGuard`) so `current_stream(cuda0)` inside `ops::matmul`
//     and the timer events land on the same CUDA stream. Without
//     that, the "ours" path queues on a different thread-local
//     stream and the event-bracketed timing measures near-nothing.
//   * We deliberately don't repeat the M2G `bench_matmul` CPU
//     columns; this file is GPU-only so `ctest -L bench_cuda` can
//     skip it cleanly on CPU hosts.
//   * The bench always runs on device 0. Multi-GPU tuning lands
//     with B-010 (pooled scratch).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublasLt.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/detail/MatMul.hpp"
#include "tesseract/ops/MatMul.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;

namespace {

// Fill a host-side vector with IID Normal(0, 1) and ship to the GPU.
// Keep the RNG seed fixed across shapes so every call sees the same
// "real numbers" — cuBLASLt's heuristic is shape-keyed but not
// data-keyed, so this doesn't affect the algorithm pick.
std::vector<float> host_gauss(std::size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = nd(rng);
  return v;
}

// Host-side dtype helpers — we only touch FP32 and FP16 here.
cudaDataType_t cuda_type_of(tesseract::DType dt) {
  switch (dt) {
    case tesseract::DType::Float32: return CUDA_R_32F;
    case tesseract::DType::Float16: return CUDA_R_16F;
    default: bench::die("cuda_type_of", "unsupported dtype");
  }
}

// Naive FP32 → FP16 conversion (IEEE-754 half binary rep) so we can
// seed FP16 buffers without pulling in a dedicated half library.
// `__half` from `<cuda_fp16.h>` would work but would add a nvcc
// dependency to this otherwise host-only TU.
uint16_t f32_to_f16(float x) {
  union { uint32_t u; float f; } u{};
  u.f = x;
  const uint32_t sign = (u.u >> 31) & 0x1;
  const uint32_t exp  = (u.u >> 23) & 0xff;
  const uint32_t mant = u.u & 0x7fffffu;
  uint16_t s = static_cast<uint16_t>(sign << 15);
  if (exp == 0xff) {
    s |= 0x7c00u;
    if (mant) s |= 0x0200u;
    return s;
  }
  int32_t e = static_cast<int32_t>(exp) - 127 + 15;
  if (e >= 31) { s |= 0x7c00u; return s; }
  if (e <= 0) { return s; }
  s |= static_cast<uint16_t>((e << 10) | (mant >> 13));
  return s;
}

struct Case {
  int64_t M, N, K;
  tesseract::DType dtype;
  const char* dtype_name;
};

struct Result {
  // `bridge` is our own `launch_matmul` on bare pointers — the
  // baseline the hard bars use. `raw` is a hand-built cuBLASLt call
  // with its own descriptor + heuristic pick — informational.
  double bridge_mean = 0.0;
  double ours_mean   = 0.0;
  double raw_mean    = 0.0;
  double bridge_min  = 0.0;
  double ours_min    = 0.0;
  double raw_min     = 0.0;
  double ratio_ours_vs_bridge = 0.0;
  double dispatch_us = 0.0;
};

// Raw cuBLASLt driver on a bare allocation — mirrors exactly what our
// cached `launch_matmul` does post-cache-hit, minus the map lookup and
// the Tensor scaffolding. We deliberately SHARE the library's cuBLASLt
// handle (see `detail/MatMul.hpp` — `get_cublaslt_handle`) instead of
// creating a second one: two handles in the same process develop
// independent heuristic state and occasionally pick different algos
// for identical shapes, which shows up as a bimodal raw-vs-ours
// ratio. One handle keeps the comparison clean.
cublasLtHandle_t g_lt = nullptr;

void ensure_lt() {
  if (g_lt) return;
  g_lt = reinterpret_cast<cublasLtHandle_t>(
      tesseract::cuda::detail::get_cublaslt_handle(0));
  if (g_lt == nullptr) {
    bench::die("get_cublaslt_handle", "returned null");
  }
}

// Persistent 4 MiB workspace on device 0 — matches
// `src/cuda/MatMul.cpp`'s `get_workspace`. Shared across all "raw"
// timings below.
void* g_ws = nullptr;
std::size_t g_ws_bytes = 0;

void ensure_ws() {
  if (g_ws) return;
  g_ws_bytes = 4 * 1024 * 1024;
  if (cudaMalloc(&g_ws, g_ws_bytes) != cudaSuccess) {
    g_ws = nullptr; g_ws_bytes = 0;
  }
}

Result run_case(const Case& c, cudaStream_t stream) {
  using namespace tesseract;

  ensure_lt();
  ensure_ws();

  const int64_t num_a = c.M * c.K;
  const int64_t num_b = c.K * c.N;
  const int64_t num_c = c.M * c.N;
  const std::size_t elem = (c.dtype == DType::Float32) ? 4 : 2;

  // Host seed (random FP32 values, then downcast for FP16 runs).
  auto host_a = host_gauss(static_cast<std::size_t>(num_a), 0x1001);
  auto host_b = host_gauss(static_cast<std::size_t>(num_b), 0x2002);

  // Device buffers for the "raw" path.
  void* da = nullptr; void* db = nullptr; void* dc = nullptr;
  bench::check_cuda(cudaMalloc(&da, num_a * elem), "cudaMalloc(da)");
  bench::check_cuda(cudaMalloc(&db, num_b * elem), "cudaMalloc(db)");
  bench::check_cuda(cudaMalloc(&dc, num_c * elem), "cudaMalloc(dc)");

  if (c.dtype == DType::Float32) {
    bench::check_cuda(cudaMemcpy(da, host_a.data(), num_a * sizeof(float),
                                 cudaMemcpyHostToDevice), "memcpy(A)");
    bench::check_cuda(cudaMemcpy(db, host_b.data(), num_b * sizeof(float),
                                 cudaMemcpyHostToDevice), "memcpy(B)");
  } else {
    std::vector<uint16_t> half_a(num_a), half_b(num_b);
    for (int64_t i = 0; i < num_a; ++i) half_a[i] = f32_to_f16(host_a[i]);
    for (int64_t i = 0; i < num_b; ++i) half_b[i] = f32_to_f16(host_b[i]);
    bench::check_cuda(cudaMemcpy(da, half_a.data(), num_a * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice), "memcpy(A/f16)");
    bench::check_cuda(cudaMemcpy(db, half_b.data(), num_b * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice), "memcpy(B/f16)");
  }

  // Build descriptors once. Row-major; no transpose — matches what our
  // op layer emits for a default `matmul(a, b)` call where both
  // operands are already in row-major layout.
  const cudaDataType_t cdt = cuda_type_of(c.dtype);
  const cublasComputeType_t ct =
      (c.dtype == DType::Float32) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F;
  const cudaDataType_t stype = CUDA_R_32F;

  cublasLtMatmulDesc_t desc = nullptr;
  if (cublasLtMatmulDescCreate(&desc, ct, stype) != CUBLAS_STATUS_SUCCESS) {
    bench::die("MatmulDescCreate", "failed");
  }
  cublasOperation_t opN = CUBLAS_OP_N;
  (void)cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &opN, sizeof(opN));
  (void)cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &opN, sizeof(opN));

  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  auto mk = [&](cublasLtMatrixLayout_t& out, uint64_t r, uint64_t cc, int64_t ld) {
    (void)cublasLtMatrixLayoutCreate(&out, cdt, r, cc, ld);
    cublasLtOrder_t ord = CUBLASLT_ORDER_ROW;
    (void)cublasLtMatrixLayoutSetAttribute(out, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                           &ord, sizeof(ord));
  };
  mk(la, static_cast<uint64_t>(c.M), static_cast<uint64_t>(c.K), c.K);
  mk(lb, static_cast<uint64_t>(c.K), static_cast<uint64_t>(c.N), c.N);
  mk(lc, static_cast<uint64_t>(c.M), static_cast<uint64_t>(c.N), c.N);

  cublasLtMatmulPreference_t pref = nullptr;
  (void)cublasLtMatmulPreferenceCreate(&pref);
  (void)cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &g_ws_bytes, sizeof(g_ws_bytes));

  cublasLtMatmulHeuristicResult_t hr{};
  int returned = 0;
  (void)cublasLtMatmulAlgoGetHeuristic(
      g_lt, desc, la, lb, lc, lc, pref, 1, &hr, &returned);
  if (returned <= 0) bench::die("AlgoGetHeuristic", "no algo");
  (void)cublasLtMatmulPreferenceDestroy(pref);

  const float alpha_f = 1.0f, beta_f = 0.0f;

  // Informational raw cuBLASLt path — hand-built descriptor + algo.
  auto raw_call = [&](cudaStream_t s) {
    (void)cublasLtMatmul(
        g_lt, desc,
        &alpha_f, da, la,
                  db, lb,
        &beta_f,  dc, lc,
                  dc, lc,
        &hr.algo, g_ws, g_ws_bytes, s);
  };

  // Primary baseline: our own `launch_matmul`, which reuses the
  // library's descriptor/algo cache. Populated on the first call
  // below; timing skips that warmup. We keep the output buffer
  // fixed (`dc`) so this path measures ONLY the cuBLASLt launch.
  auto bridge_call = [&](cudaStream_t s) {
    tesseract::cuda::detail::launch_matmul(
        c.dtype, /*device_index=*/0, c.M, c.N, c.K,
        da, tesseract::cuda::detail::MmOp::None, c.K,
        db, tesseract::cuda::detail::MmOp::None, c.N,
        dc, c.N, s);
  };

  Result r;
  auto stats_raw = bench::best_of_n_time(raw_call, stream);
  r.raw_mean = stats_raw.mean_us;
  r.raw_min  = stats_raw.min_us;

  auto stats_bridge = bench::best_of_n_time(bridge_call, stream);
  r.bridge_mean = stats_bridge.mean_us;
  r.bridge_min  = stats_bridge.min_us;

  // "Ours" path via ops::matmul. We need Tensor objects on CUDA with
  // the same contents. NoGradGuard so we don't build an autograd graph.
  Device cuda0{DeviceType::CUDA, 0};
  Tensor ta_cpu = Tensor::empty({c.M, c.K}, c.dtype);
  Tensor tb_cpu = Tensor::empty({c.K, c.N}, c.dtype);
  if (c.dtype == DType::Float32) {
    std::memcpy(ta_cpu.data_ptr<float>(), host_a.data(), num_a * sizeof(float));
    std::memcpy(tb_cpu.data_ptr<float>(), host_b.data(), num_b * sizeof(float));
  } else {
    // For FP16, we populate by H2D memcpy of the uint16 buffer into
    // a CUDA-side Tensor directly (CPU FP16 storage would need a
    // scalar type we don't expose).
  }

  Tensor ta_cuda = (c.dtype == DType::Float32) ? ta_cpu.to(cuda0)
                                               : Tensor::empty({c.M, c.K}, c.dtype, cuda0);
  Tensor tb_cuda = (c.dtype == DType::Float32) ? tb_cpu.to(cuda0)
                                               : Tensor::empty({c.K, c.N}, c.dtype, cuda0);
  if (c.dtype == DType::Float16) {
    // Seed the CUDA tensors from the same bits we shipped to the raw
    // path so both tests compute identical products.
    bench::check_cuda(cudaMemcpy(ta_cuda.raw_data(), da,
                                 num_a * elem, cudaMemcpyDeviceToDevice),
                      "memcpy(ta fp16)");
    bench::check_cuda(cudaMemcpy(tb_cuda.raw_data(), db,
                                 num_b * elem, cudaMemcpyDeviceToDevice),
                      "memcpy(tb fp16)");
  }

  auto ours_call = [&](cudaStream_t /*unused*/) {
    NoGradGuard nogg;
    Tensor out = ops::matmul(ta_cuda, tb_cuda);
    (void)out;
  };
  // First warmup call populates the descriptor cache.
  ours_call(stream);
  bench::check_cuda(cudaStreamSynchronize(stream), "first-warmup");
  auto stats_ours = bench::best_of_n_time(ours_call, stream);
  r.ours_mean = stats_ours.mean_us;
  r.ours_min  = stats_ours.min_us;

  // Ratio is measured against our own `launch_matmul` bridge, using
  // the min-over-samples which is stable across runs. The ops layer
  // can only ever be slower than `launch_matmul`, so ratio ≤ 1.0 by
  // construction.
  r.ratio_ours_vs_bridge = r.bridge_min / r.ours_min;
  r.dispatch_us = r.ours_min - r.bridge_min;

  // Cleanup descriptors (workspace + handle are process-level).
  (void)cublasLtMatrixLayoutDestroy(la);
  (void)cublasLtMatrixLayoutDestroy(lb);
  (void)cublasLtMatrixLayoutDestroy(lc);
  (void)cublasLtMatmulDescDestroy(desc);
  (void)cudaFree(da);
  (void)cudaFree(db);
  (void)cudaFree(dc);
  return r;
}

double tflops(int64_t M, int64_t N, int64_t K, double secs) {
  const double ops = 2.0 * static_cast<double>(M) * N * K;
  return ops / secs / 1e12;
}

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_matmul] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }

  bench::print_banner("bench_cuda_matmul");

  bench::BenchStream bench_stream;
  cudaStream_t stream = bench_stream.native();

  using tesseract::DType;
  const Case cases[] = {
      { 512,   512,   512, DType::Float32, "fp32" },
      {1024,  1024,  1024, DType::Float32, "fp32" },
      {2048,  2048,  2048, DType::Float32, "fp32" },
      {4096,  4096,  4096, DType::Float32, "fp32" },
      {8192,  8192,  8192, DType::Float32, "fp32" },
      { 512,   512,   512, DType::Float16, "fp16" },
      {1024,  1024,  1024, DType::Float16, "fp16" },
      {2048,  2048,  2048, DType::Float16, "fp16" },
      {4096,  4096,  4096, DType::Float16, "fp16" },
      {8192,  8192,  8192, DType::Float16, "fp16" },
  };

  std::printf("%-6s %-5s | %10s %10s %10s | %10s %10s %10s | %7s %8s\n",
              "dtype", "N",
              "raw_min", "bridge_min", "ours_min",
              "raw_TF*", "bridge_TF*", "ours_TF*",
              "ratio*", "dispatch");
  std::printf("        (* TF and ratio reported on min-over-samples; "
              "ratio* = bridge_min/ours_min)\n");
  std::printf("%s\n", "----------------------------------------"
                      "----------------------------------------"
                      "--------------------------------------");

  bool all_pass = true;
  double dispatch_at_4096_fp32 = 0.0;
  double min_ratio = 1.0;
  std::vector<double> dispatch_samples;
  dispatch_samples.reserve(sizeof(cases) / sizeof(cases[0]));

  for (const auto& c : cases) {
    Result r = run_case(c, stream);
    const double secs_raw    = r.raw_min    * 1e-6;
    const double secs_bridge = r.bridge_min * 1e-6;
    const double secs_ours   = r.ours_min   * 1e-6;
    std::printf("%-6s %-5ld | %10.2f %10.2f %10.2f | %10.2f %10.2f %10.2f | %7.4f %8.2f\n",
                c.dtype_name, long(c.M),
                r.raw_min, r.bridge_min, r.ours_min,
                tflops(c.M, c.N, c.K, secs_raw),
                tflops(c.M, c.N, c.K, secs_bridge),
                tflops(c.M, c.N, c.K, secs_ours),
                r.ratio_ours_vs_bridge, r.dispatch_us);
    if (r.ratio_ours_vs_bridge < min_ratio) min_ratio = r.ratio_ours_vs_bridge;
    dispatch_samples.push_back(r.dispatch_us);
    if (c.dtype == DType::Float32 && c.M == 4096) {
      dispatch_at_4096_fp32 = r.dispatch_us;
    }
  }
  std::printf("\n");

  // Median additive overhead. We sort the per-shape dispatch samples
  // and take the middle value — robust to one or two outliers caused
  // by cuBLASLt algo-flips or transient DVFS dips.
  std::vector<double> sorted_disp = dispatch_samples;
  std::sort(sorted_disp.begin(), sorted_disp.end());
  const double median_dispatch = sorted_disp.empty() ? 0.0
      : sorted_disp[sorted_disp.size() / 2];

  // Hard bars.
  all_pass &= bench::hard_check(median_dispatch <= 5.0,
                                "median(ours - bridge) us <= 5",
                                median_dispatch, 5.0);
  all_pass &= bench::hard_check(min_ratio >= 0.95,
                                "min(ours/bridge) per-shape >= 0.95",
                                min_ratio, 0.95);
  all_pass &= bench::hard_check(dispatch_at_4096_fp32 <= 20.0,
                                "dispatch_us @ 4096^2 FP32 <= 20",
                                dispatch_at_4096_fp32, 20.0);

  
  return all_pass ? bench::kExitOk : bench::kExitPerfMiss;
}

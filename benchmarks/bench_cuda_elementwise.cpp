// M2L.1 — bench: elementwise kernels vs cudaMemcpyAsync bandwidth.
//
// Rationale: elementwise kernels are memory-bound by construction — each
// thread reads its operand(s) and writes the output, all in global
// memory. The theoretical ceiling is the device's sustained DRAM
// bandwidth, which we probe with a large D→D `cudaMemcpyAsync`:
//
//   add / sub / mul / div : 2 reads + 1 write  → 3× operand bytes / call
//   unary (sigmoid/...)   : 1 read  + 1 write  → 2× operand bytes / call
//   memcpy                : 1 read  + 1 write  → 2× operand bytes / call
//
// We normalise every measurement to "DRAM bytes moved / second" so
// all five columns share the same y-axis and comparing an `add` row
// against the memcpy roofline is apples-to-apples.
//
// Hard bars (exit 1 if missed):
//   * Sustained add @ 64 MiB  ≥ 0.95 × memcpy DRAM BW (FP32, dense
//     contiguous). 64 MiB well exceeds the RTX Ada L2 cache (~50 MiB)
//     so this is a true DRAM-bound point.
//   * Sustained mul @ 64 MiB  ≥ 0.95 × memcpy DRAM BW (same reason).
//   * Sustained sig @ 64 MiB  ≥ 0.90 × memcpy DRAM BW (sigmoid does
//     a transcendental per element, so we allow a small ALU tax).
//
// The 1 MiB and 16 MiB rows are informational — they fit in L2 and
// report L2 bandwidth, which is useful for trend analysis but not
// for a hard gate.

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Activation.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;

namespace {

struct Case {
  const char* name;
  std::size_t bytes;  // size of each operand (same shape FP32)
};

// DRAM-bytes-moved / second. Every operand crossed (read or written)
// counts once — `reads + writes` for the op, `2` for memcpy.
double dram_gbs(std::size_t bytes_per_operand, int reads, int writes,
                double per_call_us) {
  const double secs = per_call_us * 1e-6;
  const double total_bytes = static_cast<double>(bytes_per_operand) *
                             static_cast<double>(reads + writes);
  return total_bytes / secs / 1e9;
}

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_elementwise] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_elementwise");

  bench::BenchStream bench_stream;
  cudaStream_t stream = bench_stream.native();

  const double memcpy_oneway_gbs = bench::measure_d2d_memcpy_gbs(stream);
  // memcpy reports 1-way (bytes/time). DRAM moves 1R+1W = 2× bytes.
  const double memcpy_dram_gbs = memcpy_oneway_gbs * 2.0;
  std::printf("  memcpy D2D @ 256 MiB  : %7.1f GB/s  (1-way) → %7.1f GB/s DRAM roofline\n\n",
              memcpy_oneway_gbs, memcpy_dram_gbs);

  const Case sizes[] = {
      { "1 MiB",    std::size_t(1)   << 20 },
      { "16 MiB",   std::size_t(16)  << 20 },
      { "64 MiB",   std::size_t(64)  << 20 },
      { "256 MiB",  std::size_t(256) << 20 },
  };

  std::printf("%-10s %8s | %-10s %8s %6s | %-10s %8s %6s | %-10s %8s %6s\n",
              "size", "MiB/fp32",
              "add_us", "DRAM",  "/roof",
              "mul_us", "DRAM",  "/roof",
              "sig_us", "DRAM",  "/roof");
  std::printf("        (DRAM = 3× bytes/time for add/mul, 2× for sigmoid; "
              "/roof = DRAM / memcpy_dram_gbs)\n");
  std::printf("%s\n",
    "-------------------------------------------------------------"
    "-------------------------------------------------------------");

  double add64_roof = 0.0, mul64_roof = 0.0, sig64_roof = 0.0;

  using tesseract::Tensor;
  using tesseract::DType;
  using tesseract::Device;
  using tesseract::DeviceType;
  using tesseract::NoGradGuard;
  Device cuda0{DeviceType::CUDA, 0};

  for (const auto& sc : sizes) {
    const std::size_t nelem = sc.bytes / sizeof(float);
    // Square-ish 2-D shape so `align_for_broadcast` has something
    // non-trivial to process — we still want the dense fast-path to
    // kick in (strides match sizes).
    const int64_t side = static_cast<int64_t>(std::sqrt(static_cast<double>(nelem)));
    // Pick M/N so M*N == nelem exactly (use a 1-D shape if no clean
    // factoring). FP32 sizes here are powers of two, so side*side
    // matches exactly.
    const int64_t M = side;
    const int64_t N = static_cast<int64_t>(nelem / static_cast<std::size_t>(side));
    auto A = Tensor::empty({M, N}, DType::Float32, cuda0);
    auto B = Tensor::empty({M, N}, DType::Float32, cuda0);
    // Zero-fill; for mem-bound benches the data content doesn't
    // meaningfully affect timing.
    bench::check_cuda(cudaMemsetAsync(A.raw_data(), 0, M*N*sizeof(float), stream),
                      "memset A");
    bench::check_cuda(cudaMemsetAsync(B.raw_data(), 0, M*N*sizeof(float), stream),
                      "memset B");
    bench::check_cuda(cudaStreamSynchronize(stream), "memset-sync");

    auto add_call = [&](cudaStream_t /*s*/) {
      NoGradGuard ng;
      auto C = tesseract::ops::add(A, B);
      (void)C;
    };
    auto mul_call = [&](cudaStream_t /*s*/) {
      NoGradGuard ng;
      auto C = tesseract::ops::mul(A, B);
      (void)C;
    };
    auto sig_call = [&](cudaStream_t /*s*/) {
      NoGradGuard ng;
      auto C = tesseract::ops::sigmoid(A);
      (void)C;
    };

    // best-of-N to defend against DVFS blips at the small sizes —
    // same pattern as `bench_cuda_matmul`.
    auto run = [&](auto f) { return bench::best_of_n_time(f, stream); };
    auto a = run(add_call);
    auto m = run(mul_call);
    auto g = run(sig_call);

    // Use min_us (not mean) for the headline DRAM throughput — matches
    // the "best achievable" convention the matmul bench uses.
    const double add_gbs = dram_gbs(sc.bytes, 2, 1, a.min_us);
    const double mul_gbs = dram_gbs(sc.bytes, 2, 1, m.min_us);
    const double sig_gbs = dram_gbs(sc.bytes, 1, 1, g.min_us);

    std::printf("%-10s %8zu | %10.2f %8.1f %6.3f | %10.2f %8.1f %6.3f | %10.2f %8.1f %6.3f\n",
                sc.name, sc.bytes >> 20,
                a.min_us, add_gbs, add_gbs / memcpy_dram_gbs,
                m.min_us, mul_gbs, mul_gbs / memcpy_dram_gbs,
                g.min_us, sig_gbs, sig_gbs / memcpy_dram_gbs);
    if (sc.bytes == (std::size_t(64) << 20)) {
      add64_roof = add_gbs / memcpy_dram_gbs;
      mul64_roof = mul_gbs / memcpy_dram_gbs;
      sig64_roof = sig_gbs / memcpy_dram_gbs;
    }
  }
  std::printf("\n");

  bool ok = true;
  ok &= bench::hard_check(add64_roof >= 0.95,
                          "add @ 64 MiB / memcpy_DRAM >= 0.95",
                          add64_roof, 0.95);
  ok &= bench::hard_check(mul64_roof >= 0.95,
                          "mul @ 64 MiB / memcpy_DRAM >= 0.95",
                          mul64_roof, 0.95);
  ok &= bench::hard_check(sig64_roof >= 0.90,
                          "sigmoid @ 64 MiB / memcpy_DRAM >= 0.90",
                          sig64_roof, 0.90);

  
  return ok ? bench::kExitOk : bench::kExitPerfMiss;
}

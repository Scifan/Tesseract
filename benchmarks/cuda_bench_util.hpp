// M2L.1 — shared harness for the `bench_cuda_*` executables.
//
// Every CUDA benchmark in this directory follows the same recipe:
//
//   1. Warm the GPU + driver (launch the probe once, discard the timing).
//   2. Measure N iterations in a CUDA-event sandwich.
//   3. Keep growing N until the coefficient of variation of the last
//      `kCovWindow` samples drops under `cov_target` (default 2 %), or
//      we hit `max_iters` — whichever comes first. This is the
//      steady-state detector.
//   4. Report the mean / std / min time over the accepted window.
//
// The goal is to pin down a stable per-call latency so the perf ratios
// in the M2L hard bars (≥99% cuBLASLt, ≥95% memcpy, etc.) are trustworthy
// without hand-tuning iteration counts per shape.
//
// Everything in this header is `static inline` — each bench executable
// gets its own copy, no separate TU or library target. Keeps the
// benchmark tree zero-dependency on the main library CMake (only needs
// the CUDA toolkit).

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Stream.hpp"

namespace tesseract::bench {

// ---------------------------------------------------------------------------
// Exit codes
// ---------------------------------------------------------------------------
// Shared by every M2L.1 bench main() so ctest can route them correctly.
// See `benchmarks/CMakeLists.txt` for the SKIP_RETURN_CODE + WILL_FAIL
// wiring. `kPerfMiss` is a regular test failure (exit 1) so CI fails
// the build without extra CMake plumbing.
inline constexpr int kExitOk       = 0;
inline constexpr int kExitPerfMiss = 1;
inline constexpr int kExitNoCuda   = 77;  // POSIX "test skipped"

// ---------------------------------------------------------------------------
// Error handling. We deliberately don't pull in the library's
// `tesseract::DeviceError` — bench binaries only link against the CUDA
// toolkit + the library's public API, and we want any runtime failure to
// surface as a hard crash with a clear message (not an exception that
// gets swallowed by ctest's process-exit logic).
// ---------------------------------------------------------------------------
[[noreturn]] inline void die(const char* where, const char* detail) {
  std::fprintf(stderr, "[bench] FATAL %s: %s\n", where, detail);
  std::exit(2);
}

inline void check_cuda(cudaError_t err, const char* where) {
  if (err != cudaSuccess) die(where, cudaGetErrorString(err));
}

// Probe for CUDA availability; returns the device count or 0 if no
// runtime / no visible device. Used at the top of every main() so the
// benches SKIP rather than FAIL on CPU-only CI.
inline int visible_cuda_devices() {
  int n = 0;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess) {
    // `cudaErrorNoDevice` / `cudaErrorInsufficientDriver` / `cudaErrorInvalidDevice`
    // all collapse to "no GPU to bench on" — main() should SKIP.
    return 0;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Shared stream RAII wrapper
// ---------------------------------------------------------------------------
// Every bench creates ONE non-blocking CUDA stream, installs it as the
// current stream via `StreamGuard`, and hands its raw handle to both
// the timer and any raw cuBLAS / cudaMemcpyAsync call in the run body.
// That way `ops::matmul` & friends (which resolve their stream through
// `current_stream(device)`) queue work on the SAME stream the timer
// events are recorded on — otherwise the events and the GPU work live
// on different streams and the measured time does not reflect the
// actual kernel wall time.
//
// The guard releases automatically when the `BenchStream` goes out of
// scope (end of `main` in every bench), restoring whatever thread-local
// stream was installed before.
class BenchStream {
 public:
  explicit BenchStream(Device device = Device{DeviceType::CUDA, 0})
      : stream_(Stream::create(device)), guard_(stream_) {}

  BenchStream(const BenchStream&) = delete;
  BenchStream& operator=(const BenchStream&) = delete;

  cudaStream_t native() const noexcept {
    return static_cast<cudaStream_t>(stream_.native_handle());
  }
  const Stream& stream() const noexcept { return stream_; }

 private:
  Stream stream_;
  StreamGuard guard_;
};

// ---------------------------------------------------------------------------
// Event-based timer
// ---------------------------------------------------------------------------
// CUDA events are the only timer that's both (a) asynchronous so we don't
// serialize the kernel behind a CPU clock read, and (b) accurate to ~0.5
// µs against the GPU's internal clock. `cudaEventRecord(start/stop)` on
// the same stream brackets exactly the kernel run; the device is
// synchronized between the two records by the ordering of kernel
// launches on the stream.
class CudaTimer {
 public:
  CudaTimer() {
    check_cuda(cudaEventCreate(&start_), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop_),  "cudaEventCreate(stop)");
  }
  ~CudaTimer() {
    (void)cudaEventDestroy(start_);
    (void)cudaEventDestroy(stop_);
  }

  CudaTimer(const CudaTimer&) = delete;
  CudaTimer& operator=(const CudaTimer&) = delete;

  void tic(cudaStream_t stream) {
    check_cuda(cudaEventRecord(start_, stream), "cudaEventRecord(start)");
  }
  // Returns elapsed milliseconds between the last `tic()` and this call.
  // Synchronizes the stop event so the timing is final when we return.
  float toc(cudaStream_t stream) {
    check_cuda(cudaEventRecord(stop_, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
    return ms;
  }

 private:
  cudaEvent_t start_{nullptr};
  cudaEvent_t stop_{nullptr};
};

// ---------------------------------------------------------------------------
// Steady-state timing loop
// ---------------------------------------------------------------------------
// Repeatedly runs `fn(stream)` in batches of `batch` calls per event
// window. Keeps going until either
//   (a) the coefficient of variation (stddev / mean) across the last
//       `kCovWindow` batches drops under `cov_target`, or
//   (b) we hit `max_iters` total batches.
// Returns the mean per-call time in microseconds over the accepted
// window, along with the observed cov for diagnostic printing.
//
// `batch = 0` means auto-pick: we run a 1-call timing first, then set
// batch = max(1, round(min_window_ms / per_call_ms)) so each event
// window is at least ~1 ms long. Short kernels get batched so CUDA event
// overhead (~2 µs) doesn't dominate; long kernels keep batch=1.
struct SteadyStats {
  double mean_us  = 0.0;
  double stddev_us = 0.0;
  double min_us   = 0.0;
  double cov      = 0.0;
  int    batches  = 0;
  int    batch    = 0;
};

inline constexpr int kCovWindow = 20;

template <typename F>
SteadyStats steady_state_time(F&& fn, cudaStream_t stream,
                              double cov_target = 0.02,
                              int    warmup     = 10,
                              int    max_iters  = 500,
                              int    batch      = 0,
                              double min_window_ms = 1.0) {
  CudaTimer timer;

  for (int i = 0; i < warmup; ++i) fn(stream);
  check_cuda(cudaStreamSynchronize(stream), "warmup-sync");

  if (batch <= 0) {
    // Single-call probe to size the batch.
    timer.tic(stream);
    fn(stream);
    const float probe_ms = timer.toc(stream);
    if (probe_ms >= min_window_ms) {
      batch = 1;
    } else {
      const double b = min_window_ms / std::max<float>(probe_ms, 1e-4f);
      batch = std::max(1, static_cast<int>(std::lround(b)));
    }
  }

  std::vector<double> samples;
  samples.reserve(max_iters);
  double sum = 0.0, sumsq = 0.0;
  double minv = std::numeric_limits<double>::infinity();

  for (int it = 0; it < max_iters; ++it) {
    timer.tic(stream);
    for (int k = 0; k < batch; ++k) fn(stream);
    const float win_ms = timer.toc(stream);
    const double per_us = (static_cast<double>(win_ms) / batch) * 1e3;
    samples.push_back(per_us);
    sum += per_us;
    sumsq += per_us * per_us;
    minv = std::min(minv, per_us);
    if (static_cast<int>(samples.size()) >= kCovWindow) {
      // CoV over the trailing window.
      double wsum = 0.0, wsumsq = 0.0;
      for (int i = static_cast<int>(samples.size()) - kCovWindow;
           i < static_cast<int>(samples.size()); ++i) {
        wsum   += samples[i];
        wsumsq += samples[i] * samples[i];
      }
      const double wmean = wsum / kCovWindow;
      const double wvar  = std::max(0.0, wsumsq / kCovWindow - wmean * wmean);
      const double wstd  = std::sqrt(wvar);
      const double wcov  = wmean > 0.0 ? wstd / wmean : 0.0;
      if (wcov <= cov_target && it + 1 >= kCovWindow + warmup) {
        SteadyStats s;
        s.mean_us   = wmean;
        s.stddev_us = wstd;
        s.min_us    = minv;
        s.cov       = wcov;
        s.batches   = it + 1;
        s.batch     = batch;
        return s;
      }
    }
  }

  SteadyStats s;
  const double n = static_cast<double>(samples.size());
  s.mean_us   = sum / n;
  const double var = std::max(0.0, sumsq / n - s.mean_us * s.mean_us);
  s.stddev_us = std::sqrt(var);
  s.min_us    = minv;
  s.cov       = s.mean_us > 0.0 ? s.stddev_us / s.mean_us : 0.0;
  s.batches   = static_cast<int>(samples.size());
  s.batch     = batch;
  return s;
}

// ---------------------------------------------------------------------------
// Multi-trial wrapper. Runs `steady_state_time` `trials` times and
// returns the trial with the lowest `min_us`. Defends against the
// periodic cuBLASLt algo drift + DVFS hiccups we observed on RTX Ada:
// any single `steady_state_time` call occasionally catches a streak
// of 20 samples where the heuristic picked a slightly different
// variant, which skews `min_us` upward by 20-50%. A best-of-N sweep
// gives each trial a fresh warmup + CoV-locked window and keeps the
// one where the GPU was actually running at steady state.
template <typename F>
SteadyStats best_of_n_time(F&& fn, cudaStream_t stream,
                           int trials = 5,
                           double cov_target = 0.02,
                           int    warmup     = 10,
                           int    max_iters  = 500,
                           int    batch      = 0,
                           double min_window_ms = 1.0) {
  SteadyStats best{};
  best.min_us = std::numeric_limits<double>::infinity();
  best.mean_us = std::numeric_limits<double>::infinity();
  for (int t = 0; t < trials; ++t) {
    auto s = steady_state_time(std::forward<F>(fn), stream,
                               cov_target, warmup, max_iters, batch,
                               min_window_ms);
    if (s.min_us < best.min_us) best = s;
  }
  return best;
}

// ---------------------------------------------------------------------------
// Device-to-device copy bandwidth — the "memcpy roofline" for the
// elementwise bench. Allocates a 256 MiB buffer pair and measures
// `cudaMemcpyAsync(D→D)` throughput at steady state. Returned in GB/s
// (not GiB/s; matches NVIDIA marketing numbers).
// ---------------------------------------------------------------------------
inline double measure_d2d_memcpy_gbs(cudaStream_t stream,
                                     std::size_t bytes = std::size_t(256) << 20) {
  void* src = nullptr;
  void* dst = nullptr;
  check_cuda(cudaMalloc(&src, bytes), "cudaMalloc(src)");
  check_cuda(cudaMalloc(&dst, bytes), "cudaMalloc(dst)");
  check_cuda(cudaMemsetAsync(src, 0x42, bytes, stream), "memset-src");
  check_cuda(cudaStreamSynchronize(stream), "fill-sync");
  auto copy = [&](cudaStream_t s) {
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s),
               "cudaMemcpyAsync");
  };
  auto st = steady_state_time(copy, stream, /*cov_target=*/0.02,
                              /*warmup=*/5, /*max_iters=*/200, /*batch=*/3,
                              /*min_window_ms=*/2.0);
  const double secs = st.mean_us * 1e-6;
  const double gbs  = static_cast<double>(bytes) / secs / 1e9;
  check_cuda(cudaFree(src), "cudaFree(src)");
  check_cuda(cudaFree(dst), "cudaFree(dst)");
  return gbs;
}

// ---------------------------------------------------------------------------
// Device properties snapshot
// ---------------------------------------------------------------------------
struct DeviceInfo {
  std::string name;
  int major = 0;
  int minor = 0;
  double peak_mem_bw_gbs = 0.0;
};

inline DeviceInfo query_device(int idx = 0) {
  cudaDeviceProp p{};
  check_cuda(cudaGetDeviceProperties(&p, idx), "cudaGetDeviceProperties");
  DeviceInfo d;
  d.name  = p.name;
  d.major = p.major;
  d.minor = p.minor;
  // Theoretical memory bandwidth: `memoryBusWidth` in bits × `memoryClockRate`
  // in kHz × 2 (DDR) / 8 (bits → bytes) / 1e6 (kHz → GHz; GB/s cancels).
  // Matches `nvidia-smi --query-gpu=memory_bus_width,memory_clock`.
  const double bw = 2.0 * p.memoryBusWidth * p.memoryClockRate / 8.0 / 1.0e6;
  d.peak_mem_bw_gbs = bw;
  return d;
}

// ---------------------------------------------------------------------------
// Hard-bar assertion. Exits with `kExitPerfMiss` on miss so ctest turns
// it into a FAIL without extra plumbing.
// ---------------------------------------------------------------------------
inline bool hard_check(bool cond, const char* label, double got, double want) {
  const char* verdict = cond ? "PASS" : "FAIL";
  std::printf("  [hard] %-48s got=%.4f want=%.4f  %s\n",
              label, got, want, verdict);
  return cond;
}

// ---------------------------------------------------------------------------
// Section header with device banner — keeps every bench's log uniform.
// ---------------------------------------------------------------------------
inline void print_banner(const char* bench_name) {
  const DeviceInfo d = query_device(0);
  std::printf("==================== %s ====================\n", bench_name);
  std::printf("  device   : %s  (SM %d.%d)\n", d.name.c_str(), d.major, d.minor);
  std::printf("  peak BW  : %.1f GB/s  (theoretical)\n", d.peak_mem_bw_gbs);
  std::printf("\n");
}

}  // namespace tesseract::bench

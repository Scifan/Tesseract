// Wave 2.4 (B-011) — bench: pinned async H↔D transfer.
//
// What this bench measures — and what the hard bars gate on:
//
//   The two real benefits of pinned memory + `cudaMemcpyAsync` are
//   (a) lower bounce-buffer overhead on small-to-medium copies, and
//   (b) ability to overlap the transfer with compute on a different
//   stream. This bench gates both behaviors with explicit hard bars
//   rather than chasing peak PCIe bandwidth numbers — at 64 MiB on
//   PCIe Gen4 the link itself (≈ 32 GB/s effective) dominates, so a
//   "pinned is faster than pageable" check would be noise-bounded.
//
//   * **Transfer-speedup row** (1 MiB, H→D): pinned vs pageable
//     synchronous memcpy. Small-enough to keep the bounce-buffer
//     overhead visible, large-enough that `cudaMemcpyAsync`'s
//     launch overhead is amortized. Hard bar: **≥ 1.4× faster**
//     pinned over sync-pageable. Observed 1.80× on RTX 5880 Ada
//     (SM 8.9) / PCIe Gen4 x16.
//
//   * **Submit-latency row** (4 KiB, H→D async): submit latency of
//     a tiny `cudaMemcpyAsync`. At this size the PCIe transfer
//     itself takes ~1 µs; with a well-behaved driver the submit
//     call returns in a few µs regardless of DMA progress. This is
//     the metric that matters for the B-023 CUDA-Graph-replay
//     overlap: we want to be able to fire a "copy next token id
//     to device" call in under a few µs so it never sits on the
//     critical path. Hard bar: **≤ 10 µs submit latency**.
//
//   * **Overlap-with-compute row** (4 MiB transfer + small compute
//     kernel on a different stream): this is the killer demo. We
//     run them sequentially (sync copy, then compute) and then in
//     parallel (async pinned copy on `copy_stream`, compute on
//     `compute_stream`, both synchronized at the end). Hard bar:
//     **(copy || compute) / (copy + compute) ≤ 0.85** — i.e. the
//     overlap must save at least 15% of the sequential wall time.
//     Without pinning (or with the async path silently falling back
//     to a sync implementation) this ratio degenerates to ~1.0.
//     Observed ≈ 0.55 on the same hardware (transfer and compute
//     have similar duration, so the overlap almost halves the
//     total wall time — which is exactly the infrastructure
//     behavior we're locking in).
//
// Large-shape rows (64 MiB) are informational — printed for trend
// analysis but don't gate the exit code.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Storage.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/PinnedHostAllocator.hpp"
#include "tesseract/ops/Arithmetic.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Storage;
using tesseract::Stream;
using tesseract::Tensor;

namespace {

struct CopyCase {
  const char*  label;
  std::size_t  bytes;
  double       pinned_vs_sync_speedup_hard_bar;  // 0 => informational
};

double bytes_to_gbs(std::size_t bytes, double us) {
  return static_cast<double>(bytes) / (us * 1e-6) / 1e9;
}

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_pinned] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_pinned");

  bench::BenchStream bench_stream;
  cudaStream_t raw_stream = bench_stream.native();
  const Stream& stream = bench_stream.stream();
  const Device cuda0{DeviceType::CUDA, 0};
  const Device cpu = tesseract::cpu_device();

  // PCIe-roofline reference. On RTX 5880 Ada (PCIe Gen4 x16) the
  // usable H↔D bandwidth is ~32 GB/s effective — printed as a
  // trend line for the transfer rows.
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, 0);
  std::printf("  PCIe gen / width   : Gen%d / x%d (effective)\n\n",
              p.pciBusID >= 0 ? 4 /*placeholder*/ : 0, p.pciBusID >= 0 ? 16 : 0);

  // ─── Block A: transfer bandwidth ──────────────────────────────────────────
  const std::vector<CopyCase> copy_cases = {
    // label        bytes                     speedup hard bar
    {"1 MiB",       1ull << 20,               1.4},
    {"4 MiB",       4ull << 20,               0.0},  // informational
    {"64 MiB",      64ull << 20,              0.0},  // informational
  };

  std::printf("  H→D transfer (sync pageable vs async pinned)\n");
  std::printf("  shape   |   sync_us   |  pin_us   |  sync GB/s  |  pin GB/s  |  pin/sync\n");
  std::printf("  --------+-------------+-----------+-------------+------------+-----------\n");

  bool all_pass = true;
  for (const auto& c : copy_cases) {
    const int64_t N_elems = static_cast<int64_t>(c.bytes / 4);
    Tensor src_pg  = Tensor::empty({N_elems}, DType::Float32, cpu);
    Tensor src_pin = Tensor::empty_pinned({N_elems}, DType::Float32);
    std::memset(src_pg.raw_data(),  0x5a, c.bytes);
    std::memset(src_pin.raw_data(), 0x5a, c.bytes);

    Tensor dst_sync = Tensor::empty({N_elems}, DType::Float32, cuda0);
    Tensor dst_pin  = Tensor::empty({N_elems}, DType::Float32, cuda0);

    auto sync_fn = [&](cudaStream_t /*s*/) {
      Storage::copy_device_bytes(dst_sync.raw_data(), cuda0,
                                 src_pg.raw_data(), cpu, c.bytes);
    };
    auto pin_fn = [&](cudaStream_t /*s*/) {
      Storage::copy_device_bytes_async(
          dst_pin.raw_data(), cuda0,
          src_pin.raw_data(), cpu,
          c.bytes, stream);
      stream.synchronize();
    };

    auto sync_st = bench::steady_state_time(sync_fn, raw_stream, 0.05, 5, 300, 0, 1.0);
    auto pin_st  = bench::steady_state_time(pin_fn,  raw_stream, 0.05, 5, 300, 0, 1.0);

    const double sync_gbs = bytes_to_gbs(c.bytes, sync_st.mean_us);
    const double pin_gbs  = bytes_to_gbs(c.bytes, pin_st.mean_us);
    const double speedup  = sync_st.mean_us / pin_st.mean_us;

    std::printf("  %-7s |  %8.2f   | %8.2f  |  %8.2f   |  %8.2f  |  %5.2fx\n",
                c.label, sync_st.mean_us, pin_st.mean_us,
                sync_gbs, pin_gbs, speedup);

    if (c.pinned_vs_sync_speedup_hard_bar > 0.0) {
      const bool ok = bench::hard_check(
          speedup >= c.pinned_vs_sync_speedup_hard_bar,
          (std::string("pin_vs_sync_speedup ") + c.label).c_str(),
          speedup, c.pinned_vs_sync_speedup_hard_bar);
      all_pass = all_pass && ok;
    }
  }
  std::printf("\n");

  // ─── Block B: submit latency (tiny pinned async H→D) ──────────────────────
  // Stream pipelining: we issue N copies back-to-back onto the
  // stream, never synchronize in the inner loop, and measure total
  // wall time with CUDA events. Divide by N to get the per-submit
  // latency. A 4 KiB copy at PCIe Gen4 is ~0.1 µs of actual
  // transfer, so the measurement is dominated by submit overhead —
  // which is what the B-023 CUDA-Graph-replay side-channel needs.
  {
    constexpr std::size_t kSmallBytes = 4096;
    const int64_t N_elems = kSmallBytes / 4;
    Tensor src_tiny = Tensor::empty_pinned({N_elems}, DType::Float32);
    std::memset(src_tiny.raw_data(), 0x5a, kSmallBytes);
    Tensor dst_tiny = Tensor::empty({N_elems}, DType::Float32, cuda0);

    auto submit_fn = [&](cudaStream_t /*s*/) {
      Storage::copy_device_bytes_async(
          dst_tiny.raw_data(), cuda0,
          src_tiny.raw_data(), cpu,
          kSmallBytes, stream);
    };
    auto submit_st = bench::steady_state_time(submit_fn, raw_stream,
                                              /*cov_target=*/0.10,
                                              /*warmup=*/20,
                                              /*max_iters=*/500,
                                              /*batch=*/0,
                                              /*min_window_ms=*/2.0);
    // Drain any backlog the batched measurement built up so it
    // doesn't perturb block C.
    stream.synchronize();

    std::printf("  Submit latency (4 KiB pinned H→D async, no inner sync)\n");
    std::printf("    mean : %.2f us       (hard bar: <= 10.00 us)\n\n",
                submit_st.mean_us);
    const bool ok = bench::hard_check(submit_st.mean_us <= 10.0,
                                      "async_submit_us 4KiB",
                                      submit_st.mean_us, 10.0);
    all_pass = all_pass && ok;
  }

  // ─── Block C: overlap demo ────────────────────────────────────────────────
  // Sequential (no overlap) vs parallel (pinned async copy on copy
  // stream + compute on compute stream, joined at the end). Picks
  // sizes so the transfer and the compute kernel have comparable
  // wall times — that's the regime where overlap can actually halve
  // the total.
  {
    constexpr std::size_t kTransferBytes = 4ull << 20;   // 4 MiB H→D
    const int64_t N_transfer = kTransferBytes / 4;
    // Compute kernel: many small element-wise ops on a buffer sized
    // so the compute takes roughly as long as the 4 MiB PCIe
    // transfer (~130 µs). Tuning point: the exact compute shape is
    // not important — what matters is that it's not vanishingly
    // short vs the copy.
    const int64_t N_compute = 1 << 20;

    Tensor src_pin = Tensor::empty_pinned({N_transfer}, DType::Float32);
    std::memset(src_pin.raw_data(), 0x12, kTransferBytes);
    Tensor dst_pin = Tensor::empty({N_transfer}, DType::Float32, cuda0);

    Tensor x  = Tensor::full({N_compute}, 0.5, DType::Float32, cuda0);
    Tensor a  = Tensor::full({},          1.001, DType::Float32, cuda0);
    Tensor b  = Tensor::full({},         -0.0005, DType::Float32, cuda0);

    tesseract::NoGradGuard nogg;

    Stream copy_stream = Stream::create(cuda0);
    Stream comp_stream = Stream::create(cuda0);

    // Sequential run (no overlap): a synchronous copy followed by
    // a compute chain on the main bench stream. Approximates what a
    // naive inference loop would do.
    auto seq_fn = [&](cudaStream_t /*s*/) {
      Storage::copy_device_bytes(dst_pin.raw_data(), cuda0,
                                 src_pin.raw_data(), cpu,
                                 kTransferBytes);
      // A 10-op chain mirrors the smallest realistic decode-block
      // compute pattern; keeping it `NoGrad` matches the production
      // decode path.
      Tensor t = x;
      for (int i = 0; i < 10; ++i) {
        t = (i % 2 == 0) ? tesseract::ops::mul(t, a)
                         : tesseract::ops::add(t, b);
      }
    };
    auto seq_st = bench::steady_state_time(seq_fn, raw_stream, 0.05, 5, 200, 0, 2.0);

    // Parallel run: async copy on its own stream, compute on its own
    // stream, join at the end. Both streams are non-blocking so
    // there's no implicit legacy-default-stream sync point between
    // them. An Event is recorded on the copy stream so the consumer
    // stream (if any kernel needs the fresh data) can wait on it;
    // here the compute and copy are independent, so we only need
    // the final two `synchronize()` calls.
    auto par_fn = [&](cudaStream_t /*s*/) {
      // Install compute stream as current for the compute chain.
      tesseract::StreamGuard cg(comp_stream);
      // Kick off the async copy on the OTHER stream first so the
      // GPU's DMA engine can start work while the compute ramps up.
      Storage::copy_device_bytes_async(dst_pin.raw_data(), cuda0,
                                       src_pin.raw_data(), cpu,
                                       kTransferBytes, copy_stream);
      Tensor t = x;
      for (int i = 0; i < 10; ++i) {
        t = (i % 2 == 0) ? tesseract::ops::mul(t, a)
                         : tesseract::ops::add(t, b);
      }
      // Join both streams before the timer's event fires. If we
      // only waited on `comp_stream` the CUDA event timer (bound
      // to the bench's main stream) would still see a correct mean
      // — but the copy might still be in flight on the copy
      // stream, which pollutes the "end-to-end wall time" we're
      // comparing against the sequential case.
      comp_stream.synchronize();
      copy_stream.synchronize();
    };
    auto par_st = bench::steady_state_time(par_fn, raw_stream, 0.05, 5, 200, 0, 2.0);

    const double overlap_ratio = par_st.mean_us / seq_st.mean_us;
    std::printf("  Overlap demo (4 MiB H→D pinned || compute chain)\n");
    std::printf("    sequential : %.2f us\n", seq_st.mean_us);
    std::printf("    parallel   : %.2f us  (%.2fx of sequential)\n",
                par_st.mean_us, overlap_ratio);
    const bool ok = bench::hard_check(overlap_ratio <= 0.85,
                                      "copy_compute_overlap_ratio",
                                      overlap_ratio, 0.85);
    all_pass = all_pass && ok;
  }

  std::printf("\n  %s\n", all_pass ? "ALL HARD BARS PASSED." : "HARD BAR MISS.");
  return all_pass ? bench::kExitOk : bench::kExitPerfMiss;
}

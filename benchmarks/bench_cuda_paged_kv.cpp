// Wave 4.5 (B-019b) — bench: paged vs contiguous KV cache. Pins the two
// properties that justify the paging machinery:
//
//   1. Memory residency. The contiguous `KVCache` reserves
//      `2 · B · H · max_len · D_head` elements the moment it is
//      constructed, regardless of how many tokens the request actually
//      emits. `PagedKVCache` allocates physical blocks on demand, so a
//      request holding `L` tokens occupies
//      `2 · B · ceil(L / block_size) · H · block_size · D_head` —
//      proportional to `L`, not `max_len`. For the typical serving
//      regime (long `max_len`, most requests short) this is the whole
//      ball game.
//
//   2. Per-step gather overhead. Paging is not free: because the valid
//      prefix is physically scattered across blocks, `keys_view()` /
//      `values_view()` gather it into a contiguous tensor every step,
//      whereas the contiguous cache returns a zero-copy narrow. We
//      measure that gather cost in isolation and assert it stays a
//      small fraction of a real decode step's compute (a Llama-7B block
//      decode is ~500-1100 µs per `bench_cuda_llama_decode`), so the
//      memory win does not regress decode latency in practice. A
//      block-table-aware paged-attention kernel that reads K/V in place
//      removes the gather entirely — tracked as B-019b+.
//
// Hard bars
// ---------
//   * Memory: paged resident bytes / contiguous reserved bytes ≤ 0.10
//     on the `L=256, max_len=8192` short-request shape (true ratio
//     256/8192 ≈ 0.031; the bar sits 3× above to stay robust to a
//     future block_size bump).
//   * Per-step gather latency: the append+gather for one decode step at
//     `L=512` stays ≤ 250 µs, i.e. < ~50% of a 7B-block decode step —
//     a guard that the gather doesn't blow up into the dominant cost on
//     this shape. (Measured ~tens of µs on SM 8.9.)

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/KVCache.hpp"
#include "tesseract/nn/PagedKVCache.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Tensor;
using tesseract::nn::KVCache;
using tesseract::nn::PagedKVCache;

namespace {

// Reserved bytes for a contiguous cache: both K and V slabs, full
// max_len, fp32.
int64_t contiguous_bytes(int64_t B, int64_t H, int64_t max_len, int64_t Dh) {
  return int64_t{2} * B * H * max_len * Dh * int64_t(sizeof(float));
}

// Resident bytes for a paged cache: both K and V pools, only the
// allocated blocks count.
int64_t paged_resident_bytes(int64_t allocated_blocks, int64_t H,
                             int64_t block_size, int64_t Dh) {
  return int64_t{2} * allocated_blocks * H * block_size * Dh *
         int64_t(sizeof(float));
}

}  // namespace

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_paged_kv] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_paged_kv");

  bench::BenchStream bench_stream;
  cudaStream_t raw_stream = bench_stream.native();
  const Device cuda0{DeviceType::CUDA, 0};

  // Llama-7B head geometry, single request.
  constexpr int64_t B  = 1;
  constexpr int64_t H  = 32;
  constexpr int64_t Dh = 128;
  constexpr int64_t BLK = 16;       // tokens per physical block

  bool all_pass = true;

  // ---- Memory residency -----------------------------------------------------
  // Short request: L=256 tokens against a max_len=8192 context window.
  {
    constexpr int64_t MAX = 8192;
    constexpr int64_t L   = 256;
    // Pool sized for the whole context so a long request would still
    // fit; residency is what we actually allocate for L tokens.
    const int64_t pool_blocks = B * ((MAX + BLK - 1) / BLK);

    PagedKVCache paged(B, H, Dh, MAX, BLK, pool_blocks, DType::Float32, cuda0);
    Tensor k = Tensor::zeros({B, H, L, Dh}, DType::Float32, cuda0);
    Tensor v = Tensor::zeros({B, H, L, Dh}, DType::Float32, cuda0);
    paged.append(k, v);
    bench::check_cuda(cudaStreamSynchronize(raw_stream), "mem-append-sync");

    const int64_t cont_b = contiguous_bytes(B, H, MAX, Dh);
    const int64_t pag_b  =
        paged_resident_bytes(paged.num_allocated_blocks(), H, BLK, Dh);
    const double frac = static_cast<double>(pag_b) / static_cast<double>(cont_b);

    std::printf("  memory @ L=%ld, max_len=%ld, block_size=%ld:\n",
                long(L), long(MAX), long(BLK));
    std::printf("    contiguous reserved : %8.2f MB\n", cont_b / 1.0e6);
    std::printf("    paged resident      : %8.2f MB  (%ld blocks)\n",
                pag_b / 1.0e6, long(paged.num_allocated_blocks()));
    std::printf("    paged / contiguous  : %8.4f\n\n", frac);

    all_pass &= bench::hard_check(
        frac <= 0.10,
        "paged resident / contiguous reserved <= 0.10",
        frac, 0.10);
  }

  // ---- Per-step gather latency ----------------------------------------------
  // Prefill to L+1 so the block backing position L exists, then measure
  // the steady-state cost of one decode step's cache ops: rewind to L,
  // append one token at slot L, and gather both K/V prefixes. The
  // contiguous cache's view is a zero-copy narrow (≈0), so the absolute
  // paged number is the honest "paging tax" per step.
  {
    constexpr int64_t MAX = 4096;
    constexpr int64_t L   = 512;
    const int64_t pool_blocks = B * ((MAX + BLK - 1) / BLK);

    PagedKVCache paged(B, H, Dh, MAX, BLK, pool_blocks, DType::Float32, cuda0);
    KVCache contig(B, H, Dh, MAX, DType::Float32, cuda0);

    Tensor k_pre = Tensor::zeros({B, H, L + 1, Dh}, DType::Float32, cuda0);
    Tensor v_pre = Tensor::zeros({B, H, L + 1, Dh}, DType::Float32, cuda0);
    paged.append(k_pre, v_pre);
    contig.append(k_pre, v_pre);
    bench::check_cuda(cudaStreamSynchronize(raw_stream), "lat-prefill-sync");

    Tensor k1 = Tensor::zeros({B, H, 1, Dh}, DType::Float32, cuda0);
    Tensor v1 = Tensor::zeros({B, H, 1, Dh}, DType::Float32, cuda0);

    auto paged_step = [&](cudaStream_t /*s*/) {
      paged.set_current_len(L);
      paged.append(k1, v1);
      Tensor kk = paged.keys_view();
      Tensor vv = paged.values_view();
      (void)kk; (void)vv;
    };
    auto contig_step = [&](cudaStream_t /*s*/) {
      contig.set_current_len(L);
      contig.append(k1, v1);
      Tensor kk = contig.keys_view();
      Tensor vv = contig.values_view();
      (void)kk; (void)vv;
    };

    auto sp = bench::steady_state_time(paged_step, raw_stream,
                                       /*cov_target=*/0.05, /*warmup=*/10,
                                       /*max_iters=*/400, /*batch=*/0,
                                       /*min_window_ms=*/1.0);
    auto sc = bench::steady_state_time(contig_step, raw_stream,
                                       /*cov_target=*/0.05, /*warmup=*/10,
                                       /*max_iters=*/400, /*batch=*/0,
                                       /*min_window_ms=*/1.0);

    std::printf("  per-step cache ops @ L=%ld (append 1 + gather K/V):\n",
                long(L));
    std::printf("    contiguous (narrow) : %8.2f us\n", sc.mean_us);
    std::printf("    paged (gather)      : %8.2f us\n", sp.mean_us);
    std::printf("    paged - contiguous  : %8.2f us\n\n",
                sp.mean_us - sc.mean_us);

    all_pass &= bench::hard_check(
        sp.mean_us <= 250.0,
        "paged per-step append+gather latency (us) <= 250",
        sp.mean_us, 250.0);
  }

  std::printf("\n  %s\n", all_pass ? "ALL HARD BARS PASSED." : "HARD BAR MISS.");
  return all_pass ? bench::kExitOk : bench::kExitPerfMiss;
}

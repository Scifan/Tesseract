// M4 Track B3 (B-043, occupancy-unlocked) — REAL multi-GPU tensor-parallel
// scaling on physical cards 0..N-1.
//
// Unlike the single-process `SimCommBackend` parity tests (which hold every
// rank's shard in one address space on one device), this benchmark places each
// rank's shard on its **own** CUDA device and runs the per-rank compute there,
// then performs a real cross-device all-reduce (D2D copy + sum) to combine the
// partials. It measures what tensor parallelism actually buys:
//
//   * per-GPU weight memory  = dense / N   (each card holds 1/N of the FFN)
//   * aggregate forward throughput across N cards
//
// Sharding == Megatron SwiGLU MLP: a dense `FeedForward(d_model, d_ff)` equals
// the sum over r of `FeedForward(d_model, d_ff/N)` shards (gate/up are
// column-parallel, down is row-parallel, one all-reduce). We therefore model
// rank r as an independent `FeedForward` with `d_ff/N` and all-reduce the
// per-rank outputs — numerically the Megatron MLP, physically multi-card.
//
// Forward-only: cross-device copy (`Tensor::to`) is a pure-data op in the
// current autograd (no grad-fn), so the TP *backward* parity is pinned by the
// in-process SimCommBackend tests instead. This bench is the inference-side
// scaling story; the wiring of an autograd-aware NCCL all-reduce is the
// remaining production step (documented).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/FeedForward.hpp"
#include "tesseract/ops/Arithmetic.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::NoGradGuard;
using tesseract::Tensor;
using tesseract::nn::FeedForward;
namespace ops = tesseract::ops;

namespace {

int64_t param_bytes(const FeedForward& m) {
  int64_t total = 0;
  for (const auto& [name, t] : m.named_parameters()) total += t.nbytes();
  for (const auto& [name, t] : m.named_buffers()) total += t.nbytes();
  return total;
}

void sync_devices(int n) {
  for (int r = 0; r < n; ++r) {
    bench::check_cuda(cudaSetDevice(r), "cudaSetDevice");
    bench::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }
  bench::check_cuda(cudaSetDevice(0), "cudaSetDevice(0)");
}

struct TpResult {
  int     world;
  double  step_us;
  double  tok_per_s;
  int64_t per_gpu_weight_bytes;
};

}  // namespace

int main() {
  const int avail = bench::visible_cuda_devices();
  if (avail == 0) {
    std::printf("[bench_cuda_tp_scaling] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_tp_scaling");

  // Llama-2-7B FFN shape: d_ff must stay divisible by every world size we try.
  // 11008 is not divisible by 3, so use 12288 (= 4096 * 3) which divides by
  // 1/2/3 cleanly and keeps the per-card footprint representative.
  constexpr int64_t d_model = 4096;
  constexpr int64_t d_ff    = 12288;
  constexpr int64_t T       = 4096;  // tokens per forward (batch*seq)

  const int max_world = std::min(avail, 3);
  std::printf("  config   : d_model=%ld d_ff=%ld tokens=%ld  cards=%d\n",
              long(d_model), long(d_ff), long(T), max_world);
  std::printf("  note     : dense FFN replicated as sum of d_ff/N shards "
              "(Megatron MLP), one cross-device all-reduce.\n\n");

  std::vector<TpResult> results;

  for (int world = 1; world <= max_world; ++world) {
    if (d_ff % world != 0) continue;
    const int64_t d_ff_shard = d_ff / world;

    // Build one FFN shard per rank on its own device.
    std::vector<std::shared_ptr<FeedForward>> shards;
    std::vector<Tensor> x_dev;
    for (int r = 0; r < world; ++r) {
      Device dev{DeviceType::CUDA, r};
      auto ff = std::make_shared<FeedForward>(d_model, d_ff_shard,
                                              /*use_bias=*/false,
                                              DType::Float32);
      ff->to(dev);
      ff->eval();
      shards.push_back(ff);
      Tensor x = Tensor::zeros({T, d_model}, DType::Float32);
      x_dev.push_back(x.to(dev));
    }
    const int64_t per_gpu_bytes = param_bytes(*shards[0]);

    auto run_step = [&]() {
      NoGradGuard nogg;
      std::vector<Tensor> partials;
      partials.reserve(world);
      for (int r = 0; r < world; ++r) {
        partials.push_back(shards[r]->forward(x_dev[r]));
      }
      // Real cross-device all-reduce: bring every partial to card 0 and sum.
      Tensor acc = partials[0];
      for (int r = 1; r < world; ++r) {
        acc = ops::add(acc, partials[r].to(Device{DeviceType::CUDA, 0}));
      }
      return acc;
    };

    for (int i = 0; i < 5; ++i) (void)run_step();
    sync_devices(world);

    constexpr int iters = 30;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) (void)run_step();
    sync_devices(world);
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double step_us = total_us / iters;
    const double tok_s = T / (step_us * 1e-6);
    results.push_back({world, step_us, tok_s, per_gpu_bytes});

    std::printf("  TP=%d : step=%8.1f us  throughput=%9.0f tok/s  "
                "per-GPU weight=%6.1f MB\n",
                world, step_us, tok_s,
                per_gpu_bytes / (1024.0 * 1024.0));
  }

  std::printf("\n  scaling vs TP=1:\n");
  if (!results.empty()) {
    const double base_us = results[0].step_us;
    const int64_t base_bytes = results[0].per_gpu_weight_bytes;
    for (const auto& r : results) {
      std::printf("    TP=%d : %.2fx latency, %.2fx per-GPU memory reduction\n",
                  r.world, base_us / r.step_us,
                  static_cast<double>(base_bytes) / r.per_gpu_weight_bytes);
    }
  }
  std::printf("\n");
  return bench::kExitOk;
}

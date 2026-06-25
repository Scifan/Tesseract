// M2L.1 — bench: end-to-end `nn::TransformerBlock` throughput.
//
// No hard bar. Reports forward-only and forward+backward `tokens / s`
// on a BERT-base-ish config (d_model=512, 8 heads, d_ff=2048, S=1024,
// B=16, FP32). Doubles as the top-line regression dashboard for M2K's
// composite transformer block: whenever we tighten one of the
// underlying primitives (matmul dispatch, elementwise fast-path, fused
// rms_norm / attention) this number should climb monotonically.

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/TransformerBlock.hpp"
#include "tesseract/ops/Reduction.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;

int main() {
  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_transformer_block] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_transformer_block (FP32)");

  bench::BenchStream bench_stream;
  cudaStream_t stream = bench_stream.native();

  using tesseract::Tensor;
  using tesseract::DType;
  using tesseract::Device;
  using tesseract::DeviceType;
  using tesseract::NoGradGuard;
  Device cuda0{DeviceType::CUDA, 0};

  constexpr int64_t B = 16, S = 1024, D = 512, H = 8, Dff = 2048;

  tesseract::nn::TransformerBlock block(D, H, Dff, /*norm_eps=*/1e-5,
                                        /*causal=*/true,
                                        /*use_bias=*/false,
                                        DType::Float32);
  block.to(cuda0);

  auto X_leaf = Tensor::empty({B, S, D}, DType::Float32, cuda0);
  bench::check_cuda(cudaMemsetAsync(X_leaf.raw_data(), 0,
                                    B * S * D * sizeof(float), stream),
                    "memset X");
  X_leaf.set_requires_grad(true);
  bench::check_cuda(cudaStreamSynchronize(stream), "init-sync");

  auto fwd = [&](cudaStream_t /*s*/) {
    NoGradGuard nogg;
    auto y = block.forward(X_leaf);
    (void)y;
  };
  // Reset all gradients before each step. Without this the loop leaks: the
  // engine's gradient *accumulation* (`leaf_grad = ops::add(leaf_grad, gi)`)
  // runs under active grad mode, so a persistent `.grad` chains a new graph
  // node — and every step's 536 MB attention-scores saved tensor — across
  // iterations until the card OOMs. Zeroing the grads each step makes the
  // accumulation take the move (not add) path and mirrors a real training
  // step (zero_grad -> forward -> backward).
  auto zero_all = [&]() {
    block.zero_grad();
    if (auto* am = X_leaf.mutable_autograd_meta()) am->grad = Tensor{};
  };
  auto fwd_bwd = [&](cudaStream_t /*s*/) {
    zero_all();
    auto y = block.forward(X_leaf);
    auto loss = tesseract::ops::sum(y);
    tesseract::Engine::backward(loss);
  };

  fwd(stream);
  bench::check_cuda(cudaStreamSynchronize(stream), "warm-fwd");
  auto s_fwd = bench::steady_state_time(fwd, stream,
                                        /*cov_target=*/0.03);

  fwd_bwd(stream);
  bench::check_cuda(cudaStreamSynchronize(stream), "warm-bwd");
  auto s_bwd = bench::steady_state_time(fwd_bwd, stream,
                                        /*cov_target=*/0.03);

  const double tokens = static_cast<double>(B) * S;
  const double tps_fwd = tokens / (s_fwd.mean_us * 1e-6);
  const double tps_bwd = tokens / (s_bwd.mean_us * 1e-6);
  std::printf("  config           : d_model=%ld heads=%ld d_ff=%ld  B=%ld S=%ld\n",
              long(D), long(H), long(Dff), long(B), long(S));
  std::printf("  forward only     : %9.2f µs  = %9.0f tok/s\n",
              s_fwd.mean_us, tps_fwd);
  std::printf("  forward+backward : %9.2f µs  = %9.0f tok/s\n",
              s_bwd.mean_us, tps_bwd);
  std::printf("  bwd / fwd        : %5.2f\n\n", s_bwd.mean_us / s_fwd.mean_us);
  std::printf("  (no hard bar — top-line dashboard; see "
              "docs/benchmarks/m2-cuda.md for history.)\n");

  
  return bench::kExitOk;
}

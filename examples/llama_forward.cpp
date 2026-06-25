// Minimal single-block Llama-style transformer forward demo.
//
// Builds one `nn::TransformerBlock` (RMSNorm → MHA w/ causal mask →
// RMSNorm → SwiGLU FFN, both sub-layers pre-norm + residual),
// drives a randomly-initialized input through it, and prints a few
// basic sanity numbers (output shape, finite count, a sum-loss +
// gradient norm). The default config is a toy size (d_model=64,
// heads=4, d_ff=128) so the example runs end-to-end in well under
// a second on either CPU or CUDA.
//
// Usage:
//   ./examples/tesseract_llama_forward [--device cpu|cuda] [--backward]
//                                      [--d-model N] [--heads N]
//                                      [--d-ff N] [--batch N] [--seq N]
//
// Design intent: the smallest standalone binary that exercises the
// entire M2K primitive chain (ops::rms_norm, ops::attention,
// ops::sigmoid/mul/matmul/add composing SwiGLU, residual adds) in
// eager mode, on both the CPU and CUDA backends, with autograd
// enabled. A companion test (`tests/nn/test_transformer_block.cpp`)
// pins numerical parity between the two devices.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/TransformerBlock.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/utils/Logging.hpp"

namespace {

// Deterministic fill with N(0, 1/√fan_in)-style Gaussian so the first
// forward pass has well-scaled activations. We don't rely on any
// library RNG here — the point of the demo is the transformer block,
// not distribution fidelity. `seed` is exposed so the script stays
// reproducible run-to-run.
tesseract::Tensor random_normal(tesseract::Shape shape, double stddev, uint64_t seed) {
  auto t = tesseract::Tensor::empty(shape, tesseract::DType::Float32, tesseract::cpu_device());
  float* p = t.data_ptr<float>();
  const int64_t n = t.numel();
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.0f, static_cast<float>(stddev));
  for (int64_t i = 0; i < n; ++i) p[i] = nd(rng);
  return t;
}

int64_t count_finite(const tesseract::Tensor& host) {
  const float* p = host.data_ptr<float>();
  int64_t ok = 0;
  for (int64_t i = 0; i < host.numel(); ++i) {
    if (std::isfinite(p[i])) ++ok;
  }
  return ok;
}

double l2_norm(const tesseract::Tensor& host) {
  const float* p = host.data_ptr<float>();
  double acc = 0.0;
  for (int64_t i = 0; i < host.numel(); ++i) acc += static_cast<double>(p[i]) * p[i];
  return std::sqrt(acc);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace tesseract;

  std::string device_cli = "cpu";
  int64_t d_model = 64;
  int64_t heads   = 4;
  int64_t d_ff    = 128;
  int64_t batch   = 2;
  int64_t seq     = 16;
  bool run_backward = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--device" && i + 1 < argc)    device_cli = argv[++i];
    else if (a == "--d-model" && i + 1 < argc) d_model = std::atoll(argv[++i]);
    else if (a == "--heads" && i + 1 < argc)   heads   = std::atoll(argv[++i]);
    else if (a == "--d-ff" && i + 1 < argc)    d_ff    = std::atoll(argv[++i]);
    else if (a == "--batch" && i + 1 < argc)   batch   = std::atoll(argv[++i]);
    else if (a == "--seq" && i + 1 < argc)     seq     = std::atoll(argv[++i]);
    else if (a == "--backward")                run_backward = true;
    else if (a == "--help" || a == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--device cpu|cuda] [--backward]"
                   " [--d-model N] [--heads N] [--d-ff N]"
                   " [--batch N] [--seq N]\n";
      return 0;
    } else {
      std::cerr << "error: unknown arg '" << a << "' (see --help)\n";
      return 1;
    }
  }

  Device run_device = cpu_device();
  if (device_cli == "cuda") {
    run_device = Device{DeviceType::CUDA, 0};
  } else if (device_cli != "cpu") {
    std::cerr << "error: unknown --device '" << device_cli
              << "' (expected 'cpu' or 'cuda')\n";
    return 1;
  }

  std::cout << "[llama_forward] config:\n"
            << "  device  = " << run_device.to_string() << "\n"
            << "  batch   = " << batch   << "\n"
            << "  seq     = " << seq     << "\n"
            << "  d_model = " << d_model << "\n"
            << "  heads   = " << heads   << "\n"
            << "  d_ff    = " << d_ff    << "\n"
            << "  backward= " << (run_backward ? "yes" : "no") << "\n";

  // Build the block and migrate it to the requested device. Llama
  // convention: biases off, causal attention on, norm eps = 1e-5.
  auto block = std::make_shared<nn::TransformerBlock>(
      d_model, heads, d_ff, /*norm_eps=*/1e-5,
      /*causal=*/true, /*use_bias=*/false);
  block->to(run_device);

  // Input: N(0, 1). Host-side construction, then one bulk transfer.
  Tensor x_host = random_normal({batch, seq, d_model}, /*stddev=*/1.0, /*seed=*/12345);
  Tensor x = run_device.is_cpu() ? x_host : x_host.to(run_device);
  x.set_requires_grad(run_backward);

  // Forward.
  Tensor out = block->forward(x);
  std::cout << "[llama_forward] out.shape = " << out.shape().to_string() << "\n";

  // Move output back to host for the sanity inspection prints.
  const Tensor out_host = run_device.is_cpu() ? out : out.to(cpu_device());
  std::cout << "[llama_forward] finite  = "
            << count_finite(out_host) << " / " << out_host.numel() << "\n";
  std::cout << "[llama_forward] ||out||_2 = " << l2_norm(out_host) << "\n";

  if (run_backward) {
    // Simplest possible scalar loss so `backward()` has something to
    // seed: sum(out). Autograd populates `.grad` on every parameter +
    // on `x` (since `requires_grad=true`).
    Tensor loss = ops::sum(out);
    Engine::backward(loss);

    // Per-parameter gradient norm. Keeps the demo's output compact; a
    // full param-name-annotated dump is deferred to a future
    // `nn::Module::named_parameters()` pass (backlog).
    double total_g = 0.0;
    int64_t total_n = 0;
    for (const auto& p : block->parameters()) {
      const auto* am = p.autograd_meta();
      if (am == nullptr || !am->grad.defined()) continue;
      const Tensor g_host = run_device.is_cpu() ? am->grad : am->grad.to(cpu_device());
      const int64_t nf = count_finite(g_host);
      total_g += l2_norm(g_host);
      total_n += nf;
    }
    std::cout << "[llama_forward] backward: sum(||grad||) = " << total_g
              << ", finite_grad_elems = " << total_n << "\n";
  }

  std::cout << "[llama_forward] done.\n";
  return 0;
}

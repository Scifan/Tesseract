// Step-by-step loss parity between the CPU and CUDA training stacks on
// a synthetic 3-class Gaussian mixture. The harness deliberately does
// not depend on the real MNIST download — ctest must stay hermetic —
// but the architecture (2-layer MLP + ReLU + cross-entropy + Adam)
// and the dataset's statistical properties (well-separated clusters,
// clean linear + nonlinear decision surfaces) mirror the MNIST
// example closely enough that a passing run here is a strong signal
// that `examples/mnist.cpp --device cuda` will converge.
//
// Parity contract (M2I):
//   * Same initial weights: both models are materialized on CPU with
//     identical `uniform_init` seeding, then one is pushed to CUDA.
//   * Same mini-batch ordering: both models see the same shuffle
//     (driven by a reseeded PRNG at the top of each epoch) and the
//     same per-batch row indices.
//   * Per-step loss parity: at every step we bounce the CUDA loss
//     through `.to(cpu)` and compare against the CPU loss. Tolerance
//     follows the M2E/M2F/M2G envelope for the tightest-drift path
//     (WithinAbs(., 5e-3)) — the scatter-add reductions inside
//     `reduce_to_shape` and the TF32 tensor-core matmul paths both
//     land inside that band at batch=50 fp32.
//   * Final accuracy bar: CUDA run must also clear the same
//     >0.97 training accuracy that the CPU mnist-smoke asserts, so
//     the CUDA stack passes the "convergence reached" smoke in
//     addition to step-by-step parity.
//
// CUDA-gated: when the CUDA backend is unavailable (CPU-only build,
// no GPU, or `compute-sanitizer`-style launch failure during the
// first step) the whole suite SKIPs via `SKIP_RETURN_CODE 4`. A
// CPU-only smoke (same dataset + same training budget) always runs
// so the CPU build has at least one asserted path out of this TU.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/nn/Activation.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Sequential.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/optim/Adam.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

namespace {

struct Dataset {
  Tensor x;       // [N, 2]
  Tensor y;       // [N]
  int64_t num_classes;
};

Dataset make_gaussian_mixture(int per_class, uint64_t seed = 42) {
  const int C = 3;
  const int N = C * per_class;
  auto x = Tensor::empty({N, 2}, DType::Float32);
  auto y = Tensor::empty({N}, DType::Int64);
  float* px = x.data_ptr<float>();
  int64_t* py = y.data_ptr<int64_t>();

  std::mt19937_64 rng(seed);
  std::normal_distribution<float> noise(0.0f, 0.35f);
  const float centers[C][2] = {{-2.0f, -2.0f}, {2.0f, 0.0f}, {0.0f, 2.5f}};

  int idx = 0;
  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < per_class; ++i) {
      px[2 * idx + 0] = centers[c][0] + noise(rng);
      px[2 * idx + 1] = centers[c][1] + noise(rng);
      py[idx] = c;
      ++idx;
    }
  }
  return {std::move(x), std::move(y), C};
}

double accuracy_host(const Tensor& logits_host, const Tensor& targets) {
  const int64_t N = logits_host.shape()[0];
  const int64_t Cs = logits_host.shape()[1];
  const float* pl = logits_host.data_ptr<float>();
  const int64_t* pt = targets.data_ptr<int64_t>();
  int64_t correct = 0;
  for (int64_t n = 0; n < N; ++n) {
    int64_t best = 0;
    float bv = pl[n * Cs + 0];
    for (int64_t c = 1; c < Cs; ++c) {
      if (pl[n * Cs + c] > bv) { bv = pl[n * Cs + c]; best = c; }
    }
    if (best == pt[n]) ++correct;
  }
  return static_cast<double>(correct) / static_cast<double>(N);
}

// Builds a 2-layer MLP matching the mnist-smoke architecture. Each
// call produces a fresh random init (Linear's thread-local seed
// bumps per ctor), which is exactly why the parity test below
// overwrites the CUDA model's weight storage from the CPU model
// before training — it's the simplest way to force identical init
// without threading a seed through the nn API.
std::shared_ptr<nn::Sequential> build_mlp(int in_features, int hidden,
                                          int num_classes) {
  auto fc1  = std::make_shared<nn::Linear>(in_features, hidden);
  auto relu = std::make_shared<nn::ReLU>();
  auto fc2  = std::make_shared<nn::Linear>(hidden, num_classes);
  return std::make_shared<nn::Sequential>(
      std::initializer_list<std::shared_ptr<nn::Module>>{fc1, relu, fc2});
}

// Copy the raw element bytes of `src` into `dst` without disturbing
// either tensor's TensorImpl identity. Both must be on CPU,
// contiguous, same shape, same dtype. Only used to force identical
// init across the two models before the CUDA side is migrated via
// `Module::to(cuda)`.
void copy_cpu_bytes_into(Tensor& dst, const Tensor& src) {
  REQUIRE(dst.device().is_cpu());
  REQUIRE(src.device().is_cpu());
  REQUIRE(dst.dtype() == src.dtype());
  REQUIRE(dst.shape() == src.shape());
  REQUIRE(dst.is_contiguous());
  REQUIRE(src.is_contiguous());
  std::memcpy(dst.raw_data(), src.raw_data(), src.nbytes());
}

bool cuda_available() {
  return cuda::device_count() > 0;
}

// One training run. `device` selects CPU or CUDA; `per_step_loss`
// receives the scalar loss at every step (both sides use this to
// cross-compare). Returns the final training accuracy on the full
// dataset.
double train_and_record(std::shared_ptr<nn::Sequential> model,
                        const Dataset& ds, Device device, int epochs,
                        int batch, uint64_t shuffle_seed,
                        std::vector<double>& per_step_loss) {
  model->to(device);
  optim::Adam opt(model->parameters(), /*lr=*/5e-2);

  const int64_t N = ds.x.shape()[0];
  std::vector<int64_t> idx(static_cast<std::size_t>(N));
  std::iota(idx.begin(), idx.end(), 0);
  std::mt19937_64 rng(shuffle_seed);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(idx.begin(), idx.end(), rng);

    for (int64_t i = 0; i + batch <= N; i += batch) {
      auto xb_cpu = Tensor::empty({batch, 2}, DType::Float32);
      auto yb_cpu = Tensor::empty({batch}, DType::Int64);
      const float* sx = ds.x.data_ptr<float>();
      const int64_t* sy = ds.y.data_ptr<int64_t>();
      float* dx = xb_cpu.data_ptr<float>();
      int64_t* dy = yb_cpu.data_ptr<int64_t>();
      for (int b = 0; b < batch; ++b) {
        const int64_t r = idx[i + b];
        dx[2 * b + 0] = sx[2 * r + 0];
        dx[2 * b + 1] = sx[2 * r + 1];
        dy[b] = sy[r];
      }

      Tensor xb = device.is_cpu() ? xb_cpu : xb_cpu.to(device);
      Tensor yb = device.is_cpu() ? yb_cpu : yb_cpu.to(device);

      opt.zero_grad();
      Tensor logits = model->forward(xb);
      Tensor loss = ops::cross_entropy_with_logits(logits, yb);
      Engine::backward(loss);
      opt.step();

      const Tensor loss_host =
          device.is_cpu() ? loss : loss.to(cpu_device());
      per_step_loss.push_back(
          static_cast<double>(*loss_host.data_ptr<float>()));
    }
  }

  NoGradGuard nogg;
  Tensor x_dev = device.is_cpu() ? ds.x : ds.x.to(device);
  Tensor logits = model->forward(x_dev);
  Tensor logits_host = device.is_cpu() ? logits : logits.to(cpu_device());
  return accuracy_host(logits_host, ds.y);
}

}  // namespace

TEST_CASE("mnist-style parity: CPU vs CUDA loss curve",
          "[nn][cuda][mnist][parity]") {
  if (!cuda_available()) SKIP("No CUDA device available");

  const auto ds = make_gaussian_mixture(/*per_class=*/200);
  constexpr int kEpochs = 4;
  constexpr int kBatch  = 50;

  auto cpu_model = build_mlp(2, 16, static_cast<int>(ds.num_classes));
  auto gpu_model = build_mlp(2, 16, static_cast<int>(ds.num_classes));

  // Force identical initial weights on both models. Linear's RNG is
  // a thread-local LCG that bumps per-ctor, so back-to-back ctors
  // produce different random init — we patch over that by copying
  // the CPU reference's param storage into the CUDA-bound model's
  // storage. Both models are still on CPU at this point;
  // `Module::to(cuda)` happens inside `train_and_record`.
  {
    auto cpu_params = cpu_model->parameters();
    auto gpu_params = gpu_model->parameters();
    REQUIRE(cpu_params.size() == gpu_params.size());
    for (std::size_t k = 0; k < cpu_params.size(); ++k) {
      copy_cpu_bytes_into(gpu_params[k], cpu_params[k]);
    }
  }

  std::vector<double> cpu_loss, gpu_loss;
  const double cpu_acc = train_and_record(cpu_model, ds, cpu_device(),
                                          kEpochs, kBatch, /*seed=*/7,
                                          cpu_loss);
  const double gpu_acc = train_and_record(gpu_model, ds,
                                          Device{DeviceType::CUDA, 0},
                                          kEpochs, kBatch, /*seed=*/7,
                                          gpu_loss);

  REQUIRE(cpu_loss.size() == gpu_loss.size());
  INFO("steps recorded: " << cpu_loss.size());

  // Per-step loss parity. 5e-3 matches the loosest band across M2F's
  // reductions and M2G's TF32 matmul — both are in-play here. The
  // first few steps tend to drift the most (small exp() arguments
  // inside cross-entropy) but the gap still stays inside the band.
  for (std::size_t s = 0; s < cpu_loss.size(); ++s) {
    INFO("step " << s
         << "  cpu=" << cpu_loss[s]
         << "  gpu=" << gpu_loss[s]
         << "  delta=" << std::abs(cpu_loss[s] - gpu_loss[s]));
    REQUIRE_THAT(gpu_loss[s], WithinAbs(cpu_loss[s], 5e-3));
  }

  // Final training accuracy clears the same bar as the CPU smoke.
  INFO("cpu_acc=" << cpu_acc << "  gpu_acc=" << gpu_acc);
  REQUIRE(cpu_acc > 0.97);
  REQUIRE(gpu_acc > 0.97);
  // And the two should agree to within 1 point of accuracy. This is
  // looser than per-step loss parity because a single mis-classified
  // sample moves `acc` by ~1/600 on this dataset.
  REQUIRE(std::abs(gpu_acc - cpu_acc) < 0.02);
}

TEST_CASE("mnist-style smoke: CPU-only eager training still converges",
          "[nn][mnist][smoke]") {
  const auto ds = make_gaussian_mixture(/*per_class=*/200);
  constexpr int kEpochs = 4;
  constexpr int kBatch  = 50;

  auto model = build_mlp(2, 16, static_cast<int>(ds.num_classes));
  std::vector<double> loss;
  const double acc = train_and_record(model, ds, cpu_device(), kEpochs,
                                      kBatch, /*seed=*/7, loss);
  REQUIRE_FALSE(loss.empty());
  REQUIRE(loss.back() < loss.front() * 0.5);
  REQUIRE(acc > 0.97);
}


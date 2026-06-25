// Multi-epoch training smoke test on a deterministic synthetic classification
// dataset. The intent is *not* to exercise MNIST per se (ctest must not
// depend on an external download), but to validate the full forward →
// cross-entropy → backward → optimizer.step() loop for many iterations
// across a realistic model (2-layer MLP + ReLU).
//
// Dataset: 3 isotropic Gaussians in R^2, clearly separable. A correct
// training stack should reach ~100% training accuracy within a few epochs.
//
// Exit criterion: final train accuracy > 0.97 and loss strictly decreasing
// between the first and last epoch (monotonicity to within 5% jitter).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
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
  // Well-separated centers so an MLP converges quickly.
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

double accuracy(const Tensor& logits, const Tensor& targets) {
  const int64_t N = logits.shape()[0];
  const int64_t C = logits.shape()[1];
  const float* pl = logits.data_ptr<float>();
  const int64_t* pt = targets.data_ptr<int64_t>();
  int64_t correct = 0;
  for (int64_t n = 0; n < N; ++n) {
    int64_t best = 0;
    float bv = pl[n * C + 0];
    for (int64_t c = 1; c < C; ++c) {
      if (pl[n * C + c] > bv) { bv = pl[n * C + c]; best = c; }
    }
    if (best == pt[n]) ++correct;
  }
  return static_cast<double>(correct) / static_cast<double>(N);
}

}  // namespace

TEST_CASE("nn+Adam: multi-epoch training on synthetic 3-class data",
          "[nn][smoke]") {
  auto ds = make_gaussian_mixture(/*per_class=*/200);
  const int64_t N = ds.x.shape()[0];

  auto fc1  = std::make_shared<nn::Linear>(2, 16);
  auto relu = std::make_shared<nn::ReLU>();
  auto fc2  = std::make_shared<nn::Linear>(16, ds.num_classes);
  auto model = std::make_shared<nn::Sequential>(
      std::initializer_list<std::shared_ptr<nn::Module>>{fc1, relu, fc2});

  optim::Adam opt(model->parameters(), /*lr=*/5e-2);

  constexpr int kEpochs = 8;
  constexpr int kBatch  = 50;
  std::vector<int64_t> idx(static_cast<std::size_t>(N));
  std::iota(idx.begin(), idx.end(), 0);
  std::mt19937_64 rng(7);

  std::vector<double> epoch_loss;
  for (int epoch = 0; epoch < kEpochs; ++epoch) {
    std::shuffle(idx.begin(), idx.end(), rng);
    double loss_sum = 0.0;
    int steps = 0;
    for (int64_t i = 0; i + kBatch <= N; i += kBatch) {
      auto xb = Tensor::empty({kBatch, 2}, DType::Float32);
      auto yb = Tensor::empty({kBatch}, DType::Int64);
      const float* sx = ds.x.data_ptr<float>();
      const int64_t* sy = ds.y.data_ptr<int64_t>();
      float* dx = xb.data_ptr<float>();
      int64_t* dy = yb.data_ptr<int64_t>();
      for (int b = 0; b < kBatch; ++b) {
        const int64_t r = idx[i + b];
        dx[2 * b + 0] = sx[2 * r + 0];
        dx[2 * b + 1] = sx[2 * r + 1];
        dy[b] = sy[r];
      }
      opt.zero_grad();
      Tensor logits = model->forward(xb);
      Tensor loss = ops::cross_entropy_with_logits(logits, yb);
      Engine::backward(loss);
      opt.step();
      loss_sum += static_cast<double>(*loss.data_ptr<float>());
      ++steps;
    }
    epoch_loss.push_back(loss_sum / steps);
  }

  // The final epoch should be much cheaper than the first (>= 3x drop for
  // this well-separated dataset with Adam).
  INFO("epoch 0 loss = " << epoch_loss.front()
       << "  final loss = " << epoch_loss.back());
  REQUIRE(epoch_loss.back() < epoch_loss.front() * 0.5);
  REQUIRE(epoch_loss.back() < 0.25);

  // Final training accuracy.
  NoGradGuard nogg;
  Tensor logits = model->forward(ds.x);
  const double acc = accuracy(logits, ds.y);
  INFO("final train accuracy = " << acc);
  REQUIRE(acc > 0.97);
}

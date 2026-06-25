#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Sequential.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/optim/Adam.hpp"
#include "tesseract/optim/SGD.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

namespace {

// Helper: MSE loss = mean((y_hat - y)^2).
Tensor mse(const Tensor& y_hat, const Tensor& y) {
  Tensor d = ops::sub(y_hat, y);
  Tensor sq = ops::mul(d, d);
  return ops::mean(sq);
}

// Generate a linear dataset y = a*x + b with N samples, x ~ uniform(-1, 1).
void make_dataset(int N, float a, float b, Tensor& x, Tensor& y) {
  x = Tensor::empty({N, 1}, DType::Float32);
  y = Tensor::empty({N, 1}, DType::Float32);
  uint64_t s = 0xABCDEF0123456789ULL;
  float* px = x.data_ptr<float>();
  float* py = y.data_ptr<float>();
  for (int i = 0; i < N; ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const float u =
        static_cast<float>(static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFULL) /
                           9007199254740992.0);
    px[i] = 2.0f * u - 1.0f;
    py[i] = a * px[i] + b;
  }
}

}  // namespace

TEST_CASE("nn+SGD: fit y = 3x + 2", "[nn][optim][regression]") {
  constexpr int kN = 64;
  Tensor x, y;
  make_dataset(kN, 3.0f, 2.0f, x, y);

  auto model = std::make_shared<nn::Linear>(/*in=*/1, /*out=*/1);
  optim::SGD opt(model->parameters(), /*lr=*/0.1, /*momentum=*/0.0);

  double last_loss = 0.0;
  for (int step = 0; step < 2000; ++step) {
    opt.zero_grad();
    Tensor pred = model->forward(x);
    Tensor loss = mse(pred, y);
    Engine::backward(loss);
    opt.step();
    last_loss = static_cast<double>(*loss.data_ptr<float>());
    if (last_loss < 1e-6) break;
  }

  REQUIRE(last_loss < 1e-4);
  const float w = *model->weight().data_ptr<float>();
  const float b = *model->bias().data_ptr<float>();
  REQUIRE_THAT(w, WithinAbs(3.0, 1e-2));
  REQUIRE_THAT(b, WithinAbs(2.0, 1e-2));
}

TEST_CASE("nn+Adam: fit y = 3x + 2", "[nn][optim][regression]") {
  constexpr int kN = 64;
  Tensor x, y;
  make_dataset(kN, 3.0f, 2.0f, x, y);

  auto model = std::make_shared<nn::Linear>(1, 1);
  optim::Adam opt(model->parameters(), /*lr=*/0.1);

  double last_loss = 0.0;
  for (int step = 0; step < 2000; ++step) {
    opt.zero_grad();
    Tensor pred = model->forward(x);
    Tensor loss = mse(pred, y);
    Engine::backward(loss);
    opt.step();
    last_loss = static_cast<double>(*loss.data_ptr<float>());
    if (last_loss < 1e-8) break;
  }

  REQUIRE(last_loss < 1e-4);
  const float w = *model->weight().data_ptr<float>();
  const float b = *model->bias().data_ptr<float>();
  REQUIRE_THAT(w, WithinAbs(3.0, 1e-2));
  REQUIRE_THAT(b, WithinAbs(2.0, 1e-2));
}

TEST_CASE("nn::Sequential forward dispatches in order", "[nn]") {
  auto a = std::make_shared<nn::Linear>(2, 3);
  auto b = std::make_shared<nn::Linear>(3, 1);
  auto seq = std::make_shared<nn::Sequential>(
      std::initializer_list<std::shared_ptr<nn::Module>>{a, b});
  auto params = seq->parameters();
  // Linear(2,3): weight[3,2] + bias[3] = 2; Linear(3,1): weight[1,3] + bias[1] = 2
  REQUIRE(params.size() == 4);

  auto x = Tensor::zeros({5, 2}, DType::Float32);
  auto y = seq->forward(x);
  REQUIRE(y.shape() == Shape({5, 1}));
}

TEST_CASE("Linear shapes", "[nn]") {
  nn::Linear fc(3, 4);
  // PyTorch convention: weight is stored as [out, in].
  REQUIRE(fc.weight().shape() == Shape({4, 3}));
  REQUIRE(fc.bias().shape()   == Shape({4}));
  auto x = Tensor::zeros({7, 3}, DType::Float32);
  auto y = fc.forward(x);
  REQUIRE(y.shape() == Shape({7, 4}));
}

// M4 Phase 8 — numerical backward parity: graph `build_backward` vs eager
// autograd. The IR backward rules (softmax over any dim, sum over any dim
// with/without keepdim) are mirrored between `src/graph/Autograd.cpp` and the
// MLIR `--tesseract-backward` pass; the existing coverage was FileCheck-only
// (structure). This test executes the captured backward graph through the
// interpreter and asserts the input gradient matches `autograd::Engine`
// numerically — a real value check, not just IR shape.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Autograd.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/graph/Interpreter.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/ops/Softmax.hpp"

using namespace tesseract;

namespace {

Tensor make_random(std::initializer_list<int64_t> dims, uint64_t seed) {
  auto t = Tensor::empty(dims, DType::Float32);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
  return t;
}

void require_close(const Tensor& a, const Tensor& b, double tol = 1e-5) {
  REQUIRE(a.numel() == b.numel());
  const Tensor ac = a.contiguous();
  const Tensor bc = b.contiguous();
  const float* pa = ac.data_ptr<float>();
  const float* pb = bc.data_ptr<float>();
  for (int64_t i = 0; i < ac.numel(); ++i) {
    const double d = std::fabs(static_cast<double>(pa[i]) - pb[i]);
    INFO("elem " << i << " graph=" << pa[i] << " eager=" << pb[i]
                 << " |d|=" << d);
    REQUIRE(d <= tol);
  }
}

// Eager reference gradient of  loss = sum(softmax(x, dim) * w)  w.r.t. x.
Tensor eager_softmax_grad(const Tensor& x_data, const Tensor& w, int64_t dim) {
  Tensor x = x_data.clone();
  x.set_requires_grad(true);
  Tensor y = ops::softmax(x, dim);
  Tensor loss = ops::sum(ops::mul(y, w));
  Engine::backward(loss);
  return x.grad().contiguous();
}

// Graph gradient of the same program via build_backward + interpreter.
Tensor graph_softmax_grad(const Tensor& x_data, const Tensor& w, int64_t dim) {
  graph::Graph g;
  graph::ValueId x_id = 0, w_id = 0;
  {
    graph::GraphScope scope;
    x_id = graph::bind_param(x_data, "x");
    w_id = graph::bind_input(w, "w");
    Tensor y = ops::softmax(x_data, dim);
    Tensor loss = ops::sum(ops::mul(y, w));
    graph::mark_output(loss);
    g = std::move(scope.graph_);
  }
  graph::BackwardResult bwd = graph::build_backward(g);
  REQUIRE(bwd.dparams.size() == 1);

  std::unordered_map<graph::ValueId, Tensor> bind;
  bind.emplace(x_id, x_data);
  bind.emplace(w_id, w);
  bind.emplace(bwd.cotangents[0], Tensor::ones({}, DType::Float32));
  std::vector<Tensor> outs = graph::run(g, bind);
  // outs = [forward outputs..., dparams...]; the single dparam is last.
  return outs.back().contiguous();
}

// Eager reference gradient of  loss = sum(sum(x, dim, keepdim) * w)  w.r.t. x.
Tensor eager_sum_grad(const Tensor& x_data, const Tensor& w, int64_t dim,
                      bool keepdim) {
  Tensor x = x_data.clone();
  x.set_requires_grad(true);
  Tensor r = ops::sum(x, dim, keepdim);
  Tensor loss = ops::sum(ops::mul(r, w));
  Engine::backward(loss);
  return x.grad().contiguous();
}

Tensor graph_sum_grad(const Tensor& x_data, const Tensor& w, int64_t dim,
                      bool keepdim) {
  graph::Graph g;
  graph::ValueId x_id = 0, w_id = 0;
  {
    graph::GraphScope scope;
    x_id = graph::bind_param(x_data, "x");
    w_id = graph::bind_input(w, "w");
    Tensor r = ops::sum(x_data, dim, keepdim);
    Tensor loss = ops::sum(ops::mul(r, w));
    graph::mark_output(loss);
    g = std::move(scope.graph_);
  }
  graph::BackwardResult bwd = graph::build_backward(g);
  REQUIRE(bwd.dparams.size() == 1);
  std::unordered_map<graph::ValueId, Tensor> bind;
  bind.emplace(x_id, x_data);
  bind.emplace(w_id, w);
  bind.emplace(bwd.cotangents[0], Tensor::ones({}, DType::Float32));
  std::vector<Tensor> outs = graph::run(g, bind);
  return outs.back().contiguous();
}

}  // namespace

TEST_CASE("backward parity: softmax over each dim matches eager autograd",
          "[ir][backward][parity]") {
  Tensor x = make_random({4, 5, 6}, 0xA1);
  for (int64_t dim : {0, 1, 2, -1}) {
    // Per-element upstream cotangent (non-degenerate: sum(softmax) alone has
    // an analytically-zero grad, which would hide bugs).
    Tensor w = make_random({4, 5, 6}, 0xB2 + static_cast<uint64_t>(dim + 4));
    Tensor g_graph = graph_softmax_grad(x, w, dim);
    Tensor g_eager = eager_softmax_grad(x, w, dim);
    require_close(g_graph, g_eager, 1e-5);
  }
}

TEST_CASE("backward parity: sum over each dim (keepdim) matches eager autograd",
          "[ir][backward][parity]") {
  Tensor x = make_random({3, 4, 5}, 0xC3);
  for (int64_t dim : {0, 1, 2}) {
    for (bool keepdim : {false, true}) {
      // w shape matches the reduced output.
      const std::vector<int64_t> rshape =
          keepdim
              ? std::vector<int64_t>{dim == 0 ? 1 : 3, dim == 1 ? 1 : 4,
                                     dim == 2 ? 1 : 5}
              : (dim == 0   ? std::vector<int64_t>{4, 5}
                 : dim == 1 ? std::vector<int64_t>{3, 5}
                            : std::vector<int64_t>{3, 4});
      Tensor w = Tensor::empty(Shape(rshape), DType::Float32);
      std::mt19937_64 rng(0xD4 + static_cast<uint64_t>(dim) +
                          (keepdim ? 100u : 0u));
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      for (int64_t i = 0; i < w.numel(); ++i) w.data_ptr<float>()[i] = dist(rng);

      Tensor g_graph = graph_sum_grad(x, w, dim, keepdim);
      Tensor g_eager = eager_sum_grad(x, w, dim, keepdim);
      require_close(g_graph, g_eager, 1e-5);
    }
  }
}

TEST_CASE("backward parity: sum reduce-all matches eager autograd",
          "[ir][backward][parity]") {
  Tensor x = make_random({3, 4}, 0xE5);
  Tensor w = Tensor::ones({}, DType::Float32);
  // loss = sum(x) * 1 ; dx = ones.
  Tensor g_graph = graph_sum_grad(x, w, -1, false);
  Tensor g_eager = eager_sum_grad(x, w, -1, false);
  require_close(g_graph, g_eager, 1e-5);
}

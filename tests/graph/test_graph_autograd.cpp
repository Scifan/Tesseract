// M1I.2 — graph-level autograd + interpreter parity with eager mode.
//
// These tests capture a small forward graph inside `graph::GraphScope`,
// call `graph::build_backward` to extend it with reverse-mode AD, then
// execute the extended graph via `graph::run` and compare the resulting
// parameter gradients against the equivalent eager-mode tape run. The
// two paths must match exactly (within a tight float tolerance) for the
// graph interpreter to be considered correct.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <unordered_map>
#include <vector>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Autograd.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/graph/Interpreter.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/ops/View.hpp"

using namespace tesseract;
using Catch::Approx;

namespace {

// Element-wise max-|diff| between two float tensors. Materializes both to
// contiguous layout first so that tensors produced by view-only ops
// (transpose, etc.) get traversed in logical rather than storage order.
float max_abs_diff(const Tensor& a, const Tensor& b) {
  REQUIRE(a.shape() == b.shape());
  REQUIRE(a.dtype() == DType::Float32);
  REQUIRE(b.dtype() == DType::Float32);
  Tensor ac = a.contiguous();
  Tensor bc = b.contiguous();
  const float* pa = ac.data_ptr<float>();
  const float* pb = bc.data_ptr<float>();
  float m = 0.0f;
  const int64_t n = ac.numel();
  for (int64_t i = 0; i < n; ++i) {
    const float d = std::abs(pa[i] - pb[i]);
    if (d > m) m = d;
  }
  return m;
}

Tensor make_param(std::initializer_list<float> vals, Shape shape) {
  Tensor t = Tensor::from_vector<float>(vals, shape);
  t.set_requires_grad(true);
  return t;
}

}  // namespace

TEST_CASE("graph::build_backward + run matches eager tape on add + sum",
          "[graph][autograd]") {
  // Forward: loss = sum(a * b + b).
  Tensor a = make_param({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor b = make_param({0.5f, -1.0f, 2.0f, 1.5f}, {2, 2});

  // --- Eager reference. ---
  Tensor loss_eager;
  {
    Tensor ab = ops::mul(a, b);
    Tensor s  = ops::add(ab, b);
    loss_eager = ops::sum(s);
    Engine::backward(loss_eager);
  }
  Tensor da_ref = a.grad().clone();
  Tensor db_ref = b.grad().clone();

  // --- Graph mode. ---
  graph::Graph graph_copy;
  graph::ValueId a_id, b_id, loss_id;
  {
    graph::GraphScope scope;
    a_id = graph::bind_param(a, "a");
    b_id = graph::bind_param(b, "b");
    Tensor ab = ops::mul(a, b);
    Tensor s  = ops::add(ab, b);
    Tensor loss = ops::sum(s);
    loss_id = graph::value_id_of(loss);
    scope.graph_.mark_output(loss_id);
    graph_copy = std::move(scope.graph_);
  }

  graph::BackwardResult bwd = graph::build_backward(graph_copy);
  REQUIRE(bwd.cotangents.size() == 1);
  REQUIRE(bwd.dparams.size() == 2);

  Tensor a_t = Tensor::from_vector<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Tensor b_t = Tensor::from_vector<float>({0.5f, -1.0f, 2.0f, 1.5f}, {2, 2});
  Tensor seed = Tensor::from_vector<float>({1.0f}, {});
  std::unordered_map<graph::ValueId, Tensor> bind;
  bind.emplace(a_id, a_t);
  bind.emplace(b_id, b_t);
  bind.emplace(bwd.cotangents[0], seed);

  auto outs = graph::run(graph_copy, bind);
  REQUIRE(outs.size() == 3);  // loss + dA + dB
  REQUIRE(max_abs_diff(outs[0], loss_eager) < 1e-6f);
  REQUIRE(max_abs_diff(outs[1], da_ref) < 1e-6f);
  REQUIRE(max_abs_diff(outs[2], db_ref) < 1e-6f);
}

TEST_CASE("graph::build_backward + run matches eager tape on Linear + ReLU + "
          "cross_entropy_with_logits",
          "[graph][autograd][mnist]") {
  // Small but realistic: batch=4, in=6, hidden=5, classes=3.
  const int64_t N = 4, D = 6, H = 5, C = 3;

  nn::Linear l1(D, H);
  nn::Linear l2(H, C);

  // Freeze inputs / targets for determinism.
  Tensor x = Tensor::from_vector<float>(
      {
          // 4 x 6
          0.1f, -0.3f, 0.5f,  0.2f, -0.7f, 0.4f,
          0.6f, -0.2f, -0.1f, 0.8f, 0.3f,  -0.9f,
          -0.5f, 0.1f, 0.7f,  -0.4f, 0.2f, 0.6f,
          0.3f, 0.8f,  -0.6f, 0.5f,  -0.2f, 0.1f,
      },
      {N, D});
  Tensor y = Tensor::from_vector<int64_t>({0, 1, 2, 1}, {N});

  // nn::Linear registers params in order: weight, bias.
  auto l1_params = l1.parameters();
  auto l2_params = l2.parameters();
  REQUIRE(l1_params.size() == 2);
  REQUIRE(l2_params.size() == 2);
  Tensor W1 = l1_params[0], b1 = l1_params[1];
  Tensor W2 = l2_params[0], b2 = l2_params[1];

  // --- Eager reference. ---
  Tensor loss_eager;
  {
    Tensor h1 = l1.forward(x);
    Tensor h2 = ops::relu(h1);
    Tensor logits = l2.forward(h2);
    loss_eager = ops::cross_entropy_with_logits(logits, y);
    Engine::backward(loss_eager);
  }
  // Snapshot the eager grads; we reuse the param tensors in the graph path
  // and the second backward would overwrite them.
  Tensor dW1_ref = W1.grad().clone();
  Tensor db1_ref = b1.grad().clone();
  Tensor dW2_ref = W2.grad().clone();
  Tensor db2_ref = b2.grad().clone();

  // --- Graph mode. ---
  graph::Graph graph_copy;
  graph::ValueId x_id, y_id, loss_id;
  graph::ValueId w1_id, b1_id_v, w2_id, b2_id_v;
  {
    graph::GraphScope scope;
    x_id = graph::bind_input(x, "x");
    y_id = graph::bind_input(y, "targets");
    w1_id    = graph::bind_param(W1, "W1");
    b1_id_v  = graph::bind_param(b1, "b1");
    w2_id    = graph::bind_param(W2, "W2");
    b2_id_v  = graph::bind_param(b2, "b2");

    Tensor h1 = l1.forward(x);
    Tensor h2 = ops::relu(h1);
    Tensor logits = l2.forward(h2);
    Tensor loss = ops::cross_entropy_with_logits(logits, y);

    loss_id = graph::value_id_of(loss);
    scope.graph_.mark_output(loss_id);
    graph_copy = std::move(scope.graph_);
  }

  // Compile backward.
  graph::BackwardResult bwd = graph::build_backward(graph_copy);
  REQUIRE(bwd.cotangents.size() == 1);
  REQUIRE(bwd.dparams.size() == 4);

  // Wire bindings.
  Tensor seed = Tensor::from_vector<float>({1.0f}, {});
  std::unordered_map<graph::ValueId, Tensor> bind;
  bind.emplace(x_id, x);
  bind.emplace(y_id, y);
  bind.emplace(w1_id,   W1);
  bind.emplace(b1_id_v, b1);
  bind.emplace(w2_id,   W2);
  bind.emplace(b2_id_v, b2);
  bind.emplace(bwd.cotangents[0], seed);

  auto outs = graph::run(graph_copy, bind);
  REQUIRE(outs.size() == 5);  // loss, dW1, db1, dW2, db2

  // Param declaration order inside the scope is W1, b1, W2, b2.
  const Tensor& loss_g = outs[0];
  const Tensor& dW1_g  = outs[1];
  const Tensor& db1_g  = outs[2];
  const Tensor& dW2_g  = outs[3];
  const Tensor& db2_g  = outs[4];

  const float tol = 1e-5f;
  REQUIRE(max_abs_diff(loss_g, loss_eager) < tol);
  REQUIRE(max_abs_diff(dW1_g, dW1_ref) < tol);
  REQUIRE(max_abs_diff(db1_g, db1_ref) < tol);
  REQUIRE(max_abs_diff(dW2_g, dW2_ref) < tol);
  REQUIRE(max_abs_diff(db2_g, db2_ref) < tol);
}

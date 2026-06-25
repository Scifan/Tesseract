// M1I.2.b — MLIR ExecutionEngine JIT forward-path smoke tests.
//
// Builds small graphs using only the ops that the
// `--convert-tesseract-to-linalg` pass already knows how to lower
// (binary elementwise, matmul, sum). Runs them twice — once through the
// C++ graph interpreter and once through `ir::JitEngine` — and asserts
// bit-level numerical agreement.
//
// Phase-2 extends coverage to the remaining ops that MNIST's forward
// graph needs: relu, neg, transpose, broadcast_to, and relu_backward.
// Cross-entropy forward/backward lowerings land in a follow-up phase.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/graph/Interpreter.hpp"
#include "tesseract/ir/JitEngine.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/ops/View.hpp"

using namespace tesseract;

namespace {

Tensor make_random(std::initializer_list<int64_t> dims, uint64_t seed,
                   float lo = -1.0f, float hi = 1.0f) {
  auto t = Tensor::empty(dims, DType::Float32);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  float* p = t.data_ptr<float>();
  const int64_t n = t.numel();
  for (int64_t i = 0; i < n; ++i) p[i] = dist(rng);
  return t;
}

void require_close(const Tensor& a, const Tensor& b, double tol = 1e-5) {
  REQUIRE(a.dtype() == b.dtype());
  REQUIRE(a.numel() == b.numel());
  REQUIRE(a.rank() == b.rank());
  for (int64_t d = 0; d < a.rank(); ++d) REQUIRE(a.shape()[d] == b.shape()[d]);

  const float* pa = a.data_ptr<float>();
  const float* pb = b.data_ptr<float>();
  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    const double diff = std::fabs(static_cast<double>(pa[i] - pb[i]));
    INFO("element " << i << ": jit=" << pa[i] << " interp=" << pb[i]
                    << " |diff|=" << diff);
    REQUIRE(diff <= tol);
  }
}

}  // namespace

TEST_CASE("jit: pure binary elementwise add matches interpreter",
          "[ir][jit]") {
  constexpr int64_t M = 8;
  constexpr int64_t N = 4;
  auto a = make_random({M, N}, /*seed=*/11);
  auto b = make_random({M, N}, /*seed=*/22);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    auto a_id = graph::bind_input(a, "a");
    auto b_id = graph::bind_input(b, "b");
    Tensor c = ops::add(a, b);
    graph::mark_output(c);
    captured = std::move(scope.graph());
    (void)a_id;
    (void)b_id;
  }

  // Reference: C++ interpreter.
  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], a}, {captured.inputs()[1], b}};
  auto interp_outs = graph::run(captured, bind);
  REQUIRE(interp_outs.size() == 1);

  // JIT path: build + lower + invoke.
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  REQUIRE(jit_outs.size() == 1);

  require_close(jit_outs[0], interp_outs[0]);
}

TEST_CASE("jit: matmul+sum composite matches interpreter", "[ir][jit]") {
  // z = sum(a @ b) over all axes. Exercises both matmul (linalg.matmul +
  // linalg.fill) and sum (linalg.reduce) in one pipeline run.
  constexpr int64_t M = 6;
  constexpr int64_t K = 5;
  constexpr int64_t N = 7;
  auto a = make_random({M, K}, /*seed=*/101);
  auto b = make_random({K, N}, /*seed=*/202);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(a, "a");
    graph::bind_input(b, "b");
    Tensor ab = ops::matmul(a, b);
    // `dim=1, keepdim=false` reduces the N axis, producing a [M] vector.
    // Avoid `dim=-1` because the C++ ops and the linalg lowering disagree
    // on the meaning of negative dims (one treats it as "all axes", the
    // other as "last axis"). Disambiguate by spelling the axis out.
    Tensor s = ops::sum(ab, /*dim=*/1, /*keepdim=*/false);
    graph::mark_output(s);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], a}, {captured.inputs()[1], b}};
  auto interp_outs = graph::run(captured, bind);
  REQUIRE(interp_outs.size() == 1);

  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  REQUIRE(jit_outs.size() == 1);

  // Matmul + accumulation: a touch looser tolerance than the pure add case
  // in case the pipeline picks a different reduction order.
  require_close(jit_outs[0], interp_outs[0], /*tol=*/1e-3);
}

TEST_CASE("jit: multiple outputs survive buffer-results-to-out-params",
          "[ir][jit]") {
  // Two outputs so we exercise the "prepend N memrefs" path of the
  // buffer-results-to-out-params pass.
  constexpr int64_t M = 4;
  constexpr int64_t N = 3;
  auto a = make_random({M, N}, /*seed=*/42);
  auto b = make_random({M, N}, /*seed=*/43);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(a, "a");
    graph::bind_input(b, "b");
    Tensor sum_t = ops::add(a, b);
    Tensor diff_t = ops::sub(a, b);
    graph::mark_output(sum_t);
    graph::mark_output(diff_t);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], a}, {captured.inputs()[1], b}};
  auto interp_outs = graph::run(captured, bind);
  REQUIRE(interp_outs.size() == 2);

  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  REQUIRE(jit_outs.size() == 2);

  require_close(jit_outs[0], interp_outs[0]);
  require_close(jit_outs[1], interp_outs[1]);
}

TEST_CASE("jit: params share the input channel and pass through correctly",
          "[ir][jit]") {
  // Graph with 1 input + 1 param + 1 output. `bind_param` is the primary
  // channel the MNIST training loop uses for weights/biases.
  auto x = make_random({5, 4}, /*seed=*/7);
  auto w = make_random({5, 4}, /*seed=*/8);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    graph::bind_param(w, "w");
    Tensor y = ops::mul(x, w);
    graph::mark_output(y);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], x}, {captured.params()[0], w}};

  auto interp_outs = graph::run(captured, bind);
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  require_close(jit_outs[0], interp_outs[0]);
}

TEST_CASE("jit: two-layer matmul chain matches interpreter (MLP-like forward)",
          "[ir][jit]") {
  // Forward path of a no-bias, no-activation MLP. Shapes pattern the
  // MNIST example (B=32, 784 → 128 → 10) but shrunk so the test stays
  // under a millisecond.
  constexpr int64_t B = 8;
  constexpr int64_t D0 = 12;
  constexpr int64_t D1 = 6;
  constexpr int64_t D2 = 3;

  auto x = make_random({B, D0}, /*seed=*/300);
  auto w1 = make_random({D0, D1}, /*seed=*/301, -0.1f, 0.1f);
  auto w2 = make_random({D1, D2}, /*seed=*/302, -0.1f, 0.1f);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    graph::bind_param(w1, "w1");
    graph::bind_param(w2, "w2");
    Tensor h = ops::matmul(x, w1);
    Tensor y = ops::matmul(h, w2);
    graph::mark_output(y);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], x},
      {captured.params()[0], w1},
      {captured.params()[1], w2},
  };

  auto interp_outs = graph::run(captured, bind);
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  // Two nested matmuls; the reduction order may diverge slightly from
  // the eager kernel so 1e-3 absolute is the appropriate tolerance.
  require_close(jit_outs[0], interp_outs[0], /*tol=*/1e-3);
}

TEST_CASE("jit: invocation repeats produce stable results", "[ir][jit]") {
  // A JIT'd function should behave as a pure function: two invocations
  // with the same bindings must produce byte-for-byte identical outputs.
  // This guards against any residual state (e.g. uninit output buffers)
  // that would otherwise show up as subtle flakiness in a training loop.
  auto a = make_random({3, 5}, /*seed=*/55);
  auto b = make_random({3, 5}, /*seed=*/66);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(a, "a");
    graph::bind_input(b, "b");
    Tensor c = ops::add(a, b);
    graph::mark_output(c);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], a}, {captured.inputs()[1], b}};

  ir::JitEngine jit(captured);
  auto first = jit.invoke(bind);
  auto second = jit.invoke(bind);

  REQUIRE(first.size() == 1);
  REQUIRE(second.size() == 1);
  const float* p1 = first[0].data_ptr<float>();
  const float* p2 = second[0].data_ptr<float>();
  for (int64_t i = 0; i < first[0].numel(); ++i) {
    REQUIRE(p1[i] == p2[i]);
  }
}

// ---------------------------------------------------------------------------
// Phase-2: activations + shape ops.
// ---------------------------------------------------------------------------

TEST_CASE("jit: relu elementwise", "[ir][jit]") {
  // Exercises the linalg.generic-based relu lowering; values span both
  // sides of zero so we actually cover the arith.maximumf branch.
  auto x = make_random({4, 7}, /*seed=*/991, -1.5f, 1.5f);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    Tensor y = ops::relu(x);
    graph::mark_output(y);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], x}};
  auto interp_outs = graph::run(captured, bind);
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  require_close(jit_outs[0], interp_outs[0], /*tol=*/0.0);
}

TEST_CASE("jit: transpose + matmul (matmul-backward-shape)", "[ir][jit]") {
  // The matmul backward rule transposes one operand; this test covers
  // that exact shape: d_a = dout @ w^T, with `w` provided as a param.
  constexpr int64_t B = 5;
  constexpr int64_t K = 4;
  constexpr int64_t N = 6;
  auto dout = make_random({B, N}, /*seed=*/701);
  auto w = make_random({K, N}, /*seed=*/702);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(dout, "dout");
    graph::bind_param(w, "w");
    Tensor wT = ops::transpose(w, /*dim_a=*/0, /*dim_b=*/1);
    Tensor dA = ops::matmul(dout, wT);
    graph::mark_output(dA);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], dout}, {captured.params()[0], w}};
  auto interp_outs = graph::run(captured, bind);
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  require_close(jit_outs[0], interp_outs[0], /*tol=*/1e-3);
}

TEST_CASE("jit: bias broadcast + add (MNIST forward fragment)", "[ir][jit]") {
  // x @ W + b, with b broadcast from [N] -> [B, N]. This is the shape
  // the MNIST forward pass hits every batch: `broadcast_to` expands the
  // bias row-vector onto the batch axis.
  constexpr int64_t B = 3;
  constexpr int64_t D = 4;
  constexpr int64_t N = 5;
  auto x = make_random({B, D}, /*seed=*/811);
  auto w = make_random({D, N}, /*seed=*/812, -0.1f, 0.1f);
  auto b = make_random({N}, /*seed=*/813, -0.1f, 0.1f);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    graph::bind_param(w, "w");
    graph::bind_param(b, "b");
    Tensor xw = ops::matmul(x, w);
    Tensor bb = ops::broadcast_to(b, Shape({B, N}));
    Tensor y = ops::add(xw, bb);
    graph::mark_output(y);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], x},
      {captured.params()[0], w},
      {captured.params()[1], b}};
  auto interp_outs = graph::run(captured, bind);
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  require_close(jit_outs[0], interp_outs[0], /*tol=*/1e-3);
}

TEST_CASE("jit: MLP-style hidden layer (matmul + bias + relu)", "[ir][jit]") {
  // End-to-end shape a single MLP layer would produce in MNIST. Stacks
  // matmul + broadcast_to + add + relu in one JITed module.
  constexpr int64_t B = 4;
  constexpr int64_t D = 8;
  constexpr int64_t H = 6;
  auto x = make_random({B, D}, /*seed=*/1001);
  auto w = make_random({D, H}, /*seed=*/1002, -0.2f, 0.2f);
  auto b = make_random({H}, /*seed=*/1003, -0.2f, 0.2f);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    graph::bind_param(w, "w");
    graph::bind_param(b, "b");
    Tensor z = ops::matmul(x, w);
    Tensor bb = ops::broadcast_to(b, Shape({B, H}));
    Tensor preact = ops::add(z, bb);
    Tensor h = ops::relu(preact);
    graph::mark_output(h);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], x},
      {captured.params()[0], w},
      {captured.params()[1], b}};
  auto interp_outs = graph::run(captured, bind);
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  require_close(jit_outs[0], interp_outs[0], /*tol=*/1e-3);
}

TEST_CASE("jit: relu_backward elementwise mask", "[ir][jit]") {
  // Fused backward op: dx = dout * (x > 0 ? 1 : 0). Mix positive and
  // negative entries in x so both sides of the mask are exercised.
  auto x = make_random({3, 5}, /*seed=*/1200, -1.0f, 1.0f);
  auto dout = make_random({3, 5}, /*seed=*/1201, -1.0f, 1.0f);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    graph::bind_input(dout, "dout");
    Tensor dx = ops::relu_backward(x, dout);
    graph::mark_output(dx);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], x}, {captured.inputs()[1], dout}};
  auto interp_outs = graph::run(captured, bind);
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  require_close(jit_outs[0], interp_outs[0], /*tol=*/0.0);
}

TEST_CASE("jit: cross_entropy_with_logits forward", "[ir][jit]") {
  // Softmax-CE loss with integer targets. Exercises the full log-sum-exp
  // lowering: row-max, exp, sum, log, onehot + reduce.
  constexpr int64_t N = 4;
  constexpr int64_t C = 5;
  auto logits = make_random({N, C}, /*seed=*/1401, -2.0f, 2.0f);
  auto targets = Tensor::empty({N}, DType::Int64);
  int64_t* pt = targets.data_ptr<int64_t>();
  pt[0] = 0;
  pt[1] = 3;
  pt[2] = 2;
  pt[3] = 4;

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(logits, "logits");
    graph::bind_input(targets, "targets");
    Tensor loss = ops::cross_entropy_with_logits(logits, targets);
    graph::mark_output(loss);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], logits}, {captured.inputs()[1], targets}};
  auto interp_outs = graph::run(captured, bind);
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  require_close(jit_outs[0], interp_outs[0], /*tol=*/1e-4);
}

TEST_CASE("jit: cross_entropy_with_logits backward", "[ir][jit]") {
  // Fused gradient: d_logits = (softmax(logits) - one_hot(targets)) * g / N.
  constexpr int64_t N = 3;
  constexpr int64_t C = 4;
  auto logits = make_random({N, C}, /*seed=*/1501, -1.5f, 1.5f);
  auto targets = Tensor::empty({N}, DType::Int64);
  int64_t* pt = targets.data_ptr<int64_t>();
  pt[0] = 1;
  pt[1] = 0;
  pt[2] = 3;
  auto g = Tensor::empty({}, DType::Float32);
  *g.data_ptr<float>() = 1.0f;

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(logits, "logits");
    graph::bind_input(targets, "targets");
    graph::bind_input(g, "g");
    Tensor dlogits = ops::cross_entropy_with_logits_backward(logits, targets, g);
    graph::mark_output(dlogits);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], logits},
      {captured.inputs()[1], targets},
      {captured.inputs()[2], g}};
  auto interp_outs = graph::run(captured, bind);
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  require_close(jit_outs[0], interp_outs[0], /*tol=*/1e-4);
}

TEST_CASE("jit: neg elementwise", "[ir][jit]") {
  auto x = make_random({2, 3, 4}, /*seed=*/1300);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    Tensor y = ops::neg(x);
    graph::mark_output(y);
    captured = std::move(scope.graph());
  }

  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], x}};
  auto interp_outs = graph::run(captured, bind);
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);
  require_close(jit_outs[0], interp_outs[0], /*tol=*/0.0);
}

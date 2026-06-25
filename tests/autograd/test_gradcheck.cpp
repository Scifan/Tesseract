#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <functional>
#include <vector>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Indexing.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/ops/Softmax.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

namespace {

// Double-precision finite-difference gradient for a scalar function of
// several input tensors. Each input is perturbed element-wise. Returns a
// vector of gradient tensors aligned with `inputs`.
//
// Assumes `inputs` are contiguous float64 tensors.
using ScalarFn = std::function<double(const std::vector<Tensor>&)>;

std::vector<Tensor> numeric_grad(const ScalarFn& f, std::vector<Tensor>& inputs, double eps = 1e-4) {
  std::vector<Tensor> grads;
  grads.reserve(inputs.size());
  for (const auto& in : inputs) {
    grads.push_back(Tensor::zeros(in.shape(), DType::Float64));
  }
  for (std::size_t k = 0; k < inputs.size(); ++k) {
    double* pi = inputs[k].data_ptr<double>();
    double* pg = grads[k].data_ptr<double>();
    const int64_t n = inputs[k].numel();
    for (int64_t i = 0; i < n; ++i) {
      const double save = pi[i];
      pi[i] = save + eps;
      const double fp = f(inputs);
      pi[i] = save - eps;
      const double fm = f(inputs);
      pi[i] = save;
      pg[i] = (fp - fm) / (2.0 * eps);
    }
  }
  return grads;
}

// A driver that takes a Tensor-returning function and runs gradcheck.
using ScalarTensorFn = std::function<Tensor(const std::vector<Tensor>&)>;

void gradcheck(const ScalarTensorFn& f, std::vector<Tensor> inputs, double eps = 1e-4,
               double atol = 1e-4) {
  // Promote all inputs to Float64 for numerical stability.
  for (auto& t : inputs) {
    REQUIRE(t.dtype() == DType::Float64);
    REQUIRE(t.is_contiguous());
  }

  // Numerical gradient via a scalar wrapper.
  ScalarFn scalar_f = [&](const std::vector<Tensor>& xs) {
    NoGradGuard nogg;
    Tensor out = f(xs);
    REQUIRE(out.shape() == Shape({}));
    return static_cast<double>(*out.data_ptr<double>());
  };
  std::vector<Tensor> ng = numeric_grad(scalar_f, inputs, eps);

  // Analytic gradient: mark leaves, run forward with autograd, backward.
  std::vector<Tensor> leafs;
  leafs.reserve(inputs.size());
  for (auto& t : inputs) {
    Tensor c = t.clone();
    c.set_requires_grad(true);
    leafs.push_back(c);
  }
  Tensor loss = f(leafs);
  REQUIRE(loss.shape() == Shape({}));
  Engine::backward(loss);

  for (std::size_t k = 0; k < inputs.size(); ++k) {
    const Tensor& ag = leafs[k].grad();
    REQUIRE(ag.defined());
    REQUIRE(ag.shape() == inputs[k].shape());
    const double* pa = ag.data_ptr<double>();
    const double* pn = ng[k].data_ptr<double>();
    const int64_t n = inputs[k].numel();
    for (int64_t i = 0; i < n; ++i) {
      INFO("input " << k << " elem " << i << " analytic=" << pa[i] << " numeric=" << pn[i]);
      REQUIRE_THAT(pa[i], WithinAbs(pn[i], atol));
    }
  }
}

Tensor rand_like(const Shape& s, double lo = -1.0, double hi = 1.0) {
  Tensor t = Tensor::empty(s, DType::Float64);
  const int64_t n = t.numel();
  double* p = t.data_ptr<double>();
  // Deterministic pseudo-random sequence so tests are reproducible.
  uint64_t seed = 0x9E3779B97F4A7C15ULL;
  for (int64_t i = 0; i < n; ++i) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u = static_cast<double>((seed >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0;
    p[i] = lo + (hi - lo) * u;
  }
  return t;
}

}  // namespace

TEST_CASE("gradcheck: sum of add", "[autograd][gradcheck]") {
  auto a = rand_like({2, 3});
  auto b = rand_like({2, 3});
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::add(xs[0], xs[1])); },
            {a, b});
}

TEST_CASE("gradcheck: sum of add with broadcast", "[autograd][gradcheck]") {
  auto a = rand_like({2, 3});
  auto b = rand_like({3});
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::add(xs[0], xs[1])); },
            {a, b});
}

TEST_CASE("gradcheck: sum of mul", "[autograd][gradcheck]") {
  auto a = rand_like({2, 3});
  auto b = rand_like({2, 3});
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::mul(xs[0], xs[1])); },
            {a, b});
}

TEST_CASE("gradcheck: sum of sub", "[autograd][gradcheck]") {
  auto a = rand_like({2, 3});
  auto b = rand_like({3});
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::sub(xs[0], xs[1])); },
            {a, b});
}

TEST_CASE("gradcheck: sum of div (positive b)", "[autograd][gradcheck]") {
  auto a = rand_like({2, 3});
  auto b = rand_like({2, 3}, 1.0, 2.0);  // keep away from 0 for stability
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::div(xs[0], xs[1])); },
            {a, b});
}

TEST_CASE("gradcheck: sum of neg", "[autograd][gradcheck]") {
  auto a = rand_like({4});
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::neg(xs[0])); }, {a});
}

TEST_CASE("gradcheck: sum of matmul", "[autograd][gradcheck]") {
  auto a = rand_like({3, 4});
  auto b = rand_like({4, 2});
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::matmul(xs[0], xs[1])); },
            {a, b});
}

TEST_CASE("gradcheck: relu", "[autograd][gradcheck]") {
  // Avoid exactly-zero inputs where the subgradient is not well-defined.
  auto a = rand_like({5}, 0.1, 1.0);
  auto b = rand_like({5}, -1.0, -0.1);
  // Combine so the tensor has both positive and negative entries.
  double* pa = a.data_ptr<double>();
  double* pb = b.data_ptr<double>();
  for (int i = 0; i < 5; i += 2) pa[i] = pb[i];
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::relu(xs[0])); }, {a});
}

TEST_CASE("gradcheck: sigmoid", "[autograd][gradcheck]") {
  auto a = rand_like({6});
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::sigmoid(xs[0])); }, {a});
}

TEST_CASE("gradcheck: tanh", "[autograd][gradcheck]") {
  auto a = rand_like({6});
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::tanh(xs[0])); }, {a});
}

TEST_CASE("gradcheck: exp / log composition", "[autograd][gradcheck]") {
  auto a = rand_like({5}, 0.5, 2.0);  // positive for log
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::log(ops::exp(xs[0]))); }, {a});
}

TEST_CASE("gradcheck: softmax + mean", "[autograd][gradcheck]") {
  auto a = rand_like({3, 4});
  gradcheck([](const std::vector<Tensor>& xs) { return ops::mean(ops::softmax(xs[0], 1)); }, {a});
}

TEST_CASE("gradcheck: log_softmax + sum", "[autograd][gradcheck]") {
  auto a = rand_like({3, 4});
  gradcheck([](const std::vector<Tensor>& xs) { return ops::sum(ops::log_softmax(xs[0], 1)); }, {a});
}

TEST_CASE("gradcheck: cross_entropy_with_logits", "[autograd][gradcheck]") {
  auto logits = rand_like({4, 3});
  auto targets = Tensor::from_vector<int64_t>({0, 2, 1, 2}, {4});
  gradcheck([&targets](const std::vector<Tensor>& xs) {
              return ops::cross_entropy_with_logits(xs[0], targets);
            },
            {logits});
}

TEST_CASE("gradcheck: view", "[autograd][gradcheck][view]") {
  auto a = rand_like({2, 3});
  // Composition: reshape to [3,2], then sum. Tests that the reshape backward
  // correctly routes the [3,2] gradient back to [2,3].
  gradcheck([](const std::vector<Tensor>& xs) {
              return ops::sum(ops::view(xs[0], {3, 2}));
            },
            {a});
}

TEST_CASE("gradcheck: reshape", "[autograd][gradcheck][view]") {
  auto a = rand_like({2, 3, 2});
  gradcheck([](const std::vector<Tensor>& xs) {
              return ops::sum(ops::reshape(xs[0], {4, 3}));
            },
            {a});
}

TEST_CASE("gradcheck: transpose + matmul", "[autograd][gradcheck][view]") {
  // This is the nn::Linear pattern: loss = sum(x @ w^T).
  auto x = rand_like({3, 4});
  auto w = rand_like({2, 4});  // [out, in] PyTorch convention
  gradcheck([](const std::vector<Tensor>& xs) {
              Tensor y = ops::matmul(xs[0], ops::transpose(xs[1], 0, 1));
              return ops::sum(y);
            },
            {x, w});
}

TEST_CASE("gradcheck: permute", "[autograd][gradcheck][view]") {
  auto a = rand_like({2, 3, 4});
  gradcheck([](const std::vector<Tensor>& xs) {
              return ops::sum(ops::permute(xs[0], {2, 0, 1}));
            },
            {a});
}

TEST_CASE("gradcheck: contiguous is identity", "[autograd][gradcheck][view]") {
  auto a = rand_like({3, 4});
  gradcheck([](const std::vector<Tensor>& xs) {
              // contiguous() on an already-contiguous input + on a permuted
              // (non-contig) input — both must pass.
              Tensor y = ops::contiguous(ops::transpose(xs[0], 0, 1));
              return ops::sum(y);
            },
            {a});
}

// -----------------------------------------------------------------------------
// B-003: cat / split / index_select / gather gradcheck coverage.
// The split backward is particularly load-bearing because it relies on the
// engine summing N zero-padded `SplitChunkBackward` contributions at the
// shared parent edge — the gradcheck exercise pins that behaviour end-to-end.
// -----------------------------------------------------------------------------

TEST_CASE("gradcheck: cat along dim 0", "[autograd][gradcheck][indexing]") {
  auto a = rand_like({2, 3});
  auto b = rand_like({3, 3});
  gradcheck([](const std::vector<Tensor>& xs) {
              return ops::sum(ops::cat({xs[0], xs[1]}, 0));
            },
            {a, b});
}

TEST_CASE("gradcheck: cat along dim 1 with negative index",
          "[autograd][gradcheck][indexing]") {
  auto a = rand_like({3, 2});
  auto b = rand_like({3, 4});
  gradcheck([](const std::vector<Tensor>& xs) {
              // Non-linear combination so the gradient isn't the same in every
              // slab — catches backward slicing bugs.
              Tensor y = ops::cat({xs[0], ops::mul(xs[1], xs[1])}, -1);
              return ops::sum(y);
            },
            {a, b});
}

TEST_CASE("gradcheck: split + elementwise combine",
          "[autograd][gradcheck][indexing]") {
  auto a = rand_like({2, 6});
  gradcheck([](const std::vector<Tensor>& xs) {
              auto parts = ops::split(xs[0], 2, 1);
              // 3 chunks of [2, 2]; combine non-linearly so every chunk's
              // gradient is distinct (would collapse to constant otherwise).
              Tensor y = ops::add(ops::mul(parts[0], parts[1]), parts[2]);
              return ops::sum(y);
            },
            {a});
}

TEST_CASE("gradcheck: split_with_sizes irregular",
          "[autograd][gradcheck][indexing]") {
  auto a = rand_like({5, 3});
  gradcheck([](const std::vector<Tensor>& xs) {
              auto parts = ops::split_with_sizes(xs[0], {1, 2, 2}, 0);
              Tensor s0 = ops::sum(parts[0]);
              Tensor s1 = ops::sum(ops::mul(parts[1], parts[1]));
              Tensor s2 = ops::sum(parts[2]);
              return ops::add(ops::add(s0, s1), s2);
            },
            {a});
}

TEST_CASE("gradcheck: cat(split(x)) is identity on gradient",
          "[autograd][gradcheck][indexing]") {
  auto a = rand_like({3, 4});
  gradcheck([](const std::vector<Tensor>& xs) {
              auto parts = ops::split(xs[0], 2, 1);
              Tensor y = ops::cat(parts, 1);
              return ops::sum(ops::mul(y, y));  // sum(x^2), grad = 2x
            },
            {a});
}

TEST_CASE("gradcheck: index_select rows", "[autograd][gradcheck][indexing]") {
  auto a = rand_like({4, 3});
  Tensor idx = Tensor::from_vector<int64_t>({0, 2, 1, 2, 0}, {5});
  gradcheck([&idx](const std::vector<Tensor>& xs) {
              // Duplicate indices (0 and 2 appear twice) ensure the scatter-add
              // backward is truly additive, not a copy.
              Tensor y = ops::index_select(xs[0], 0, idx);
              return ops::sum(ops::mul(y, y));
            },
            {a});
}

TEST_CASE("gradcheck: index_select columns along dim 1",
          "[autograd][gradcheck][indexing]") {
  auto a = rand_like({3, 5});
  Tensor idx = Tensor::from_vector<int64_t>({4, 0, 2, 0}, {4});
  gradcheck([&idx](const std::vector<Tensor>& xs) {
              Tensor y = ops::index_select(xs[0], 1, idx);
              return ops::sum(y);
            },
            {a});
}

TEST_CASE("gradcheck: gather dim 1", "[autograd][gradcheck][indexing]") {
  auto a = rand_like({3, 4});
  // Shape matches output: [3, 2]. Some duplicates across rows.
  Tensor idx = Tensor::from_vector<int64_t>({0, 0, 3, 1, 2, 2}, {3, 2});
  gradcheck([&idx](const std::vector<Tensor>& xs) {
              Tensor y = ops::gather(xs[0], 1, idx);
              return ops::sum(ops::mul(y, y));
            },
            {a});
}

TEST_CASE("gradcheck: gather dim 0", "[autograd][gradcheck][indexing]") {
  auto a = rand_like({4, 3});
  Tensor idx = Tensor::from_vector<int64_t>({0, 3, 1, 0, 2, 2}, {2, 3});
  gradcheck([&idx](const std::vector<Tensor>& xs) {
              Tensor y = ops::gather(xs[0], 0, idx);
              return ops::sum(y);
            },
            {a});
}

TEST_CASE("autograd: NoGradGuard prevents graph construction", "[autograd]") {
  auto a = Tensor::from_vector<double>({1.0, 2.0}, {2});
  a.set_requires_grad(true);
  {
    NoGradGuard nogg;
    auto b = ops::mul(a, a);
    REQUIRE_FALSE(b.requires_grad());
  }
}

TEST_CASE("autograd: grad accumulates on leaf", "[autograd]") {
  auto a = Tensor::from_vector<double>({2.0}, {1});
  a.set_requires_grad(true);
  auto b = ops::mul(a, a);  // dL/da = 2a
  auto c = ops::mul(a, a);
  auto d = ops::add(b, c);  // derivative = 4a
  auto loss = ops::sum(d);
  Engine::backward(loss);
  REQUIRE_THAT(*a.grad().data_ptr<double>(), WithinAbs(8.0, 1e-9));  // 4*2
}

// -----------------------------------------------------------------------------
// Batched matmul (B-004) — each of these tests pins a different wiring of the
// broadcast rules against numerical gradients. Shapes are kept small so the
// O(n^2) `numeric_grad` stays fast, but big enough that bugs in batch-stride
// math or broadcast-reduction would be visible.
// -----------------------------------------------------------------------------

TEST_CASE("gradcheck: batched matmul [B,M,K]@[B,K,N] sum", "[autograd][gradcheck][matmul][batched]") {
  // Both operands carry a real batch dim; exercises the regular
  // batched-forward path with no broadcasting.
  auto a = rand_like({2, 3, 4});
  auto b = rand_like({2, 4, 3});
  gradcheck([](const std::vector<Tensor>& xs) {
              return ops::sum(ops::matmul(xs[0], xs[1]));
            },
            {a, b});
}

TEST_CASE("gradcheck: batched matmul [M,K]@[B,K,N]", "[autograd][gradcheck][matmul][batched]") {
  // Shared rank-2 lhs, batched rhs. The backward must sum the broadcast
  // batch axis out of `grad_lhs` so it matches `lhs.shape()`.
  auto a = rand_like({3, 4});
  auto b = rand_like({2, 4, 3});
  gradcheck([](const std::vector<Tensor>& xs) {
              return ops::sum(ops::matmul(xs[0], xs[1]));
            },
            {a, b});
}

TEST_CASE("gradcheck: batched matmul [B,M,K]@[K,N]", "[autograd][gradcheck][matmul][batched]") {
  // Sequence-apply-projection shape: per-batch activations, shared weights.
  // Mirror of the case above — now `grad_rhs` is what gets reduced.
  auto a = rand_like({2, 3, 4});
  auto b = rand_like({4, 3});
  gradcheck([](const std::vector<Tensor>& xs) {
              return ops::sum(ops::matmul(xs[0], xs[1]));
            },
            {a, b});
}

TEST_CASE("gradcheck: batched matmul two-level broadcast", "[autograd][gradcheck][matmul][batched]") {
  // lhs=[2,1,M,K], rhs=[1,3,K,N] → out=[2,3,M,N]. Every leading axis is
  // broadcast on exactly one side; both grads must reduce different axes.
  auto a = rand_like({2, 1, 3, 4});
  auto b = rand_like({1, 3, 4, 2});
  gradcheck([](const std::vector<Tensor>& xs) {
              return ops::sum(ops::matmul(xs[0], xs[1]));
            },
            {a, b});
}

TEST_CASE("gradcheck: batched matmul composed with tanh", "[autograd][gradcheck][matmul][batched]") {
  // Post-activation so both operands see a non-identity upstream gradient —
  // catches bugs that would otherwise be hidden when grad_out == ones.
  auto a = rand_like({2, 3, 4});
  auto b = rand_like({2, 4, 3});
  gradcheck([](const std::vector<Tensor>& xs) {
              return ops::sum(ops::tanh(ops::matmul(xs[0], xs[1])));
            },
            {a, b});
}

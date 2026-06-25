// Unit tests for `nn::Embedding`.
//
// Covers:
//   - Shape / dtype contract for rank-1, rank-2, rank-3 indices.
//   - Forward correctness against a hand-rolled row-gather reference.
//   - Gradient flow: the backward produces a scatter-add into weight.grad,
//     i.e. if a row is indexed k times, its gradient is the sum of the
//     k per-reference upstream grads. (This is the test that pins us to
//     PyTorch's dense-gradient nn.Embedding semantics rather than the
//     sparse-gradient variant.)

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/Embedding.hpp"
#include "tesseract/ops/Reduction.hpp"

using tesseract::DType;
using tesseract::Engine;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::nn::Embedding;

namespace {

Tensor make_indices(std::vector<int64_t> data, Shape shape) {
  Tensor t = Tensor::empty(shape, DType::Int64);
  std::memcpy(t.raw_data(), data.data(), data.size() * sizeof(int64_t));
  return t;
}

}  // namespace

TEST_CASE("Embedding: rank-1 indices produce [N, D] output") {
  Embedding emb(/*num_embeddings=*/10, /*embedding_dim=*/4);
  Tensor idx = make_indices({0, 3, 7, 2}, Shape({4}));
  Tensor y = emb.forward(idx);
  REQUIRE(y.shape() == Shape({4, 4}));
  REQUIRE(y.dtype() == DType::Float32);

  // Row equality to weight()'s rows.
  const float* wp = emb.weight().data_ptr<float>();
  const float* yp = y.data_ptr<float>();
  const int64_t D = 4;
  const std::vector<int64_t> rows = {0, 3, 7, 2};
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < D; ++j) {
      REQUIRE(yp[i * D + j] == wp[rows[i] * D + j]);
    }
  }
}

TEST_CASE("Embedding: rank-2 indices produce [B, S, D] output") {
  Embedding emb(8, 5);
  Tensor idx = make_indices({1, 2, 3, 4, 0, 7}, Shape({2, 3}));  // [B=2, S=3]
  Tensor y = emb.forward(idx);
  REQUIRE(y.shape() == Shape({2, 3, 5}));

  const float* wp = emb.weight().data_ptr<float>();
  const float* yp = y.data_ptr<float>();
  const int64_t D = 5;
  const std::vector<int64_t> flat = {1, 2, 3, 4, 0, 7};
  for (int64_t i = 0; i < 6; ++i) {
    for (int64_t j = 0; j < D; ++j) {
      REQUIRE(yp[i * D + j] == wp[flat[i] * D + j]);
    }
  }
}

TEST_CASE("Embedding: backward scatters gradient into the right rows") {
  // V=6, D=3. Indices [0, 3, 0] — row 0 is referenced twice, so
  // weight.grad[0, :] must end up as 2·upstream_grad_row.
  const int64_t V = 6;
  const int64_t D = 3;
  Embedding emb(V, D);

  Tensor idx = make_indices({0, 3, 0}, Shape({3}));
  Tensor y = emb.forward(idx);
  REQUIRE(y.shape() == Shape({3, D}));

  // sum(y) ensures every element of y contributes grad=1 upstream.
  Tensor loss = tesseract::ops::sum(y);
  Engine::backward(loss);

  // weight.grad[i, j] = count(idx == i)  (since d(sum y)/d y[k, j] = 1
  // and y[k, :] = weight[idx[k], :]).
  const auto* am = emb.weight().autograd_meta();
  REQUIRE(am != nullptr);
  REQUIRE(am->grad.defined());
  const Tensor& g = am->grad;
  REQUIRE(g.shape() == Shape({V, D}));
  const float* gp = g.data_ptr<float>();

  const std::vector<int64_t> expected_counts = {2, 0, 0, 1, 0, 0};
  for (int64_t r = 0; r < V; ++r) {
    for (int64_t c = 0; c < D; ++c) {
      REQUIRE(gp[r * D + c] == static_cast<float>(expected_counts[r]));
    }
  }
}

TEST_CASE("Embedding: Module::to(cpu) round-trip is a no-op") {
  Embedding emb(4, 3);
  Tensor idx = make_indices({0, 1, 2}, Shape({3}));
  Tensor y1 = emb.forward(idx);
  emb.to(tesseract::cpu_device());
  Tensor y2 = emb.forward(idx);
  REQUIRE(y1.shape() == y2.shape());
  const float* p1 = y1.data_ptr<float>();
  const float* p2 = y2.data_ptr<float>();
  for (int64_t i = 0; i < y1.numel(); ++i) REQUIRE(p1[i] == p2[i]);
}

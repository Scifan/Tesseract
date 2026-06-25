#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Loss.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

TEST_CASE("cross_entropy_with_logits: hand computed", "[ops][loss]") {
  // Batch of 2, 3 classes. Targets = [2, 0].
  auto logits = Tensor::from_vector<float>({1.0f, 2.0f, 3.0f,
                                            0.0f, -1.0f, 0.5f}, {2, 3});
  auto targets = Tensor::from_vector<int64_t>({2, 0}, {2});
  auto loss = ops::cross_entropy_with_logits(logits, targets);
  REQUIRE(loss.shape() == Shape({}));

  // Row 0: log(sum(exp([1,2,3]))) - 3
  const double z0 = std::log(std::exp(1.0) + std::exp(2.0) + std::exp(3.0)) - 3.0;
  // Row 1: log(sum(exp([0,-1,0.5]))) - 0
  const double z1 = std::log(std::exp(0.0) + std::exp(-1.0) + std::exp(0.5)) - 0.0;
  const double expected = (z0 + z1) / 2.0;
  REQUIRE_THAT(*loss.data_ptr<float>(), WithinAbs(expected, 1e-5));
}

TEST_CASE("cross_entropy_with_logits: dtype / shape checks", "[ops][loss]") {
  auto logits = Tensor::from_vector<float>({1, 2, 3, 4}, {2, 2});
  auto bad_t_dtype = Tensor::from_vector<float>({0, 1}, {2});  // wrong dtype
  REQUIRE_THROWS_AS(ops::cross_entropy_with_logits(logits, bad_t_dtype), tesseract::Error);

  auto bad_t_shape = Tensor::from_vector<int64_t>({0, 1, 0}, {3});
  REQUIRE_THROWS_AS(ops::cross_entropy_with_logits(logits, bad_t_shape), tesseract::Error);

  auto oob = Tensor::from_vector<int64_t>({0, 5}, {2});
  REQUIRE_THROWS_AS(ops::cross_entropy_with_logits(logits, oob), tesseract::Error);
}

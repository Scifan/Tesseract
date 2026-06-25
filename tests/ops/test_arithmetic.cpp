#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/utils/Logging.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

namespace {
constexpr double kEps = 1e-6;
}

TEST_CASE("add: same shape float32", "[ops][arith]") {
  auto a = Tensor::from_vector<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  auto b = Tensor::from_vector<float>({10.0f, 20.0f, 30.0f, 40.0f}, {2, 2});
  auto c = ops::add(a, b);
  REQUIRE(c.shape() == Shape({2, 2}));
  const float* p = c.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(11.0, kEps));
  REQUIRE_THAT(p[1], WithinAbs(22.0, kEps));
  REQUIRE_THAT(p[2], WithinAbs(33.0, kEps));
  REQUIRE_THAT(p[3], WithinAbs(44.0, kEps));
}

TEST_CASE("add: broadcast row-vec to matrix", "[ops][arith]") {
  auto a = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto b = Tensor::from_vector<float>({10, 20, 30}, {1, 3});
  auto c = ops::add(a, b);
  REQUIRE(c.shape() == Shape({2, 3}));
  const float* p = c.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(11.0, kEps));
  REQUIRE_THAT(p[1], WithinAbs(22.0, kEps));
  REQUIRE_THAT(p[2], WithinAbs(33.0, kEps));
  REQUIRE_THAT(p[3], WithinAbs(14.0, kEps));
  REQUIRE_THAT(p[4], WithinAbs(25.0, kEps));
  REQUIRE_THAT(p[5], WithinAbs(36.0, kEps));
}

TEST_CASE("add: broadcast scalar to 2x3", "[ops][arith]") {
  auto a = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto b = Tensor::full({}, 100.0f);
  auto c = ops::add(a, b);
  REQUIRE(c.shape() == Shape({2, 3}));
  const float* p = c.data_ptr<float>();
  for (int i = 0; i < 6; ++i) {
    REQUIRE_THAT(p[i], WithinAbs(static_cast<double>(i + 101), kEps));
  }
}

TEST_CASE("sub / mul / div elementwise", "[ops][arith]") {
  auto a = Tensor::from_vector<float>({6.0f, 8.0f, 10.0f}, {3});
  auto b = Tensor::from_vector<float>({2.0f, 4.0f, 5.0f}, {3});

  auto s = ops::sub(a, b);
  auto m = ops::mul(a, b);
  auto d = ops::div(a, b);
  const float* ps = s.data_ptr<float>();
  const float* pm = m.data_ptr<float>();
  const float* pd = d.data_ptr<float>();
  REQUIRE_THAT(ps[0], WithinAbs(4.0, kEps));
  REQUIRE_THAT(ps[1], WithinAbs(4.0, kEps));
  REQUIRE_THAT(ps[2], WithinAbs(5.0, kEps));
  REQUIRE_THAT(pm[0], WithinAbs(12.0, kEps));
  REQUIRE_THAT(pm[1], WithinAbs(32.0, kEps));
  REQUIRE_THAT(pm[2], WithinAbs(50.0, kEps));
  REQUIRE_THAT(pd[0], WithinAbs(3.0, kEps));
  REQUIRE_THAT(pd[1], WithinAbs(2.0, kEps));
  REQUIRE_THAT(pd[2], WithinAbs(2.0, kEps));
}

TEST_CASE("neg: float32", "[ops][arith]") {
  auto a = Tensor::from_vector<float>({1.0f, -2.0f, 3.5f, -4.5f}, {4});
  auto n = ops::neg(a);
  const float* p = n.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(-1.0, kEps));
  REQUIRE_THAT(p[1], WithinAbs(2.0, kEps));
  REQUIRE_THAT(p[2], WithinAbs(-3.5, kEps));
  REQUIRE_THAT(p[3], WithinAbs(4.5, kEps));
}

TEST_CASE("broadcast: shape mismatch throws", "[ops][arith]") {
  auto a = Tensor::zeros({2, 3}, DType::Float32);
  auto b = Tensor::zeros({4, 3}, DType::Float32);
  REQUIRE_THROWS_AS(ops::add(a, b), tesseract::Error);
}

TEST_CASE("reduce_to_shape: 2x3 -> 3", "[ops][arith]") {
  auto a = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto r = ops::reduce_to_shape(a, Shape{3});
  REQUIRE(r.shape() == Shape({3}));
  const float* p = r.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(5.0, kEps));
  REQUIRE_THAT(p[1], WithinAbs(7.0, kEps));
  REQUIRE_THAT(p[2], WithinAbs(9.0, kEps));
}

TEST_CASE("reduce_to_shape: 2x3 -> scalar", "[ops][arith]") {
  auto a = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto r = ops::reduce_to_shape(a, Shape{});
  REQUIRE(r.shape() == Shape({}));
  REQUIRE_THAT(*r.data_ptr<float>(), WithinAbs(21.0, kEps));
}

TEST_CASE("reduce_to_shape: 2x3 -> 2x1", "[ops][arith]") {
  auto a = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto r = ops::reduce_to_shape(a, Shape{2, 1});
  REQUIRE(r.shape() == Shape({2, 1}));
  const float* p = r.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(6.0, kEps));
  REQUIRE_THAT(p[1], WithinAbs(15.0, kEps));
}

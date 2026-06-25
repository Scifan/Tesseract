#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Activation.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

TEST_CASE("relu", "[ops][act]") {
  auto x = Tensor::from_vector<float>({-1.0f, 0.0f, 1.0f, -2.5f, 3.25f}, {5});
  auto y = ops::relu(x);
  const float* p = y.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(0.0,  1e-6));
  REQUIRE_THAT(p[1], WithinAbs(0.0,  1e-6));
  REQUIRE_THAT(p[2], WithinAbs(1.0,  1e-6));
  REQUIRE_THAT(p[3], WithinAbs(0.0,  1e-6));
  REQUIRE_THAT(p[4], WithinAbs(3.25, 1e-6));
}

TEST_CASE("sigmoid", "[ops][act]") {
  auto x = Tensor::from_vector<float>({0.0f, 1.0f, -1.0f}, {3});
  auto y = ops::sigmoid(x);
  const float* p = y.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(0.5, 1e-6));
  REQUIRE_THAT(p[1], WithinAbs(1.0 / (1.0 + std::exp(-1.0)), 1e-6));
  REQUIRE_THAT(p[2], WithinAbs(1.0 / (1.0 + std::exp(1.0)),  1e-6));
}

TEST_CASE("tanh / exp / log", "[ops][act]") {
  auto x = Tensor::from_vector<float>({0.0f, 1.0f, -1.0f}, {3});
  auto t = ops::tanh(x);
  const float* pt = t.data_ptr<float>();
  REQUIRE_THAT(pt[0], WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(pt[1], WithinAbs(std::tanh(1.0),  1e-6));
  REQUIRE_THAT(pt[2], WithinAbs(std::tanh(-1.0), 1e-6));

  auto e = ops::exp(x);
  REQUIRE_THAT(*e.data_ptr<float>(), WithinAbs(1.0, 1e-6));

  auto lx = Tensor::from_vector<double>({1.0, std::exp(1.0)}, {2});
  auto l = ops::log(lx);
  const double* pl = l.data_ptr<double>();
  REQUIRE_THAT(pl[0], WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(pl[1], WithinAbs(1.0, 1e-12));
}

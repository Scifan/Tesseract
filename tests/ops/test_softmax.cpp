#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Softmax.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

TEST_CASE("softmax: 1D", "[ops][softmax]") {
  auto x = Tensor::from_vector<float>({1.0f, 2.0f, 3.0f}, {3});
  auto y = ops::softmax(x, 0);
  const float* p = y.data_ptr<float>();
  const double e1 = std::exp(1.0), e2 = std::exp(2.0), e3 = std::exp(3.0);
  const double s = e1 + e2 + e3;
  REQUIRE_THAT(p[0], WithinAbs(e1 / s, 1e-6));
  REQUIRE_THAT(p[1], WithinAbs(e2 / s, 1e-6));
  REQUIRE_THAT(p[2], WithinAbs(e3 / s, 1e-6));
  REQUIRE_THAT(p[0] + p[1] + p[2], WithinAbs(1.0, 1e-6));
}

TEST_CASE("softmax: 2D along last dim", "[ops][softmax]") {
  auto x = Tensor::from_vector<float>({1, 2, 3,
                                       -1, 0, 1}, {2, 3});
  auto y = ops::softmax(x, 1);
  REQUIRE(y.shape() == Shape({2, 3}));
  const float* p = y.data_ptr<float>();
  REQUIRE_THAT(p[0] + p[1] + p[2], WithinAbs(1.0, 1e-6));
  REQUIRE_THAT(p[3] + p[4] + p[5], WithinAbs(1.0, 1e-6));
}

TEST_CASE("softmax: numerical stability with large values", "[ops][softmax]") {
  auto x = Tensor::from_vector<float>({1000.0f, 1001.0f, 1002.0f}, {3});
  auto y = ops::softmax(x, 0);
  const float* p = y.data_ptr<float>();
  REQUIRE(std::isfinite(p[0]));
  REQUIRE(std::isfinite(p[1]));
  REQUIRE(std::isfinite(p[2]));
  REQUIRE_THAT(p[0] + p[1] + p[2], WithinAbs(1.0, 1e-6));
  // Equivalent to softmax([0,1,2]).
  const double e = std::exp(1.0), e2 = std::exp(2.0);
  const double s = 1.0 + e + e2;
  REQUIRE_THAT(p[2], WithinAbs(e2 / s, 1e-6));
}

TEST_CASE("log_softmax: matches log(softmax)", "[ops][softmax]") {
  auto x = Tensor::from_vector<double>({0.5, 1.5, -0.5, 2.0}, {4});
  auto l = ops::log_softmax(x, 0);
  auto s = ops::softmax(x, 0);
  const double* pl = l.data_ptr<double>();
  const double* ps = s.data_ptr<double>();
  for (int i = 0; i < 4; ++i) {
    REQUIRE_THAT(pl[i], WithinAbs(std::log(ps[i]), 1e-9));
  }
}

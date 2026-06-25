#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Reduction.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

TEST_CASE("sum: all reduce", "[ops][reduce]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto s = ops::sum(x);
  REQUIRE(s.shape() == Shape({}));
  REQUIRE_THAT(*s.data_ptr<float>(), WithinAbs(21.0, 1e-6));
}

TEST_CASE("sum: dim=0 no keepdim", "[ops][reduce]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto s = ops::sum(x, 0);
  REQUIRE(s.shape() == Shape({3}));
  const float* p = s.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(5.0, 1e-6));
  REQUIRE_THAT(p[1], WithinAbs(7.0, 1e-6));
  REQUIRE_THAT(p[2], WithinAbs(9.0, 1e-6));
}

TEST_CASE("sum: dim=1 keepdim", "[ops][reduce]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto s = ops::sum(x, 1, /*keepdim=*/true);
  REQUIRE(s.shape() == Shape({2, 1}));
  const float* p = s.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(6.0, 1e-6));
  REQUIRE_THAT(p[1], WithinAbs(15.0, 1e-6));
}

TEST_CASE("mean: all reduce", "[ops][reduce]") {
  auto x = Tensor::from_vector<float>({2, 4, 6, 8}, {4});
  auto m = ops::mean(x);
  REQUIRE_THAT(*m.data_ptr<float>(), WithinAbs(5.0, 1e-6));
}

TEST_CASE("mean: dim=0", "[ops][reduce]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto m = ops::mean(x, 0);
  const float* p = m.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(2.5, 1e-6));
  REQUIRE_THAT(p[1], WithinAbs(3.5, 1e-6));
  REQUIRE_THAT(p[2], WithinAbs(4.5, 1e-6));
}

TEST_CASE("max: all reduce and dim", "[ops][reduce]") {
  auto x = Tensor::from_vector<float>({1, 5, 2, 9, 3, 4}, {2, 3});
  REQUIRE_THAT(*ops::max(x).data_ptr<float>(), WithinAbs(9.0, 1e-6));
  auto m = ops::max(x, 1);
  REQUIRE(m.shape() == Shape({2}));
  const float* p = m.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(5.0, 1e-6));
  REQUIRE_THAT(p[1], WithinAbs(9.0, 1e-6));
}

// Forward kernel unit tests for B-003 ops: cat / split / index_select /
// gather. Gradcheck lives in tests/autograd/test_gradcheck.cpp so the
// numerical + analytic agreement is shared across all autograd ops.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Indexing.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

namespace {
constexpr double kEps = 1e-6;
}

TEST_CASE("cat: along dim 0", "[ops][indexing][cat]") {
  auto a = Tensor::from_vector<float>({1, 2, 3, 4}, {2, 2});
  auto b = Tensor::from_vector<float>({5, 6, 7, 8, 9, 10}, {3, 2});
  Tensor c = ops::cat({a, b}, 0);
  REQUIRE(c.shape() == Shape({5, 2}));
  const float* p = c.data_ptr<float>();
  for (int i = 0; i < 10; ++i) REQUIRE_THAT(p[i], WithinAbs(i + 1.0, kEps));
}

TEST_CASE("cat: along dim 1 with negative dim", "[ops][indexing][cat]") {
  auto a = Tensor::from_vector<float>({1, 2, 3, 4}, {2, 2});
  auto b = Tensor::from_vector<float>({10, 20, 30, 40, 50, 60}, {2, 3});
  Tensor c = ops::cat({a, b}, -1);
  REQUIRE(c.shape() == Shape({2, 5}));
  const float* p = c.data_ptr<float>();
  // Row 0: [1, 2, 10, 20, 30]; Row 1: [3, 4, 40, 50, 60]
  const float expected[] = {1, 2, 10, 20, 30, 3, 4, 40, 50, 60};
  for (int i = 0; i < 10; ++i) REQUIRE_THAT(p[i], WithinAbs(expected[i], kEps));
}

TEST_CASE("cat: single tensor is identity (but independent storage)",
          "[ops][indexing][cat]") {
  auto a = Tensor::from_vector<float>({1, 2, 3, 4}, {2, 2});
  Tensor c = ops::cat({a}, 0);
  REQUIRE(c.shape() == a.shape());
  const float* pa = a.data_ptr<float>();
  const float* pc = c.data_ptr<float>();
  for (int i = 0; i < 4; ++i) REQUIRE_THAT(pc[i], WithinAbs(pa[i], kEps));
  REQUIRE(pa != pc);
}

TEST_CASE("cat: shape / dtype mismatches throw", "[ops][indexing][cat]") {
  auto a = Tensor::from_vector<float>({1, 2, 3, 4}, {2, 2});
  auto b = Tensor::from_vector<float>({5, 6, 7, 8, 9}, {1, 5});
  REQUIRE_THROWS(ops::cat({a, b}, 0));  // non-cat dim differs
}

TEST_CASE("split: equal-sized chunks", "[ops][indexing][split]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {6});
  auto parts = ops::split(x, 2, 0);
  REQUIRE(parts.size() == 3);
  for (int i = 0; i < 3; ++i) REQUIRE(parts[i].shape() == Shape({2}));
  const float* p0 = parts[0].data_ptr<float>();
  const float* p1 = parts[1].data_ptr<float>();
  const float* p2 = parts[2].data_ptr<float>();
  REQUIRE_THAT(p0[0], WithinAbs(1.0, kEps));
  REQUIRE_THAT(p0[1], WithinAbs(2.0, kEps));
  REQUIRE_THAT(p1[0], WithinAbs(3.0, kEps));
  REQUIRE_THAT(p1[1], WithinAbs(4.0, kEps));
  REQUIRE_THAT(p2[0], WithinAbs(5.0, kEps));
  REQUIRE_THAT(p2[1], WithinAbs(6.0, kEps));
}

TEST_CASE("split: uneven last chunk", "[ops][indexing][split]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5}, {5});
  auto parts = ops::split(x, 2, 0);
  REQUIRE(parts.size() == 3);
  REQUIRE(parts[0].shape() == Shape({2}));
  REQUIRE(parts[1].shape() == Shape({2}));
  REQUIRE(parts[2].shape() == Shape({1}));
  REQUIRE_THAT(*parts[2].data_ptr<float>(), WithinAbs(5.0, kEps));
}

TEST_CASE("split_with_sizes: irregular splits", "[ops][indexing][split]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6, 7, 8}, {2, 4});
  auto parts = ops::split_with_sizes(x, {1, 3}, 1);
  REQUIRE(parts.size() == 2);
  REQUIRE(parts[0].shape() == Shape({2, 1}));
  REQUIRE(parts[1].shape() == Shape({2, 3}));
  const float* p0 = parts[0].data_ptr<float>();
  const float* p1 = parts[1].data_ptr<float>();
  REQUIRE_THAT(p0[0], WithinAbs(1.0, kEps));
  REQUIRE_THAT(p0[1], WithinAbs(5.0, kEps));
  REQUIRE_THAT(p1[0], WithinAbs(2.0, kEps));
  REQUIRE_THAT(p1[1], WithinAbs(3.0, kEps));
  REQUIRE_THAT(p1[2], WithinAbs(4.0, kEps));
  REQUIRE_THAT(p1[3], WithinAbs(6.0, kEps));
  REQUIRE_THAT(p1[4], WithinAbs(7.0, kEps));
  REQUIRE_THAT(p1[5], WithinAbs(8.0, kEps));
}

TEST_CASE("split_with_sizes: wrong total throws", "[ops][indexing][split]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4}, {4});
  REQUIRE_THROWS(ops::split_with_sizes(x, {1, 2}, 0));  // 1 + 2 != 4
}

TEST_CASE("cat . split is identity", "[ops][indexing][cat][split]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                       {3, 4});
  auto parts = ops::split(x, 2, 1);
  Tensor y = ops::cat(parts, 1);
  REQUIRE(y.shape() == x.shape());
  const float* px = x.data_ptr<float>();
  const float* py = y.data_ptr<float>();
  for (int i = 0; i < 12; ++i) REQUIRE_THAT(py[i], WithinAbs(px[i], kEps));
}

TEST_CASE("index_select: basic rows", "[ops][indexing][index_select]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
  auto idx = Tensor::from_vector<int64_t>({2, 0, 2}, {3});
  Tensor y = ops::index_select(x, 0, idx);
  REQUIRE(y.shape() == Shape({3, 3}));
  const float* p = y.data_ptr<float>();
  const float expected[] = {7, 8, 9, 1, 2, 3, 7, 8, 9};
  for (int i = 0; i < 9; ++i) REQUIRE_THAT(p[i], WithinAbs(expected[i], kEps));
}

TEST_CASE("index_select: columns along dim 1", "[ops][indexing][index_select]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto idx = Tensor::from_vector<int64_t>({1, 0, 2, 1}, {4});
  Tensor y = ops::index_select(x, 1, idx);
  REQUIRE(y.shape() == Shape({2, 4}));
  const float* p = y.data_ptr<float>();
  // row 0: [2, 1, 3, 2]; row 1: [5, 4, 6, 5]
  const float expected[] = {2, 1, 3, 2, 5, 4, 6, 5};
  for (int i = 0; i < 8; ++i) REQUIRE_THAT(p[i], WithinAbs(expected[i], kEps));
}

TEST_CASE("index_select: out-of-range index throws",
          "[ops][indexing][index_select]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4}, {2, 2});
  auto bad = Tensor::from_vector<int64_t>({2}, {1});  // 2 >= 2
  REQUIRE_THROWS(ops::index_select(x, 0, bad));
}

TEST_CASE("gather: dim 1 element-wise", "[ops][indexing][gather]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto idx = Tensor::from_vector<int64_t>({0, 2, 1, 0}, {2, 2});
  Tensor y = ops::gather(x, 1, idx);
  REQUIRE(y.shape() == Shape({2, 2}));
  const float* p = y.data_ptr<float>();
  // row 0: [x[0,0]=1, x[0,2]=3]; row 1: [x[1,1]=5, x[1,0]=4]
  REQUIRE_THAT(p[0], WithinAbs(1.0, kEps));
  REQUIRE_THAT(p[1], WithinAbs(3.0, kEps));
  REQUIRE_THAT(p[2], WithinAbs(5.0, kEps));
  REQUIRE_THAT(p[3], WithinAbs(4.0, kEps));
}

TEST_CASE("gather: dim 0 picks rows by column", "[ops][indexing][gather]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {3, 2});
  auto idx = Tensor::from_vector<int64_t>({2, 0, 1, 2}, {2, 2});
  Tensor y = ops::gather(x, 0, idx);
  REQUIRE(y.shape() == Shape({2, 2}));
  const float* p = y.data_ptr<float>();
  // out[i,j] = x[idx[i,j], j]
  // out[0,0] = x[2,0] = 5; out[0,1] = x[0,1] = 2
  // out[1,0] = x[1,0] = 3; out[1,1] = x[2,1] = 6
  REQUIRE_THAT(p[0], WithinAbs(5.0, kEps));
  REQUIRE_THAT(p[1], WithinAbs(2.0, kEps));
  REQUIRE_THAT(p[2], WithinAbs(3.0, kEps));
  REQUIRE_THAT(p[3], WithinAbs(6.0, kEps));
}

TEST_CASE("gather: mismatched rank throws", "[ops][indexing][gather]") {
  auto x = Tensor::from_vector<float>({1, 2, 3, 4}, {2, 2});
  auto bad_idx = Tensor::from_vector<int64_t>({0, 1}, {2});  // rank 1 vs 2
  REQUIRE_THROWS(ops::gather(x, 0, bad_idx));
}

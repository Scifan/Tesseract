#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/utils/Logging.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

TEST_CASE("matmul: 2x3 @ 3x2 float32", "[ops][matmul]") {
  auto a = Tensor::from_vector<float>({1, 2, 3,
                                       4, 5, 6}, {2, 3});
  auto b = Tensor::from_vector<float>({7,  8,
                                       9,  10,
                                       11, 12}, {3, 2});
  auto c = ops::matmul(a, b);
  REQUIRE(c.shape() == Shape({2, 2}));
  const float* p = c.data_ptr<float>();
  // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
  REQUIRE_THAT(p[0], WithinAbs(58.0,  1e-5));
  REQUIRE_THAT(p[1], WithinAbs(64.0,  1e-5));
  REQUIRE_THAT(p[2], WithinAbs(139.0, 1e-5));
  REQUIRE_THAT(p[3], WithinAbs(154.0, 1e-5));
}

TEST_CASE("matmul: identity * x = x", "[ops][matmul]") {
  auto eye = Tensor::from_vector<float>({1, 0, 0,
                                         0, 1, 0,
                                         0, 0, 1}, {3, 3});
  auto x = Tensor::from_vector<float>({1, 2,
                                       3, 4,
                                       5, 6}, {3, 2});
  auto r = ops::matmul(eye, x);
  REQUIRE(r.shape() == Shape({3, 2}));
  const float* p = r.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(1.0, 1e-6));
  REQUIRE_THAT(p[1], WithinAbs(2.0, 1e-6));
  REQUIRE_THAT(p[5], WithinAbs(6.0, 1e-6));
}

TEST_CASE("matmul: float64", "[ops][matmul]") {
  auto a = Tensor::from_vector<double>({1, 2, 3, 4}, {2, 2});
  auto b = Tensor::from_vector<double>({5, 6, 7, 8}, {2, 2});
  auto c = ops::matmul(a, b);
  const double* p = c.data_ptr<double>();
  REQUIRE_THAT(p[0], WithinAbs(19.0, 1e-12));
  REQUIRE_THAT(p[1], WithinAbs(22.0, 1e-12));
  REQUIRE_THAT(p[2], WithinAbs(43.0, 1e-12));
  REQUIRE_THAT(p[3], WithinAbs(50.0, 1e-12));
}

TEST_CASE("matmul: non-contiguous input is materialized", "[ops][matmul]") {
  auto a = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto at = a.transpose(0, 1);  // 3x2 non-contig
  auto b = Tensor::from_vector<float>({1, 2, 3, 4}, {2, 2});
  auto c = ops::matmul(at, b);
  REQUIRE(c.shape() == Shape({3, 2}));
  // at = [[1,4],[2,5],[3,6]]; at @ [[1,2],[3,4]] = [[13,18],[17,24],[21,30]]
  const float* p = c.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(13.0, 1e-6));
  REQUIRE_THAT(p[1], WithinAbs(18.0, 1e-6));
  REQUIRE_THAT(p[2], WithinAbs(17.0, 1e-6));
  REQUIRE_THAT(p[3], WithinAbs(24.0, 1e-6));
  REQUIRE_THAT(p[4], WithinAbs(21.0, 1e-6));
  REQUIRE_THAT(p[5], WithinAbs(30.0, 1e-6));
}

TEST_CASE("matmul: shape mismatch throws", "[ops][matmul]") {
  auto a = Tensor::zeros({2, 3}, DType::Float32);
  auto b = Tensor::zeros({4, 2}, DType::Float32);
  REQUIRE_THROWS_AS(ops::matmul(a, b), tesseract::Error);
}

namespace {

// Reference batched matmul built out of per-slab `ops::matmul`. It explicitly
// walks the broadcast grid so tests can pin the element-wise behaviour of the
// batched kernel against a known-good rank-2 implementation.
//
// Intentionally slow and independent from the code-under-test so it catches
// regressions in either direction.
tesseract::Tensor batched_matmul_reference(const tesseract::Tensor& lhs,
                                           const tesseract::Tensor& rhs) {
  using namespace tesseract;
  const auto& ls = lhs.shape();
  const auto& rs = rhs.shape();
  const int64_t M = ls[ls.rank() - 2];
  const int64_t K = ls[ls.rank() - 1];
  const int64_t N = rs[rs.rank() - 1];
  // Broadcast batch shapes (same logic the kernel uses; fine to duplicate
  // here since we want the reference to be obviously correct).
  const std::size_t lb_r = ls.rank() - 2;
  const std::size_t rb_r = rs.rank() - 2;
  const std::size_t ob_r = std::max(lb_r, rb_r);
  std::vector<int64_t> ob(ob_r, 1);
  for (std::size_t i = 0; i < ob_r; ++i) {
    const int64_t da = (i < ob_r - lb_r) ? 1 : ls[i - (ob_r - lb_r)];
    const int64_t db = (i < ob_r - rb_r) ? 1 : rs[i - (ob_r - rb_r)];
    ob[i] = std::max(da, db);
  }
  Shape out_shape;
  out_shape.resize(ob_r + 2);
  for (std::size_t i = 0; i < ob_r; ++i) out_shape[i] = ob[i];
  out_shape[ob_r + 0] = M;
  out_shape[ob_r + 1] = N;
  Tensor out = Tensor::zeros(out_shape, lhs.dtype(), lhs.device());

  // Iterate out-batch grid. For each batch position, pick the right (M,K)
  // and (K,N) slabs and write into the (M,N) slab of out.
  const int64_t batch_numel = [&] { int64_t p = 1; for (auto d : ob) p *= d; return p; }();
  for (int64_t flat = 0; flat < batch_numel; ++flat) {
    std::vector<int64_t> idx(ob_r, 0);
    int64_t rem = flat;
    for (std::size_t d = ob_r; d-- > 0;) { idx[d] = rem % ob[d]; rem /= ob[d]; }
    // Pick the lhs / rhs slice indices (broadcast: clamp to 0 where dim==1).
    auto pick = [&](const Shape& shape_full, std::size_t batch_rank)
        -> std::vector<int64_t> {
      std::vector<int64_t> s(batch_rank);
      for (std::size_t d = 0; d < batch_rank; ++d) {
        const int64_t dim = shape_full[d];
        const int64_t i = idx[d + (ob_r - batch_rank)];
        s[d] = (dim == 1) ? 0 : i;
      }
      return s;
    };
    std::vector<int64_t> li = pick(ls, lb_r);
    std::vector<int64_t> ri = pick(rs, rb_r);

    // Compute linear offsets into the *contiguous* lhs/rhs buffers for the
    // picked batch slab.
    int64_t a_off = 0;
    {
      int64_t stride = M * K;
      for (std::size_t d = lb_r; d-- > 0;) {
        a_off += li[d] * stride;
        stride *= ls[d];
      }
    }
    int64_t b_off = 0;
    {
      int64_t stride = K * N;
      for (std::size_t d = rb_r; d-- > 0;) {
        b_off += ri[d] * stride;
        stride *= rs[d];
      }
    }
    const float* pa = lhs.data_ptr<float>() + a_off;
    const float* pb = rhs.data_ptr<float>() + b_off;
    float* pc = out.data_ptr<float>() + flat * M * N;
    // Scalar triple loop. Order matters only for numerical noise; we test
    // against `1e-5` so this naive loop nest is fine.
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) acc += pa[m * K + k] * pb[k * N + n];
        pc[m * N + n] = acc;
      }
    }
  }
  return out;
}

}  // namespace

TEST_CASE("matmul: rank-3 both operands batched", "[ops][matmul][batched]") {
  // A: [B=2, M=3, K=4], B: [B=2, K=4, N=5] -> out: [2, 3, 5]
  auto a = Tensor::empty({2, 3, 4}, DType::Float32);
  auto b = Tensor::empty({2, 4, 5}, DType::Float32);
  // Fill with a deterministic LCG so we exercise distinct batch slabs.
  uint64_t s = 0xDEADBEEFCAFEBABEULL;
  auto fill = [&](Tensor& t) {
    const int64_t n = t.numel();
    float* p = t.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      p[i] = static_cast<float>(int(s >> 32) % 11) * 0.1f;
    }
  };
  fill(a);
  fill(b);

  auto out = ops::matmul(a, b);
  REQUIRE(out.shape() == Shape({2, 3, 5}));

  auto ref = batched_matmul_reference(a, b);
  const float* po = out.data_ptr<float>();
  const float* pr = ref.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); ++i) {
    REQUIRE_THAT(po[i], WithinAbs(pr[i], 1e-5));
  }
}

TEST_CASE("matmul: rank-2 lhs broadcasts across rank-3 rhs",
          "[ops][matmul][batched][broadcast]") {
  // Classic "embedding table times batch of hidden states" shape. lhs is a
  // single [K, N] weight; rhs is [B, M, K] — but we express it the other way
  // around: lhs = [M, K] shared, rhs = [B, K, N]. Result = [B, M, N].
  auto lhs = Tensor::empty({3, 4}, DType::Float32);
  auto rhs = Tensor::empty({2, 4, 5}, DType::Float32);
  uint64_t s = 0x123456789ABCDEF0ULL;
  auto fill = [&](Tensor& t) {
    float* p = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      p[i] = static_cast<float>(int(s >> 32) % 7) * 0.25f;
    }
  };
  fill(lhs);
  fill(rhs);
  auto out = ops::matmul(lhs, rhs);
  REQUIRE(out.shape() == Shape({2, 3, 5}));
  auto ref = batched_matmul_reference(lhs, rhs);
  const float* po = out.data_ptr<float>();
  const float* pr = ref.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); ++i) {
    REQUIRE_THAT(po[i], WithinAbs(pr[i], 1e-5));
  }
}

TEST_CASE("matmul: rank-3 lhs against rank-2 rhs (projection)",
          "[ops][matmul][batched][broadcast]") {
  // Typical "apply the same projection to every sequence position":
  // lhs = [B, M, K], rhs = [K, N] -> out = [B, M, N].
  auto lhs = Tensor::empty({2, 3, 4}, DType::Float32);
  auto rhs = Tensor::empty({4, 5}, DType::Float32);
  uint64_t s = 0xABCDEF01234567UL;
  auto fill = [&](Tensor& t) {
    float* p = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      p[i] = static_cast<float>(int(s >> 32) % 9) * 0.125f;
    }
  };
  fill(lhs);
  fill(rhs);
  auto out = ops::matmul(lhs, rhs);
  REQUIRE(out.shape() == Shape({2, 3, 5}));
  auto ref = batched_matmul_reference(lhs, rhs);
  const float* po = out.data_ptr<float>();
  const float* pr = ref.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); ++i) {
    REQUIRE_THAT(po[i], WithinAbs(pr[i], 1e-5));
  }
}

TEST_CASE("matmul: rank-4 two-level broadcast",
          "[ops][matmul][batched][broadcast]") {
  // lhs = [2, 1, M=3, K=4]  (second batch dim broadcasts)
  // rhs = [1, 3, K=4, N=5]  (first batch dim broadcasts)
  // out = [2, 3, 3, 5]. Stresses broadcast on multiple leading axes.
  auto lhs = Tensor::empty({2, 1, 3, 4}, DType::Float32);
  auto rhs = Tensor::empty({1, 3, 4, 5}, DType::Float32);
  uint64_t s = 0x55AA55AA00FF00FFULL;
  auto fill = [&](Tensor& t) {
    float* p = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      p[i] = static_cast<float>(int(s >> 32) % 5) * 0.1f;
    }
  };
  fill(lhs);
  fill(rhs);
  auto out = ops::matmul(lhs, rhs);
  REQUIRE(out.shape() == Shape({2, 3, 3, 5}));
  auto ref = batched_matmul_reference(lhs, rhs);
  const float* po = out.data_ptr<float>();
  const float* pr = ref.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); ++i) {
    REQUIRE_THAT(po[i], WithinAbs(pr[i], 1e-5));
  }
}

TEST_CASE("matmul: batched inner-dim mismatch throws",
          "[ops][matmul][batched]") {
  auto a = Tensor::zeros({2, 3, 4}, DType::Float32);
  auto b = Tensor::zeros({2, 5, 6}, DType::Float32);
  REQUIRE_THROWS_AS(ops::matmul(a, b), tesseract::Error);
}

TEST_CASE("matmul: batch dim broadcast mismatch throws",
          "[ops][matmul][batched]") {
  // Leading batch dims [2] vs [3] don't broadcast — neither is 1.
  auto a = Tensor::zeros({2, 3, 4}, DType::Float32);
  auto b = Tensor::zeros({3, 4, 5}, DType::Float32);
  REQUIRE_THROWS_AS(ops::matmul(a, b), tesseract::Error);
}

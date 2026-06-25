// M2J — CPU-side correctness + autograd for `ops::attention(q, k, v,
// mask, causal, dropout_p)`. The composite implementation (matmul +
// softmax + matmul, Q pre-scaled by 1/√d, optional additive + causal
// masks) runs entirely through already-validated M0/M1 primitives, so
// these tests anchor the contract (shapes / mask semantics / grad
// flow) rather than re-validate the primitives themselves. The FA3
// kernel vendoring + numerical parity with `flash-attention-3`'s
// reference kernel land in M2L behind a Hopper-only guard.

#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Attention.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Softmax.hpp"
#include "tesseract/utils/Logging.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

namespace {

// Hand-computed reference SDPA: out = softmax(Q·Kᵀ / √d + mask + causal) · V.
// Operates on float data in rank-3 [B, S, D] / [B, S_q, S_k] layouts.
// Uses stable softmax (subtract per-row max). `mask` may be empty.
std::vector<float> reference_sdpa(
    const std::vector<float>& q, int64_t B, int64_t S_q, int64_t D_q,
    const std::vector<float>& k, int64_t S_k,
    const std::vector<float>& v, int64_t D_v,
    const std::vector<float>* mask,  // [B, S_q, S_k] or null
    bool causal) {
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D_q));
  const float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> out(static_cast<std::size_t>(B) * S_q * D_v, 0.0f);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < S_q; ++i) {
      std::vector<float> s(S_k);
      for (int64_t j = 0; j < S_k; ++j) {
        float acc = 0.0f;
        for (int64_t d = 0; d < D_q; ++d) {
          acc += q[(b * S_q + i) * D_q + d] * k[(b * S_k + j) * D_q + d];
        }
        s[j] = acc * inv_sqrt_d;
        if (mask) s[j] += (*mask)[(b * S_q + i) * S_k + j];
        if (causal && j > i) s[j] = neg_inf;
      }
      float m = s[0];
      for (int64_t j = 1; j < S_k; ++j) m = std::max(m, s[j]);
      float z = 0.0f;
      for (int64_t j = 0; j < S_k; ++j) z += std::exp(s[j] - m);
      for (int64_t d = 0; d < D_v; ++d) {
        float acc = 0.0f;
        for (int64_t j = 0; j < S_k; ++j) {
          const float p = std::exp(s[j] - m) / z;
          acc += p * v[(b * S_k + j) * D_v + d];
        }
        out[(b * S_q + i) * D_v + d] = acc;
      }
    }
  }
  return out;
}

}  // namespace

TEST_CASE("attention: rank-3 forward matches hand-computed reference",
          "[ops][attention]") {
  const int64_t B = 2, S = 4, D = 6;

  std::vector<float> q(B * S * D), k(B * S * D), v(B * S * D);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = 0.01f * (i - 0.5f * q.size());
  for (std::size_t i = 0; i < k.size(); ++i) k[i] = 0.02f * (i - 0.5f * k.size());
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.03f * (i - 0.5f * v.size());

  Tensor tq = Tensor::from_vector(q, {B, S, D});
  Tensor tk = Tensor::from_vector(k, {B, S, D});
  Tensor tv = Tensor::from_vector(v, {B, S, D});
  Tensor out = ops::attention(tq, tk, tv);
  REQUIRE(out.shape() == Shape({B, S, D}));

  auto ref = reference_sdpa(q, B, S, D, k, S, v, D, /*mask=*/nullptr, /*causal=*/false);
  const float* got = out.data_ptr<float>();
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(got[i], WithinAbs(ref[i], 1e-5f));
  }
}

TEST_CASE("attention: causal mask blocks future tokens",
          "[ops][attention][causal]") {
  const int64_t B = 1, S = 5, D = 4;

  std::vector<float> q(B * S * D), k(B * S * D), v(B * S * D);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = 0.05f * i;
  for (std::size_t i = 0; i < k.size(); ++i) k[i] = -0.03f * i + 0.1f;
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.02f * i - 0.5f;

  Tensor tq = Tensor::from_vector(q, {B, S, D});
  Tensor tk = Tensor::from_vector(k, {B, S, D});
  Tensor tv = Tensor::from_vector(v, {B, S, D});
  Tensor out = ops::attention(tq, tk, tv, /*mask=*/Tensor{}, /*causal=*/true);
  REQUIRE(out.shape() == Shape({B, S, D}));

  auto ref = reference_sdpa(q, B, S, D, k, S, v, D, /*mask=*/nullptr, /*causal=*/true);
  const float* got = out.data_ptr<float>();
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(got[i], WithinAbs(ref[i], 1e-5f));
  }

  // Sanity: the first query row should depend ONLY on the first V row
  // (since softmax over a single unmasked position is a delta). Perturb
  // V[1:] and confirm out[0,0,:] is unchanged.
  std::vector<float> v2 = v;
  for (int64_t i = 1; i < S; ++i) {
    for (int64_t d = 0; d < D; ++d) v2[i * D + d] *= 10.0f;
  }
  Tensor tv2 = Tensor::from_vector(v2, {B, S, D});
  Tensor out2 = ops::attention(tq, tk, tv2, /*mask=*/Tensor{}, /*causal=*/true);
  const float* got2 = out2.data_ptr<float>();
  for (int64_t d = 0; d < D; ++d) {
    REQUIRE_THAT(got2[d], WithinAbs(got[d], 1e-5f));
  }
}

TEST_CASE("attention: additive -inf mask zeros out specific keys",
          "[ops][attention][mask]") {
  const int64_t B = 1, S_q = 2, S_k = 3, D = 4;

  std::vector<float> q(B * S_q * D), k(B * S_k * D), v(B * S_k * D);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = 0.1f * i + 0.01f;
  for (std::size_t i = 0; i < k.size(); ++i) k[i] = -0.05f * i + 0.2f;
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.03f * i;

  // Mask out key position 2 for both queries.
  const float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> mask(B * S_q * S_k, 0.0f);
  mask[0 * S_k + 2] = neg_inf;
  mask[1 * S_k + 2] = neg_inf;

  Tensor tq = Tensor::from_vector(q, {B, S_q, D});
  Tensor tk = Tensor::from_vector(k, {B, S_k, D});
  Tensor tv = Tensor::from_vector(v, {B, S_k, D});
  Tensor tm = Tensor::from_vector(mask, {B, S_q, S_k});
  Tensor out = ops::attention(tq, tk, tv, tm);
  REQUIRE(out.shape() == Shape({B, S_q, D}));

  auto ref = reference_sdpa(q, B, S_q, D, k, S_k, v, D, &mask, /*causal=*/false);
  const float* got = out.data_ptr<float>();
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(got[i], WithinAbs(ref[i], 1e-5f));
  }

  // Also sanity-check: perturbing V[:, 2, :] must NOT change the output
  // because key 2 is fully masked.
  std::vector<float> v2 = v;
  for (int64_t d = 0; d < D; ++d) v2[2 * D + d] = 99.0f;
  Tensor tv2 = Tensor::from_vector(v2, {B, S_k, D});
  Tensor out2 = ops::attention(tq, tk, tv2, tm);
  const float* got2 = out2.data_ptr<float>();
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(got2[i], WithinAbs(got[i], 1e-5f));
  }
}

TEST_CASE("attention: rank-4 batched with head dim",
          "[ops][attention]") {
  const int64_t B = 2, H = 3, S = 5, D = 4;

  std::vector<float> q(B * H * S * D), k(B * H * S * D), v(B * H * S * D);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = 0.01f * static_cast<float>(i);
  for (std::size_t i = 0; i < k.size(); ++i) k[i] = -0.02f * static_cast<float>(i);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.015f * static_cast<float>(i);

  Tensor tq = Tensor::from_vector(q, {B, H, S, D});
  Tensor tk = Tensor::from_vector(k, {B, H, S, D});
  Tensor tv = Tensor::from_vector(v, {B, H, S, D});
  Tensor out = ops::attention(tq, tk, tv);
  REQUIRE(out.shape() == Shape({B, H, S, D}));

  // Compare against the same reference, flattening the (B, H) batch.
  auto ref = reference_sdpa(q, B * H, S, D, k, S, v, D, /*mask=*/nullptr, /*causal=*/false);
  const float* got = out.data_ptr<float>();
  for (std::size_t i = 0; i < ref.size(); ++i) {
    REQUIRE_THAT(got[i], WithinAbs(ref[i], 1e-5f));
  }
}

TEST_CASE("attention: autograd backward flows through composite",
          "[ops][attention][autograd]") {
  const int64_t B = 1, S = 3, D = 4;

  std::vector<float> q(B * S * D), k(B * S * D), v(B * S * D);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = 0.1f * static_cast<float>(i) + 0.01f;
  for (std::size_t i = 0; i < k.size(); ++i) k[i] = -0.05f * static_cast<float>(i) + 0.1f;
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.02f * static_cast<float>(i) - 0.3f;

  Tensor tq = Tensor::from_vector(q, {B, S, D});
  Tensor tk = Tensor::from_vector(k, {B, S, D});
  Tensor tv = Tensor::from_vector(v, {B, S, D});
  tq.set_requires_grad(true);
  tk.set_requires_grad(true);
  tv.set_requires_grad(true);
  Tensor y = ops::attention(tq, tk, tv);

  // Seed grad = ones so gradients are dominated by the chain structure.
  std::vector<float> g_data(B * S * D, 1.0f);
  Tensor g = Tensor::from_vector(g_data, {B, S, D});
  Engine::backward(y, g);

  REQUIRE(tq.grad().defined());
  REQUIRE(tk.grad().defined());
  REQUIRE(tv.grad().defined());
  REQUIRE(tq.grad().shape() == tq.shape());
  REQUIRE(tk.grad().shape() == tk.shape());
  REQUIRE(tv.grad().shape() == tv.shape());

  // Every gradient must be finite (no NaN / Inf from the softmax
  // saturation path we deliberately avoided by scaling Q by 1/√d).
  auto check_finite = [](const Tensor& gt) {
    const float* p = gt.data_ptr<float>();
    for (int64_t i = 0; i < gt.numel(); ++i) {
      REQUIRE(std::isfinite(p[i]));
    }
  };
  check_finite(tq.grad());
  check_finite(tk.grad());
  check_finite(tv.grad());

  // Numerical check of the V gradient specifically: V's gradient is
  // P^T · g_out (no second-order terms through softmax), so we can
  // sanity-check it without finite-difference on the full attention
  // tape. Compute P explicitly via softmax(Q·Kᵀ/√d) with gradients off.
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
  std::vector<float> scores(B * S * S);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < S; ++i) {
      for (int64_t j = 0; j < S; ++j) {
        float acc = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
          acc += q[(b * S + i) * D + d] * k[(b * S + j) * D + d];
        }
        scores[(b * S + i) * S + j] = acc * inv_sqrt_d;
      }
    }
  }
  std::vector<float> p(B * S * S);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < S; ++i) {
      float m = scores[(b * S + i) * S];
      for (int64_t j = 1; j < S; ++j) m = std::max(m, scores[(b * S + i) * S + j]);
      float z = 0.0f;
      for (int64_t j = 0; j < S; ++j) z += std::exp(scores[(b * S + i) * S + j] - m);
      for (int64_t j = 0; j < S; ++j) {
        p[(b * S + i) * S + j] = std::exp(scores[(b * S + i) * S + j] - m) / z;
      }
    }
  }
  // grad_V[b, j, d] = sum_i P[b, i, j] · g[b, i, d]   (with g = 1 everywhere)
  //                 = (sum_i P[b, i, j]) · 1
  const float* got_gv = tv.grad().data_ptr<float>();
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t j = 0; j < S; ++j) {
      float expected = 0.0f;
      for (int64_t i = 0; i < S; ++i) expected += p[(b * S + i) * S + j];
      for (int64_t d = 0; d < D; ++d) {
        REQUIRE_THAT(got_gv[(b * S + j) * D + d], WithinAbs(expected, 1e-5f));
      }
    }
  }
}

TEST_CASE("attention: input validation",
          "[ops][attention]") {
  const int64_t B = 1, S = 2, D = 3;
  Tensor tq = Tensor::zeros({B, S, D});
  Tensor tk = Tensor::zeros({B, S, D});
  Tensor tv = Tensor::zeros({B, S, D});

  // dropout_p != 0 is explicitly unsupported in M2J.
  REQUIRE_THROWS_AS(ops::attention(tq, tk, tv, Tensor{}, false, 0.1),
                    tesseract::Error);

  // causal=true with S_q != S_k is rejected.
  Tensor tk_short = Tensor::zeros({B, S + 1, D});
  Tensor tv_short = Tensor::zeros({B, S + 1, D});
  REQUIRE_THROWS_AS(ops::attention(tq, tk_short, tv_short, Tensor{}, true),
                    tesseract::Error);

  // dtype mismatch between q and v is rejected.
  Tensor tv_f64 = Tensor::zeros({B, S, D}, DType::Float64);
  REQUIRE_THROWS_AS(ops::attention(tq, tk, tv_f64),
                    tesseract::Error);

  // head_dim mismatch (q D vs k D).
  Tensor tk_badD = Tensor::zeros({B, S, D + 1});
  REQUIRE_THROWS_AS(ops::attention(tq, tk_badD, tv),
                    tesseract::Error);
}

// M4 Track C1 (B-044) — transformer FFN + normalization primitives through
// the CPU JitEngine.
//
// These build the exact primitive chains that `ops::rms_norm` and
// `ops::swiglu_silu_gate` decompose into (mul / mean / add / sqrt / div / mul
// and sigmoid / mul / mul), capture them through `GraphScope`, JIT-compile via
// `ir::JitEngine`, and assert the result matches the eager op within FP32
// tolerance. This is the concrete down payment on the "one IR" thesis: a real
// transformer sub-block executes through the dialect → linalg → LLVM pipeline.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ir/JitEngine.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Attention.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/ops/RotaryEmbedding.hpp"
#include "tesseract/ops/Softmax.hpp"
#include "tesseract/ops/View.hpp"

using namespace tesseract;

namespace {

Tensor make_random(std::initializer_list<int64_t> dims, uint64_t seed,
                   float lo = -1.0f, float hi = 1.0f) {
  auto t = Tensor::empty(dims, DType::Float32);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
  return t;
}

void require_close(const Tensor& a, const Tensor& b, double tol) {
  REQUIRE(a.numel() == b.numel());
  const float* pa = a.data_ptr<float>();
  const float* pb = b.data_ptr<float>();
  for (int64_t i = 0; i < a.numel(); ++i) {
    const double diff = std::fabs(static_cast<double>(pa[i] - pb[i]));
    INFO("element " << i << " jit=" << pa[i] << " eager=" << pb[i]
                    << " diff=" << diff);
    REQUIRE(diff <= tol);
  }
}

}  // namespace

TEST_CASE("jit: sigmoid matches eager", "[ir][jit][transformer]") {
  auto x = make_random({4, 5}, /*seed=*/2001, -3.0f, 3.0f);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    Tensor y = ops::sigmoid(x);
    graph::mark_output(y);
    captured = std::move(scope.graph());
  }
  std::unordered_map<graph::ValueId, Tensor> bind = {{captured.inputs()[0], x}};
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  Tensor eager = ops::sigmoid(x);
  require_close(jit_outs[0], eager, /*tol=*/1e-6);
}

TEST_CASE("jit: SwiGLU activation matches eager", "[ir][jit][transformer]") {
  // silu(gate) * up = (gate * sigmoid(gate)) * up — the Llama FFN activation.
  auto gate = make_random({3, 8}, /*seed=*/2101, -2.0f, 2.0f);
  auto up = make_random({3, 8}, /*seed=*/2102, -2.0f, 2.0f);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(gate, "gate");
    graph::bind_input(up, "up");
    Tensor s = ops::sigmoid(gate);
    Tensor silu = ops::mul(gate, s);
    Tensor out = ops::mul(silu, up);
    graph::mark_output(out);
    captured = std::move(scope.graph());
  }
  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], gate}, {captured.inputs()[1], up}};
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  Tensor eager = ops::swiglu_silu_gate(gate, up);
  require_close(jit_outs[0], eager, /*tol=*/1e-5);
}

TEST_CASE("jit: RMSNorm primitive chain matches eager",
          "[ir][jit][transformer]") {
  // RMSNorm(x) = (x / sqrt(mean(x*x, -1) + eps)) * weight, built from the
  // exact primitive chain `ops::rms_norm` records. eps is bound as a
  // pre-shaped [N,1] input so the add stays same-shape; the div/mul rely on
  // the existing broadcast_to lowering for [N,1]→[N,D] and [D]→[N,D].
  constexpr int64_t N = 4;
  constexpr int64_t D = 6;
  const double eps = 1e-5;
  auto x = make_random({N, D}, /*seed=*/2201, -1.5f, 1.5f);
  auto weight = make_random({D}, /*seed=*/2202, 0.5f, 1.5f);
  auto eps_t = Tensor::full({N, 1}, eps, DType::Float32);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    graph::bind_param(weight, "weight");
    graph::bind_input(eps_t, "eps");
    Tensor sq = ops::mul(x, x);
    Tensor ms = ops::mean(sq, /*dim=*/1, /*keepdim=*/true);  // [N,1]
    Tensor denom = ops::sqrt(ops::add(ms, eps_t));           // [N,1]
    Tensor yhat = ops::div(x, denom);                        // [N,D] / [N,1]
    Tensor y = ops::mul(yhat, weight);                       // [N,D] * [D]
    graph::mark_output(y);
    captured = std::move(scope.graph());
  }
  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], x},
      {captured.params()[0], weight},
      {captured.inputs()[1], eps_t}};
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  // Eager reference: same formula.
  Tensor sq = ops::mul(x, x);
  Tensor ms = ops::mean(sq, 1, true);
  Tensor denom = ops::sqrt(ops::add(ms, eps_t));
  Tensor yhat = ops::div(x, denom);
  Tensor eager = ops::mul(yhat, weight);

  require_close(jit_outs[0], eager, /*tol=*/1e-4);
}

TEST_CASE("jit: single-head softmax matches eager", "[ir][jit][transformer]") {
  auto x = make_random({5, 7}, /*seed=*/2301, -4.0f, 4.0f);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    Tensor y = ops::softmax(x, /*dim=*/-1);
    graph::mark_output(y);
    captured = std::move(scope.graph());
  }
  std::unordered_map<graph::ValueId, Tensor> bind = {{captured.inputs()[0], x}};
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  Tensor eager = ops::softmax(x, -1);
  require_close(jit_outs[0], eager, /*tol=*/1e-6);
}

TEST_CASE("jit: single-head scaled-dot-product attention matches eager",
          "[ir][jit][transformer]") {
  // out = softmax(scale·(Q·Kᵀ) + mask)·V, the attention compute path expressed
  // with rank-2 matmul + transpose + mul + add + softmax + matmul. Compared to
  // the eager `ops::attention` op (which applies the same 1/√d scale + additive
  // mask), so this validates the whole attention block through the JIT.
  constexpr int64_t S = 6;
  constexpr int64_t D = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  auto q = make_random({S, D}, /*seed=*/2401, -1.0f, 1.0f);
  auto k = make_random({S, D}, /*seed=*/2402, -1.0f, 1.0f);
  auto v = make_random({S, D}, /*seed=*/2403, -1.0f, 1.0f);
  auto scale_t = Tensor::full({1, 1}, scale, DType::Float32);
  auto zero_mask = Tensor::full({S, S}, 0.0, DType::Float32);

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(q, "q");
    graph::bind_input(k, "k");
    graph::bind_input(v, "v");
    graph::bind_input(scale_t, "scale");
    graph::bind_input(zero_mask, "mask");
    Tensor kt = ops::transpose(k, 0, 1);          // [D, S]
    Tensor scores = ops::matmul(q, kt);           // [S, S]
    Tensor scaled = ops::mul(scores, scale_t);    // broadcast [1,1]→[S,S]
    Tensor masked = ops::add(scaled, zero_mask);  // [S, S]
    Tensor probs = ops::softmax(masked, -1);      // [S, S]
    Tensor out = ops::matmul(probs, v);           // [S, D]
    graph::mark_output(out);
    captured = std::move(scope.graph());
  }
  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], q},       {captured.inputs()[1], k},
      {captured.inputs()[2], v},       {captured.inputs()[3], scale_t},
      {captured.inputs()[4], zero_mask}};
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  Tensor eager = ops::attention(q, k, v);
  require_close(jit_outs[0], eager, /*tol=*/1e-5);
}

TEST_CASE("jit: causal single-head attention matches eager",
          "[ir][jit][transformer]") {
  // Same chain with an upper-triangular -inf additive mask; compared to the
  // eager attention's built-in causal masking.
  constexpr int64_t S = 5;
  constexpr int64_t D = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const float kNegInf = -std::numeric_limits<float>::infinity();
  auto q = make_random({S, D}, /*seed=*/2501, -1.0f, 1.0f);
  auto k = make_random({S, D}, /*seed=*/2502, -1.0f, 1.0f);
  auto v = make_random({S, D}, /*seed=*/2503, -1.0f, 1.0f);
  auto scale_t = Tensor::full({1, 1}, scale, DType::Float32);

  auto mask = Tensor::empty({S, S}, DType::Float32);
  float* mp = mask.data_ptr<float>();
  for (int64_t i = 0; i < S; ++i)
    for (int64_t j = 0; j < S; ++j) mp[i * S + j] = (j > i) ? kNegInf : 0.0f;

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(q, "q");
    graph::bind_input(k, "k");
    graph::bind_input(v, "v");
    graph::bind_input(scale_t, "scale");
    graph::bind_input(mask, "mask");
    Tensor kt = ops::transpose(k, 0, 1);
    Tensor scores = ops::matmul(q, kt);
    Tensor scaled = ops::mul(scores, scale_t);
    Tensor masked = ops::add(scaled, mask);
    Tensor probs = ops::softmax(masked, -1);
    Tensor out = ops::matmul(probs, v);
    graph::mark_output(out);
    captured = std::move(scope.graph());
  }
  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], q},       {captured.inputs()[1], k},
      {captured.inputs()[2], v},       {captured.inputs()[3], scale_t},
      {captured.inputs()[4], mask}};
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  Tensor eager = ops::attention(q, k, v, /*mask=*/Tensor{}, /*causal=*/true);
  require_close(jit_outs[0], eager, /*tol=*/1e-5);
}

TEST_CASE("jit: multi-head attention (reshape/permute/batched matmul) "
          "matches eager",
          "[ir][jit][transformer]") {
  // Full multi-head SDPA built from the head-split path:
  //   [S, H*Dh] --view--> [S, H, Dh] --permute--> [H, S, Dh]
  //   scores = bmm(Qh, Khᵀ) · scale ; probs = softmax(scores)
  //   Oh = bmm(probs, Vh) --permute--> [S, H, Dh] --view--> [S, H*Dh]
  // Exercises ViewOp/PermuteOp lowering + rank-3 (batched) matmul through the
  // CPU JIT, compared against the same op chain run eagerly. This is the
  // multi-head reshape/permute path issue.md C1 asked for.
  constexpr int64_t S = 4;
  constexpr int64_t H = 2;
  constexpr int64_t Dh = 4;
  constexpr int64_t D = H * Dh;
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
  auto q = make_random({S, D}, /*seed=*/2701, -1.0f, 1.0f);
  auto k = make_random({S, D}, /*seed=*/2702, -1.0f, 1.0f);
  auto v = make_random({S, D}, /*seed=*/2703, -1.0f, 1.0f);
  auto scale_t = Tensor::full({1, 1, 1}, scale, DType::Float32);

  // Build the head-split attention given the three projected tensors.
  auto mha = [&](const Tensor& qi, const Tensor& ki, const Tensor& vi,
                 const Tensor& sc) {
    Tensor qh = ops::permute(ops::view(qi, {S, H, Dh}), {1, 0, 2});  // [H,S,Dh]
    Tensor kh = ops::permute(ops::view(ki, {S, H, Dh}), {1, 0, 2});
    Tensor vh = ops::permute(ops::view(vi, {S, H, Dh}), {1, 0, 2});
    Tensor kt = ops::transpose(kh, 1, 2);                 // [H, Dh, S]
    Tensor scores = ops::matmul(qh, kt);                  // [H, S, S]
    Tensor scaled = ops::mul(scores, sc);                 // broadcast scale
    Tensor probs = ops::softmax(scaled, -1);              // [H, S, S]
    Tensor oh = ops::matmul(probs, vh);                   // [H, S, Dh]
    Tensor o = ops::reshape(ops::permute(oh, {1, 0, 2}), {S, D});  // [S, D]
    return o;
  };

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(q, "q");
    graph::bind_input(k, "k");
    graph::bind_input(v, "v");
    graph::bind_input(scale_t, "scale");
    Tensor out = mha(q, k, v, scale_t);
    graph::mark_output(out);
    captured = std::move(scope.graph());
  }
  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], q},
      {captured.inputs()[1], k},
      {captured.inputs()[2], v},
      {captured.inputs()[3], scale_t}};
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  Tensor eager = mha(q, k, v, scale_t);
  require_close(jit_outs[0], eager, /*tol=*/1e-5);
}

TEST_CASE("jit: full multi-head Llama block matches eager",
          "[ir][jit][transformer]") {
  // End-to-end Llama block through the CPU JIT:
  //   h  = x + Wo · MHA(RoPE(x·Wq), RoPE(x·Wk), x·Wv)
  //   y  = h + Wd · (silu(h·Wg) ⊙ (h·Wu))
  // (RMSNorm omitted here to keep the binding set small; it is covered by its
  // own JIT test above. This validates that QKV proj + multi-head reshape/
  // permute + RoPE + batched SDPA + O-proj + residual + SwiGLU FFN compose and
  // execute through the dialect→linalg→LLVM pipeline at eager parity.)
  constexpr int64_t S = 4;
  constexpr int64_t H = 2;
  constexpr int64_t Dh = 4;
  constexpr int64_t D = H * Dh;
  constexpr int64_t Dff = 16;
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

  auto x = make_random({S, D}, /*seed=*/2801, -1.0f, 1.0f);
  auto wq = make_random({D, D}, /*seed=*/2802, -0.3f, 0.3f);
  auto wk = make_random({D, D}, /*seed=*/2803, -0.3f, 0.3f);
  auto wv = make_random({D, D}, /*seed=*/2804, -0.3f, 0.3f);
  auto wo = make_random({D, D}, /*seed=*/2805, -0.3f, 0.3f);
  auto wg = make_random({D, Dff}, /*seed=*/2806, -0.3f, 0.3f);
  auto wu = make_random({D, Dff}, /*seed=*/2807, -0.3f, 0.3f);
  auto wd = make_random({Dff, D}, /*seed=*/2808, -0.3f, 0.3f);
  auto scale_t = Tensor::full({1, 1, 1}, scale, DType::Float32);

  // RoPE tables over the per-head dim Dh.
  auto cos = Tensor::empty({S, Dh}, DType::Float32);
  auto sin = Tensor::empty({S, Dh}, DType::Float32);
  {
    float* cp = cos.data_ptr<float>();
    float* sp = sin.data_ptr<float>();
    for (int64_t p = 0; p < S; ++p)
      for (int64_t j = 0; j < Dh / 2; ++j) {
        const double theta =
            static_cast<double>(p) *
            std::pow(10000.0, -2.0 * static_cast<double>(j) /
                                  static_cast<double>(Dh));
        const float c = static_cast<float>(std::cos(theta));
        const float s = static_cast<float>(std::sin(theta));
        cp[p * Dh + 2 * j] = c;     cp[p * Dh + 2 * j + 1] = c;
        sp[p * Dh + 2 * j] = s;     sp[p * Dh + 2 * j + 1] = s;
      }
  }

  auto block = [&](const Tensor& xi) {
    Tensor q = ops::matmul(xi, wq);  // [S, D]
    Tensor k = ops::matmul(xi, wk);
    Tensor vv = ops::matmul(xi, wv);
    // Head split + RoPE per head: [S,D]->[S,H,Dh]->[H,S,Dh].
    Tensor qh = ops::permute(ops::view(q, {S, H, Dh}), {1, 0, 2});
    Tensor kh = ops::permute(ops::view(k, {S, H, Dh}), {1, 0, 2});
    Tensor vh = ops::permute(ops::view(vv, {S, H, Dh}), {1, 0, 2});
    qh = ops::rotary_embedding(qh, cos, sin);  // RoPE broadcasts [S,Dh] over H
    kh = ops::rotary_embedding(kh, cos, sin);
    Tensor kt = ops::transpose(kh, 1, 2);                 // [H, Dh, S]
    Tensor scores = ops::mul(ops::matmul(qh, kt), scale_t);
    Tensor probs = ops::softmax(scores, -1);
    Tensor oh = ops::matmul(probs, vh);                   // [H, S, Dh]
    Tensor o = ops::reshape(ops::permute(oh, {1, 0, 2}), {S, D});
    Tensor attn = ops::matmul(o, wo);                     // O proj
    Tensor h = ops::add(xi, attn);                        // residual
    // SwiGLU FFN.
    Tensor g = ops::matmul(h, wg);                        // [S, Dff]
    Tensor u = ops::matmul(h, wu);
    Tensor act = ops::mul(ops::mul(g, ops::sigmoid(g)), u);
    Tensor ff = ops::matmul(act, wd);                     // [S, D]
    return ops::add(h, ff);                               // residual
  };

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    graph::bind_param(wq, "wq");
    graph::bind_param(wk, "wk");
    graph::bind_param(wv, "wv");
    graph::bind_param(wo, "wo");
    graph::bind_param(wg, "wg");
    graph::bind_param(wu, "wu");
    graph::bind_param(wd, "wd");
    graph::bind_input(scale_t, "scale");
    graph::bind_input(cos, "cos");
    graph::bind_input(sin, "sin");
    Tensor y = block(x);
    graph::mark_output(y);
    captured = std::move(scope.graph());
  }
  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], x},
      {captured.params()[0], wq},  {captured.params()[1], wk},
      {captured.params()[2], wv},  {captured.params()[3], wo},
      {captured.params()[4], wg},  {captured.params()[5], wu},
      {captured.params()[6], wd},
      {captured.inputs()[1], scale_t},
      {captured.inputs()[2], cos},
      {captured.inputs()[3], sin}};
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  Tensor eager = block(x);
  require_close(jit_outs[0], eager, /*tol=*/1e-4);
}

TEST_CASE("jit: rotary embedding matches eager", "[ir][jit][transformer]") {
  // RoPE on a [B, S, D] activation against [S, D] cos/sin tables. Validates the
  // strided-slice + rotate_half + broadcast multiply-add lowering end-to-end
  // through the CPU JIT (bufferization of tensor.extract_slice/insert_slice).
  constexpr int64_t B = 2;
  constexpr int64_t S = 4;
  constexpr int64_t D = 8;
  auto x = make_random({B, S, D}, /*seed=*/2601, -1.0f, 1.0f);

  auto cos = Tensor::empty({S, D}, DType::Float32);
  auto sin = Tensor::empty({S, D}, DType::Float32);
  float* cp = cos.data_ptr<float>();
  float* sp = sin.data_ptr<float>();
  for (int64_t p = 0; p < S; ++p) {
    for (int64_t j = 0; j < D / 2; ++j) {
      const double theta =
          static_cast<double>(p) *
          std::pow(10000.0, -2.0 * static_cast<double>(j) / static_cast<double>(D));
      const float c = static_cast<float>(std::cos(theta));
      const float s = static_cast<float>(std::sin(theta));
      cp[p * D + 2 * j] = c;
      cp[p * D + 2 * j + 1] = c;
      sp[p * D + 2 * j] = s;
      sp[p * D + 2 * j + 1] = s;
    }
  }

  graph::Graph captured;
  {
    graph::GraphScope scope;
    graph::bind_input(x, "x");
    graph::bind_input(cos, "cos");
    graph::bind_input(sin, "sin");
    Tensor y = ops::rotary_embedding(x, cos, sin);
    graph::mark_output(y);
    captured = std::move(scope.graph());
  }
  std::unordered_map<graph::ValueId, Tensor> bind = {
      {captured.inputs()[0], x},
      {captured.inputs()[1], cos},
      {captured.inputs()[2], sin}};
  ir::JitEngine jit(captured);
  auto jit_outs = jit.invoke(bind);

  Tensor eager = ops::rotary_embedding(x, cos, sin);
  require_close(jit_outs[0], eager, /*tol=*/1e-5);
}

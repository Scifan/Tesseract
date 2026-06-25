// Wave 8 (B-030) — grouped-query attention (GQA).
//
// GQA shrinks K/V to `num_kv_heads < num_heads` heads, each shared by
// `num_heads / num_kv_heads` query heads. Correctness has two pillars:
//   1. head-sharing order: query head h must use KV head h / G. We prove
//      it by building a plain-MHA reference whose K/V weights are the GQA
//      weights with each KV head's rows replicated G times — the two
//      modules must then produce identical outputs.
//   2. decode parity: GQA forward_step (chunked + token-by-token) must
//      match the one-shot GQA forward, so the KV cache (which stores only
//      Hkv heads) + repeat path is self-consistent.
// Plus shape/divisibility checks and a Llama-level GQA generate run.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/nn/KVCache.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"

using namespace tesseract;

namespace {

std::shared_ptr<nn::Linear> as_linear(const std::shared_ptr<nn::Module>& m) {
  return std::dynamic_pointer_cast<nn::Linear>(m);
}

void copy_full(const Tensor& dst, const Tensor& src) {
  Tensor d = dst;  // shared-storage handle
  REQUIRE(d.numel() == src.numel());
  std::memcpy(d.raw_data(), src.raw_data(),
              static_cast<std::size_t>(src.numel()) * sizeof(float));
}

// dst [H*Dh, D] <- src [Hkv*Dh, D], replicating each KV head's Dh-row
// block across the G query heads that share it (head h -> kv h/G).
void expand_kv(const Tensor& dst, const Tensor& src, int64_t H, int64_t Hkv,
               int64_t Dh, int64_t D) {
  Tensor d = dst;
  const float* sp = src.data_ptr<float>();
  float* dp = d.data_ptr<float>();
  const int64_t G = H / Hkv;
  const int64_t block = Dh * D;
  for (int64_t h = 0; h < H; ++h) {
    const int64_t kv = h / G;
    std::memcpy(dp + h * block, sp + kv * block,
                static_cast<std::size_t>(block) * sizeof(float));
  }
}

Tensor gaussian(Shape s, uint64_t seed) {
  Tensor t = Tensor::empty(std::move(s), DType::Float32);
  float* p = t.data_ptr<float>();
  uint64_t st = seed;
  for (int64_t i = 0; i < t.numel(); ++i) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u = static_cast<double>((st >> 11) & 0x1FFFFFFFFFFFFFULL) /
                     9007199254740992.0;
    p[i] = static_cast<float>(u * 2.0 - 1.0);
  }
  return t;
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
  REQUIRE(a.numel() == b.numel());
  const float* pa = a.data_ptr<float>();
  const float* pb = b.data_ptr<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < a.numel(); ++i) m = std::max(m, std::abs(pa[i] - pb[i]));
  return m;
}

}  // namespace

TEST_CASE("GQA: k/v projections shrink to num_kv_heads", "[nn][gqa]") {
  const int64_t D = 32, H = 8, Hkv = 2, Dh = D / H;
  nn::MultiHeadAttention mha(D, H, /*use_bias=*/false, /*causal=*/true,
                             DType::Float32, /*rope_base=*/0.0,
                             /*rope_max_seq=*/0, /*num_kv_heads=*/Hkv);
  REQUIRE(mha.num_heads() == H);
  REQUIRE(mha.num_kv_heads() == Hkv);
  REQUIRE(as_linear(mha.q_proj())->weight().shape()[0] == D);        // H*Dh
  REQUIRE(as_linear(mha.k_proj())->weight().shape()[0] == Hkv * Dh); // shrunk
  REQUIRE(as_linear(mha.v_proj())->weight().shape()[0] == Hkv * Dh);
  REQUIRE(as_linear(mha.o_proj())->weight().shape()[0] == D);
}

TEST_CASE("GQA: matches a weight-replicated plain-MHA reference", "[nn][gqa]") {
  const int64_t B = 2, S = 6, D = 16, H = 4, Hkv = 2, Dh = D / H;

  nn::MultiHeadAttention g(D, H, /*use_bias=*/false, /*causal=*/true,
                           DType::Float32, /*rope_base=*/0.0,
                           /*rope_max_seq=*/0, /*num_kv_heads=*/Hkv);
  nn::MultiHeadAttention ref(D, H, /*use_bias=*/false, /*causal=*/true,
                             DType::Float32);  // plain MHA (Hkv == H)

  // Make `ref` mathematically equal to `g`: same Q/O weights, and K/V
  // weights with each KV head replicated across its G query heads.
  copy_full(as_linear(ref.q_proj())->weight(), as_linear(g.q_proj())->weight());
  copy_full(as_linear(ref.o_proj())->weight(), as_linear(g.o_proj())->weight());
  expand_kv(as_linear(ref.k_proj())->weight(), as_linear(g.k_proj())->weight(),
            H, Hkv, Dh, D);
  expand_kv(as_linear(ref.v_proj())->weight(), as_linear(g.v_proj())->weight(),
            H, Hkv, Dh, D);

  Tensor x = gaussian({B, S, D}, /*seed=*/2024);
  Tensor out_g = g.forward(x);
  Tensor out_ref = ref.forward(x);
  REQUIRE(out_g.shape()[0] == B);
  REQUIRE(out_g.shape()[1] == S);
  REQUIRE(out_g.shape()[2] == D);
  REQUIRE(max_abs_diff(out_g, out_ref) < 1e-5f);  // identical math
}

TEST_CASE("GQA: forward_step (decode) matches one-shot forward", "[nn][gqa]") {
  const int64_t B = 1, S = 7, D = 16, H = 4, Hkv = 2, Dh = D / H;
  const double rope_base = 10000.0;
  const int64_t rope_max = 64;

  nn::MultiHeadAttention mha(D, H, /*use_bias=*/false, /*causal=*/true,
                             DType::Float32, rope_base, rope_max,
                             /*num_kv_heads=*/Hkv);

  Tensor x = gaussian({B, S, D}, /*seed=*/77);
  Tensor full = mha.forward(x);  // [B, S, D] causal

  // Token-by-token decode through a KV cache holding only Hkv heads.
  nn::KVCache cache(B, Hkv, Dh, /*max_len=*/S, DType::Float32, cpu_device());
  for (int64_t t = 0; t < S; ++t) {
    Tensor xt = x.narrow(/*dim=*/1, t, 1);  // [B, 1, D]
    Tensor step = mha.forward_step(xt, cache);
    Tensor ref_t = full.narrow(1, t, 1);
    REQUIRE(max_abs_diff(step, ref_t) < 2e-4f);
  }
  REQUIRE(cache.num_heads() == Hkv);
}

TEST_CASE("GQA: num_kv_heads must divide num_heads", "[nn][gqa]") {
  REQUIRE_THROWS(nn::MultiHeadAttention(16, 4, false, true, DType::Float32,
                                        0.0, 0, /*num_kv_heads=*/3));
  // 0 ⇒ defaults to plain MHA, no throw.
  REQUIRE_NOTHROW(nn::MultiHeadAttention(16, 4, false, true, DType::Float32,
                                         0.0, 0, /*num_kv_heads=*/0));
}

TEST_CASE("GQA: Llama model with GQA generates deterministically", "[models][gqa]") {
  models::LlamaConfig cfg;
  cfg.vocab_size = 48;
  cfg.hidden_size = 32;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 8;
  cfg.num_key_value_heads = 2;  // GQA
  cfg.intermediate_size = 64;
  cfg.max_position_embeddings = 64;
  cfg.rope_theta = 10000.0;
  cfg.tie_word_embeddings = false;
  REQUIRE(cfg.kv_heads() == 2);

  auto model = std::make_shared<models::LlamaModel>(cfg);
  const std::vector<int32_t> prompt = {3, 9, 21};

  models::LlamaModel::GenerateConfig g;
  g.max_new_tokens = 10;
  const auto a = model->generate(prompt, g);
  const auto b = model->generate(prompt, g);
  REQUIRE(a == b);
  REQUIRE(static_cast<int64_t>(a.size()) == static_cast<int64_t>(prompt.size()) + 10);

  // Chunked-prefill forward_step parity: one-shot forward over the prompt
  // equals stepping the prompt through GQA caches.
  Tensor toks = Tensor::empty({1, static_cast<int64_t>(prompt.size())},
                              DType::Int64, cpu_device());
  for (std::size_t i = 0; i < prompt.size(); ++i)
    toks.data_ptr<int64_t>()[i] = prompt[i];
  Tensor one_shot = model->forward(toks);
  auto caches = model->make_kv_caches(1, 32);
  Tensor stepped = model->forward_step(toks, caches);
  REQUIRE(max_abs_diff(one_shot, stepped) < 2e-4f);
  REQUIRE(caches[0]->num_heads() == 2);  // cache stores KV heads
}

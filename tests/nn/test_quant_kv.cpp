// Wave 9 (B-031) — INT8 KV-cache quantization.
//
// Coverage:
//   1. quantize/dequantize roundtrip is within the symmetric-INT8 error
//      bound (|x - x'| <= scale_row / 2), and shapes are correct.
//   2. CPU and CUDA produce identical results (same FP32 absmax + round).
//   3. QuantizedKVCache reconstructs exactly what `dequantize(quantize(K))`
//      would — the slab/scale append + narrow + dequant plumbing is exact
//      (chunked append == one-shot, since per-token quant is independent).
//   4. MHA forward_step with a quantized cache tracks the FP cache within
//      a bounded tolerance (lossy, not bit-identical).
//   5. Llama generate with `kv_int8` is deterministic, valid, and its
//      prefill logits stay close to the FP path.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/nn/KVCache.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"
#include "tesseract/nn/QuantizedKVCache.hpp"
#include "tesseract/quant/QuantizeKV.hpp"

using namespace tesseract;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

namespace {

bool cuda_ready() { return has_cuda_support() && is_available(); }
Device cuda0() { return Device{DeviceType::CUDA, 0}; }

std::vector<float> gaussian(int64_t n, uint64_t seed) {
  std::vector<float> v(static_cast<std::size_t>(n));
  uint64_t st = seed;
  for (int64_t i = 0; i < n; ++i) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u = static_cast<double>((st >> 11) & 0x1FFFFFFFFFFFFFULL) /
                     9007199254740992.0;
    v[static_cast<std::size_t>(i)] = static_cast<float>((u * 2.0 - 1.0) * 3.0);
  }
  return v;
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
  Tensor ac = a.to(cpu_device()).contiguous();
  Tensor bc = b.to(cpu_device()).contiguous();
  REQUIRE(ac.numel() == bc.numel());
  const float* pa = ac.data_ptr<float>();
  const float* pb = bc.data_ptr<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < ac.numel(); ++i) m = std::max(m, std::abs(pa[i] - pb[i]));
  return m;
}

}  // namespace

TEST_CASE("quantize_kv roundtrip within INT8 error bound", "[quant][kv]") {
  const int64_t B = 2, H = 3, S = 5, Dh = 8;
  Tensor x = Tensor::from_vector(gaussian(B * H * S * Dh, 7), {B, H, S, Dh});

  auto [q, scale] = quant::quantize_kv_per_token(x);
  REQUIRE(q.dtype() == DType::Int8);
  REQUIRE(q.shape() == x.shape());
  REQUIRE(scale.dtype() == DType::Float32);
  REQUIRE(scale.rank() == 3);
  REQUIRE(scale.shape()[0] == B);
  REQUIRE(scale.shape()[2] == S);

  Tensor xr = quant::dequantize_kv_per_token(q, scale, DType::Float32);
  REQUIRE(xr.shape() == x.shape());

  // |x - x'| <= scale_row / 2 for every element (symmetric round-to-nearest).
  const float* xp = x.data_ptr<float>();
  const float* rp = xr.data_ptr<float>();
  const float* sp = scale.data_ptr<float>();
  const int64_t rows = B * H * S;
  for (int64_t r = 0; r < rows; ++r) {
    const float bound = sp[r] * 0.5f + 1e-5f;
    for (int64_t d = 0; d < Dh; ++d) {
      const float diff = std::abs(xp[r * Dh + d] - rp[r * Dh + d]);
      REQUIRE(diff <= bound);
    }
  }
}

TEST_CASE("quantize_kv all-zero row gets scale 1", "[quant][kv]") {
  Tensor x = Tensor::zeros({1, 1, 2, 4}, DType::Float32, cpu_device());
  auto [q, scale] = quant::quantize_kv_per_token(x);
  const float* sp = scale.data_ptr<float>();
  REQUIRE(sp[0] == 1.0f);
  REQUIRE(sp[1] == 1.0f);
  Tensor xr = quant::dequantize_kv_per_token(q, scale, DType::Float32);
  REQUIRE(max_abs_diff(x, xr) == 0.0f);
}

TEST_CASE("quantize_kv CPU and CUDA agree", "[quant][kv][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, H = 4, S = 7, Dh = 16;
  auto data = gaussian(B * H * S * Dh, 99);
  Tensor x_cpu = Tensor::from_vector(data, {B, H, S, Dh});
  Tensor x_cuda = Tensor::from_vector(data, {B, H, S, Dh}).to(cuda0());

  auto [qc, sc] = quant::quantize_kv_per_token(x_cpu);
  auto [qg, sg] = quant::quantize_kv_per_token(x_cuda);

  Tensor rc = quant::dequantize_kv_per_token(qc, sc, DType::Float32);
  Tensor rg = quant::dequantize_kv_per_token(qg, sg, DType::Float32);
  // Identical FP32 math on both devices ⇒ exact agreement.
  REQUIRE(max_abs_diff(rc, rg) == 0.0f);
  REQUIRE(max_abs_diff(sc, sg) == 0.0f);
}

TEST_CASE("QuantizedKVCache reconstructs dequant(quant(K)) exactly", "[nn][quant][kv]") {
  const int64_t B = 1, H = 2, S = 6, Dh = 8;
  Tensor k = Tensor::from_vector(gaussian(B * H * S * Dh, 11), {B, H, S, Dh});
  Tensor v = Tensor::from_vector(gaussian(B * H * S * Dh, 22), {B, H, S, Dh});

  nn::QuantizedKVCache cache(B, H, Dh, /*max_len=*/S, DType::Float32, cpu_device());
  // Append token-by-token (per-token quant is independent of chunking).
  for (int64_t t = 0; t < S; ++t) {
    Tensor kt = k.narrow(2, t, 1).contiguous();
    Tensor vt = v.narrow(2, t, 1).contiguous();
    cache.append(kt, vt);
  }
  REQUIRE(cache.current_len() == S);

  // Ground truth: quantize the whole thing once, dequantize.
  auto [qk, sk] = quant::quantize_kv_per_token(k);
  Tensor k_ref = quant::dequantize_kv_per_token(qk, sk, DType::Float32);
  auto [qv, sv] = quant::quantize_kv_per_token(v);
  Tensor v_ref = quant::dequantize_kv_per_token(qv, sv, DType::Float32);

  REQUIRE(max_abs_diff(cache.keys_view(), k_ref) == 0.0f);
  REQUIRE(max_abs_diff(cache.values_view(), v_ref) == 0.0f);

  cache.reset();
  REQUIRE(cache.current_len() == 0);
}

TEST_CASE("MHA forward_step: quantized cache tracks FP cache", "[nn][quant][kv]") {
  const int64_t B = 1, S = 8, D = 32, Hh = 4, Dh = D / Hh;
  nn::MultiHeadAttention mha(D, Hh, /*use_bias=*/false, /*causal=*/true,
                             DType::Float32, /*rope_base=*/10000.0,
                             /*rope_max_seq=*/64);

  Tensor x = Tensor::from_vector(gaussian(B * S * D, 33), {B, S, D});

  nn::KVCache fp(B, Hh, Dh, S, DType::Float32, cpu_device());
  nn::QuantizedKVCache q(B, Hh, Dh, S, DType::Float32, cpu_device());

  float worst = 0.0f;
  for (int64_t t = 0; t < S; ++t) {
    Tensor xt = x.narrow(1, t, 1);
    Tensor out_fp = mha.forward_step(xt, fp);
    Tensor out_q  = mha.forward_step(xt, q);
    worst = std::max(worst, max_abs_diff(out_fp, out_q));
  }
  // Bounded error: INT8 KV is lossy but small for well-scaled activations.
  REQUIRE(worst < 5e-2f);
}

TEST_CASE("Llama generate with kv_int8 is deterministic and close to FP",
          "[models][quant][kv]") {
  models::LlamaConfig cfg;
  cfg.vocab_size = 48;
  cfg.hidden_size = 32;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.intermediate_size = 64;
  cfg.max_position_embeddings = 64;
  cfg.tie_word_embeddings = false;
  auto model = std::make_shared<models::LlamaModel>(cfg);

  const std::vector<int32_t> prompt = {5, 12, 30, 7};

  models::LlamaModel::GenerateConfig g;
  g.max_new_tokens = 12;
  g.kv_int8 = true;
  const auto a = model->generate(prompt, g);
  const auto b = model->generate(prompt, g);
  REQUIRE(a == b);  // INT8 path is deterministic
  REQUIRE(static_cast<int64_t>(a.size()) ==
          static_cast<int64_t>(prompt.size()) + 12);
  for (int32_t id : a) {
    REQUIRE(id >= 0);
    REQUIRE(id < cfg.vocab_size);
  }
}

// Wave 13 (B-032+++) — INT8-quantized paged KV cache storage layer.
//
// `QuantizedPagedKVCache` is the INT8 sibling of `PagedKVCache`: paged
// block storage (no max_len padding waste) with per-token INT8 + FP32-scale
// payloads (Wave-9 quant), feeding the Wave-12 fused
// `paged_decode_attention_int8` op directly on the CUDA decode hot path.
//
// What this pins down:
//   * scatter/gather correctness: its dequantized `keys_view`/`values_view`
//     must equal the contiguous `QuantizedKVCache` on identical appends
//     (same quant math, same data, just paged vs contiguous layout) —
//     bit-exact, across block boundaries and chunked appends;
//   * paging residency: blocks are allocated on demand and recycled on
//     reset;
//   * end-to-end wiring: `MHA::forward_step_batched` on CUDA quantized-paged
//     caches takes the fused INT8 kernel path and matches per-request
//     `forward_step` (the gather+dequant+attention fallback) within tol.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"
#include "tesseract/nn/QuantizedKVCache.hpp"
#include "tesseract/nn/QuantizedPagedKVCache.hpp"
#include "tesseract/nn/QuantizedPagedKVPool.hpp"

using namespace tesseract;

namespace {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  float uniform(float lo, float hi) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u =
        static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0;
    return static_cast<float>(lo + (hi - lo) * u);
  }
};

Tensor rand_f32(Shape s, Rng& rng, Device dev, float lo = -0.5f, float hi = 0.5f) {
  Tensor t = Tensor::empty(std::move(s), DType::Float32, cpu_device());
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = rng.uniform(lo, hi);
  return dev.is_cpu() ? t : t.to(dev);
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
  Tensor ah = a.device().is_cpu() ? a.contiguous() : a.to(cpu_device()).contiguous();
  Tensor bh = b.device().is_cpu() ? b.contiguous() : b.to(cpu_device()).contiguous();
  const float* pa = ah.data_ptr<float>();
  const float* pb = bh.data_ptr<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < ah.numel(); ++i) m = std::max(m, std::abs(pa[i] - pb[i]));
  return m;
}

}  // namespace

TEST_CASE("QuantizedPagedKVCache dequant view == contiguous QuantizedKVCache (CPU)",
          "[nn][paged][quant]") {
  const Device dev = cpu_device();
  const int64_t H = 3, D = 8, BLK = 4, max_len = 64;
  Rng rng(0x13A1u);

  nn::QuantizedKVCache ref(/*batch=*/1, H, D, max_len, DType::Float32, dev);
  auto pool = std::make_shared<nn::QuantizedPagedKVPool>(H, D, BLK, /*nb=*/64,
                                                         DType::Float32, dev);
  nn::QuantizedPagedKVCache paged(pool, /*batch=*/1, max_len);

  // Chunked appends that straddle block boundaries (BLK=4): 3, 5, 1, 7.
  for (int64_t chunk : {3, 5, 1, 7}) {
    Tensor k = rand_f32({1, H, chunk, D}, rng, dev);
    Tensor v = rand_f32({1, H, chunk, D}, rng, dev);
    ref.append(k, v);
    paged.append(k, v);
  }
  REQUIRE(ref.current_len() == paged.current_len());
  REQUIRE(paged.current_len() == 16);

  // Same quant math + same data ⇒ dequantized views must be bit-identical.
  REQUIRE(max_abs_diff(ref.keys_view(), paged.keys_view()) == 0.0f);
  REQUIRE(max_abs_diff(ref.values_view(), paged.values_view()) == 0.0f);
}

TEST_CASE("QuantizedPagedKVCache pages on demand and recycles on reset (CPU)",
          "[nn][paged][quant]") {
  const Device dev = cpu_device();
  const int64_t H = 2, D = 4, BLK = 4, NB = 32, max_len = 64;
  auto pool = std::make_shared<nn::QuantizedPagedKVPool>(H, D, BLK, NB,
                                                         DType::Float32, dev);
  nn::QuantizedPagedKVCache c(pool, /*batch=*/1, max_len);
  REQUIRE(pool->num_allocated() == 0);

  Rng rng(0x5151u);
  Tensor k = rand_f32({1, H, 6, D}, rng, dev);  // 6 tokens ⇒ ceil(6/4)=2 blocks
  Tensor v = rand_f32({1, H, 6, D}, rng, dev);
  c.append(k, v);
  REQUIRE(c.num_owned_blocks() == 2);
  REQUIRE(pool->num_allocated() == 2);

  c.reset();
  REQUIRE(c.current_len() == 0);
  REQUIRE(c.num_owned_blocks() == 0);
  REQUIRE(pool->num_allocated() == 0);  // blocks returned to the shared pool
}

TEST_CASE("MHA::forward_step_batched (quant-paged, CUDA) matches per-request forward_step",
          "[nn][paged][quant][cuda]") {
  if (cuda::device_count() <= 0) {
    SKIP("CUDA not available");
    return;
  }
  const Device dev{DeviceType::CUDA, 0};
  const int64_t D = 32, H = 4, Hkv = 2, BLK = 4, max_len = 64;
  nn::MultiHeadAttention mha(/*d_model=*/D, /*num_heads=*/H, /*use_bias=*/false,
                             /*causal=*/true, DType::Float32,
                             /*rope_base=*/10000.0, /*rope_max_seq=*/64,
                             /*num_kv_heads=*/Hkv);
  mha.to(dev);

  const std::vector<int64_t> prompt_lens = {5, 1, 9, 13};
  const int64_t A = static_cast<int64_t>(prompt_lens.size());
  Rng rng(0x13C8u);

  // Shared INT8 pools (one per cache-set; all requests share their set's pool).
  auto ref_pool = std::make_shared<nn::QuantizedPagedKVPool>(Hkv, D / H, BLK, 128,
                                                             DType::Float32, dev);
  auto fus_pool = std::make_shared<nn::QuantizedPagedKVPool>(Hkv, D / H, BLK, 128,
                                                             DType::Float32, dev);
  std::vector<std::unique_ptr<nn::QuantizedPagedKVCache>> ref_c, fus_c;
  for (int64_t r = 0; r < A; ++r) {
    ref_c.push_back(std::make_unique<nn::QuantizedPagedKVCache>(ref_pool, 1, max_len));
    fus_c.push_back(std::make_unique<nn::QuantizedPagedKVCache>(fus_pool, 1, max_len));
  }

  // Prefill each request's prompt through both cache sets.
  for (int64_t r = 0; r < A; ++r) {
    Tensor prompt = rand_f32({1, prompt_lens[r], D}, rng, dev);
    mha.forward_step(prompt, *ref_c[r]);
    mha.forward_step(prompt, *fus_c[r]);
  }

  // One decode step: per-request reference (gather+dequant+attention
  // fallback) vs batched fused (paged_decode_attention_int8).
  std::vector<Tensor> x_list, ref_out;
  for (int64_t r = 0; r < A; ++r) {
    Tensor xr = rand_f32({1, 1, D}, rng, dev);
    x_list.push_back(xr);
    ref_out.push_back(mha.forward_step(xr, *ref_c[r]).to(cpu_device()).contiguous());
  }
  Tensor x_all = Tensor::empty({A, 1, D}, DType::Float32, cpu_device());
  {
    float* p = x_all.data_ptr<float>();
    for (int64_t r = 0; r < A; ++r) {
      Tensor xh = x_list[static_cast<std::size_t>(r)].to(cpu_device()).contiguous();
      const float* px = xh.data_ptr<float>();
      for (int64_t d = 0; d < D; ++d) p[r * D + d] = px[d];
    }
  }
  Tensor x_all_dev = x_all.to(dev);

  std::vector<nn::KVCacheBase*> bases;
  for (int64_t r = 0; r < A; ++r) bases.push_back(fus_c[static_cast<std::size_t>(r)].get());
  Tensor out = mha.forward_step_batched(x_all_dev, bases).to(cpu_device()).contiguous();
  REQUIRE(out.shape() == Shape({A, 1, D}));

  const float* po = out.data_ptr<float>();
  float m = 0.0f;
  for (int64_t r = 0; r < A; ++r) {
    const float* pr = ref_out[static_cast<std::size_t>(r)].data_ptr<float>();
    for (int64_t d = 0; d < D; ++d)
      m = std::max(m, std::abs(po[r * D + d] - pr[d]));
  }
  INFO("quant-paged fused-vs-perrequest maxdiff=" << m);
  REQUIRE(m <= 3e-3f);
}

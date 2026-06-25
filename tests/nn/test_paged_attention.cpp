// Wave 11 (B-032+) — fused ragged paged decode-attention op.
//
// `nn::paged_decode_attention` computes one decode query's attention for
// every (request, head) in the active set in a single pass, reading K/V
// straight out of the shared paged pool via each request's block table
// (no gather, no `repeat_kv` materialization, online softmax). Its
// correctness target is the production decode recipe it replaces: gather
// the request's KV prefix (`PagedKVCache::keys_view/values_view`) and run
// `ops::attention` per (request, head) with the GQA head mapping
// `h → h/group`. This must match within the usual softmax tolerance —
// CPU near-exact, CUDA within float tolerance.
//
// What this pins down:
//   * MHA (group==1) ragged mixed-length parity vs gather+attention;
//   * GQA (group>1) parity — the on-the-fly KV-head mapping;
//   * CUDA parity;
//   * a zero-length request yields a zero output row;
//   * shape / dtype / device validation.

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
#include "tesseract/nn/PagedAttention.hpp"
#include "tesseract/nn/PagedKVCache.hpp"
#include "tesseract/nn/PagedKVPool.hpp"
#include "tesseract/ops/Attention.hpp"
#include "tesseract/quant/QuantizeKV.hpp"

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

Tensor int32_host(const std::vector<int32_t>& v, Shape s, Device dev) {
  Tensor t = Tensor::empty(std::move(s), DType::Int32, cpu_device());
  int32_t* p = t.data_ptr<int32_t>();
  for (std::size_t i = 0; i < v.size(); ++i) p[i] = v[i];
  return dev.is_cpu() ? t : t.to(dev);
}

// Build A single-request paged caches sharing one pool, fill each with
// `lens[r]` random K/V tokens, run paged_decode_attention, and compare to
// the per-(request,head) gather + ops::attention reference.
void run_parity(Device dev, int64_t H, int64_t Hkv, const std::vector<int64_t>& lens,
                float tol, uint64_t seed) {
  const int64_t A   = static_cast<int64_t>(lens.size());
  const int64_t D   = 16;
  const int64_t BLK = 4;
  const int64_t group = H / Hkv;
  const int64_t max_len = 64;
  Rng rng(seed);

  auto pool = std::make_shared<nn::PagedKVPool>(Hkv, D, BLK, /*num_blocks=*/64,
                                                DType::Float32, dev);
  std::vector<std::unique_ptr<nn::PagedKVCache>> caches;
  for (int64_t r = 0; r < A; ++r) {
    auto c = std::make_unique<nn::PagedKVCache>(pool, /*batch=*/1, max_len);
    if (lens[r] > 0) {
      Tensor k = rand_f32({1, Hkv, lens[r], D}, rng, dev);
      Tensor v = rand_f32({1, Hkv, lens[r], D}, rng, dev);
      c->append(k, v);
    }
    caches.push_back(std::move(c));
  }

  Tensor q = rand_f32({A, H, D}, rng, dev);
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));

  // Assemble the [A, max_logical] block table + [A] lens for the op.
  int64_t max_logical = 1;
  for (int64_t r = 0; r < A; ++r)
    max_logical = std::max<int64_t>(max_logical,
                                    (lens[r] + BLK - 1) / BLK);
  std::vector<int32_t> table(static_cast<std::size_t>(A * max_logical), 0);
  std::vector<int32_t> len_vec(static_cast<std::size_t>(A));
  for (int64_t r = 0; r < A; ++r) {
    len_vec[static_cast<std::size_t>(r)] = static_cast<int32_t>(lens[r]);
    const std::vector<int32_t>& bt = caches[static_cast<std::size_t>(r)]->block_table(0);
    for (std::size_t i = 0; i < bt.size(); ++i)
      table[static_cast<std::size_t>(r) * max_logical + i] = bt[i];
  }
  Tensor block_tables = int32_host(table, {A, max_logical}, dev);
  Tensor lens_t       = int32_host(len_vec, {A}, dev);

  Tensor out = nn::paged_decode_attention(q, pool->keys(), pool->values(),
                                          block_tables, lens_t, scale, group);
  REQUIRE(out.shape() == Shape({A, H, D}));

  // Reference: per (request, head) gather + ops::attention with h→h/group,
  // accumulated into a host buffer (zero rows for empty requests).
  std::vector<float> ref(static_cast<std::size_t>(A * H * D), 0.0f);
  for (int64_t r = 0; r < A; ++r) {
    if (lens[r] == 0) continue;  // zero output row
    Tensor q_r = q.narrow(0, r, 1);                                   // [1,H,D]
    Tensor k_all = caches[static_cast<std::size_t>(r)]->keys_view();  // [1,Hkv,len,D]
    Tensor v_all = caches[static_cast<std::size_t>(r)]->values_view();
    for (int64_t h = 0; h < H; ++h) {
      const int64_t hkv = h / group;
      Tensor qh = q_r.narrow(1, h, 1).reshape(Shape({1, 1, 1, D}));   // [1,1,1,D]
      Tensor kh = k_all.narrow(1, hkv, 1).contiguous();               // [1,1,len,D]
      Tensor vh = v_all.narrow(1, hkv, 1).contiguous();
      Tensor oh = ops::attention(qh, kh, vh, Tensor{}, /*causal=*/false,
                                 /*dropout_p=*/0.0);                   // [1,1,1,D]
      Tensor ohh = oh.device().is_cpu() ? oh.contiguous()
                                        : oh.to(cpu_device()).contiguous();
      const float* p = ohh.data_ptr<float>();
      for (int64_t d = 0; d < D; ++d)
        ref[static_cast<std::size_t>((r * H + h) * D + d)] = p[d];
    }
  }

  Tensor out_h = out.device().is_cpu() ? out.contiguous()
                                       : out.to(cpu_device()).contiguous();
  const float* po = out_h.data_ptr<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < A * H * D; ++i)
    m = std::max(m, std::abs(po[i] - ref[static_cast<std::size_t>(i)]));
  INFO("H=" << H << " Hkv=" << Hkv << " A=" << A << " maxdiff=" << m);
  REQUIRE(m <= tol);
}

// Wave 12 (B-032++): the INT8-direct op must equal the Wave-11 FP op run on
// the *dequantized* pool — same int8*scale values, same online softmax — so
// the only delta is the fused inline dequant. CPU bit-exact, CUDA float tol.
void run_int8_parity(Device dev, int64_t H, int64_t Hkv,
                     const std::vector<int64_t>& lens, float tol,
                     uint64_t seed) {
  const int64_t A     = static_cast<int64_t>(lens.size());
  const int64_t D     = 16;
  const int64_t BLK   = 4;
  const int64_t group = H / Hkv;
  const int64_t max_len = 64;
  Rng rng(seed);

  auto pool = std::make_shared<nn::PagedKVPool>(Hkv, D, BLK, /*num_blocks=*/64,
                                                DType::Float32, dev);
  std::vector<std::unique_ptr<nn::PagedKVCache>> caches;
  for (int64_t r = 0; r < A; ++r) {
    auto c = std::make_unique<nn::PagedKVCache>(pool, /*batch=*/1, max_len);
    if (lens[r] > 0) {
      Tensor k = rand_f32({1, Hkv, lens[r], D}, rng, dev);
      Tensor v = rand_f32({1, Hkv, lens[r], D}, rng, dev);
      c->append(k, v);
    }
    caches.push_back(std::move(c));
  }

  Tensor q = rand_f32({A, H, D}, rng, dev);
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));

  int64_t max_logical = 1;
  for (int64_t r = 0; r < A; ++r)
    max_logical = std::max<int64_t>(max_logical, (lens[r] + BLK - 1) / BLK);
  std::vector<int32_t> table(static_cast<std::size_t>(A * max_logical), 0);
  std::vector<int32_t> len_vec(static_cast<std::size_t>(A));
  for (int64_t r = 0; r < A; ++r) {
    len_vec[static_cast<std::size_t>(r)] = static_cast<int32_t>(lens[r]);
    const std::vector<int32_t>& bt = caches[static_cast<std::size_t>(r)]->block_table(0);
    for (std::size_t i = 0; i < bt.size(); ++i)
      table[static_cast<std::size_t>(r) * max_logical + i] = bt[i];
  }
  Tensor block_tables = int32_host(table, {A, max_logical}, dev);
  Tensor lens_t       = int32_host(len_vec, {A}, dev);

  // Quantize the whole FP pool (last-dim per-vector scale == per-(block,head,
  // slot) scale, exactly the INT8 op's layout), then dequantize for the ref.
  auto [kq, ksc] = quant::quantize_kv_per_token(pool->keys());
  auto [vq, vsc] = quant::quantize_kv_per_token(pool->values());
  Tensor dq_k = quant::dequantize_kv_per_token(kq, ksc, DType::Float32);
  Tensor dq_v = quant::dequantize_kv_per_token(vq, vsc, DType::Float32);

  Tensor out_i8 = nn::paged_decode_attention_int8(
      q, kq, ksc, vq, vsc, block_tables, lens_t, scale, group);
  REQUIRE(out_i8.shape() == Shape({A, H, D}));
  Tensor out_fp = nn::paged_decode_attention(q, dq_k, dq_v, block_tables,
                                             lens_t, scale, group);

  Tensor i8h = out_i8.device().is_cpu() ? out_i8.contiguous()
                                        : out_i8.to(cpu_device()).contiguous();
  Tensor fph = out_fp.device().is_cpu() ? out_fp.contiguous()
                                        : out_fp.to(cpu_device()).contiguous();
  const float* pi = i8h.data_ptr<float>();
  const float* pf = fph.data_ptr<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < A * H * D; ++i) m = std::max(m, std::abs(pi[i] - pf[i]));
  INFO("INT8 H=" << H << " Hkv=" << Hkv << " A=" << A << " maxdiff=" << m);
  REQUIRE(m <= tol);
}

// Wave 14 (B-032++++): fused paged PREFILL parity. Fill each request's cache
// with `kv_lens[r]` K/V tokens, then run `paged_prefill_attention` over a
// [A, S, H, D] query block. Reference: per (request, query s, head) gather +
// single-query attention over the causal prefix `[0, kv_lens[r]-S+s]`.
void run_prefill_parity(Device dev, int64_t H, int64_t Hkv, int64_t S,
                        const std::vector<int64_t>& kv_lens, float tol,
                        uint64_t seed) {
  const int64_t A     = static_cast<int64_t>(kv_lens.size());
  const int64_t D     = 16;
  const int64_t BLK   = 4;
  const int64_t group = H / Hkv;
  const int64_t max_len = 128;
  Rng rng(seed);

  auto pool = std::make_shared<nn::PagedKVPool>(Hkv, D, BLK, /*num_blocks=*/128,
                                                DType::Float32, dev);
  std::vector<std::unique_ptr<nn::PagedKVCache>> caches;
  for (int64_t r = 0; r < A; ++r) {
    REQUIRE(kv_lens[r] >= S);  // prefill precondition
    auto c = std::make_unique<nn::PagedKVCache>(pool, /*batch=*/1, max_len);
    Tensor k = rand_f32({1, Hkv, kv_lens[r], D}, rng, dev);
    Tensor v = rand_f32({1, Hkv, kv_lens[r], D}, rng, dev);
    c->append(k, v);
    caches.push_back(std::move(c));
  }

  Tensor q = rand_f32({A, S, H, D}, rng, dev);
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));

  int64_t max_logical = 1;
  for (int64_t r = 0; r < A; ++r)
    max_logical = std::max<int64_t>(max_logical, (kv_lens[r] + BLK - 1) / BLK);
  std::vector<int32_t> table(static_cast<std::size_t>(A * max_logical), 0);
  std::vector<int32_t> kvl_vec(static_cast<std::size_t>(A));
  for (int64_t r = 0; r < A; ++r) {
    kvl_vec[static_cast<std::size_t>(r)] = static_cast<int32_t>(kv_lens[r]);
    const std::vector<int32_t>& bt = caches[static_cast<std::size_t>(r)]->block_table(0);
    for (std::size_t i = 0; i < bt.size(); ++i)
      table[static_cast<std::size_t>(r) * max_logical + i] = bt[i];
  }
  Tensor block_tables = int32_host(table, {A, max_logical}, dev);
  Tensor kv_lens_t    = int32_host(kvl_vec, {A}, dev);

  Tensor out = nn::paged_prefill_attention(q, pool->keys(), pool->values(),
                                           block_tables, kv_lens_t, scale, group);
  REQUIRE(out.shape() == Shape({A, S, H, D}));

  std::vector<float> ref(static_cast<std::size_t>(A * S * H * D), 0.0f);
  Tensor q_h = q.device().is_cpu() ? q.contiguous() : q.to(cpu_device()).contiguous();
  for (int64_t r = 0; r < A; ++r) {
    Tensor k_all = caches[static_cast<std::size_t>(r)]->keys_view();    // [1,Hkv,kv,D]
    Tensor v_all = caches[static_cast<std::size_t>(r)]->values_view();
    for (int64_t sq = 0; sq < S; ++sq) {
      const int64_t len = kv_lens[r] - S + sq + 1;  // causal bound
      for (int64_t h = 0; h < H; ++h) {
        const int64_t hkv = h / group;
        // q[r, sq, h] → [1,1,1,D]
        Tensor qslab = q.narrow(0, r, 1).narrow(1, sq, 1).narrow(2, h, 1);
        Tensor qh = qslab.reshape(Shape({1, 1, 1, D}));
        Tensor kh = k_all.narrow(1, hkv, 1).narrow(2, 0, len).contiguous();  // [1,1,len,D]
        Tensor vh = v_all.narrow(1, hkv, 1).narrow(2, 0, len).contiguous();
        Tensor oh = ops::attention(qh, kh, vh, Tensor{}, /*causal=*/false,
                                   /*dropout_p=*/0.0);                  // [1,1,1,D]
        Tensor ohh = oh.device().is_cpu() ? oh.contiguous()
                                          : oh.to(cpu_device()).contiguous();
        const float* p = ohh.data_ptr<float>();
        for (int64_t d = 0; d < D; ++d)
          ref[static_cast<std::size_t>(((r * S + sq) * H + h) * D + d)] = p[d];
      }
    }
  }

  Tensor out_h = out.device().is_cpu() ? out.contiguous()
                                       : out.to(cpu_device()).contiguous();
  const float* po = out_h.data_ptr<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < A * S * H * D; ++i)
    m = std::max(m, std::abs(po[i] - ref[static_cast<std::size_t>(i)]));
  INFO("PREFILL H=" << H << " Hkv=" << Hkv << " S=" << S << " A=" << A
       << " maxdiff=" << m);
  REQUIRE(m <= tol);
}

// INT8 prefill: equals the FP prefill op on the dequantized pool.
void run_prefill_int8_parity(Device dev, int64_t H, int64_t Hkv, int64_t S,
                             const std::vector<int64_t>& kv_lens, float tol,
                             uint64_t seed) {
  const int64_t A     = static_cast<int64_t>(kv_lens.size());
  const int64_t D     = 16;
  const int64_t BLK   = 4;
  const int64_t group = H / Hkv;
  const int64_t max_len = 128;
  Rng rng(seed);

  auto pool = std::make_shared<nn::PagedKVPool>(Hkv, D, BLK, /*num_blocks=*/128,
                                                DType::Float32, dev);
  std::vector<std::unique_ptr<nn::PagedKVCache>> caches;
  for (int64_t r = 0; r < A; ++r) {
    REQUIRE(kv_lens[r] >= S);
    auto c = std::make_unique<nn::PagedKVCache>(pool, /*batch=*/1, max_len);
    Tensor k = rand_f32({1, Hkv, kv_lens[r], D}, rng, dev);
    Tensor v = rand_f32({1, Hkv, kv_lens[r], D}, rng, dev);
    c->append(k, v);
    caches.push_back(std::move(c));
  }

  Tensor q = rand_f32({A, S, H, D}, rng, dev);
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));

  int64_t max_logical = 1;
  for (int64_t r = 0; r < A; ++r)
    max_logical = std::max<int64_t>(max_logical, (kv_lens[r] + BLK - 1) / BLK);
  std::vector<int32_t> table(static_cast<std::size_t>(A * max_logical), 0);
  std::vector<int32_t> kvl_vec(static_cast<std::size_t>(A));
  for (int64_t r = 0; r < A; ++r) {
    kvl_vec[static_cast<std::size_t>(r)] = static_cast<int32_t>(kv_lens[r]);
    const std::vector<int32_t>& bt = caches[static_cast<std::size_t>(r)]->block_table(0);
    for (std::size_t i = 0; i < bt.size(); ++i)
      table[static_cast<std::size_t>(r) * max_logical + i] = bt[i];
  }
  Tensor block_tables = int32_host(table, {A, max_logical}, dev);
  Tensor kv_lens_t    = int32_host(kvl_vec, {A}, dev);

  auto [kq, ksc] = quant::quantize_kv_per_token(pool->keys());
  auto [vq, vsc] = quant::quantize_kv_per_token(pool->values());
  Tensor dq_k = quant::dequantize_kv_per_token(kq, ksc, DType::Float32);
  Tensor dq_v = quant::dequantize_kv_per_token(vq, vsc, DType::Float32);

  Tensor out_i8 = nn::paged_prefill_attention_int8(
      q, kq, ksc, vq, vsc, block_tables, kv_lens_t, scale, group);
  REQUIRE(out_i8.shape() == Shape({A, S, H, D}));
  Tensor out_fp = nn::paged_prefill_attention(q, dq_k, dq_v, block_tables,
                                              kv_lens_t, scale, group);

  Tensor i8h = out_i8.device().is_cpu() ? out_i8.contiguous()
                                        : out_i8.to(cpu_device()).contiguous();
  Tensor fph = out_fp.device().is_cpu() ? out_fp.contiguous()
                                        : out_fp.to(cpu_device()).contiguous();
  const float* pi = i8h.data_ptr<float>();
  const float* pf = fph.data_ptr<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < A * S * H * D; ++i) m = std::max(m, std::abs(pi[i] - pf[i]));
  INFO("PREFILL-INT8 H=" << H << " Hkv=" << Hkv << " S=" << S << " A=" << A
       << " maxdiff=" << m);
  REQUIRE(m <= tol);
}

}  // namespace

TEST_CASE("paged_prefill_attention matches gather+attention (CPU, MHA)",
          "[nn][paged][attention][prefill]") {
  run_prefill_parity(cpu_device(), /*H=*/4, /*Hkv=*/4, /*S=*/4, {6, 4, 9, 13},
                     1e-5f, 0x9000u);
}

TEST_CASE("paged_prefill_attention matches gather+attention (CPU, GQA)",
          "[nn][paged][attention][prefill]") {
  run_prefill_parity(cpu_device(), /*H=*/8, /*Hkv=*/2, /*S=*/3, {3, 16, 7, 11},
                     1e-5f, 0x9001u);
}

TEST_CASE("paged_prefill_attention matches gather+attention (CUDA)",
          "[nn][paged][attention][prefill][cuda]") {
  if (cuda::device_count() <= 0) {
    SKIP("CUDA not available");
    return;
  }
  const Device dev{DeviceType::CUDA, 0};
  run_prefill_parity(dev, /*H=*/4, /*Hkv=*/4, /*S=*/4, {6, 4, 9, 13}, 3e-3f, 0x9002u);
  run_prefill_parity(dev, /*H=*/8, /*Hkv=*/2, /*S=*/5, {5, 16, 7, 11}, 3e-3f, 0x9003u);
}

TEST_CASE("paged_prefill_attention_int8 == FP op on dequant pool (CPU)",
          "[nn][paged][attention][prefill][quant]") {
  run_prefill_int8_parity(cpu_device(), /*H=*/8, /*Hkv=*/2, /*S=*/3,
                          {5, 16, 7, 11}, 1e-5f, 0x9004u);
}

TEST_CASE("paged_prefill_attention_int8 == FP op on dequant pool (CUDA)",
          "[nn][paged][attention][prefill][quant][cuda]") {
  if (cuda::device_count() <= 0) {
    SKIP("CUDA not available");
    return;
  }
  const Device dev{DeviceType::CUDA, 0};
  run_prefill_int8_parity(dev, /*H=*/8, /*Hkv=*/2, /*S=*/3, {5, 16, 7, 11},
                          3e-3f, 0x9005u);
}

TEST_CASE("paged_decode_attention matches gather+attention (CPU, MHA)",
          "[nn][paged][attention]") {
  run_parity(cpu_device(), /*H=*/4, /*Hkv=*/4, {5, 1, 9, 13}, 1e-5f, 0xA11CEu);
}

TEST_CASE("paged_decode_attention matches gather+attention (CPU, GQA)",
          "[nn][paged][attention]") {
  run_parity(cpu_device(), /*H=*/8, /*Hkv=*/2, {3, 16, 7, 1, 11}, 1e-5f, 0xBEEFu);
}

TEST_CASE("paged_decode_attention handles a zero-length request (CPU)",
          "[nn][paged][attention]") {
  run_parity(cpu_device(), /*H=*/4, /*Hkv=*/4, {0, 6, 2}, 1e-5f, 0xC0DEu);
}

TEST_CASE("paged_decode_attention matches gather+attention (CUDA)",
          "[nn][paged][attention][cuda]") {
  if (cuda::device_count() <= 0) {
    SKIP("CUDA not available");
    return;
  }
  const Device dev{DeviceType::CUDA, 0};
  run_parity(dev, /*H=*/4, /*Hkv=*/4, {5, 1, 9, 13}, 3e-3f, 0xA11CEu);
  run_parity(dev, /*H=*/8, /*Hkv=*/2, {3, 16, 7, 1, 11}, 3e-3f, 0xBEEFu);
}

// End-to-end wiring: MHA::forward_step_batched on CUDA paged caches must
// take the fused kernel path and match per-request forward_step (the
// fallback path) within float tolerance.
TEST_CASE("MHA::forward_step_batched (paged, CUDA) matches per-request forward_step",
          "[nn][paged][attention][cuda]") {
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
  Rng rng(0x5150u);

  // Separate ref + fused pools (one pool each, all requests share it).
  auto ref_pool = std::make_shared<nn::PagedKVPool>(Hkv, D / H, BLK, 128, DType::Float32, dev);
  auto fus_pool = std::make_shared<nn::PagedKVPool>(Hkv, D / H, BLK, 128, DType::Float32, dev);
  std::vector<std::unique_ptr<nn::PagedKVCache>> ref_c, fus_c;
  for (int64_t r = 0; r < A; ++r) {
    ref_c.push_back(std::make_unique<nn::PagedKVCache>(ref_pool, 1, max_len));
    fus_c.push_back(std::make_unique<nn::PagedKVCache>(fus_pool, 1, max_len));
  }

  // Prefill each request with its own prompt through both cache sets.
  for (int64_t r = 0; r < A; ++r) {
    Tensor prompt = rand_f32({1, prompt_lens[r], D}, rng, dev);
    mha.forward_step(prompt, *ref_c[r]);
    mha.forward_step(prompt, *fus_c[r]);
  }

  // One decode step: per-request reference vs batched fused.
  std::vector<Tensor> x_list;
  std::vector<Tensor> ref_out;
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
  INFO("fused-vs-perrequest maxdiff=" << m);
  REQUIRE(m <= 3e-3f);
}

// Wave 14 (B-032++++): end-to-end wiring of the fused paged PREFILL path.
// MHA::forward_step_batched with Sn>1 (chunked prefill) over CUDA paged
// caches must take the prefill kernel and match per-request forward_step.
TEST_CASE("MHA::forward_step_batched (paged prefill Sn>1, CUDA) matches per-request",
          "[nn][paged][attention][prefill][cuda]") {
  if (cuda::device_count() <= 0) {
    SKIP("CUDA not available");
    return;
  }
  const Device dev{DeviceType::CUDA, 0};
  const int64_t D = 32, H = 4, Hkv = 2, BLK = 4, max_len = 128;
  const int64_t Sn = 6;  // chunk of 6 new tokens per request
  nn::MultiHeadAttention mha(/*d_model=*/D, /*num_heads=*/H, /*use_bias=*/false,
                             /*causal=*/true, DType::Float32,
                             /*rope_base=*/10000.0, /*rope_max_seq=*/128,
                             /*num_kv_heads=*/Hkv);
  mha.to(dev);

  // Pre-existing context per request (so the chunk attends to a real prefix).
  const std::vector<int64_t> ctx_lens = {4, 0, 9, 2};
  const int64_t A = static_cast<int64_t>(ctx_lens.size());
  Rng rng(0x71F1u);

  auto ref_pool = std::make_shared<nn::PagedKVPool>(Hkv, D / H, BLK, 256, DType::Float32, dev);
  auto fus_pool = std::make_shared<nn::PagedKVPool>(Hkv, D / H, BLK, 256, DType::Float32, dev);
  std::vector<std::unique_ptr<nn::PagedKVCache>> ref_c, fus_c;
  for (int64_t r = 0; r < A; ++r) {
    ref_c.push_back(std::make_unique<nn::PagedKVCache>(ref_pool, 1, max_len));
    fus_c.push_back(std::make_unique<nn::PagedKVCache>(fus_pool, 1, max_len));
    if (ctx_lens[r] > 0) {
      Tensor ctx = rand_f32({1, ctx_lens[r], D}, rng, dev);
      mha.forward_step(ctx, *ref_c[r]);
      mha.forward_step(ctx, *fus_c[r]);
    }
  }

  // One chunked-prefill step of Sn new tokens: per-request ref vs batched fused.
  std::vector<Tensor> x_list;
  std::vector<Tensor> ref_out;
  for (int64_t r = 0; r < A; ++r) {
    Tensor xr = rand_f32({1, Sn, D}, rng, dev);
    x_list.push_back(xr);
    ref_out.push_back(mha.forward_step(xr, *ref_c[r]).to(cpu_device()).contiguous());
  }
  Tensor x_all = Tensor::empty({A, Sn, D}, DType::Float32, cpu_device());
  {
    float* p = x_all.data_ptr<float>();
    for (int64_t r = 0; r < A; ++r) {
      Tensor xh = x_list[static_cast<std::size_t>(r)].to(cpu_device()).contiguous();
      const float* px = xh.data_ptr<float>();
      for (int64_t i = 0; i < Sn * D; ++i) p[r * Sn * D + i] = px[i];
    }
  }
  Tensor x_all_dev = x_all.to(dev);

  std::vector<nn::KVCacheBase*> bases;
  for (int64_t r = 0; r < A; ++r) bases.push_back(fus_c[static_cast<std::size_t>(r)].get());
  Tensor out = mha.forward_step_batched(x_all_dev, bases).to(cpu_device()).contiguous();
  REQUIRE(out.shape() == Shape({A, Sn, D}));

  const float* po = out.data_ptr<float>();
  float m = 0.0f;
  for (int64_t r = 0; r < A; ++r) {
    const float* pr = ref_out[static_cast<std::size_t>(r)].data_ptr<float>();
    for (int64_t i = 0; i < Sn * D; ++i)
      m = std::max(m, std::abs(po[r * Sn * D + i] - pr[i]));
  }
  INFO("prefill fused-vs-perrequest maxdiff=" << m);
  REQUIRE(m <= 3e-3f);
}

TEST_CASE("paged_decode_attention_int8 == FP op on dequant pool (CPU, MHA)",
          "[nn][paged][attention][quant]") {
  run_int8_parity(cpu_device(), /*H=*/4, /*Hkv=*/4, {5, 1, 9, 13}, 1e-5f, 0x12C8u);
}

TEST_CASE("paged_decode_attention_int8 == FP op on dequant pool (CPU, GQA)",
          "[nn][paged][attention][quant]") {
  run_int8_parity(cpu_device(), /*H=*/8, /*Hkv=*/2, {3, 16, 7, 1, 11}, 1e-5f, 0x12C9u);
}

TEST_CASE("paged_decode_attention_int8 handles a zero-length request (CPU)",
          "[nn][paged][attention][quant]") {
  run_int8_parity(cpu_device(), /*H=*/4, /*Hkv=*/4, {0, 6, 2}, 1e-5f, 0x12CAu);
}

TEST_CASE("paged_decode_attention_int8 == FP op on dequant pool (CUDA)",
          "[nn][paged][attention][quant][cuda]") {
  if (cuda::device_count() <= 0) {
    SKIP("CUDA not available");
    return;
  }
  const Device dev{DeviceType::CUDA, 0};
  run_int8_parity(dev, /*H=*/4, /*Hkv=*/4, {5, 1, 9, 13}, 3e-3f, 0x12C8u);
  run_int8_parity(dev, /*H=*/8, /*Hkv=*/2, {3, 16, 7, 1, 11}, 3e-3f, 0x12C9u);
}

TEST_CASE("paged_decode_attention_int8 validates operands",
          "[nn][paged][attention][quant]") {
  const Device dev = cpu_device();
  const int64_t A = 2, H = 4, Hkv = 4, D = 16, BLK = 4;
  auto pool = std::make_shared<nn::PagedKVPool>(Hkv, D, BLK, 16, DType::Float32, dev);
  Rng rng(7u);
  Tensor q = rand_f32({A, H, D}, rng, dev);
  Tensor bt = int32_host({0, 0}, {A, 1}, dev);
  Tensor lens = int32_host({0, 0}, {A}, dev);
  auto [kq, ksc] = quant::quantize_kv_per_token(pool->keys());
  auto [vq, vsc] = quant::quantize_kv_per_token(pool->values());
  const double scale = 0.25;

  // FP pool passed where Int8 is required.
  REQUIRE_THROWS(nn::paged_decode_attention_int8(
      q, pool->keys(), ksc, vq, vsc, bt, lens, scale, /*group=*/1));
  // group mismatch.
  REQUIRE_THROWS(nn::paged_decode_attention_int8(
      q, kq, ksc, vq, vsc, bt, lens, scale, /*group=*/3));
  // scale numel mismatch.
  Tensor ksc_bad = rand_f32({1}, rng, dev);
  REQUIRE_THROWS(nn::paged_decode_attention_int8(
      q, kq, ksc_bad, vq, vsc, bt, lens, scale, /*group=*/1));
}

TEST_CASE("paged_decode_attention validates operands", "[nn][paged][attention]") {
  const Device dev = cpu_device();
  const int64_t A = 2, H = 4, Hkv = 4, D = 16, BLK = 4;
  auto pool = std::make_shared<nn::PagedKVPool>(Hkv, D, BLK, 16, DType::Float32, dev);
  Rng rng(1u);
  Tensor q = rand_f32({A, H, D}, rng, dev);
  Tensor bt = int32_host({0, 0}, {A, 1}, dev);
  Tensor lens = int32_host({0, 0}, {A}, dev);
  const double scale = 0.25;

  // group mismatch: H must equal Hkv*group.
  REQUIRE_THROWS(nn::paged_decode_attention(q, pool->keys(), pool->values(),
                                            bt, lens, scale, /*group=*/3));
  // wrong q rank.
  Tensor q2 = q.reshape(Shape({A * H, D}));
  REQUIRE_THROWS(nn::paged_decode_attention(q2, pool->keys(), pool->values(),
                                            bt, lens, scale, /*group=*/1));
  // lens length mismatch.
  Tensor lens_bad = int32_host({0}, {1}, dev);
  REQUIRE_THROWS(nn::paged_decode_attention(q, pool->keys(), pool->values(),
                                            bt, lens_bad, scale, /*group=*/1));
}

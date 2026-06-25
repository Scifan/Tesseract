// Wave 4.5 (B-019b) — PagedKVCache + BlockAllocator.
//
// PagedKVCache is a drop-in for the Wave 2.1 KVCache that stores K/V in
// a fixed physical block pool indexed by a per-request block table
// instead of a contiguous max_len slab. The whole point is that it must
// be *behaviorally indistinguishable* from KVCache on the decode path —
// otherwise generated tokens silently diverge — while using memory
// proportional to the actual sequence length rather than max_len.
//
// What this file pins down:
//
//   * BlockAllocator free-list correctness: distinct ids, recycling,
//     exhaustion throw, double-free throw.
//   * PagedKVCache::append + keys_view/values_view reproduce the exact
//     same [B, H, current_len, Dh] prefix that KVCache produces, across
//     a mixed chunk schedule that crosses block boundaries.
//   * MultiHeadAttention::forward_step with a PagedKVCache equals the
//     one-shot forward() within FP32 abs 1e-5 — the same bar KVCache
//     meets.
//   * Memory residency: allocated blocks track ceil(current_len /
//     block_size) per request, and reset() recycles them.
//   * CPU <-> CUDA parity on the paged decode path.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/BlockAllocator.hpp"
#include "tesseract/nn/KVCache.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"
#include "tesseract/nn/PagedKVCache.hpp"
#include "tesseract/utils/Logging.hpp"

#include "tesseract/cuda/CudaRuntime.hpp"

using namespace tesseract;

namespace {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  float uniform(float lo, float hi) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u =
        static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFULL) /
        9007199254740992.0;
    return static_cast<float>(lo + (hi - lo) * u);
  }
};

Tensor make_random_f32(Shape s, Rng& rng, float lo = -0.5f, float hi = 0.5f) {
  Tensor t = Tensor::empty(std::move(s), DType::Float32);
  float* p = t.data_ptr<float>();
  const int64_t n = t.numel();
  for (int64_t i = 0; i < n; ++i) p[i] = rng.uniform(lo, hi);
  return t;
}

Tensor slice_seq(const Tensor& x, int64_t lo, int64_t len) {
  return x.narrow(/*dim=*/1, lo, len);
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
  REQUIRE(a.numel() == b.numel());
  Tensor a_cpu = a.device().is_cpu() ? a.contiguous()
                                     : a.to(cpu_device()).contiguous();
  Tensor b_cpu = b.device().is_cpu() ? b.contiguous()
                                     : b.to(cpu_device()).contiguous();
  const float* pa = a_cpu.data_ptr<float>();
  const float* pb = b_cpu.data_ptr<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < a_cpu.numel(); ++i) {
    m = std::max(m, std::abs(pa[i] - pb[i]));
  }
  return m;
}

}  // namespace

TEST_CASE("BlockAllocator: free-list hands out distinct ids and recycles",
          "[nn][paged][allocator]") {
  nn::BlockAllocator alloc(/*num_blocks=*/4);
  REQUIRE(alloc.num_blocks() == 4);
  REQUIRE(alloc.num_free() == 4);
  REQUIRE(alloc.num_allocated() == 0);

  std::vector<int32_t> ids;
  for (int i = 0; i < 4; ++i) ids.push_back(alloc.allocate());
  // All four distinct, all in range.
  std::sort(ids.begin(), ids.end());
  REQUIRE(ids == std::vector<int32_t>({0, 1, 2, 3}));
  REQUIRE(alloc.num_free() == 0);
  REQUIRE(alloc.num_allocated() == 4);

  // Pool exhausted.
  REQUIRE_THROWS(alloc.allocate());

  // Free one and re-allocate — it comes back.
  alloc.free(2);
  REQUIRE(alloc.num_free() == 1);
  const int32_t reused = alloc.allocate();
  REQUIRE(reused == 2);

  // Double-free + out-of-range throw.
  alloc.free(0);
  REQUIRE_THROWS(alloc.free(0));      // already free
  REQUIRE_THROWS(alloc.free(-1));     // out of range
  REQUIRE_THROWS(alloc.free(4));      // out of range

  alloc.free_all();
  REQUIRE(alloc.num_free() == 4);
  REQUIRE(alloc.num_allocated() == 0);
}

TEST_CASE("PagedKVCache: keys/values gather matches contiguous KVCache",
          "[nn][paged][kvcache]") {
  constexpr int64_t B = 2, H = 3, Dh = 4, MAX = 8, BLK = 3;
  // Pool sized so a full batch (2 requests * ceil(8/3)=3 blocks each = 6)
  // fits with headroom.
  nn::PagedKVCache paged(B, H, Dh, MAX, /*block_size=*/BLK, /*num_blocks=*/8,
                         DType::Float32, cpu_device());
  nn::KVCache contig(B, H, Dh, MAX, DType::Float32, cpu_device());

  REQUIRE(paged.current_len() == 0);
  REQUIRE(paged.keys_view().shape() == Shape({B, H, 0, Dh}));
  REQUIRE(paged.num_allocated_blocks() == 0);

  Rng rng(0xC0FFEEu);
  // Mixed schedule: 2, then 1, then 3, then 2 = 8 tokens. The chunk of
  // 3 starting at pos=3 straddles the block_size=3 boundary, exercising
  // the multi-block append path.
  for (int64_t c : {int64_t{2}, int64_t{1}, int64_t{3}, int64_t{2}}) {
    Tensor k = make_random_f32({B, H, c, Dh}, rng);
    Tensor v = make_random_f32({B, H, c, Dh}, rng);
    paged.append(k, v);
    contig.append(k, v);
    REQUIRE(paged.current_len() == contig.current_len());
    // keys/values prefix must match the contiguous cache byte-for-byte.
    REQUIRE(max_abs_diff(paged.keys_view(), contig.keys_view()) == 0.0f);
    REQUIRE(max_abs_diff(paged.values_view(), contig.values_view()) == 0.0f);
  }
  REQUIRE(paged.current_len() == 8);

  // Memory residency: each request holds ceil(8/3) = 3 blocks → 6 total.
  REQUIRE(paged.num_allocated_blocks() == B * 3);

  // reset() recycles every block.
  paged.reset();
  REQUIRE(paged.current_len() == 0);
  REQUIRE(paged.num_allocated_blocks() == 0);
  REQUIRE(paged.keys_view().shape() == Shape({B, H, 0, Dh}));
}

TEST_CASE("PagedKVCache: residency tracks ceil(len/block_size), not max_len",
          "[nn][paged][kvcache]") {
  constexpr int64_t B = 1, H = 2, Dh = 4, MAX = 64, BLK = 8;
  nn::PagedKVCache paged(B, H, Dh, MAX, BLK, /*num_blocks=*/16,
                         DType::Float32, cpu_device());
  Rng rng(11u);

  // Emit 5 tokens — way short of max_len=64. The contiguous cache would
  // reserve all 64; paged holds ceil(5/8) = 1 block.
  Tensor k = make_random_f32({B, H, 5, Dh}, rng);
  Tensor v = make_random_f32({B, H, 5, Dh}, rng);
  paged.append(k, v);
  REQUIRE(paged.num_allocated_blocks() == 1);

  // 4 more tokens (total 9) crosses into a 2nd block.
  Tensor k2 = make_random_f32({B, H, 4, Dh}, rng);
  Tensor v2 = make_random_f32({B, H, 4, Dh}, rng);
  paged.append(k2, v2);
  REQUIRE(paged.current_len() == 9);
  REQUIRE(paged.num_allocated_blocks() == 2);  // ceil(9/8) = 2
}

TEST_CASE("PagedKVCache: append rejects overflow and pool exhaustion",
          "[nn][paged][kvcache]") {
  Rng rng(7u);
  // max_len overflow.
  {
    nn::PagedKVCache cache(/*B=*/1, /*H=*/2, /*Dh=*/4, /*MAX=*/3, /*BLK=*/2,
                           /*num_blocks=*/8);
    Tensor k = make_random_f32({1, 2, 2, 4}, rng);
    Tensor v = make_random_f32({1, 2, 2, 4}, rng);
    cache.append(k, v);
    Tensor k2 = make_random_f32({1, 2, 2, 4}, rng);
    Tensor v2 = make_random_f32({1, 2, 2, 4}, rng);
    REQUIRE_THROWS(cache.append(k2, v2));  // 2 + 2 > max_len=3
  }
  // Pool exhaustion: 1 block of size 2 can hold 2 tokens; the 3rd throws.
  {
    nn::PagedKVCache cache(/*B=*/1, /*H=*/1, /*Dh=*/2, /*MAX=*/16, /*BLK=*/2,
                           /*num_blocks=*/1);
    Tensor k = make_random_f32({1, 1, 2, 2}, rng);
    Tensor v = make_random_f32({1, 1, 2, 2}, rng);
    cache.append(k, v);  // fills the single block
    Tensor k2 = make_random_f32({1, 1, 1, 2}, rng);
    Tensor v2 = make_random_f32({1, 1, 1, 2}, rng);
    REQUIRE_THROWS(cache.append(k2, v2));  // needs a 2nd block; none free
  }
}

TEST_CASE("MHA::forward_step with PagedKVCache matches full forward (CPU)",
          "[nn][paged][mha]") {
  constexpr int64_t B = 2, S = 6, D = 16, H = 4, BLK = 4;
  nn::MultiHeadAttention mha(
      /*d_model=*/D, /*num_heads=*/H, /*use_bias=*/false,
      /*causal=*/true, DType::Float32,
      /*rope_base=*/10000.0, /*rope_max_seq=*/16);

  Rng rng(0xDEADBEEFu);
  Tensor x = make_random_f32({B, S, D}, rng);
  Tensor out_full = mha.forward(x);

  // Stream token-by-token through a paged cache; block_size=4 with S=6
  // means the 2nd block kicks in mid-decode.
  nn::PagedKVCache cache(B, H, D / H, /*max_len=*/S, BLK,
                         /*num_blocks=*/B * 2, DType::Float32, cpu_device());
  Tensor acc = Tensor::empty({B, S, D}, DType::Float32);
  float* p_acc = acc.data_ptr<float>();
  for (int64_t t = 0; t < S; ++t) {
    Tensor xt = slice_seq(x, t, 1).contiguous();
    Tensor yt = mha.forward_step(xt, cache);
    REQUIRE(yt.shape() == Shape({B, 1, D}));
    const float* p = yt.contiguous().data_ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t d = 0; d < D; ++d) {
        p_acc[(b * S + t) * D + d] = p[b * D + d];
      }
    }
  }
  REQUIRE(max_abs_diff(out_full, acc) < 1e-5f);
}

TEST_CASE("MHA::forward_step with PagedKVCache matches KVCache (CUDA)",
          "[nn][paged][mha][cuda]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  constexpr int64_t B = 1, S = 8, D = 32, H = 4, BLK = 3;
  nn::MultiHeadAttention mha(
      /*d_model=*/D, /*num_heads=*/H, /*use_bias=*/false,
      /*causal=*/true, DType::Float32,
      /*rope_base=*/10000.0, /*rope_max_seq=*/32);
  const Device cuda0{DeviceType::CUDA, 0};
  mha.to(cuda0);

  Rng rng(0xFEEDFACEu);
  Tensor x_cpu = make_random_f32({B, S, D}, rng);
  Tensor x = x_cpu.to(cuda0);

  Tensor out_full = mha.forward(x).to(cpu_device());

  nn::PagedKVCache cache(B, H, D / H, /*max_len=*/S, BLK,
                         /*num_blocks=*/B * 4, DType::Float32, cuda0);
  Tensor acc = Tensor::empty({B, S, D}, DType::Float32);
  float* p_acc = acc.data_ptr<float>();
  for (int64_t t = 0; t < S; ++t) {
    Tensor xt = slice_seq(x, t, 1).contiguous();
    Tensor yt = mha.forward_step(xt, cache).to(cpu_device()).contiguous();
    const float* p = yt.data_ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t d = 0; d < D; ++d) {
        p_acc[(b * S + t) * D + d] = p[b * D + d];
      }
    }
  }
  REQUIRE(max_abs_diff(out_full, acc) < 5e-5f);
}

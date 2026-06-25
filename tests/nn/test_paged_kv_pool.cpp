// Wave 7 (B-029) — shared PagedKVPool across many PagedKVCaches.
//
// The continuous-batching memory win comes from MANY requests drawing
// blocks from ONE pool so a finished request's blocks recycle into the
// shared budget. This file pins down:
//   * many caches share one pool's block budget;
//   * reset() returns ONLY a cache's own blocks (never others');
//   * freed blocks are reused by the next cache (recycling);
//   * a shared-pool cache produces byte-identical keys/values views to a
//     private-pool cache fed the same appends (no cross-talk).

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/PagedKVCache.hpp"
#include "tesseract/nn/PagedKVPool.hpp"
#include "tesseract/utils/Logging.hpp"

using namespace tesseract;

namespace {

// Fill a [1, H, S, Dh] tensor with a deterministic ramp keyed by `base`
// so different appends are distinguishable in the gather.
Tensor ramp(int64_t H, int64_t S, int64_t Dh, float base) {
  Tensor t = Tensor::empty({1, H, S, Dh}, DType::Float32);
  float* p = t.data_ptr<float>();
  const int64_t n = t.numel();
  for (int64_t i = 0; i < n; ++i) p[i] = base + static_cast<float>(i);
  return t;
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
  REQUIRE(a.numel() == b.numel());
  const float* pa = a.data_ptr<float>();
  const float* pb = b.data_ptr<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < a.numel(); ++i)
    m = std::max(m, std::abs(pa[i] - pb[i]));
  return m;
}

}  // namespace

TEST_CASE("PagedKVPool: many caches share one block budget", "[nn][paged][pool]") {
  const int64_t H = 2, Dh = 4, block_size = 4, num_blocks = 8;
  auto pool = std::make_shared<nn::PagedKVPool>(H, Dh, block_size, num_blocks,
                                                DType::Float32, cpu_device());

  nn::PagedKVCache a(pool, /*batch=*/1, /*max_len=*/32);
  nn::PagedKVCache b(pool, /*batch=*/1, /*max_len=*/32);

  REQUIRE(pool->num_allocated() == 0);
  REQUIRE(pool->num_free() == num_blocks);

  // a: 6 tokens -> ceil(6/4) = 2 blocks.
  a.append(ramp(H, 6, Dh, 100.0f), ramp(H, 6, Dh, 200.0f));
  REQUIRE(a.current_len() == 6);
  REQUIRE(a.num_owned_blocks() == 2);
  REQUIRE(pool->num_allocated() == 2);

  // b: 3 tokens -> 1 block. Pool budget is shared.
  b.append(ramp(H, 3, Dh, 300.0f), ramp(H, 3, Dh, 400.0f));
  REQUIRE(b.num_owned_blocks() == 1);
  REQUIRE(pool->num_allocated() == 3);
  REQUIRE(pool->num_free() == num_blocks - 3);
}

TEST_CASE("PagedKVPool: reset frees only the cache's own blocks", "[nn][paged][pool]") {
  const int64_t H = 2, Dh = 4, block_size = 4, num_blocks = 8;
  auto pool = std::make_shared<nn::PagedKVPool>(H, Dh, block_size, num_blocks,
                                                DType::Float32, cpu_device());
  nn::PagedKVCache a(pool, 1, 32);
  nn::PagedKVCache b(pool, 1, 32);

  a.append(ramp(H, 6, Dh, 1.0f), ramp(H, 6, Dh, 2.0f));  // 2 blocks
  b.append(ramp(H, 3, Dh, 3.0f), ramp(H, 3, Dh, 4.0f));  // 1 block
  REQUIRE(pool->num_allocated() == 3);

  a.reset();  // returns a's 2 blocks only
  REQUIRE(a.current_len() == 0);
  REQUIRE(a.num_owned_blocks() == 0);
  REQUIRE(pool->num_allocated() == 1);   // b's block survives
  REQUIRE(b.current_len() == 3);
  REQUIRE(b.num_owned_blocks() == 1);

  // b's data is untouched by a's reset.
  Tensor b_keys = b.keys_view();
  REQUIRE(b_keys.shape()[2] == 3);
  REQUIRE(max_abs_diff(b_keys, ramp(H, 3, Dh, 3.0f)) == 0.0f);
}

TEST_CASE("PagedKVPool: freed blocks are recycled by the next cache", "[nn][paged][pool]") {
  const int64_t H = 1, Dh = 2, block_size = 2, num_blocks = 3;
  auto pool = std::make_shared<nn::PagedKVPool>(H, Dh, block_size, num_blocks,
                                                DType::Float32, cpu_device());
  // Exhaust the pool with one cache (3 blocks = 6 tokens).
  nn::PagedKVCache a(pool, 1, 16);
  a.append(ramp(H, 6, Dh, 10.0f), ramp(H, 6, Dh, 20.0f));
  REQUIRE(pool->num_free() == 0);

  // A fresh append on a second cache would now throw (pool dry).
  nn::PagedKVCache b(pool, 1, 16);
  REQUIRE_THROWS(b.append(ramp(H, 2, Dh, 30.0f), ramp(H, 2, Dh, 40.0f)));

  // Recycle a's blocks; b can now allocate from the same physical pool.
  a.reset();
  REQUIRE(pool->num_free() == num_blocks);
  REQUIRE_NOTHROW(b.append(ramp(H, 4, Dh, 50.0f), ramp(H, 4, Dh, 60.0f)));
  REQUIRE(b.current_len() == 4);
  REQUIRE(max_abs_diff(b.keys_view(), ramp(H, 4, Dh, 50.0f)) == 0.0f);
}

TEST_CASE("PagedKVPool: shared-pool cache matches private-pool cache", "[nn][paged][pool]") {
  const int64_t H = 3, Dh = 8, block_size = 4, num_blocks = 16;

  // Private-pool cache (Wave 4.5 ctor).
  nn::PagedKVCache priv(/*batch=*/1, H, Dh, /*max_len=*/64, block_size,
                        num_blocks, DType::Float32, cpu_device());
  // Shared-pool cache (Wave 7 ctor).
  auto pool = std::make_shared<nn::PagedKVPool>(H, Dh, block_size, num_blocks,
                                                DType::Float32, cpu_device());
  nn::PagedKVCache shared(pool, 1, 64);

  // Identical mixed-chunk append schedule crossing block boundaries.
  const std::vector<int64_t> chunks = {1, 3, 4, 2, 5};
  float base = 0.0f;
  for (int64_t s : chunks) {
    Tensor k = ramp(H, s, Dh, base);
    Tensor v = ramp(H, s, Dh, base + 1000.0f);
    priv.append(k, v);
    shared.append(k, v);
    base += 100.0f;
  }
  REQUIRE(priv.current_len() == shared.current_len());
  REQUIRE(max_abs_diff(priv.keys_view(), shared.keys_view()) == 0.0f);
  REQUIRE(max_abs_diff(priv.values_view(), shared.values_view()) == 0.0f);
}

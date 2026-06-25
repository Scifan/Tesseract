#pragma once

// Wave 4.5 (B-019b): vLLM-style paged KV cache.
//
// The Wave 2.1 `nn::KVCache` pre-allocates a contiguous
// `[B, H, max_len, D_head]` slab per (keys, values). That is simple and
// fast for a single long request, but wastes memory the moment you
// serve *many* requests whose lengths vary: every request reserves the
// full `max_len` whether it emits 8 tokens or 8192. For a server the
// resident KV memory is `num_requests · max_len`, dominated by padding.
//
// `PagedKVCache` swaps the contiguous slab for a fixed-size physical
// **block pool** plus a per-request **block table** (the PagedAttention
// design). Storage is carved into `num_blocks` blocks of `block_size`
// tokens each; a request holding `S` tokens occupies
// `ceil(S / block_size)` blocks pulled from a shared `BlockAllocator`.
// Resident memory becomes `Σ_r ceil(S_r / block_size) · block_size`
// instead of `num_requests · max_len` — the win continuous batching is
// built on.
//
// Drop-in contract
// ----------------
// The public surface mirrors `KVCache` exactly — `append(k_new, v_new)`,
// `keys_view()`, `values_view()`, `reset()`, `current_len()`, the dim
// accessors — so `MultiHeadAttention::forward_step` consumes either
// cache through the same calls with no code change. The one behavioral
// difference is documented on `keys_view()`: because the valid prefix is
// physically scattered across blocks, the view is **materialized** by a
// gather into a fresh contiguous `[B, H, current_len, D_head]` tensor
// rather than returned as a zero-copy narrow. That keeps `ops::attention`
// (and the Wave 4.2 fused-attention kernel) unchanged. A
// block-table-aware paged-attention kernel that reads K/V in place — no
// gather — is the B-019b+ performance follow-up.
//
// MVP scope: uniform batch length (all `B` requests share one
// `current_len_`, matching `KVCache` semantics) so the attention
// consumption path is untouched. Ragged per-request lengths — the full
// continuous-batching story — pairs with the scheduler and is tracked as
// B-019b+. Even with uniform length the paging win is real: physical
// blocks are allocated on demand proportional to `current_len`, so a
// short prompt never reserves the `max_len` tail.

#include <cstdint>
#include <memory>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/KVCacheBase.hpp"
#include "tesseract/nn/PagedKVPool.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

class PagedKVCache : public KVCacheBase {
 public:
  // Allocate a paged cache backed by its OWN private block pool.
  // `block_size` tokens per physical block; `num_blocks` total blocks.
  // `max_len` bounds the per-request sequence length (used for `append`
  // overflow checks and `set_current_len` bounds). To guarantee a single
  // full-length batch never exhausts the pool, size `num_blocks >=
  // batch * ceil(max_len / block_size)`; smaller pools are legal and
  // simply rely on requests staying short (a deliberate paging choice —
  // `append` throws cleanly via the allocator if the pool runs dry).
  PagedKVCache(int64_t batch, int64_t num_heads, int64_t head_dim,
               int64_t max_len, int64_t block_size, int64_t num_blocks,
               DType dtype = DType::Float32, Device device = cpu_device());

  // Wave 7 (B-029): allocate a cache that draws blocks from an EXISTING
  // shared `PagedKVPool` (one pool per transformer layer, shared across
  // every request in a continuous batch). The cache's geometry (heads,
  // head_dim, block_size, dtype, device) is inherited from the pool; it
  // keeps only its own block table + length. `reset()` returns this
  // cache's blocks to the shared pool without touching other requests'.
  PagedKVCache(std::shared_ptr<PagedKVPool> pool, int64_t batch,
               int64_t max_len);

  // Same contract as KVCache::append: copy `[B, H, S_new, D_head]`
  // contiguous K/V into seq positions `[current_len_, current_len_ +
  // S_new)`, allocating physical blocks on demand, and advance
  // `current_len_`. Returns the updated `current_len_`.
  int64_t append(const Tensor& k_new, const Tensor& v_new) override;

  // Gather the valid `[B, H, current_len_, D_head]` prefix out of the
  // scattered physical blocks into a fresh contiguous tensor. NOTE:
  // unlike KVCache this is a copy, not a zero-copy narrow (see header
  // note). Safe to feed straight into `ops::attention`.
  Tensor keys_view() const override;
  Tensor values_view() const override;

  // Recycle every block back to the allocator and rewind to empty.
  // Storage (the pool) is retained for the next prompt.
  void reset();

  // Capture-only position override, mirroring KVCache::set_current_len.
  // Requires the blocks backing `[0, len)` to already be allocated
  // (i.e. a prior `append` reached at least `len`); throws otherwise.
  void set_current_len(int64_t len);

  int64_t batch()        const noexcept override { return batch_; }
  int64_t num_heads()    const noexcept override { return pool_->num_heads(); }
  int64_t head_dim()     const noexcept override { return pool_->head_dim(); }
  int64_t current_len()  const noexcept override { return current_len_; }
  int64_t max_len()      const noexcept { return max_len_; }
  int64_t block_size()   const noexcept { return pool_->block_size(); }
  int64_t num_blocks()   const noexcept { return pool_->num_blocks(); }
  DType   dtype()        const noexcept { return pool_->dtype(); }
  Device  device()       const noexcept { return pool_->device(); }

  // The shared (or private) pool backing this cache. Lets a scheduler
  // bind several caches to the same pool.
  const std::shared_ptr<PagedKVPool>& pool() const noexcept { return pool_; }

  // Physical blocks currently handed out across the whole backing pool —
  // the memory-residency metric the bench gates on. For a private-pool
  // cache this equals this cache's own blocks
  // (`batch * ceil(current_len_ / block_size)` in the uniform-length
  // MVP); for a shared pool it is the pool-wide live count.
  int64_t num_allocated_blocks() const noexcept {
    return pool_->num_allocated();
  }

  // Blocks this cache alone holds (its block-table footprint). Useful for
  // a scheduler reasoning about a single request's residency.
  int64_t num_owned_blocks() const noexcept;

  // Wave 11 (B-032+): request `b`'s logical→physical block mapping (size =
  // number of logical blocks allocated so far). Lets a batched paged
  // decode-attention path assemble the `[A, max_logical]` block-table
  // tensor the fused kernel indexes, reading K/V in place from `pool()`
  // instead of gathering each request's prefix.
  const std::vector<int32_t>& block_table(int64_t b) const {
    TESSERACT_CHECK(b >= 0 && b < batch_,
                    "PagedKVCache::block_table: batch index {} out of range "
                    "[0, {})", b, batch_);
    return block_table_[static_cast<std::size_t>(b)];
  }

 private:
  // Ensure `block_table_[b]` has a physical block for logical index
  // `logical`, allocating from the pool as needed. Returns the physical
  // block id.
  int32_t ensure_block(int64_t b, int64_t logical);

  // Shared gather/scatter worker for keys_view / values_view.
  Tensor gather(const Tensor& pool) const;

  std::shared_ptr<PagedKVPool> pool_;  // shared (or private) block storage
  int64_t batch_;
  int64_t max_len_;
  int64_t current_len_{0};

  // block_table_[b][logical] = physical block id for request b's
  // logical block `logical`.
  std::vector<std::vector<int32_t>> block_table_;

  // CUDA-only: device-resident copy of the flattened block table
  // `[batch, max_logical]` (Int32), uploaded from `block_table_` on each
  // gather so the single-launch gather kernel can index the pool without
  // a host round-trip per element. Unallocated on CPU caches (the CPU
  // gather walks `block_table_` directly).
  int64_t max_logical_{0};
  // `mutable`: refreshed inside the const `gather()` (it's a scratch
  // staging buffer, not part of the cache's logical state).
  mutable Tensor block_table_dev_;
};

}  // namespace tesseract::nn

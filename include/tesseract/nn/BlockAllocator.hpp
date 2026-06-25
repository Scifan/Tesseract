#pragma once

// Wave 4.5 (B-019b): physical-block free-list allocator for PagedKVCache.
//
// vLLM-style paged KV storage carves a fixed-size physical pool of
// `num_blocks` blocks (each block holds `block_size` tokens worth of
// K/V for every head) and hands them out on demand. A request that
// has only emitted `S` tokens occupies `ceil(S / block_size)` physical
// blocks rather than the `max_len`-sized contiguous slab the Wave 2.1
// cache always pre-allocated. The allocator is the bookkeeping core of
// that scheme: it owns nothing but integer block ids and a free-list.
//
// It is deliberately device- and dtype-agnostic — the physical pool
// tensor lives in `PagedKVCache`; the allocator only decides *which*
// block index a writer may use next. That keeps it trivially unit-
// testable on its own (no CUDA, no Tensor) and lets a future
// continuous-batching scheduler reuse the same allocator across
// several caches sharing one global pool.
//
// Allocation policy: LIFO free-list (a stack). Freed blocks are pushed
// back and reused most-recently-first, which keeps a hot working set
// resident in the same physical blocks across `reset()` cycles — good
// for the "same model, many short prompts" serving pattern. The policy
// is an implementation detail; callers must not depend on which
// concrete id `allocate()` returns, only that it is free and distinct
// from every other currently-allocated id.

#include <cstdint>
#include <vector>

namespace tesseract::nn {

class BlockAllocator {
 public:
  // Construct an allocator owning `num_blocks` physical blocks
  // (ids `0 .. num_blocks-1`), all initially free. `num_blocks`
  // must be > 0.
  explicit BlockAllocator(int32_t num_blocks);

  // Hand out a free physical block id. Throws if the pool is
  // exhausted — callers (PagedKVCache::append) translate that into a
  // clear "KV cache out of blocks, raise num_blocks or evict" error.
  int32_t allocate();

  // Return a previously-allocated block id to the free-list. Throws
  // if `id` is out of range or is already free (double-free is a
  // bookkeeping bug we want to surface loudly, not paper over).
  void free(int32_t id);

  // Return every currently-allocated block to the free-list in one
  // shot. Used by PagedKVCache::reset() to recycle a finished
  // request's blocks without walking its block table.
  void free_all() noexcept;

  int32_t num_blocks()    const noexcept { return num_blocks_; }
  int32_t num_free()      const noexcept {
    return static_cast<int32_t>(free_list_.size());
  }
  int32_t num_allocated() const noexcept {
    return num_blocks_ - static_cast<int32_t>(free_list_.size());
  }

 private:
  int32_t              num_blocks_;
  std::vector<int32_t> free_list_;   // stack of free ids
  std::vector<bool>    allocated_;   // allocated_[id] == true ⇒ in use
};

}  // namespace tesseract::nn

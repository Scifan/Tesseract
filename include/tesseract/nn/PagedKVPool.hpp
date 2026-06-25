#pragma once

// Wave 7 (B-029): shared physical KV block pool.
//
// Wave 4.5 baked the physical block storage (the `[num_blocks, H,
// block_size, D_head]` K/V tensors) and the `BlockAllocator` *inside*
// each `PagedKVCache`. That is correct for a single request but it
// prevents the one thing continuous batching exists for: **letting many
// requests share one pool of blocks** so a finished request's blocks are
// immediately reusable by another. With per-cache pools the resident
// memory is still `Σ_r reserved_r`; with a shared pool it is bounded by
// the live token count across *all* requests at once.
//
// `PagedKVPool` factors that storage + allocator out into a standalone,
// reference-counted object (one per transformer layer). Many
// `PagedKVCache` handles — one per (request, layer) — point at the same
// pool, each keeping only its own block table + length. The pool owns:
//   * the two physical pool tensors (K and V),
//   * the `BlockAllocator` that hands out / reclaims block ids,
//   * the geometry every sharing cache must agree on (heads, head_dim,
//     block_size, dtype, device).
//
// The allocator's own header already anticipated this: "lets a future
// continuous-batching scheduler reuse the same allocator across several
// caches sharing one global pool."

#include <cstdint>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/BlockAllocator.hpp"

namespace tesseract::nn {

class PagedKVPool {
 public:
  // Allocate the physical pool: `num_blocks` blocks of `block_size`
  // tokens each, holding K and V for every one of `num_heads` heads with
  // `head_dim` channels. Both pool tensors are zero-initialized so the
  // never-written tail of a partially-filled block reads as a
  // deterministic 0 (matches KVCache / PagedKVCache safety).
  PagedKVPool(int64_t num_heads, int64_t head_dim, int64_t block_size,
              int64_t num_blocks, DType dtype = DType::Float32,
              Device device = cpu_device());

  // Block bookkeeping — thin pass-throughs to the owned allocator.
  int32_t allocate() { return allocator_.allocate(); }
  void    free(int32_t id) { allocator_.free(id); }

  int64_t num_heads()   const noexcept { return num_heads_; }
  int64_t head_dim()    const noexcept { return head_dim_; }
  int64_t block_size()  const noexcept { return block_size_; }
  int64_t num_blocks()  const noexcept { return allocator_.num_blocks(); }
  int64_t num_free()    const noexcept { return allocator_.num_free(); }
  int64_t num_allocated() const noexcept { return allocator_.num_allocated(); }
  DType   dtype()       const noexcept { return dtype_; }
  Device  device()      const noexcept { return device_; }

  // Physical storage. Layout is row-major `[num_blocks, H, block_size,
  // D_head]`; block `p`, head `h`, slot `s`, channel `c` lives at element
  // offset `((p*H + h)*block_size + s)*D_head + c`.
  const Tensor& keys()   const noexcept { return keys_pool_; }
  const Tensor& values() const noexcept { return values_pool_; }
  Tensor&       keys()         noexcept { return keys_pool_; }
  Tensor&       values()       noexcept { return values_pool_; }

 private:
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t block_size_;
  DType   dtype_;
  Device  device_;

  BlockAllocator allocator_;
  Tensor keys_pool_;
  Tensor values_pool_;
};

}  // namespace tesseract::nn

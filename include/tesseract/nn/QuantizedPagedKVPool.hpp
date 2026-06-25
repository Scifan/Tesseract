#pragma once

// Wave 13 (B-032+++): shared physical KV block pool, INT8 storage.
//
// The INT8 sibling of `PagedKVPool` (Wave 7). It carves the same fixed
// `num_blocks × block_size` physical block grid, but each K/V slot holds an
// **INT8** payload `[num_blocks, H, block_size, D_head]` plus one FP32
// **scale** per `(block, head, slot)` `[num_blocks, H, block_size]` — the
// Wave-9 per-token, per-head symmetric scheme (one scale per `D_head`
// vector). That is exactly the layout the Wave-12 fused op
// `nn::paged_decode_attention_int8` reads, so an INT8 paged pool feeds
// attention with no FP-prefix transient: combining paging's
// no-padding-waste residency with quant's ~4× (vs FP32) / ~2× (vs FP16)
// per-token KV shrink.
//
// As with `PagedKVPool`, one pool per transformer layer is shared by many
// `QuantizedPagedKVCache` handles (one per request·layer); each keeps only
// its own block table + length, and a single owned `BlockAllocator` hands
// out / reclaims block ids across all of them. The INT8 payload tensors and
// their parallel FP32 scale tensors share the *same* block ids (the
// allocator is the single source of truth) — index `(p, h, s)` of
// `keys()`/`values()` pairs with `(p, h, s)` of `key_scale()`/
// `value_scale()`.

#include <cstdint>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/BlockAllocator.hpp"

namespace tesseract::nn {

class QuantizedPagedKVPool {
 public:
  // Allocate the physical pool: `num_blocks` blocks of `block_size` tokens,
  // each holding INT8 K and V for `num_heads` heads × `head_dim` channels,
  // plus one FP32 scale per (block, head, slot). All four tensors are
  // zero-initialized so a never-written slot reads as a deterministic 0
  // (quantized 0 dequantizes to 0 for any scale; matches PagedKVPool
  // safety). `dtype` is the *floating* dtype attention produces/consumes
  // (Float32 / Float16 / BFloat16) — used for append validation and the
  // dequantized `keys_view`; storage is always INT8 + FP32 regardless.
  QuantizedPagedKVPool(int64_t num_heads, int64_t head_dim, int64_t block_size,
                       int64_t num_blocks, DType dtype = DType::Float32,
                       Device device = cpu_device());

  // Block bookkeeping — thin pass-throughs to the owned allocator. INT8
  // payload and FP32 scale share the returned id.
  int32_t allocate() { return allocator_.allocate(); }
  void    free(int32_t id) { allocator_.free(id); }

  int64_t num_heads()     const noexcept { return num_heads_; }
  int64_t head_dim()      const noexcept { return head_dim_; }
  int64_t block_size()    const noexcept { return block_size_; }
  int64_t num_blocks()    const noexcept { return allocator_.num_blocks(); }
  int64_t num_free()      const noexcept { return allocator_.num_free(); }
  int64_t num_allocated() const noexcept { return allocator_.num_allocated(); }
  DType   dtype()         const noexcept { return dtype_; }
  Device  device()        const noexcept { return device_; }

  // INT8 payload pools. Layout `[num_blocks, H, block_size, D_head]`,
  // row-major: block `p`, head `h`, slot `s`, channel `c` at element offset
  // `((p*H + h)*block_size + s)*D_head + c`.
  const Tensor& keys()   const noexcept { return keys_q_; }
  const Tensor& values() const noexcept { return values_q_; }
  Tensor&       keys()         noexcept { return keys_q_; }
  Tensor&       values()       noexcept { return values_q_; }

  // FP32 scale pools. Layout `[num_blocks, H, block_size]`, row-major: one
  // scale per `(p, h, s)` — the scale for the `D_head` vector at the same
  // index in `keys()` / `values()`.
  const Tensor& key_scale()   const noexcept { return key_scale_; }
  const Tensor& value_scale() const noexcept { return value_scale_; }
  Tensor&       key_scale()         noexcept { return key_scale_; }
  Tensor&       value_scale()       noexcept { return value_scale_; }

 private:
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t block_size_;
  DType   dtype_;  // floating dtype attention uses; storage is INT8 + FP32
  Device  device_;

  BlockAllocator allocator_;
  Tensor keys_q_;       // [num_blocks, H, block_size, D_head] Int8
  Tensor values_q_;     // same
  Tensor key_scale_;    // [num_blocks, H, block_size] Float32
  Tensor value_scale_;  // same
};

}  // namespace tesseract::nn

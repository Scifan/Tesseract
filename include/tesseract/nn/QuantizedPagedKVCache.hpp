#pragma once

// Wave 13 (B-032+++): INT8-quantized paged KV cache.
//
// The cache that finally unifies the three KV fast paths the framework
// built separately: GQA (Wave 8), KV-quant (Wave 9), and paged attention
// (Waves 7/11/12). It is the INT8 sibling of `PagedKVCache` (Wave 4.5/7):
// per-(request, layer) handles draw blocks from a shared
// `QuantizedPagedKVPool`, keeping only a block table + length — but storage
// is INT8 + FP32 scale instead of full-precision floats.
//
// Drop-in `KVCacheBase`:
//   * `append([B,H,Sn,D] FP)` quantizes per-(token, head) to INT8 + FP32
//     scale (`quant::quantize_kv_per_token`) and scatters the INT8 payload
//     + scale into on-demand pool blocks.
//   * `keys_view()` / `values_view()` gather the scattered INT8 + scale
//     prefix and dequantize to a fresh contiguous `[B,H,L,D]` FP tensor —
//     so `MultiHeadAttention::forward_step` consumes it through the exact
//     same calls as every other cache (the CPU / non-fused fallback).
//
// The win lands on the CUDA decode hot path: `MultiHeadAttention::
// forward_step_batched` reads the pool's INT8 payload + scale **directly**
// via `nn::paged_decode_attention_int8` (no gather, no FP-prefix
// transient), so resident KV is ~4× (vs FP32) / ~2× (vs FP16) smaller AND
// read at that lower bandwidth straight into attention.
//
// Numerics: per-token symmetric INT8 — lossy, so parity against an FP cache
// is bounded-error, not bit-identical (same contract as `QuantizedKVCache`).

#include <cstdint>
#include <memory>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/KVCacheBase.hpp"
#include "tesseract/nn/QuantizedPagedKVPool.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

class QuantizedPagedKVCache : public KVCacheBase {
 public:
  // Allocate a quantized paged cache backed by its OWN private INT8 pool.
  QuantizedPagedKVCache(int64_t batch, int64_t num_heads, int64_t head_dim,
                        int64_t max_len, int64_t block_size, int64_t num_blocks,
                        DType dtype = DType::Float32,
                        Device device = cpu_device());

  // Allocate a cache that draws blocks from an EXISTING shared
  // `QuantizedPagedKVPool` (one pool per transformer layer, shared across a
  // continuous batch). Geometry is inherited from the pool; the cache keeps
  // only its block table + length. `reset()` returns this cache's blocks to
  // the shared pool without touching other requests'.
  QuantizedPagedKVCache(std::shared_ptr<QuantizedPagedKVPool> pool,
                        int64_t batch, int64_t max_len);

  // Quantize `[B, H, S_new, D_head]` FP K/V to INT8 + per-(token, head)
  // FP32 scale and store at positions `[current_len, current_len + S_new)`,
  // allocating physical blocks on demand. Returns the new length. Inputs
  // must be contiguous, on the pool device, in the pool's float dtype.
  int64_t append(const Tensor& k_new, const Tensor& v_new) override;

  // Gather + dequantize the valid `[B, H, current_len, D_head]` prefix into
  // a fresh contiguous FP tensor (in the pool's float dtype). Materialized
  // (copy), not a view — the CPU / non-fused fallback path.
  Tensor keys_view() const override;
  Tensor values_view() const override;

  // Recycle every block back to the (possibly shared) allocator and rewind
  // to empty. Storage (the pool) is retained for the next prompt.
  void reset();

  // Capture-only position override (graph-rewind contract): the blocks
  // backing `[0, len)` must already be allocated; throws otherwise.
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

  // The shared (or private) INT8 pool backing this cache.
  const std::shared_ptr<QuantizedPagedKVPool>& pool() const noexcept {
    return pool_;
  }

  int64_t num_allocated_blocks() const noexcept { return pool_->num_allocated(); }
  int64_t num_owned_blocks() const noexcept;

  // Request `b`'s logical→physical block mapping — lets the batched fused
  // INT8 path assemble the `[A, max_logical]` block-table tensor that
  // `paged_decode_attention_int8` indexes.
  const std::vector<int32_t>& block_table(int64_t b) const {
    TESSERACT_CHECK(b >= 0 && b < batch_,
                    "QuantizedPagedKVCache::block_table: batch index {} out of "
                    "range [0, {})", b, batch_);
    return block_table_[static_cast<std::size_t>(b)];
  }

 private:
  int32_t ensure_block(int64_t b, int64_t logical);

  // Scatter a contiguous source `[B, H, S_new, inner]` into a pool tensor
  // `[num_blocks, H, block_size, inner]` using the (already-allocated)
  // block table for sequence positions `[current_len, current_len + Sn)`.
  void scatter(void* pool_base, const void* src_base, std::size_t elem,
               int64_t inner, int64_t B, int64_t H, int64_t Sn) const;

  // Gather a pool tensor `[num_blocks, H, block_size, inner]` into a fresh
  // contiguous `[B, H, current_len, inner]` tensor of `out_dtype`.
  Tensor gather(const Tensor& pool, int64_t inner, DType out_dtype) const;

  // keys_view / values_view worker: gather INT8 payload + scale, dequantize.
  Tensor dequant_view(const Tensor& payload_pool, const Tensor& scale_pool) const;

  std::shared_ptr<QuantizedPagedKVPool> pool_;
  int64_t batch_;
  int64_t max_len_;
  int64_t current_len_{0};

  // block_table_[b][logical] = physical block id for request b.
  std::vector<std::vector<int32_t>> block_table_;
};

}  // namespace tesseract::nn

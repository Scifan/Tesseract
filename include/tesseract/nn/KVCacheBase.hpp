#pragma once

// Wave 4.5 (B-019b): abstract KV-cache interface.
//
// `MultiHeadAttention::forward_step` consumes a KV cache through a small,
// fixed set of calls — `append`, `keys_view`, `values_view`, plus the
// shape accessors. Wave 2.1 shipped exactly one implementation
// (`KVCache`, contiguous slabs). Wave 4.5 adds `PagedKVCache` (vLLM-style
// paged storage). Rather than overload `forward_step` per concrete cache
// or template it, both caches implement this interface and the decode
// path takes a `KVCacheBase&` — so swapping contiguous ↔ paged storage
// is a one-line change at the call site with zero attention-code churn.
//
// The interface is intentionally minimal: only what the decode path
// actually touches. Capture-specific knobs (`set_current_len`),
// construction, and storage-introspection accessors stay on the concrete
// classes.

#include <cstdint>

#include "tesseract/core/Tensor.hpp"

namespace tesseract::nn {

class KVCacheBase {
 public:
  virtual ~KVCacheBase() = default;

  // Copy `[B, H, S_new, D_head]` contiguous K/V into seq positions
  // `[current_len, current_len + S_new)` and advance the length.
  // Returns the updated current length.
  virtual int64_t append(const Tensor& k_new, const Tensor& v_new) = 0;

  // Read-only `[B, H, current_len, D_head]` prefix of the cached K / V.
  // May be a zero-copy view (KVCache) or a materialized gather
  // (PagedKVCache) — callers must treat it as read-only either way.
  virtual Tensor keys_view() const = 0;
  virtual Tensor values_view() const = 0;

  virtual int64_t batch()       const noexcept = 0;
  virtual int64_t num_heads()   const noexcept = 0;
  virtual int64_t head_dim()    const noexcept = 0;
  virtual int64_t current_len() const noexcept = 0;
};

}  // namespace tesseract::nn

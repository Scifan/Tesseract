#pragma once

// Wave 9 (B-031): INT8-quantized KV cache for long-context decode.
//
// A drop-in `KVCacheBase` that stores K/V as **INT8** (per-token,
// per-head symmetric) instead of full-precision floats — a ~4× (vs FP32)
// or ~2× (vs FP16) cut in *persistent* KV memory, which is what bounds
// concurrent requests and context length on the Wave-7 pools. The cache
// quantizes the projected K/V on `append` and dequantizes the prefix on
// `keys_view()` / `values_view()`, so `MultiHeadAttention::forward_step`
// consumes it through the exact same interface as the FP `KVCache`
// (Wave 2.1) and `PagedKVCache` (Wave 4.5) — no attention-code change.
//
// Memory model:
//   * Persistent storage is INT8 K/V slabs `[B, H, max_len, D_head]`
//     plus FP32 per-(b, h, token) scale slabs `[B, H, max_len]`. That
//     is the long-lived footprint across all layers and the full
//     `max_len` reservation — where the win lands.
//   * `keys_view()` dequantizes the *current prefix* into a fresh FP
//     `[B, H, L, D_head]` tensor each call. That transient is one
//     layer's prefix (FP-sized), produced and consumed within the step.
//     Fusing dequant into the attention matmul (a quantized-attention
//     kernel) is the deferred throughput/footprint follow-up (B-031+).
//
// Numerics: per-token symmetric INT8 (`scale = absmax/127`, banker's
// round, clamp [-127, 127]) via `quant::quantize_kv_per_token` /
// `dequantize_kv_per_token` — identical math on CPU and CUDA. Output of
// `keys_view()` is in the cache's float `dtype`. This is lossy: scheduler
// / generate parity against an FP cache is *bounded-error*, not
// bit-identical (see `tests/nn/test_quant_kv.cpp`).
//
// Lifetime / autograd: same contract as `KVCache` — owns its storage,
// `reset()` rewinds without reallocating, every stored tensor is
// detached (pure inference structure).

#include <cstdint>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/KVCacheBase.hpp"

namespace tesseract::nn {

class QuantizedKVCache : public KVCacheBase {
 public:
  // `dtype` is the *floating* dtype the attention block produces and
  // consumes (Float32 / Float16 / BFloat16); storage is always INT8 +
  // FP32 scales regardless. `append` validates incoming K/V against it.
  QuantizedKVCache(int64_t batch, int64_t num_heads, int64_t head_dim,
                   int64_t max_len, DType dtype = DType::Float32,
                   Device device = cpu_device());

  // Quantize `[B, H, S_new, D_head]` FP K/V and store at positions
  // `[current_len_, current_len_ + S_new)`; advance and return the new
  // length. Inputs must be contiguous, on the cache device, in the
  // cache dtype.
  int64_t append(const Tensor& k_new, const Tensor& v_new) override;

  // Dequantized `[B, H, current_len_, D_head]` prefix in the cache
  // dtype. **Materialized** (not a view) — see header note.
  Tensor keys_view() const override;
  Tensor values_view() const override;

  void reset() noexcept { current_len_ = 0; }
  void set_current_len(int64_t len);

  int64_t batch()       const noexcept override { return batch_; }
  int64_t num_heads()   const noexcept override { return num_heads_; }
  int64_t head_dim()    const noexcept override { return head_dim_; }
  int64_t max_len()     const noexcept { return max_len_; }
  int64_t current_len() const noexcept override { return current_len_; }
  DType   dtype()       const noexcept { return dtype_; }
  Device  device()      const noexcept { return device_; }

 private:
  Tensor dequant_prefix(const Tensor& q_slab, const Tensor& scale_slab) const;

  int64_t batch_;
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t max_len_;
  DType   dtype_;
  Device  device_;
  int64_t current_len_{0};
  Tensor  keys_q_;      // [B, H, max_len, D_head] Int8
  Tensor  values_q_;    // same
  Tensor  key_scale_;   // [B, H, max_len] Float32
  Tensor  value_scale_; // same
};

}  // namespace tesseract::nn

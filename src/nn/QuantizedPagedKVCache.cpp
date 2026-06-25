#include "tesseract/nn/QuantizedPagedKVCache.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Storage.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/quant/QuantizeKV.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

QuantizedPagedKVCache::QuantizedPagedKVCache(int64_t batch, int64_t num_heads,
                                             int64_t head_dim, int64_t max_len,
                                             int64_t block_size,
                                             int64_t num_blocks, DType dtype,
                                             Device device)
    : QuantizedPagedKVCache(
          std::make_shared<QuantizedPagedKVPool>(num_heads, head_dim, block_size,
                                                 num_blocks, dtype, device),
          batch, max_len) {}

QuantizedPagedKVCache::QuantizedPagedKVCache(
    std::shared_ptr<QuantizedPagedKVPool> pool, int64_t batch, int64_t max_len)
    : pool_(std::move(pool)),
      batch_(batch),
      max_len_(max_len),
      block_table_(static_cast<std::size_t>(batch)) {
  TESSERACT_CHECK(pool_ != nullptr, "QuantizedPagedKVCache: pool must be non-null");
  TESSERACT_CHECK(batch > 0 && max_len > 0,
                  "QuantizedPagedKVCache: dims must be positive "
                  "(batch={}, max_len={})", batch, max_len);
}

int32_t QuantizedPagedKVCache::ensure_block(int64_t b, int64_t logical) {
  auto& table = block_table_[static_cast<std::size_t>(b)];
  while (static_cast<int64_t>(table.size()) <= logical) {
    table.push_back(pool_->allocate());
  }
  return table[static_cast<std::size_t>(logical)];
}

int64_t QuantizedPagedKVCache::num_owned_blocks() const noexcept {
  int64_t n = 0;
  for (const auto& table : block_table_) n += static_cast<int64_t>(table.size());
  return n;
}

void QuantizedPagedKVCache::scatter(void* pool_base, const void* src_base,
                                    std::size_t elem, int64_t inner, int64_t B,
                                    int64_t H, int64_t Sn) const {
  const int64_t block_size = pool_->block_size();
  const Device  device     = pool_->device();
  const bool    on_cpu     = device.is_cpu();
  const int64_t pool_block_stride = H * block_size * inner;
  const int64_t pool_head_stride  = block_size * inner;

  auto* dst_base = static_cast<std::byte*>(pool_base);
  const auto* s_base = static_cast<const std::byte*>(src_base);

  for (int64_t b = 0; b < B; ++b) {
    const auto& table = block_table_[static_cast<std::size_t>(b)];
    int64_t i = 0;
    while (i < Sn) {
      const int64_t gpos    = current_len_ + i;
      const int64_t logical = gpos / block_size;
      const int64_t slot    = gpos % block_size;
      const int64_t run     = std::min(block_size - slot, Sn - i);
      const int32_t p       = table[static_cast<std::size_t>(logical)];
      const std::size_t run_bytes =
          static_cast<std::size_t>(run) * static_cast<std::size_t>(inner) * elem;

      for (int64_t h = 0; h < H; ++h) {
        const int64_t dst_off =
            p * pool_block_stride + h * pool_head_stride + slot * inner;
        const int64_t src_off = ((b * H + h) * Sn + i) * inner;
        std::byte* dst = dst_base + static_cast<std::size_t>(dst_off) * elem;
        const std::byte* src = s_base + static_cast<std::size_t>(src_off) * elem;
        if (on_cpu) {
          Storage::copy_device_bytes(dst, device, src, device, run_bytes);
        } else {
          const Stream& s = current_stream(device);
          Storage::copy_device_bytes_async(dst, device, src, device, run_bytes, s);
        }
      }
      i += run;
    }
  }
}

int64_t QuantizedPagedKVCache::append(const Tensor& k_new, const Tensor& v_new) {
  TESSERACT_CHECK(k_new.defined() && v_new.defined(),
                  "QuantizedPagedKVCache::append: k_new/v_new must be defined");
  TESSERACT_CHECK(k_new.rank() == 4 && v_new.rank() == 4,
                  "QuantizedPagedKVCache::append: expected rank-4 [B,H,S,Dh], "
                  "got k={} v={}", k_new.shape().to_string(),
                  v_new.shape().to_string());
  const DType   dtype_     = pool_->dtype();
  const Device  device_    = pool_->device();
  const int64_t num_heads_ = pool_->num_heads();
  const int64_t head_dim_  = pool_->head_dim();
  const int64_t block_size = pool_->block_size();

  TESSERACT_CHECK(k_new.dtype() == dtype_ && v_new.dtype() == dtype_,
                  "QuantizedPagedKVCache::append: dtype mismatch (cache={}, "
                  "k={}, v={})", dtype_name(dtype_), dtype_name(k_new.dtype()),
                  dtype_name(v_new.dtype()));
  TESSERACT_CHECK(k_new.device() == device_ && v_new.device() == device_,
                  "QuantizedPagedKVCache::append: device mismatch");

  const int64_t B  = k_new.shape()[0];
  const int64_t H  = k_new.shape()[1];
  const int64_t Sn = k_new.shape()[2];
  const int64_t D  = k_new.shape()[3];

  TESSERACT_CHECK(B == batch_ && H == num_heads_ && D == head_dim_,
                  "QuantizedPagedKVCache::append: k_new shape {} incompatible "
                  "with cache (B={}, H={}, Dh={})", k_new.shape().to_string(),
                  batch_, num_heads_, head_dim_);
  TESSERACT_CHECK(v_new.shape()[0] == B && v_new.shape()[1] == H &&
                  v_new.shape()[2] == Sn && v_new.shape()[3] == D,
                  "QuantizedPagedKVCache::append: v_new shape {} disagrees with "
                  "k_new {}", v_new.shape().to_string(), k_new.shape().to_string());
  TESSERACT_CHECK(current_len_ + Sn <= max_len_,
                  "QuantizedPagedKVCache::append: would exceed max_len "
                  "({} + {} > {})", current_len_, Sn, max_len_);
  TESSERACT_CHECK(k_new.is_contiguous() && v_new.is_contiguous(),
                  "QuantizedPagedKVCache::append: k_new/v_new must be contiguous");

  if (Sn == 0) return current_len_;

  // Quantize on-device: q [B,H,Sn,D] Int8, scale [B,H,Sn] Float32 — the
  // Wave-9 per-token, per-head symmetric scheme (one scale per D-vector).
  auto [kq, ksc] = quant::quantize_kv_per_token(k_new);
  auto [vq, vsc] = quant::quantize_kv_per_token(v_new);

  // Allocate every physical block the new range [current_len, +Sn) touches.
  const int64_t first_logical = current_len_ / block_size;
  const int64_t last_logical  = (current_len_ + Sn - 1) / block_size;
  for (int64_t b = 0; b < B; ++b)
    for (int64_t logical = first_logical; logical <= last_logical; ++logical)
      ensure_block(b, logical);

  const std::size_t i8 = dtype_size(DType::Int8);
  const std::size_t f4 = dtype_size(DType::Float32);
  scatter(pool_->keys().raw_data(),   kq.raw_data(),  i8, D, B, H, Sn);
  scatter(pool_->key_scale().raw_data(),   ksc.raw_data(), f4, 1, B, H, Sn);
  scatter(pool_->values().raw_data(), vq.raw_data(),  i8, D, B, H, Sn);
  scatter(pool_->value_scale().raw_data(), vsc.raw_data(), f4, 1, B, H, Sn);

  current_len_ += Sn;
  return current_len_;
}

Tensor QuantizedPagedKVCache::gather(const Tensor& pool, int64_t inner,
                                     DType out_dtype) const {
  const int64_t B = batch_;
  const int64_t H = pool_->num_heads();
  const int64_t L = current_len_;
  const int64_t block_size = pool_->block_size();
  const Device  device     = pool_->device();

  Tensor out = Tensor::empty({B, H, L, inner}, out_dtype, device);
  if (L == 0) return out;

  const int64_t num_logical = (L + block_size - 1) / block_size;
  const std::size_t elem = dtype_size(out_dtype);
  const bool on_cpu = device.is_cpu();
  const int64_t pool_block_stride = H * block_size * inner;
  const int64_t pool_head_stride  = block_size * inner;

  auto* out_base = static_cast<std::byte*>(out.raw_data());
  const auto* pool_base = static_cast<const std::byte*>(pool.raw_data());

  // Per-head byte copies (CPU memcpy / CUDA async). The launch-collapsed
  // gather kernel only handles 2/4/8-byte elements, so the INT8 payload
  // (1 byte) can't use it — and gather is the *fallback* path anyway (the
  // fused INT8 decode reads the pool in place, never gathering), so
  // correctness over peak speed is the right call here.
  for (int64_t b = 0; b < B; ++b) {
    const auto& table = block_table_[static_cast<std::size_t>(b)];
    for (int64_t logical = 0; logical < num_logical; ++logical) {
      const int64_t seq_start = logical * block_size;
      const int64_t run = std::min(block_size, L - seq_start);
      const int32_t p = table[static_cast<std::size_t>(logical)];
      const std::size_t run_bytes =
          static_cast<std::size_t>(run) * static_cast<std::size_t>(inner) * elem;
      for (int64_t h = 0; h < H; ++h) {
        const int64_t src_off = p * pool_block_stride + h * pool_head_stride;
        const int64_t dst_off = ((b * H + h) * L + seq_start) * inner;
        std::byte* dst = out_base + static_cast<std::size_t>(dst_off) * elem;
        const std::byte* src = pool_base + static_cast<std::size_t>(src_off) * elem;
        if (on_cpu) {
          Storage::copy_device_bytes(dst, device, src, device, run_bytes);
        } else {
          const Stream& s = current_stream(device);
          Storage::copy_device_bytes_async(dst, device, src, device, run_bytes, s);
        }
      }
    }
  }
  return out;
}

Tensor QuantizedPagedKVCache::dequant_view(const Tensor& payload_pool,
                                           const Tensor& scale_pool) const {
  const int64_t B = batch_;
  const int64_t H = pool_->num_heads();
  const int64_t L = current_len_;
  const int64_t D = pool_->head_dim();
  if (L == 0)
    return Tensor::empty({B, H, L, D}, pool_->dtype(), pool_->device());

  Tensor q  = gather(payload_pool, D, DType::Int8);    // [B, H, L, D] Int8
  Tensor sc = gather(scale_pool, 1, DType::Float32);   // [B, H, L, 1] Float32
  Tensor sc3 = ops::reshape(sc, Shape({B, H, L}));     // [B, H, L]
  return quant::dequantize_kv_per_token(q, sc3, pool_->dtype());
}

Tensor QuantizedPagedKVCache::keys_view() const {
  return dequant_view(pool_->keys(), pool_->key_scale());
}
Tensor QuantizedPagedKVCache::values_view() const {
  return dequant_view(pool_->values(), pool_->value_scale());
}

void QuantizedPagedKVCache::reset() {
  for (auto& table : block_table_) {
    for (int32_t id : table) pool_->free(id);
    table.clear();
  }
  current_len_ = 0;
}

void QuantizedPagedKVCache::set_current_len(int64_t len) {
  TESSERACT_CHECK(len >= 0 && len <= max_len_,
                  "QuantizedPagedKVCache::set_current_len: len {} out of range "
                  "[0, {}]", len, max_len_);
  const int64_t block_size = pool_->block_size();
  const int64_t need = (len + block_size - 1) / block_size;
  for (int64_t b = 0; b < batch_; ++b) {
    TESSERACT_CHECK(
        static_cast<int64_t>(block_table_[static_cast<std::size_t>(b)].size()) >= need,
        "QuantizedPagedKVCache::set_current_len: request {} has {} blocks but "
        "len {} needs {} — append before rewinding", b,
        block_table_[static_cast<std::size_t>(b)].size(), len, need);
  }
  current_len_ = len;
}

}  // namespace tesseract::nn

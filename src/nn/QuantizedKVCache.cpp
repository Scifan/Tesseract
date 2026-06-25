#include "tesseract/nn/QuantizedKVCache.hpp"

#include <cstddef>
#include <utility>

#include "tesseract/core/Storage.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/quant/QuantizeKV.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

namespace {

// Copy a contiguous `[B, H, Sn, inner]` source into a `[B, H, max_len,
// inner]` destination slab at seq-offset `pos`, per-(b, h) sub-block.
// Mirrors `KVCache::append`'s strided insert (the slab seq stride is
// max_len·inner, the source seq stride is Sn·inner, so the insert is not
// a single contiguous block). `inner == head_dim` for the INT8 K/V
// slabs, `inner == 1` for the FP32 scale slabs.
void copy_into_slab(Tensor& dst, const Tensor& src, int64_t pos,
                    int64_t B, int64_t H, int64_t Sn, int64_t inner,
                    int64_t max_len, std::size_t elem, Device device) {
  if (Sn == 0) return;
  const std::size_t block_bytes =
      static_cast<std::size_t>(Sn) * static_cast<std::size_t>(inner) * elem;
  auto* dst_base = static_cast<std::byte*>(dst.raw_data());
  const auto* src_base = static_cast<const std::byte*>(src.raw_data());
  const int64_t dst_h_stride   = max_len * inner;
  const int64_t dst_b_stride   = H * dst_h_stride;
  const int64_t dst_seq_stride = inner;
  const int64_t src_h_stride   = Sn * inner;

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      const int64_t dst_off =
          b * dst_b_stride + h * dst_h_stride + pos * dst_seq_stride;
      const int64_t src_off = (b * H + h) * src_h_stride;
      std::byte* d = dst_base + static_cast<std::size_t>(dst_off) * elem;
      const std::byte* s = src_base + static_cast<std::size_t>(src_off) * elem;
      if (device.is_cpu()) {
        Storage::copy_device_bytes(d, device, s, device, block_bytes);
      } else {
        const Stream& st = current_stream(device);
        Storage::copy_device_bytes_async(d, device, s, device, block_bytes, st);
      }
    }
  }
}

}  // namespace

QuantizedKVCache::QuantizedKVCache(int64_t batch, int64_t num_heads,
                                   int64_t head_dim, int64_t max_len,
                                   DType dtype, Device device)
    : batch_(batch),
      num_heads_(num_heads),
      head_dim_(head_dim),
      max_len_(max_len),
      dtype_(dtype),
      device_(device) {
  TESSERACT_CHECK(batch > 0 && num_heads > 0 && head_dim > 0 && max_len > 0,
                  "QuantizedKVCache: dims must be positive "
                  "(batch={}, num_heads={}, head_dim={}, max_len={})",
                  batch, num_heads, head_dim, max_len);
  TESSERACT_CHECK(dtype == DType::Float32 || dtype == DType::Float16 ||
                  dtype == DType::BFloat16,
                  "QuantizedKVCache: dtype must be Float32/Float16/BFloat16, "
                  "got {}", dtype_name(dtype));

  keys_q_     = Tensor::zeros({batch, num_heads, max_len, head_dim},
                              DType::Int8, device);
  values_q_   = Tensor::zeros({batch, num_heads, max_len, head_dim},
                              DType::Int8, device);
  key_scale_  = Tensor::zeros({batch, num_heads, max_len}, DType::Float32, device);
  value_scale_= Tensor::zeros({batch, num_heads, max_len}, DType::Float32, device);
}

void QuantizedKVCache::set_current_len(int64_t len) {
  TESSERACT_CHECK(len >= 0 && len <= max_len_,
                  "QuantizedKVCache::set_current_len: len {} out of range "
                  "[0, {}]", len, max_len_);
  current_len_ = len;
}

int64_t QuantizedKVCache::append(const Tensor& k_new, const Tensor& v_new) {
  TESSERACT_CHECK(k_new.defined() && v_new.defined(),
                  "QuantizedKVCache::append: k_new and v_new must be defined");
  TESSERACT_CHECK(k_new.rank() == 4 && v_new.rank() == 4,
                  "QuantizedKVCache::append: expected rank-4 [B, H, S, Dh] "
                  "inputs, got k={} v={}",
                  k_new.shape().to_string(), v_new.shape().to_string());
  TESSERACT_CHECK(k_new.dtype() == dtype_ && v_new.dtype() == dtype_,
                  "QuantizedKVCache::append: dtype mismatch (cache={}, k={}, "
                  "v={})", dtype_name(dtype_), dtype_name(k_new.dtype()),
                  dtype_name(v_new.dtype()));
  TESSERACT_CHECK(k_new.device() == device_ && v_new.device() == device_,
                  "QuantizedKVCache::append: device mismatch (cache={}, k={}, "
                  "v={})", device_.to_string(), k_new.device().to_string(),
                  v_new.device().to_string());

  const int64_t B  = k_new.shape()[0];
  const int64_t H  = k_new.shape()[1];
  const int64_t Sn = k_new.shape()[2];
  const int64_t D  = k_new.shape()[3];

  TESSERACT_CHECK(B == batch_ && H == num_heads_ && D == head_dim_,
                  "QuantizedKVCache::append: k_new shape {} incompatible with "
                  "cache (B={}, H={}, Dh={})", k_new.shape().to_string(),
                  batch_, num_heads_, head_dim_);
  TESSERACT_CHECK(v_new.shape()[0] == B && v_new.shape()[1] == H &&
                  v_new.shape()[2] == Sn && v_new.shape()[3] == D,
                  "QuantizedKVCache::append: v_new shape {} disagrees with "
                  "k_new {}", v_new.shape().to_string(), k_new.shape().to_string());
  TESSERACT_CHECK(current_len_ + Sn <= max_len_,
                  "QuantizedKVCache::append: would exceed max_len ({} + {} > {})",
                  current_len_, Sn, max_len_);
  TESSERACT_CHECK(k_new.is_contiguous() && v_new.is_contiguous(),
                  "QuantizedKVCache::append: k_new / v_new must be contiguous");

  if (Sn == 0) return current_len_;

  // Quantize the new slab (per-token INT8) then byte-copy the INT8
  // payload + FP32 scales into their slabs at the seq offset.
  auto [qk, sk] = quant::quantize_kv_per_token(k_new);  // [B,H,Sn,Dh] i8, [B,H,Sn] f32
  auto [qv, sv] = quant::quantize_kv_per_token(v_new);

  copy_into_slab(keys_q_,   qk, current_len_, B, H, Sn, D, max_len_,
                 /*elem=*/1, device_);
  copy_into_slab(values_q_, qv, current_len_, B, H, Sn, D, max_len_,
                 /*elem=*/1, device_);
  copy_into_slab(key_scale_,   sk, current_len_, B, H, Sn, /*inner=*/1,
                 max_len_, /*elem=*/sizeof(float), device_);
  copy_into_slab(value_scale_, sv, current_len_, B, H, Sn, /*inner=*/1,
                 max_len_, /*elem=*/sizeof(float), device_);

  current_len_ += Sn;
  return current_len_;
}

Tensor QuantizedKVCache::dequant_prefix(const Tensor& q_slab,
                                        const Tensor& scale_slab) const {
  // Narrow to the valid prefix (strided views), realize contiguous, then
  // dequantize back to the float dtype.
  Tensor q_pref = q_slab.narrow(/*dim=*/2, /*start=*/0, /*len=*/current_len_)
                      .contiguous();
  Tensor s_pref = scale_slab.narrow(/*dim=*/2, /*start=*/0, /*len=*/current_len_)
                      .contiguous();
  return quant::dequantize_kv_per_token(q_pref, s_pref, dtype_);
}

Tensor QuantizedKVCache::keys_view() const {
  return dequant_prefix(keys_q_, key_scale_);
}

Tensor QuantizedKVCache::values_view() const {
  return dequant_prefix(values_q_, value_scale_);
}

}  // namespace tesseract::nn

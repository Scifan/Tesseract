#include "tesseract/nn/KVCache.hpp"

#include <cstddef>

#include "tesseract/core/Storage.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

void KVCache::set_current_len(int64_t len) {
  TESSERACT_CHECK(len >= 0 && len <= max_len_,
                  "KVCache::set_current_len: len {} out of range [0, {}]",
                  len, max_len_);
  current_len_ = len;
}

KVCache::KVCache(int64_t batch, int64_t num_heads, int64_t head_dim,
                 int64_t max_len, DType dtype, Device device)
    : batch_(batch),
      num_heads_(num_heads),
      head_dim_(head_dim),
      max_len_(max_len),
      dtype_(dtype),
      device_(device) {
  TESSERACT_CHECK(batch > 0 && num_heads > 0 && head_dim > 0 && max_len > 0,
                  "KVCache: dims must be positive "
                  "(batch={}, num_heads={}, head_dim={}, max_len={})",
                  batch, num_heads, head_dim, max_len);
  TESSERACT_CHECK(dtype_is_floating(dtype),
                  "KVCache: dtype must be floating-point, got {}",
                  dtype_name(dtype));

  // Pre-allocate both slabs up-front so `append()` is a pure byte
  // copy; zero-initialize so that any attention math over the
  // never-filled tail (shouldn't happen, but e.g. if a test
  // accidentally queries past current_len_) sees a deterministic 0
  // rather than uninitialized memory.
  keys_   = Tensor::zeros({batch, num_heads, max_len, head_dim}, dtype, device);
  values_ = Tensor::zeros({batch, num_heads, max_len, head_dim}, dtype, device);
}

int64_t KVCache::append(const Tensor& k_new, const Tensor& v_new) {
  TESSERACT_CHECK(k_new.defined() && v_new.defined(),
                  "KVCache::append: k_new and v_new must be defined");
  TESSERACT_CHECK(k_new.rank() == 4 && v_new.rank() == 4,
                  "KVCache::append: expected rank-4 [B, H, S, Dh] inputs, "
                  "got k={} v={}",
                  k_new.shape().to_string(), v_new.shape().to_string());
  TESSERACT_CHECK(k_new.dtype() == dtype_ && v_new.dtype() == dtype_,
                  "KVCache::append: dtype mismatch (cache={}, k={}, v={})",
                  dtype_name(dtype_), dtype_name(k_new.dtype()),
                  dtype_name(v_new.dtype()));
  TESSERACT_CHECK(k_new.device() == device_ && v_new.device() == device_,
                  "KVCache::append: device mismatch (cache={}, k={}, v={})",
                  device_.to_string(), k_new.device().to_string(),
                  v_new.device().to_string());

  const int64_t B  = k_new.shape()[0];
  const int64_t H  = k_new.shape()[1];
  const int64_t Sn = k_new.shape()[2];
  const int64_t D  = k_new.shape()[3];

  TESSERACT_CHECK(B == batch_ && H == num_heads_ && D == head_dim_,
                  "KVCache::append: k_new shape {} incompatible with cache "
                  "(B={}, H={}, Dh={})",
                  k_new.shape().to_string(), batch_, num_heads_, head_dim_);
  TESSERACT_CHECK(v_new.shape()[0] == B && v_new.shape()[1] == H &&
                  v_new.shape()[2] == Sn && v_new.shape()[3] == D,
                  "KVCache::append: v_new shape {} disagrees with k_new {}",
                  v_new.shape().to_string(), k_new.shape().to_string());
  TESSERACT_CHECK(current_len_ + Sn <= max_len_,
                  "KVCache::append: would exceed max_len ({} + {} > {})",
                  current_len_, Sn, max_len_);

  if (Sn == 0) return current_len_;

  // Both slabs are row-major contiguous `[B, H, max_len, Dh]`; an
  // `append` of `[B, H, Sn, Dh]` at seq-offset `current_len_` is
  // NOT a single contiguous block when Sn < max_len_ (each
  // (b, h, :, :) sub-slab is strided by `max_len_ * Dh` instead of
  // `Sn * Dh`). We handle that with a per-(b, h) byte copy — `B * H`
  // calls, each `Sn * Dh * itemsize` bytes. For Llama-7B this is
  // 32 heads × 1 batch = 32 copies per step, each 64 B for Sn=1 /
  // Dh=128 / fp16 → trivially cheap vs. the attention math.
  //
  // Contiguous-input fast path: when `Sn == max_len_` (prefill of
  // the whole cache at once) the per-(b,h) block is a full slab
  // and the strides collapse; the loop below still does the right
  // thing but degenerates to `B * H` adjacent copies. The kernel
  // path is unchanged.
  TESSERACT_CHECK(k_new.is_contiguous() && v_new.is_contiguous(),
                  "KVCache::append: k_new / v_new must be contiguous "
                  "(shape k={}, v={})",
                  k_new.shape().to_string(), v_new.shape().to_string());

  const std::size_t elem = dtype_size(dtype_);
  const std::size_t row_bytes = static_cast<std::size_t>(Sn) *
                                static_cast<std::size_t>(D) * elem;

  const int64_t cache_row_stride =
      keys_.strides()[1];  // [B, H, max_len, Dh] → stride over H dim
  const int64_t cache_seq_stride =
      keys_.strides()[2];  // stride over the seq dim (always == Dh when contig)

  // New-row strides: k_new is contiguous `[B, H, Sn, Dh]`; block
  // stride over H equals `Sn * Dh`.
  const int64_t new_h_stride = Sn * D;

  auto* keys_base = static_cast<std::byte*>(keys_.raw_data());
  auto* vals_base = static_cast<std::byte*>(values_.raw_data());
  const auto* k_new_base = static_cast<const std::byte*>(k_new.raw_data());
  const auto* v_new_base = static_cast<const std::byte*>(v_new.raw_data());

  // Destination base for (b=0, h=0) at the current write offset; the
  // per-(b,h) sub-block start is `dst_base + (b*H + h) * cache_row_stride`
  // because the slab is contiguous `[B, H, max_len, Dh]` so
  // `strides()[0] == H * cache_row_stride`. Source is the contiguous
  // `[B, H, Sn, Dh]` new slab with row stride `new_h_stride`.
  const int64_t dst_base_off = current_len_ * cache_seq_stride;
  std::byte* k_dst0 = keys_base + static_cast<std::size_t>(dst_base_off) * elem;
  std::byte* v_dst0 = vals_base + static_cast<std::size_t>(dst_base_off) * elem;

  // CPU path: plain host memcpy via `copy_device_bytes` (which detects
  // CPU↔CPU and falls through to `std::memcpy`). CUDA path: a single
  // strided `cudaMemcpy2DAsync` per slab covering all B*H rows on the
  // per-device current stream. `k_new` / `v_new` were produced by
  // kernels on that same stream, so stream ordering alone guarantees
  // those kernels finish before the copy reads their outputs.
  //
  // The async strided copy is the capture-safe option AND collapses what
  // used to be `2 * B * H` tiny per-(b,h) memcpy nodes into 2 nodes. The
  // old per-row loop made graph capture both slow (B*H*2 nodes) and
  // fragile under replay at large head counts; one strided node per slab
  // fixes both. Wave 4.3 (B-023b): `MHA::forward_step` can be wrapped in
  // `CudaGraph::capture(...)` and the append rides the graph cleanly.
  const std::size_t rows = static_cast<std::size_t>(B) * static_cast<std::size_t>(H);
  const std::size_t dpitch = static_cast<std::size_t>(cache_row_stride) * elem;
  const std::size_t spitch = static_cast<std::size_t>(new_h_stride) * elem;
  if (device_.is_cpu()) {
    for (std::size_t r = 0; r < rows; ++r) {
      std::byte* kd = k_dst0 + r * dpitch;
      std::byte* vd = v_dst0 + r * dpitch;
      const std::byte* ks = k_new_base + r * spitch;
      const std::byte* vs = v_new_base + r * spitch;
      Storage::copy_device_bytes(kd, device_, ks, device_, row_bytes);
      Storage::copy_device_bytes(vd, device_, vs, device_, row_bytes);
    }
  } else {
    const Stream& s = current_stream(device_);
    Storage::copy_device_bytes_2d_async(k_dst0, dpitch, k_new_base, spitch,
                                        row_bytes, rows, device_, s);
    Storage::copy_device_bytes_2d_async(v_dst0, dpitch, v_new_base, spitch,
                                        row_bytes, rows, device_, s);
  }

  current_len_ += Sn;
  return current_len_;
}

Tensor KVCache::keys_view() const {
  return keys_.narrow(/*dim=*/2, /*start=*/0, /*len=*/current_len_);
}

Tensor KVCache::values_view() const {
  return values_.narrow(/*dim=*/2, /*start=*/0, /*len=*/current_len_);
}

}  // namespace tesseract::nn

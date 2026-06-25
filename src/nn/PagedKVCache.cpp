#include "tesseract/nn/PagedKVCache.hpp"

#include <algorithm>
#include <cstddef>

#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Storage.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/PagedKV.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

PagedKVCache::PagedKVCache(int64_t batch, int64_t num_heads, int64_t head_dim,
                           int64_t max_len, int64_t block_size,
                           int64_t num_blocks, DType dtype, Device device)
    : PagedKVCache(std::make_shared<PagedKVPool>(num_heads, head_dim,
                                                 block_size, num_blocks, dtype,
                                                 device),
                   batch, max_len) {}

PagedKVCache::PagedKVCache(std::shared_ptr<PagedKVPool> pool, int64_t batch,
                           int64_t max_len)
    : pool_(std::move(pool)),
      batch_(batch),
      max_len_(max_len),
      block_table_(static_cast<std::size_t>(batch)) {
  TESSERACT_CHECK(pool_ != nullptr, "PagedKVCache: pool must be non-null");
  TESSERACT_CHECK(batch > 0 && max_len > 0,
                  "PagedKVCache: dims must be positive (batch={}, max_len={})",
                  batch, max_len);

  // On CUDA the gather is done by a single kernel launch that needs the
  // block table in device memory. Pre-allocate the `[batch, max_logical]`
  // Int32 staging tensor; `gather()` refreshes it from `block_table_`
  // before each launch. (CPU caches walk `block_table_` directly and
  // leave this undefined.)
  max_logical_ = (max_len + pool_->block_size() - 1) / pool_->block_size();
  if (!pool_->device().is_cpu()) {
    block_table_dev_ =
        Tensor::zeros({batch, max_logical_}, DType::Int32, pool_->device());
  }
}

int32_t PagedKVCache::ensure_block(int64_t b, int64_t logical) {
  auto& table = block_table_[static_cast<std::size_t>(b)];
  while (static_cast<int64_t>(table.size()) <= logical) {
    table.push_back(pool_->allocate());
  }
  return table[static_cast<std::size_t>(logical)];
}

int64_t PagedKVCache::num_owned_blocks() const noexcept {
  int64_t n = 0;
  for (const auto& table : block_table_) n += static_cast<int64_t>(table.size());
  return n;
}

int64_t PagedKVCache::append(const Tensor& k_new, const Tensor& v_new) {
  TESSERACT_CHECK(k_new.defined() && v_new.defined(),
                  "PagedKVCache::append: k_new and v_new must be defined");
  TESSERACT_CHECK(k_new.rank() == 4 && v_new.rank() == 4,
                  "PagedKVCache::append: expected rank-4 [B, H, S, Dh] inputs, "
                  "got k={} v={}",
                  k_new.shape().to_string(), v_new.shape().to_string());
  const DType   dtype_      = pool_->dtype();
  const Device  device_     = pool_->device();
  const int64_t num_heads_  = pool_->num_heads();
  const int64_t head_dim_   = pool_->head_dim();
  const int64_t block_size_ = pool_->block_size();

  TESSERACT_CHECK(k_new.dtype() == dtype_ && v_new.dtype() == dtype_,
                  "PagedKVCache::append: dtype mismatch (cache={}, k={}, v={})",
                  dtype_name(dtype_), dtype_name(k_new.dtype()),
                  dtype_name(v_new.dtype()));
  TESSERACT_CHECK(k_new.device() == device_ && v_new.device() == device_,
                  "PagedKVCache::append: device mismatch (cache={}, k={}, v={})",
                  device_.to_string(), k_new.device().to_string(),
                  v_new.device().to_string());

  const int64_t B  = k_new.shape()[0];
  const int64_t H  = k_new.shape()[1];
  const int64_t Sn = k_new.shape()[2];
  const int64_t D  = k_new.shape()[3];

  TESSERACT_CHECK(B == batch_ && H == num_heads_ && D == head_dim_,
                  "PagedKVCache::append: k_new shape {} incompatible with cache "
                  "(B={}, H={}, Dh={})",
                  k_new.shape().to_string(), batch_, num_heads_, head_dim_);
  TESSERACT_CHECK(v_new.shape()[0] == B && v_new.shape()[1] == H &&
                  v_new.shape()[2] == Sn && v_new.shape()[3] == D,
                  "PagedKVCache::append: v_new shape {} disagrees with k_new {}",
                  v_new.shape().to_string(), k_new.shape().to_string());
  TESSERACT_CHECK(current_len_ + Sn <= max_len_,
                  "PagedKVCache::append: would exceed max_len ({} + {} > {})",
                  current_len_, Sn, max_len_);
  TESSERACT_CHECK(k_new.is_contiguous() && v_new.is_contiguous(),
                  "PagedKVCache::append: k_new / v_new must be contiguous "
                  "(shape k={}, v={})",
                  k_new.shape().to_string(), v_new.shape().to_string());

  if (Sn == 0) return current_len_;

  const std::size_t elem = dtype_size(dtype_);
  // Pool layout [num_blocks, H, block_size, Dh] is row-major contiguous;
  // block `p`, head `h`, slot `s` lives at element offset
  //   ((p * H + h) * block_size + s) * Dh.
  const int64_t pool_block_stride = num_heads_ * block_size_ * D;
  const int64_t pool_head_stride  = block_size_ * D;

  auto* keys_base = static_cast<std::byte*>(pool_->keys().raw_data());
  auto* vals_base = static_cast<std::byte*>(pool_->values().raw_data());
  const auto* k_src_base = static_cast<const std::byte*>(k_new.raw_data());
  const auto* v_src_base = static_cast<const std::byte*>(v_new.raw_data());

  const bool on_cpu = device_.is_cpu();

  for (int64_t b = 0; b < B; ++b) {
    // Walk the new tokens [0, Sn), grouping consecutive tokens that fall
    // in the same physical block into one copy per head. For decode
    // (Sn=1) this is a single run; for chunked prefill it splits at
    // every block_size boundary.
    int64_t i = 0;
    while (i < Sn) {
      const int64_t gpos    = current_len_ + i;
      const int64_t logical = gpos / block_size_;
      const int64_t slot    = gpos % block_size_;
      const int64_t run     = std::min(block_size_ - slot, Sn - i);
      const int32_t p       = ensure_block(b, logical);

      const std::size_t run_bytes =
          static_cast<std::size_t>(run) * static_cast<std::size_t>(D) * elem;

      for (int64_t h = 0; h < H; ++h) {
        const int64_t dst_off =
            p * pool_block_stride + h * pool_head_stride + slot * D;
        const int64_t src_off = ((b * H + h) * Sn + i) * D;

        std::byte* k_dst = keys_base + static_cast<std::size_t>(dst_off) * elem;
        std::byte* v_dst = vals_base + static_cast<std::size_t>(dst_off) * elem;
        const std::byte* k_src =
            k_src_base + static_cast<std::size_t>(src_off) * elem;
        const std::byte* v_src =
            v_src_base + static_cast<std::size_t>(src_off) * elem;

        if (on_cpu) {
          Storage::copy_device_bytes(k_dst, device_, k_src, device_, run_bytes);
          Storage::copy_device_bytes(v_dst, device_, v_src, device_, run_bytes);
        } else {
          const Stream& s = current_stream(device_);
          Storage::copy_device_bytes_async(k_dst, device_, k_src, device_,
                                           run_bytes, s);
          Storage::copy_device_bytes_async(v_dst, device_, v_src, device_,
                                           run_bytes, s);
        }
      }
      i += run;
    }
  }

  current_len_ += Sn;
  return current_len_;
}

Tensor PagedKVCache::gather(const Tensor& pool) const {
  const int64_t B = batch_;
  const int64_t H = pool_->num_heads();
  const int64_t D = pool_->head_dim();
  const int64_t L = current_len_;
  const int64_t block_size_ = pool_->block_size();
  const DType   dtype_  = pool_->dtype();
  const Device  device_ = pool_->device();

  Tensor out = Tensor::empty({B, H, L, D}, dtype_, device_);
  if (L == 0) return out;

  const int64_t num_logical = (L + block_size_ - 1) / block_size_;

  if (!device_.is_cpu()) {
    // CUDA: refresh the device block table from `block_table_`, then do
    // the whole gather in one kernel launch. The per-block memcpy loop
    // (B·H·num_logical tiny async copies) is launch-overhead-bound and
    // dominated the decode step on profiling — the kernel collapses it
    // to a single launch indexed by the block table.
    std::vector<int32_t> flat(static_cast<std::size_t>(B * num_logical), 0);
    for (int64_t b = 0; b < B; ++b) {
      const auto& table = block_table_[static_cast<std::size_t>(b)];
      for (int64_t logical = 0; logical < num_logical; ++logical) {
        flat[static_cast<std::size_t>(b * num_logical + logical)] =
            table[static_cast<std::size_t>(logical)];
      }
    }
    // Upload into the leading `[B, num_logical]` corner of the
    // `[B, max_logical]` staging tensor. We upload row by row so the
    // device layout matches the kernel's `b * num_logical + logical`
    // indexing even when num_logical < max_logical_.
    auto* dev_base = static_cast<std::byte*>(block_table_dev_.raw_data());
    const std::size_t row_bytes =
        static_cast<std::size_t>(num_logical) * sizeof(int32_t);
    for (int64_t b = 0; b < B; ++b) {
      std::byte* dst =
          dev_base + static_cast<std::size_t>(b * num_logical) * sizeof(int32_t);
      const std::byte* src = reinterpret_cast<const std::byte*>(
          flat.data() + static_cast<std::size_t>(b * num_logical));
      Storage::copy_device_bytes(dst, device_, src, cpu_device(), row_bytes);
    }

    const Stream& s = current_stream(device_);
    cuda::detail::launch_paged_gather(
        device_.index, static_cast<int64_t>(dtype_size(dtype_)),
        pool.raw_data(), out.raw_data(),
        static_cast<const int32_t*>(block_table_dev_.raw_data()),
        B, H, L, D, block_size_, pool_->num_blocks(), num_logical,
        s.native_handle());
    return out;
  }

  // CPU: walk the block table directly with host memcpys.
  const std::size_t elem = dtype_size(dtype_);
  const int64_t pool_block_stride = H * block_size_ * D;
  const int64_t pool_head_stride  = block_size_ * D;

  auto* out_base = static_cast<std::byte*>(out.raw_data());
  const auto* pool_base = static_cast<const std::byte*>(pool.raw_data());

  for (int64_t b = 0; b < B; ++b) {
    const auto& table = block_table_[static_cast<std::size_t>(b)];
    for (int64_t logical = 0; logical < num_logical; ++logical) {
      const int64_t seq_start = logical * block_size_;
      const int64_t run = std::min(block_size_, L - seq_start);
      const int32_t p = table[static_cast<std::size_t>(logical)];

      const std::size_t run_bytes =
          static_cast<std::size_t>(run) * static_cast<std::size_t>(D) * elem;

      for (int64_t h = 0; h < H; ++h) {
        const int64_t src_off = p * pool_block_stride + h * pool_head_stride;
        const int64_t dst_off = ((b * H + h) * L + seq_start) * D;

        std::byte* dst = out_base + static_cast<std::size_t>(dst_off) * elem;
        const std::byte* src =
            pool_base + static_cast<std::size_t>(src_off) * elem;
        Storage::copy_device_bytes(dst, device_, src, device_, run_bytes);
      }
    }
  }
  return out;
}

Tensor PagedKVCache::keys_view() const { return gather(pool_->keys()); }
Tensor PagedKVCache::values_view() const { return gather(pool_->values()); }

void PagedKVCache::reset() {
  // Return only THIS cache's blocks to the (possibly shared) pool — never
  // free_all(), which would yank blocks out from under other requests
  // sharing the pool.
  for (auto& table : block_table_) {
    for (int32_t id : table) pool_->free(id);
    table.clear();
  }
  current_len_ = 0;
}

void PagedKVCache::set_current_len(int64_t len) {
  TESSERACT_CHECK(len >= 0 && len <= max_len_,
                  "PagedKVCache::set_current_len: len {} out of range [0, {}]",
                  len, max_len_);
  // Every request must already have physical blocks covering [0, len);
  // unlike the contiguous cache there is no pre-allocated tail to rewind
  // into. This is the graph-capture rewind contract — the caller is
  // expected to have appended at least `len` tokens before rewinding.
  const int64_t block_size_ = pool_->block_size();
  const int64_t need = (len + block_size_ - 1) / block_size_;
  for (int64_t b = 0; b < batch_; ++b) {
    TESSERACT_CHECK(
        static_cast<int64_t>(block_table_[static_cast<std::size_t>(b)].size()) >= need,
        "PagedKVCache::set_current_len: request {} has {} blocks but len {} "
        "needs {} — append before rewinding",
        b, block_table_[static_cast<std::size_t>(b)].size(), len, need);
  }
  current_len_ = len;
}

}  // namespace tesseract::nn

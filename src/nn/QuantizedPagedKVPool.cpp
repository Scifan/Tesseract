#include "tesseract/nn/QuantizedPagedKVPool.hpp"

#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

QuantizedPagedKVPool::QuantizedPagedKVPool(int64_t num_heads, int64_t head_dim,
                                           int64_t block_size,
                                           int64_t num_blocks, DType dtype,
                                           Device device)
    : num_heads_(num_heads),
      head_dim_(head_dim),
      block_size_(block_size),
      dtype_(dtype),
      device_(device),
      allocator_(static_cast<int32_t>(num_blocks)) {
  TESSERACT_CHECK(num_heads > 0 && head_dim > 0 && block_size > 0,
                  "QuantizedPagedKVPool: dims must be positive "
                  "(num_heads={}, head_dim={}, block_size={})",
                  num_heads, head_dim, block_size);
  TESSERACT_CHECK(num_blocks > 0 && num_blocks <= INT32_MAX,
                  "QuantizedPagedKVPool: num_blocks must be in (0, INT32_MAX], "
                  "got {}", num_blocks);
  TESSERACT_CHECK(dtype_is_floating(dtype),
                  "QuantizedPagedKVPool: dtype (the float type attention uses) "
                  "must be floating-point, got {}", dtype_name(dtype));

  keys_q_   = Tensor::zeros({num_blocks, num_heads, block_size, head_dim},
                            DType::Int8, device);
  values_q_ = Tensor::zeros({num_blocks, num_heads, block_size, head_dim},
                            DType::Int8, device);
  key_scale_   = Tensor::zeros({num_blocks, num_heads, block_size},
                               DType::Float32, device);
  value_scale_ = Tensor::zeros({num_blocks, num_heads, block_size},
                               DType::Float32, device);
}

}  // namespace tesseract::nn

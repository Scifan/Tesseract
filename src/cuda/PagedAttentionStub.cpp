// CPU-only build stub for the Wave-11 (B-032+) paged decode-attention
// bridge.
//
// Same always-compiled stub-vs-kernel pairing as `QuantizeKVStub.cpp` /
// `PagedKVStub.cpp`: this TU is always in the `tesseract_cuda` source
// list, but its body is gated on `!TESSERACT_HAS_CUDA`. When CUDA is
// compiled in, the real kernel in `PagedAttention.cu` owns this symbol and
// this TU collapses to empty. The op layer only reaches for the launcher
// on a CUDA device, so the throw is a safety net.

#include <cstdint>

#include "tesseract/cuda/detail/PagedAttention.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

void launch_paged_decode_attention(
    DType /*dtype*/, int /*device_index*/,
    int64_t /*A*/, int64_t /*H*/, int64_t /*Hkv*/, int64_t /*D*/,
    int64_t /*block_size*/, int64_t /*num_blocks*/, int64_t /*max_logical*/,
    int /*group*/, float /*scale*/,
    const void* /*q*/, const void* /*k_pool*/, const void* /*v_pool*/,
    const int32_t* /*block_tables*/, const int32_t* /*lens*/,
    void* /*o*/, void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] paged_decode_attention invoked but the CUDA backend was "
      "not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_paged_decode_attention_int8(
    DType /*dtype*/, int /*device_index*/,
    int64_t /*A*/, int64_t /*H*/, int64_t /*Hkv*/, int64_t /*D*/,
    int64_t /*block_size*/, int64_t /*num_blocks*/, int64_t /*max_logical*/,
    int /*group*/, float /*scale*/,
    const void* /*q*/, const int8_t* /*k_pool*/, const float* /*k_scale*/,
    const int8_t* /*v_pool*/, const float* /*v_scale*/,
    const int32_t* /*block_tables*/, const int32_t* /*lens*/,
    void* /*o*/, void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] paged_decode_attention_int8 invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_paged_prefill_attention(
    DType /*dtype*/, int /*device_index*/,
    int64_t /*A*/, int64_t /*S*/, int64_t /*H*/, int64_t /*Hkv*/, int64_t /*D*/,
    int64_t /*block_size*/, int64_t /*num_blocks*/, int64_t /*max_logical*/,
    int /*group*/, float /*scale*/,
    const void* /*q*/, const void* /*k_pool*/, const void* /*v_pool*/,
    const int32_t* /*block_tables*/, const int32_t* /*kv_lens*/,
    void* /*o*/, void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] paged_prefill_attention invoked but the CUDA backend was "
      "not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_paged_prefill_attention_int8(
    DType /*dtype*/, int /*device_index*/,
    int64_t /*A*/, int64_t /*S*/, int64_t /*H*/, int64_t /*Hkv*/, int64_t /*D*/,
    int64_t /*block_size*/, int64_t /*num_blocks*/, int64_t /*max_logical*/,
    int /*group*/, float /*scale*/,
    const void* /*q*/, const int8_t* /*k_pool*/, const float* /*k_scale*/,
    const int8_t* /*v_pool*/, const float* /*v_scale*/,
    const int32_t* /*block_tables*/, const int32_t* /*kv_lens*/,
    void* /*o*/, void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] paged_prefill_attention_int8 invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

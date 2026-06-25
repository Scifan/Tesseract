// CPU-only build stub for the Wave-9 (B-031) KV quantization bridge.
//
// Same always-compiled stub-vs-kernel pairing as `PagedKVStub.cpp` /
// `DequantMatMulStub.cpp`: this TU is always in the `tesseract_cuda`
// source list, but its body is gated on `!TESSERACT_HAS_CUDA`. When CUDA
// is compiled in, the real kernels in `QuantizeKV.cu` own these symbols
// and this TU collapses to empty. `QuantizedKVCache` only reaches for
// these launchers on a CUDA device, so the throw is a safety net.

#include <cstdint>

#include "tesseract/cuda/detail/QuantizeKV.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

void launch_quantize_kv_per_token(DType /*dtype*/, int /*device_index*/,
                                   int64_t /*rows*/, int64_t /*head_dim*/,
                                   const void* /*x*/, int8_t* /*q*/,
                                   float* /*scale*/, void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] quantize_kv invoked but the CUDA backend was not "
      "compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_dequantize_kv_per_token(DType /*dtype*/, int /*device_index*/,
                                     int64_t /*rows*/, int64_t /*head_dim*/,
                                     const int8_t* /*q*/, const float* /*scale*/,
                                     void* /*out*/, void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] dequantize_kv invoked but the CUDA backend was not "
      "compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

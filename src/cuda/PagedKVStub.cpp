// CPU-only build stub for the Wave-4.5 paged KV-cache gather bridge.
//
// Same always-compiled stub-vs-kernel pairing as `SwiGLUStub.cpp` /
// `DequantMatMulStub.cpp`: this TU is always in the `tesseract_cuda`
// source list, but its body is gated on `!TESSERACT_HAS_CUDA`. When CUDA
// is compiled in, the real kernel in `PagedKV.cu` owns the
// `launch_paged_gather` symbol and this TU collapses to an empty
// translation unit. PagedKVCache only reaches for this launcher on a
// CUDA device, so the throw is purely a safety net for a logic bug.

#include <cstdint>

#include "tesseract/cuda/detail/PagedKV.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

void launch_paged_gather(int /*device_index*/, int64_t /*elem_size*/,
                         const void* /*pool*/, void* /*out*/,
                         const int32_t* /*block_table*/,
                         int64_t /*B*/, int64_t /*H*/, int64_t /*L*/,
                         int64_t /*head_dim*/, int64_t /*block_size*/,
                         int64_t /*num_blocks*/, int64_t /*num_logical*/,
                         void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] paged_gather invoked but the CUDA backend was not "
      "compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

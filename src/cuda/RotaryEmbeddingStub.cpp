// CPU-only build stub for the B-014 rotary-embedding CUDA bridge. Same
// pattern as `SoftmaxStub.cpp`: this TU only emits a throwing symbol
// when `TESSERACT_HAS_CUDA` is *not* defined. The real implementation
// lives in `RotaryEmbedding.cu` and only compiles under the CUDA
// backend.

#include <cstdint>

#include "tesseract/cuda/detail/RotaryEmbedding.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA rotary_embedding kernel invoked but the CUDA "
      "backend was not compiled in (rebuild with "
      "-DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_rotary_embedding(DType /*dtype*/, int /*device_index*/,
                             int64_t /*outer*/, int64_t /*seq*/, int64_t /*dim*/,
                             const void* /*x*/, const void* /*cos*/,
                             const void* /*sin*/, void* /*out*/,
                             void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

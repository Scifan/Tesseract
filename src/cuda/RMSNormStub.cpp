// CPU-only build stubs for the Wave-2 fused RMSNorm / LayerNorm bridge.
//
// Same pattern as `ElementwiseStub.cpp` and `RotaryEmbeddingStub.cpp`:
// this TU is always in the `tesseract_cuda` source list, but its body
// is gated on `!TESSERACT_HAS_CUDA`. When CUDA is compiled in, the
// real kernels in `RMSNorm.cu` participate in the link and the stubs
// collapse to an empty TU so we never double-define the launchers.

#include <cstdint>

#include "tesseract/cuda/detail/RMSNorm.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] fused CUDA norm kernel invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_rms_norm(DType /*dtype*/, int /*device_index*/,
                     int64_t /*outer*/, int64_t /*D*/,
                     const void* /*x*/, const void* /*weight*/,
                     double /*eps*/, void* /*out*/, void* /*stream*/) {
  throw_not_built();
}

void launch_layer_norm(DType /*dtype*/, int /*device_index*/,
                       int64_t /*outer*/, int64_t /*D*/,
                       const void* /*x*/, const void* /*weight*/,
                       const void* /*bias*/, double /*eps*/,
                       void* /*out*/, void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

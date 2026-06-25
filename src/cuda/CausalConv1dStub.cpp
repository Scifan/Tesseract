// CPU-only build stub for the Phase 5 causal conv1d CUDA bridge. The CUDA
// branch in nn::Mamba::conv1d_forward guards on device().is_cuda(), so this
// is never called in CPU-only builds; the symbol must still link.

#include <cstdint>

#include "tesseract/cuda/detail/CausalConv1d.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

void launch_causal_conv1d(int /*device_index*/, int64_t /*B*/, int64_t /*L*/,
                          int64_t /*channels*/, int64_t /*K*/,
                          const float* /*x*/, const float* /*weight*/,
                          const float* /*bias*/, float* /*out*/,
                          void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] CUDA causal conv1d invoked but the CUDA backend was not "
      "compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

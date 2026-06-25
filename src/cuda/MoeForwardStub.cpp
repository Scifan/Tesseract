// CPU-only build stub for the Phase 4 fused GPU MoE forward. The CUDA
// branch in nn::MoEFeedForward guards on device().is_cuda(), so this is
// never called in CPU-only builds; the symbol must still link.

#include <cstdint>

#include "tesseract/cuda/detail/MoeForward.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

bool launch_moe_grouped_ffn(int /*device_index*/, int64_t /*T*/, int64_t /*D*/,
                            int64_t /*dff*/, int64_t /*E*/, int64_t /*k*/,
                            const float* /*x*/, const float* /*gates*/,
                            const float* /*mask*/, const float* const* /*Wg*/,
                            const float* const* /*Wu*/, const float* const* /*Wd*/,
                            float* /*y*/, void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] fused GPU MoE invoked but the CUDA backend was not "
      "compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

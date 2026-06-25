// CPU-only build stub for the Phase 4 MoE routing CUDA bridge. Same
// pattern as `SoftmaxStub.cpp`: in CPU-only builds `nn::MoEFeedForward`
// never reaches the CUDA branch (it guards on `device().is_cuda()`), but
// the symbol must still resolve at link time.

#include <cstdint>

#include "tesseract/cuda/detail/MoeRoute.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

void launch_moe_route(int /*device_index*/, int64_t /*num_tokens*/,
                      int64_t /*num_experts*/, int64_t /*top_k*/,
                      const float* /*logits*/, float* /*gates*/,
                      float* /*mask*/, void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] CUDA MoE routing kernel invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

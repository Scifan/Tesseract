// CPU-only build stubs for the M2F CUDA cross-entropy bridge. Same
// pattern as `ElementwiseStub.cpp`; see that file's comment for the
// rationale.

#include <cstdint>

#include "tesseract/cuda/detail/Loss.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA cross-entropy kernel invoked but the CUDA "
      "backend was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_ce_forward(DType /*dtype*/, int /*device_index*/,
                       int64_t /*N*/, int64_t /*C*/,
                       const void* /*logits*/,
                       const int64_t* /*targets*/,
                       void* /*loss_out*/, void* /*probs_out*/,
                       void* /*stream*/) {
  throw_not_built();
}

void launch_ce_backward(DType /*dtype*/, int /*device_index*/,
                        int64_t /*N*/, int64_t /*C*/,
                        const void* /*probs*/,
                        const int64_t* /*targets*/,
                        const void* /*grad_scalar*/,
                        void* /*dlogits_out*/,
                        void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

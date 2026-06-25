// CPU-only build stub for the M2I Adam bridge. Same pattern as
// `ShapeStub.cpp`: unconditionally added to the CUDA target's
// source list, body gated on `!TESSERACT_HAS_CUDA` so the CUDA build
// picks up `Optim.cu`'s real implementation instead.

#include "tesseract/cuda/detail/Optim.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA optim kernel invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_adam_step(DType /*dtype*/, int /*device_index*/,
                      int64_t /*n*/,
                      void* /*param*/,
                      const void* /*grad*/,
                      void* /*m_buf*/,
                      void* /*v_buf*/,
                      double /*lr*/,
                      double /*beta1*/,
                      double /*beta2*/,
                      double /*eps*/,
                      double /*bc1*/,
                      double /*bc2*/,
                      void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

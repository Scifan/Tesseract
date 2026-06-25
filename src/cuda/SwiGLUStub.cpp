// CPU-only build stub for the Wave-4.1 fused SwiGLU CUDA bridge.
//
// Same always-compiled stub-vs-kernel pairing as `RMSNormStub.cpp` /
// `DequantMatMulStub.cpp`: this TU is always in the `tesseract_cuda`
// source list, but its body is gated on `!TESSERACT_HAS_CUDA`. When
// CUDA is compiled in, the real kernel in `SwiGLU.cu` owns the
// `launch_swiglu_silu_gate` symbol and this TU collapses to an
// empty translation unit so the linker never sees two definitions.

#include <cstdint>

#include "tesseract/cuda/detail/SwiGLU.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] fused CUDA swiglu_silu_gate invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_swiglu_silu_gate(DType /*dtype*/, int /*device_index*/,
                             int64_t /*numel*/,
                             const void* /*gate*/,
                             const void* /*up*/,
                             void* /*out*/,
                             void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

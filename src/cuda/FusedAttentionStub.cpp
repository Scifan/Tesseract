// CPU-only build stub for the Wave-4.2 fused attention CUDA bridge.
//
// Same always-compiled stub-vs-kernel pairing as `SwiGLUStub.cpp` /
// `RMSNormStub.cpp`: this TU is always in the `tesseract_cuda`
// source list, but its body is gated on `!TESSERACT_HAS_CUDA`. When
// CUDA is compiled in, the real kernel in `FusedAttention.cu` owns
// the `launch_fused_attention` symbol and this TU collapses to an
// empty translation unit so the linker never sees two definitions.

#include <cstdint>

#include "tesseract/cuda/detail/FusedAttention.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] fused CUDA attention invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_fused_attention(DType /*dtype*/, int /*device_index*/,
                            int64_t /*B*/, int64_t /*H*/, int64_t /*H_kv*/,
                            int64_t /*S_q*/, int64_t /*S_k*/,
                            int64_t /*D*/,
                            float /*scale_inv_sqrt_d*/, bool /*causal*/,
                            const void* /*q*/, const void* /*k*/,
                            const void* /*v*/,
                            void* /*o*/, void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

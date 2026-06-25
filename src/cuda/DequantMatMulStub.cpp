// CPU-only build stub for the Wave-3.1 INT8 dequant-matmul bridge.
//
// Same always-compiled pattern as `RMSNormStub.cpp` /
// `RotaryEmbeddingStub.cpp`: this TU is always in the `tesseract_cuda`
// source list, but the body only exists under `!TESSERACT_HAS_CUDA`.
// When CUDA is compiled in, the real kernel in `DequantMatMul.cu`
// participates in the link and this stub collapses to an empty TU, so
// we never double-define `launch_dequant_matmul_int8`.

#include <cstdint>

#include "tesseract/cuda/detail/DequantMatMul.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] dequant-matmul CUDA kernel invoked but the CUDA "
      "backend was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_dequant_matmul_int8(DType /*dtype*/, int /*device_index*/,
                                int64_t /*M*/, int64_t /*N*/, int64_t /*K*/,
                                const void* /*x*/, const void* /*q_w*/,
                                const void* /*scale*/, void* /*y*/,
                                void* /*stream*/) {
  throw_not_built();
}

void launch_dequant_matmul_int4_group(DType /*dtype*/, int /*device_index*/,
                                      int64_t /*M*/, int64_t /*N*/,
                                      int64_t /*K*/, int64_t /*group_size*/,
                                      const void* /*x*/,
                                      const void* /*q_packed*/,
                                      const void* /*scale*/, void* /*y*/,
                                      void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

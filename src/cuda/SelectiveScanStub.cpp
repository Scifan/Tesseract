// CPU-only build stub for the M4 Track A2 (B-039) selective-scan CUDA bridge.
//
// Same always-compiled stub-vs-kernel pairing as `SwiGLUStub.cpp`: this TU is
// always in the `tesseract_cuda` source list, but its body is gated on
// `!TESSERACT_HAS_CUDA`. When CUDA is compiled in, the real kernel in
// `SelectiveScan.cu` owns the `launch_selective_scan` symbol and this TU
// collapses to an empty translation unit.

#include <cstdint>

#include "tesseract/cuda/detail/SelectiveScan.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

void launch_selective_scan(DType /*dtype*/, int /*device_index*/,
                           int64_t /*B*/, int64_t /*L*/, int64_t /*D*/,
                           int64_t /*N*/, const void* /*u*/,
                           const void* /*delta*/, const void* /*A*/,
                           const void* /*Bm*/, const void* /*Cm*/,
                           const void* /*Dskip*/, const void* /*state_in*/,
                           void* /*y*/, void* /*state_out*/, void* /*stream*/) {
  throw tesseract::DeviceError(
      "[tesseract] CUDA selective_scan invoked but the CUDA backend was not "
      "compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

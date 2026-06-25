// CPU-only build stubs for the M2F CUDA softmax bridge. Same pattern
// as `ElementwiseStub.cpp`; see that file's comment for the rationale.

#include <cstdint>

#include "tesseract/cuda/detail/Softmax.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA softmax kernel invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_softmax(bool /*take_log*/, DType /*dtype*/,
                    int /*device_index*/, int /*ndim*/, int /*dim*/,
                    const int64_t* /*in_sizes*/,
                    const int64_t* /*in_strides*/,
                    const void* /*x*/, void* /*out*/, void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

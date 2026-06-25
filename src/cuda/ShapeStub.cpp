// CPU-only build stubs for the M2H CUDA shape bridge. Same pattern
// as `ElementwiseStub.cpp`: unconditionally added to the CUDA target's
// source list, body gated on `!TESSERACT_HAS_CUDA` so the CUDA build
// picks up `Shape.cu`'s real implementations instead. See
// `src/cuda/ElementwiseStub.cpp` for the rationale.

#include "tesseract/cuda/detail/Shape.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA shape kernel invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_strided_copy(DType /*dtype*/, int /*device_index*/,
                         int /*ndim*/,
                         const int64_t* /*sizes*/,
                         const int64_t* /*src_strides*/,
                         const int64_t* /*dst_strides*/,
                         const void* /*src*/, void* /*dst*/,
                         void* /*stream*/) {
  throw_not_built();
}

void launch_strided_scatter_add(DType /*dtype*/, int /*device_index*/,
                                int /*ndim*/,
                                const int64_t* /*sizes*/,
                                const int64_t* /*src_strides*/,
                                const int64_t* /*dst_strides*/,
                                const void* /*src*/, void* /*dst*/,
                                void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

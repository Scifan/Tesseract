// CPU-only build stubs for the M2H CUDA indexing bridge. Same pattern
// as `ElementwiseStub.cpp`: unconditionally added to the CUDA target's
// source list, body gated on `!TESSERACT_HAS_CUDA` so the CUDA build
// picks up `Indexing.cu`'s real implementations instead.

#include "tesseract/cuda/detail/Indexing.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA indexing kernel invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_index_select(DType /*dtype*/, int /*device_index*/,
                         int /*ndim*/, int /*dim*/,
                         const int64_t* /*out_sizes*/,
                         const int64_t* /*src_strides*/,
                         const int64_t* /*out_strides*/,
                         const void* /*src*/,
                         const int64_t* /*indices*/,
                         void* /*out*/,
                         void* /*stream*/) {
  throw_not_built();
}

void launch_scatter_add_at_dim(DType /*dtype*/, int /*device_index*/,
                               int /*ndim*/, int /*dim*/,
                               const int64_t* /*grad_sizes*/,
                               const int64_t* /*grad_strides*/,
                               const int64_t* /*dst_strides*/,
                               const void* /*grad*/,
                               const int64_t* /*indices*/,
                               void* /*dst*/,
                               void* /*stream*/) {
  throw_not_built();
}

void launch_gather(DType /*dtype*/, int /*device_index*/,
                   int /*ndim*/, int /*dim*/,
                   const int64_t* /*out_sizes*/,
                   const int64_t* /*src_strides*/,
                   const int64_t* /*idx_strides*/,
                   const int64_t* /*out_strides*/,
                   const void* /*src*/,
                   const int64_t* /*indices*/,
                   void* /*out*/,
                   void* /*stream*/) {
  throw_not_built();
}

void launch_gather_scatter_add(DType /*dtype*/, int /*device_index*/,
                               int /*ndim*/, int /*dim*/,
                               const int64_t* /*grad_sizes*/,
                               const int64_t* /*grad_strides*/,
                               const int64_t* /*idx_strides*/,
                               const int64_t* /*dst_strides*/,
                               const void* /*grad*/,
                               const int64_t* /*indices*/,
                               void* /*dst*/,
                               void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

// CPU-only build stubs for the M2F CUDA reduction bridge. Same
// "always-compiled, body gated on !TESSERACT_HAS_CUDA" pattern as
// `ElementwiseStub.cpp`: in the CUDA-ON build this TU collapses to an
// empty source (the real definitions live in `Reduction.cu`), so we
// never double-define the `detail::launch_reduce_*` symbols.
//
// Reaching these bodies would mean the op-layer dispatch in
// `src/ops/cpu/Reduction.cpp` grew a new case that forgot the
// `is_cuda()` guard — the `throw` makes that failure loud, with a
// helpful "rebuild with `-DTESSERACT_ENABLE_CUDA=ON`" hint.

#include <cstdint>

#include "tesseract/cuda/detail/Reduction.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA reduction kernel invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_reduce_all(ReduceKind /*op*/, DType /*dtype*/,
                       int /*device_index*/, int64_t /*nelem*/,
                       const void* /*x*/, void* /*out*/, void* /*stream*/) {
  throw_not_built();
}

void launch_reduce_dim(ReduceKind /*op*/, DType /*dtype*/,
                       int /*device_index*/, int /*ndim*/, int /*dim*/,
                       const int64_t* /*in_sizes*/,
                       const int64_t* /*in_strides*/,
                       const void* /*x*/, void* /*out*/, void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

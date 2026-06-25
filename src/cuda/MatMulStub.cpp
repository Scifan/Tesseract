// CPU-only build stub for the M2G CUDA matmul bridge.
//
// Same shape as `ElementwiseStub.cpp` / `ReductionStub.cpp` / etc.:
// unconditionally added to `tesseract_cuda`'s source list, but the
// body is gated on `!TESSERACT_HAS_CUDA`. When CUDA is compiled in,
// the real `launch_matmul` implementation lives in `MatMul.cpp` (which
// only builds under TESSERACT_ENABLE_CUDA=ON), and this TU collapses
// to an empty translation unit so the linker never sees two
// definitions.
//
// The stub throws a clear `DeviceError`. The op-layer dispatch in
// `src/ops/cpu/MatMul.cpp` already guards CUDA tensors behind an
// `is_cuda()` check, so reaching this stub in production means
// someone constructed a CUDA-tagged tensor in a CPU-only build (which
// `CudaAllocator::instance_for` already refuses), or the dispatch
// grew a new case that forgot the device check — which tests catch.

#include "tesseract/cuda/detail/MatMul.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA matmul (cuBLASLt) invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_matmul(DType /*dtype*/, int /*device_index*/,
                   int64_t /*M*/, int64_t /*N*/, int64_t /*K*/,
                   const void* /*a*/, MmOp /*op_a*/, int64_t /*lda*/,
                   const void* /*b*/, MmOp /*op_b*/, int64_t /*ldb*/,
                   void* /*c*/, int64_t /*ldc*/,
                   void* /*stream_handle*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

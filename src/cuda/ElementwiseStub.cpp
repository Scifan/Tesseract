// CPU-only build stubs for the M2E CUDA elementwise bridge.
//
// Same pattern as `Memcpy.cpp`'s "CPU-only branch": this TU is
// unconditionally added to `tesseract_cuda`'s source list, but its
// body is gated on `!TESSERACT_HAS_CUDA`. When CUDA is compiled in
// (i.e. `Elementwise.cu` participates in the link), this file
// collapses to an empty TU so we never double-define the launchers.
//
// The stub bodies throw a clear `DeviceError` — the op-layer dispatch
// in `src/ops/cpu/Arithmetic.cpp` / `Activation.cpp` guards CUDA
// tensors behind an `is_cuda()` check, so reaching these stubs means
// either:
//   * someone hand-constructed a CUDA-tagged tensor in a CPU-only
//     build (impossible via `CudaAllocator::instance_for()`, which
//     already throws), or
//   * the dispatch switch grew a new case that forgot the device
//     check — which the tests would catch immediately.

#include <cstddef>

#include "tesseract/cuda/detail/Elementwise.hpp"
#include "tesseract/utils/Logging.hpp"

#if !defined(TESSERACT_HAS_CUDA)

namespace tesseract::cuda::detail {

[[noreturn]] static void throw_not_built() {
  throw tesseract::DeviceError(
      "[tesseract] CUDA elementwise kernel invoked but the CUDA backend "
      "was not compiled in (rebuild with -DTESSERACT_ENABLE_CUDA=ON)");
}

void launch_binary_elementwise(BinaryKind /*op*/, DType /*dtype*/,
                               int /*device_index*/, int /*ndim*/,
                               const int64_t* /*out_sizes*/,
                               const int64_t* /*a_strides*/,
                               const int64_t* /*b_strides*/,
                               void* /*out*/, const void* /*a*/,
                               const void* /*b*/, void* /*stream*/) {
  throw_not_built();
}

void launch_unary_elementwise(UnaryKind /*op*/, DType /*dtype*/,
                              int /*device_index*/, int /*ndim*/,
                              const int64_t* /*shape*/,
                              const int64_t* /*x_strides*/,
                              void* /*out*/, const void* /*x*/,
                              void* /*stream*/) {
  throw_not_built();
}

void launch_fill(DType /*dtype*/, int /*device_index*/,
                 std::size_t /*nelem*/, void* /*out*/, double /*value*/,
                 void* /*stream*/) {
  throw_not_built();
}

}  // namespace tesseract::cuda::detail

#endif  // !TESSERACT_HAS_CUDA

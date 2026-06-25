#include "tesseract/cuda/CudaRuntime.hpp"

#include <fmt/format.h>

#include "tesseract/utils/Logging.hpp"

#if defined(TESSERACT_HAS_CUDA)
#include "Internal.hpp"
#endif

// Routing layer for the CUDA probe API. This translation unit is always
// compiled with a plain C++ compiler (no nvcc) and therefore never pulls
// `<cuda_runtime.h>` into the CPU-only build. When the library is built
// with TESSERACT_HAS_CUDA=1, the bodies forward to free functions
// declared in `Internal.hpp` and defined by the nvcc-compiled
// `Probe.cu`; when built without, they return stub values.
//
// This lets user code call `tesseract::cuda::is_available()` and make a
// sensible decision (skip a test, print a banner, etc.) without having
// to sprinkle `#ifdef TESSERACT_HAS_CUDA` through every call site.

namespace tesseract::cuda {

bool has_cuda_support() noexcept {
#if defined(TESSERACT_HAS_CUDA)
  return true;
#else
  return false;
#endif
}

bool is_available() noexcept {
#if defined(TESSERACT_HAS_CUDA)
  return detail::real_device_count() > 0;
#else
  return false;
#endif
}

int device_count() noexcept {
#if defined(TESSERACT_HAS_CUDA)
  return detail::real_device_count();
#else
  return 0;
#endif
}

DeviceInfo device_info(int index) {
#if defined(TESSERACT_HAS_CUDA)
  DeviceInfo info{};
  if (!detail::real_device_info(index, &info)) {
    throw DeviceError(fmt::format(
        "[tesseract] cuda::device_info({}) failed: driver reported no such device "
        "(device_count()={})",
        index, detail::real_device_count()));
  }
  return info;
#else
  (void)index;
  throw DeviceError(
      "[tesseract] cuda::device_info called on a CPU-only build "
      "(rebuild with -DTESSERACT_ENABLE_CUDA=ON to get a CUDA backend)");
#endif
}

std::string runtime_version_string() {
#if defined(TESSERACT_HAS_CUDA)
  return detail::real_runtime_version_string();
#else
  return "tesseract: built without CUDA support";
#endif
}

}  // namespace tesseract::cuda

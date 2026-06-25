// M2B probe: smallest possible nvcc-compiled TU that proves the CUDA build
// pipeline works end-to-end. Also provides the real-driver implementation of
// `tesseract::cuda::detail::real_*` that `CudaRuntime.cpp` routes to when
// TESSERACT_HAS_CUDA=1.
//
// Scope for M2B is deliberately tiny: no memory allocation, no streams, no
// kernels that touch user data. The only `__global__` defined here is a
// "hello, world"-style kernel that we launch once in
// `real_runtime_version_string()` so the nvcc device codegen path is
// exercised at least once per build. Every subsequent track (M2C streams,
// M2E elementwise, M2G cuBLAS) adds its own `.cu` files in this directory.

#include <cstdint>
#include <cstdio>
#include <string>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "Internal.hpp"

namespace tesseract::cuda::detail {
namespace {

// Swallow any "no CUDA device" / driver-not-found error into a 0 count.
// Tests use `is_available()` to decide whether to skip; everyone else
// surfaces driver errors through higher-level APIs (the real error
// reporting layer arrives in M2C alongside the allocator).
int query_device_count() noexcept {
  int n = 0;
  const cudaError_t rc = cudaGetDeviceCount(&n);
  if (rc != cudaSuccess) {
    // Discard the sticky error so later calls aren't poisoned.
    (void)cudaGetLastError();
    return 0;
  }
  return n;
}

__global__ void noop_kernel(int* flag) {
  if (flag != nullptr) {
    *flag = 1;
  }
}

// Force-instantiate the no-op kernel by taking its address. Keeps the
// "nvcc pipeline is actually wired up" signal load-bearing rather than
// theoretical without requiring a runtime launch (which would need a
// device context). If nvcc / linker ever drops device code from this TU,
// this symbol won't resolve and the link will fail.
[[maybe_unused]] void (* const kForceDeviceCodegen)(int*) = noop_kernel;

}  // namespace

int real_device_count() noexcept {
  return query_device_count();
}

bool real_device_info(int index, DeviceInfo* out) noexcept {
  if (out == nullptr) return false;
  const int n = query_device_count();
  if (index < 0 || index >= n) return false;

  cudaDeviceProp prop{};
  const cudaError_t rc = cudaGetDeviceProperties(&prop, index);
  if (rc != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }

  out->index = index;
  out->name = prop.name;
  out->compute_capability_major = prop.major;
  out->compute_capability_minor = prop.minor;
  out->total_global_memory_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
  return true;
}

std::string real_runtime_version_string() {
  int runtime_version = 0;
  int driver_version = 0;
  (void)cudaRuntimeGetVersion(&runtime_version);
  (void)cudaDriverGetVersion(&driver_version);

  const int rt_major = runtime_version / 1000;
  const int rt_minor = (runtime_version % 1000) / 10;
  const int drv_major = driver_version / 1000;
  const int drv_minor = (driver_version % 1000) / 10;

  const int n = query_device_count();
  std::string devices;
  for (int i = 0; i < n; ++i) {
    DeviceInfo info{};
    if (real_device_info(i, &info)) {
      if (!devices.empty()) devices += ", ";
      devices += fmt::format("#{} {} (sm_{}{})", info.index, info.name,
                             info.compute_capability_major,
                             info.compute_capability_minor);
    }
  }

  return fmt::format(
      "CUDA runtime {}.{} / driver {}.{}; visible devices: [{}]",
      rt_major, rt_minor, drv_major, drv_minor,
      devices.empty() ? std::string{"none"} : devices);
}

}  // namespace tesseract::cuda::detail

#pragma once

// Shared CUDA-side helpers used by the M2E and M2F kernel TUs
// (`Elementwise.cu`, `Reduction.cu`, `Softmax.cu`, `Loss.cu`). Kept
// deliberately small and C++17-parseable so nvcc — configured with
// `CUDA_STANDARD=17` in `src/cuda/CMakeLists.txt` — never has to see
// `std::span` or explicit-template lambdas. The same local mirror of
// `Shape::kMaxRank` that `Elementwise.cu` uses lives here so host-side
// mismatches surface in the `TESSERACT_CHECK(ndim <= kMaxRank, ...)`
// assertions in each launcher.
//
// Only included from `.cu` TUs; do **not** include this from a plain
// `.cpp` or anywhere under `include/`. The `DeviceGuard` / `check_*`
// helpers rely on `<cuda_runtime.h>` and `fmt/format.h`, and the
// `flat_to_offset` helper is `__device__`-qualified.

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "tesseract/utils/Logging.hpp"

namespace tesseract::cuda::detail {

// Local mirror of `tesseract::Shape::kMaxRank`. We do not include
// Shape.hpp here because it pulls in `std::span` (C++20); the bridge
// surface already hands us raw `int64_t*` + `int ndim`.
constexpr int kMaxRank = 8;

// Trivially-copyable kernel-arg descriptor for shape / stride data.
// Passed by value to `__global__`s.
struct ShapePod {
  int64_t sizes[kMaxRank];
  int ndim;
};

// Given a flat output index `i`, an output shape (`sizes`), and a
// per-operand stride array (`strides`), compute the flat byte-element
// offset into that operand. Handles broadcast (stride==0) and strided
// (arbitrary stride) inputs uniformly. Identical to the one in
// `Elementwise.cu`; duplicated here rather than made a device-func
// reference so each TU can inline it into its own kernels.
__device__ __forceinline__ int64_t
flat_to_offset(int64_t i, const ShapePod& sizes, const ShapePod& strides) {
  int64_t off = 0;
  int64_t rem = i;
#pragma unroll
  for (int d_rev = 0; d_rev < kMaxRank; ++d_rev) {
    const int d = sizes.ndim - 1 - d_rev;
    if (d < 0) break;
    const int64_t dim = sizes.sizes[d];
    const int64_t c = rem % dim;
    rem /= dim;
    off += c * strides.sizes[d];
  }
  return off;
}

// Element-count for a ShapePod; equivalent to `Shape::numel()` but
// device/host-neutral.
inline int64_t numel_pod(const ShapePod& s) {
  int64_t n = 1;
  for (int d = 0; d < s.ndim; ++d) n *= s.sizes[d];
  return n;
}

// Pack a raw `int64_t[ndim]` array into a ShapePod. Used by every
// M2F launcher to translate the op-layer's `const int64_t*` + `int
// ndim` descriptor into a trivially-copyable kernel arg.
inline ShapePod pack_raw(const int64_t* data, int ndim) {
  ShapePod pod{};
  pod.ndim = ndim;
  for (int d = 0; d < ndim; ++d) pod.sizes[d] = data[d];
  return pod;
}

// RAII device-context switch. Restores the caller's previous current
// device on destruction so call-site side effects don't leak across
// launchers (op code in `tesseract_core` does not itself call
// `cudaSetDevice`, so restoring is mainly a hygiene thing for tests
// that iterate over multiple devices).
struct DeviceGuard {
  int previous{-1};

  explicit DeviceGuard(int target) {
    cudaError_t err = cudaGetDevice(&previous);
    if (err != cudaSuccess) previous = -1;
    err = cudaSetDevice(target);
    if (err != cudaSuccess) {
      throw DeviceError(fmt::format(
          "[tesseract] cudaSetDevice({}) failed: {}",
          target, cudaGetErrorString(err)));
    }
  }
  ~DeviceGuard() { if (previous >= 0) (void)cudaSetDevice(previous); }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
};

// Check the post-launch error state. CUDA kernel launch errors (bad
// grid/block config, launching into a destroyed stream, ...) show up
// via `cudaGetLastError`; actual kernel-interior memory errors only
// surface on the next synchronize (or under compute-sanitizer), but
// our callers cross the device boundary via `Storage::copy_device_bytes`
// which syncs the current stream anyway.
inline void check_launch(const char* op_name) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw DeviceError(fmt::format(
        "[tesseract] CUDA {} kernel launch failed: {}",
        op_name, cudaGetErrorString(err)));
  }
}

// Default block size. 256 is the sweet spot for M2F reductions on Ada
// (fills a warp group, leaves headroom for register pressure). Each
// kernel may override by constructing its own launch config.
constexpr int kBlockSize = 256;

// Convert an element count into a 1-D grid size, clamped to the CUDA
// grid.x limit. 2^31-1 blocks is enough for any tensor we can
// allocate today (would saturate 256*(2^31-1) ≈ 0.5T elements).
inline int launch_grid(int64_t total, int block = kBlockSize) {
  if (total <= 0) return 1;
  const int64_t g = (total + block - 1) / block;
  return static_cast<int>(std::min<int64_t>(g, 2'147'483'647LL));
}

}  // namespace tesseract::cuda::detail

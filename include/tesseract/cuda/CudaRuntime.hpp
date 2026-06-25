#pragma once

// Public probe surface for the CUDA backend.
//
// This header is intentionally plain C++20: it does NOT include
// <cuda_runtime.h>, <cuda.h>, or any CUDA-specific type. That keeps the
// CPU-only build (TESSERACT_ENABLE_CUDA=OFF) free of any CUDA Toolkit
// dependency, while still letting downstream code ask at runtime "is the
// CUDA backend available in this build, and if so what devices are visible?"
// without an `#ifdef` at every call site.
//
// Semantics:
//   * In a CPU-only build, `is_available()` always returns false,
//     `device_count()` always returns 0, and `device_info(i)` throws. No
//     actual CUDA runtime call is made — the stubs short-circuit.
//   * In a CUDA-enabled build, these functions forward to the CUDA driver
//     API via symbols supplied by `tesseract_cuda`. They never throw on
//     a "no GPU present" condition; they return false / 0 and let callers
//     (tests in particular) decide how to react.
//
// This is the M2.α probe surface only (M2B). Allocators, streams, and
// data-copy operations are introduced in M2C / M2D and live next to this
// header in tesseract/cuda/.

#include <cstdint>
#include <string>

namespace tesseract::cuda {

// True iff the binary was compiled with TESSERACT_ENABLE_CUDA=ON AND the
// CUDA driver on the host can enumerate at least one device. A false return
// can mean either (a) the build is CPU-only, or (b) the build has CUDA but
// the host has no visible GPU / driver.
bool is_available() noexcept;

// True iff the binary was compiled with TESSERACT_ENABLE_CUDA=ON, regardless
// of whether the current host has a GPU visible to the driver. Useful for
// tests that want to distinguish "this binary can't talk to CUDA at all"
// from "this binary could, but we're running on a headless box — skip".
bool has_cuda_support() noexcept;

// Number of devices visible to the CUDA driver, or 0 if the build has no
// CUDA support. Never throws; a driver error (e.g. "no CUDA-capable device")
// is reported as 0.
int device_count() noexcept;

// Per-device descriptor populated by `device_info`. Kept deliberately minimal
// in M2B — we only need enough to decide "can we run the tests here" and
// to print a human-readable summary. More fields (warp size, SM count,
// clock rate, L2 cache size, ...) arrive alongside the op kernels in M2E+.
struct DeviceInfo {
  int index{-1};
  std::string name;
  int compute_capability_major{0};
  int compute_capability_minor{0};
  std::uint64_t total_global_memory_bytes{0};
};

// Describe device `index`. Throws `DeviceError` when the build is CPU-only,
// when `index` is out of range, or when the CUDA driver reports an error
// for that device.
DeviceInfo device_info(int index);

// Textual "CUDA X.Y, nvcc Z.Z on host <os>"-style summary; safe to call in
// both build configurations. Used by the test skipper and by any future
// `--version` banner.
std::string runtime_version_string();

}  // namespace tesseract::cuda

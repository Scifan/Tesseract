#pragma once

// Internal CUDA bridge for the M2I optimizer kernels. Same `detail/`
// visibility / C++17 contract as `Elementwise.hpp` / `Shape.hpp` etc.:
// lives under `include/` so the op-layer TU (`src/optim/Adam.cpp`)
// can call into it without touching private include dirs, but it is
// NOT part of the public API.
//
// Scope (M2I): one launcher, `launch_adam_step`, implementing the
// fused Adam parameter update
//
//    m ← β₁·m + (1-β₁)·g
//    v ← β₂·v + (1-β₂)·g²
//    p ← p - lr · (m / (1-β₁^t)) / (√(v / (1-β₂^t)) + ε)
//
// as a single elementwise CUDA kernel. The bias-correction factors are
// precomputed on the host and passed in as `bc1` / `bc2` so the kernel
// stays branchless and dispatch-free.
//
// Contract:
//   * All six tensors (`param`, `grad`, `m`, `v` buffers) must be
//     contiguous, same shape, same dtype, and resident on the same
//     CUDA device (the caller's `nn::Module::to(device)` setup).
//   * Supported dtypes: Float32, Float64. Half / BFloat16 optimizer
//     state is numerically unsafe for Adam's second-moment
//     accumulator (the `g²` term underflows rapidly on typical
//     gradient magnitudes); the full-precision path is the one
//     documented in the M2 exit bar.
//   * Stream handle crosses as `void*` (raw `cudaStream_t`), same as
//     the rest of the M2 bridges. `launch_adam_step` queues work on
//     that stream and returns without syncing.

#include <cstddef>
#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

void launch_adam_step(DType dtype, int device_index,
                      int64_t n,
                      void* param,
                      const void* grad,
                      void* m_buf,
                      void* v_buf,
                      double lr,
                      double beta1,
                      double beta2,
                      double eps,
                      double bc1,
                      double bc2,
                      void* stream);

}  // namespace tesseract::cuda::detail

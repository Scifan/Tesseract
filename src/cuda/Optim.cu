// M2I: CUDA Adam kernel. Only compiled when TESSERACT_ENABLE_CUDA=ON
// (see `src/cuda/CMakeLists.txt`); the CPU-only stub lives in
// `OptimStub.cpp`.
//
// One fused elementwise kernel per param:
//
//   m ← β₁·m + (1-β₁)·g
//   v ← β₂·v + (1-β₂)·g²
//   p ← p - lr · (m / bc1) / (√(v / bc2) + ε)
//
// `bc1` / `bc2` are the bias-correction factors `(1 - β^t)`; they are
// precomputed on the host once per `step()` and passed as constants so
// the kernel stays branchless and dispatch-free. No grid-wide
// reductions here — Adam is per-element — so this is a plain 1-D grid.
//
// Dtype coverage matches the bridge contract: Float32 + Float64 only.
// Half / BFloat16 Adam state is numerically unsafe for the `g²`
// second-moment term (rapid underflow on typical gradient magnitudes);
// the M2 exit bar expects full-precision optimizer state. Float
// parameters with Float grads stay on the FP32 kernel.

#include <cstdint>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/Optim.hpp"
#include "tesseract/utils/Logging.hpp"

#include "KernelUtils.cuh"

namespace tesseract::cuda::detail {

namespace {

template <typename T>
__global__ void adam_step_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    T* __restrict__ m_buf,
    T* __restrict__ v_buf,
    T lr, T b1, T b2, T eps,
    T ibc1, T ibc2,
    int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const T g = grad[i];
  T m_new = b1 * m_buf[i] + (T{1} - b1) * g;
  T v_new = b2 * v_buf[i] + (T{1} - b2) * g * g;
  m_buf[i] = m_new;
  v_buf[i] = v_new;

  const T m_hat = m_new * ibc1;
  const T v_hat = v_new * ibc2;
  // `sqrt` picks up the right overload via ADL: `::sqrt(float)` for
  // `T=float` and `::sqrt(double)` for `T=double`. Avoid `std::sqrt`
  // in device code so we don't accidentally pull in a host-only
  // overload on older toolkits.
  param[i] = param[i] - lr * m_hat / (sqrt(v_hat) + eps);
}

}  // namespace

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
                      void* stream_handle) {
  if (n <= 0) return;
  TESSERACT_CHECK(param != nullptr && grad != nullptr &&
                      m_buf != nullptr && v_buf != nullptr,
                  "[tesseract] launch_adam_step: null buffer pointer");
  TESSERACT_CHECK(bc1 > 0.0 && bc2 > 0.0,
                  "[tesseract] launch_adam_step: bias-correction factors "
                  "must be positive (got bc1={}, bc2={})", bc1, bc2);

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  const int grid = launch_grid(n);

  const double ibc1 = 1.0 / bc1;
  const double ibc2 = 1.0 / bc2;

  switch (dtype) {
    case DType::Float32:
      adam_step_kernel<float><<<grid, kBlockSize, 0, stream>>>(
          static_cast<float*>(param),
          static_cast<const float*>(grad),
          static_cast<float*>(m_buf),
          static_cast<float*>(v_buf),
          static_cast<float>(lr), static_cast<float>(beta1),
          static_cast<float>(beta2), static_cast<float>(eps),
          static_cast<float>(ibc1), static_cast<float>(ibc2),
          n);
      break;
    case DType::Float64:
      adam_step_kernel<double><<<grid, kBlockSize, 0, stream>>>(
          static_cast<double*>(param),
          static_cast<const double*>(grad),
          static_cast<double*>(m_buf),
          static_cast<double*>(v_buf),
          lr, beta1, beta2, eps, ibc1, ibc2, n);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA Adam on dtype {} is not implemented in M2I "
          "— optimizer state uses full-precision accumulation (Float32 "
          "or Float64). Cast params + grads to Float32 on host first.",
          dtype_name(dtype)));
  }
  check_launch("adam-step");
}

}  // namespace tesseract::cuda::detail

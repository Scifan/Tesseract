// M4 perf-closeout Phase 5 — causal depthwise conv1d kernel for Mamba.
// See include/tesseract/cuda/detail/CausalConv1d.hpp.

#include <cstdint>

#include <cuda_runtime.h>

#include "tesseract/cuda/detail/CausalConv1d.hpp"

#include "KernelUtils.cuh"

namespace tesseract::cuda::detail {

namespace {

__global__ void causal_conv1d_kernel(int64_t B, int64_t L, int64_t C, int K,
                                     const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     const float* __restrict__ bias,
                                     float* __restrict__ out) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * L * C) return;
  const int64_t d = idx % C;
  const int64_t t = (idx / C) % L;
  const int64_t b = idx / (C * L);

  float acc = bias ? bias[d] : 0.0f;
  // out[b,t,d] = Σ_k w[d,k] · x[b, t-(K-1)+k, d]
  for (int k = 0; k < K; ++k) {
    const int64_t src_t = t - (K - 1) + k;
    if (src_t >= 0) {
      acc += weight[d * K + k] * x[(b * L + src_t) * C + d];
    }
  }
  out[idx] = acc;
}

}  // namespace

void launch_causal_conv1d(int device_index, int64_t B, int64_t L,
                          int64_t channels, int64_t K, const float* x,
                          const float* weight, const float* bias, float* out,
                          void* stream_handle) {
  const int64_t total = B * L * channels;
  if (total == 0) return;
  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  constexpr int TPB = 256;
  const unsigned blocks = static_cast<unsigned>((total + TPB - 1) / TPB);
  causal_conv1d_kernel<<<blocks, TPB, 0, stream>>>(
      B, L, channels, static_cast<int>(K), x, weight, bias, out);
  check_launch("causal_conv1d_kernel");
}

}  // namespace tesseract::cuda::detail

// M4 perf-closeout Phase 4 — device-side MoE top-k routing kernel.
//
// One block per token, `E` threads (E <= 256). Each thread owns one expert
// logit. The block:
//   1. block-max over logits (numerically stable softmax),
//   2. exp(logit - max) per slot, block-sum → full softmax probs,
//   3. iterative top-k selection: k rounds of "pick current max
//      (value-desc, index-asc tie-break), mark it, exclude it",
//   4. renormalize the k winners' probs to sum 1 → gates; write mask.
//
// Steps 1–2 give the same probs as the host `softmax`; step 3 the same
// winners as the host `partial_sort` tie-break; step 4 the same gate
// renormalization as `masked / sum(masked)`. Parity with the CPU reference
// is therefore exact up to fp rounding.

#include <cstdint>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "tesseract/cuda/detail/MoeRoute.hpp"

#include "KernelUtils.cuh"

namespace tesseract::cuda::detail {

namespace {

constexpr int kMaxExperts = 256;

__global__ void moe_route_kernel(int64_t T, int E, int k,
                                 const float* __restrict__ logits,
                                 float* __restrict__ gates,
                                 float* __restrict__ mask) {
  const int64_t t = blockIdx.x;
  if (t >= T) return;
  const int e = threadIdx.x;

  __shared__ float s_logit[kMaxExperts];
  __shared__ float s_prob[kMaxExperts];
  __shared__ int   s_chosen[kMaxExperts];  // 0/1 selected
  __shared__ float s_red[kMaxExperts];     // reduction scratch

  const float lg = (e < E) ? logits[t * E + e] : -CUDART_INF_F;
  if (e < E) {
    s_logit[e] = lg;
    s_chosen[e] = 0;
  }
  __syncthreads();

  // --- block max over logits ---
  s_red[e] = (e < E) ? lg : -CUDART_INF_F;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (e < s) s_red[e] = fmaxf(s_red[e], s_red[e + s]);
    __syncthreads();
  }
  const float mx = s_red[0];
  __syncthreads();

  // --- exp(logit - max), block sum ---
  const float ex = (e < E) ? __expf(lg - mx) : 0.0f;
  s_red[e] = ex;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (e < s) s_red[e] += s_red[e + s];
    __syncthreads();
  }
  const float denom = s_red[0];
  if (e < E) {
    s_prob[e] = ex / denom;     // full softmax prob
    gates[t * E + e] = 0.0f;
    mask[t * E + e] = 0.0f;
  }
  __syncthreads();

  // --- k rounds of argmax (value-desc, index-asc), single-threaded ---
  // k is tiny (1–2 typically); doing it on thread 0 avoids a tricky
  // index-tie-break parallel reduction and keeps exact parity.
  if (e == 0) {
    float chosen_sum = 0.0f;
    for (int r = 0; r < k; ++r) {
      float best = -CUDART_INF_F;
      int best_i = -1;
      for (int j = 0; j < E; ++j) {
        if (s_chosen[j]) continue;
        const float v = s_logit[j];
        if (v > best) { best = v; best_i = j; }  // index-asc on ties (strict >)
      }
      if (best_i >= 0) {
        s_chosen[best_i] = 1;
        chosen_sum += s_prob[best_i];
      }
    }
    s_red[0] = chosen_sum;
  }
  __syncthreads();

  // --- renormalize winners → gates; write mask ---
  const float gsum = s_red[0];
  if (e < E && s_chosen[e]) {
    mask[t * E + e] = 1.0f;
    gates[t * E + e] = (gsum > 0.0f) ? (s_prob[e] / gsum) : 0.0f;
  }
}

}  // namespace

void launch_moe_route(int device_index, int64_t num_tokens, int64_t num_experts,
                      int64_t top_k, const float* logits, float* gates,
                      float* mask, void* stream_handle) {
  if (num_tokens == 0) return;
  TESSERACT_CHECK(num_experts >= 1 && num_experts <= kMaxExperts,
                  "[tesseract] launch_moe_route: num_experts={} out of range "
                  "[1, {}]", num_experts, kMaxExperts);
  TESSERACT_CHECK(top_k >= 1 && top_k <= num_experts,
                  "[tesseract] launch_moe_route: top_k={} out of range [1, {}]",
                  top_k, num_experts);

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  // Round threads up to a power of two so the tree reductions are clean.
  int threads = 1;
  while (threads < num_experts) threads <<= 1;
  const dim3 grid(static_cast<unsigned>(num_tokens));
  moe_route_kernel<<<grid, threads, 0, stream>>>(
      num_tokens, static_cast<int>(num_experts), static_cast<int>(top_k),
      logits, gates, mask);
  check_launch("moe_route_kernel");
}

}  // namespace tesseract::cuda::detail

// M4 perf-closeout Phase 4 — fully fused GPU MoE inference forward.
// See include/tesseract/cuda/detail/MoeForward.hpp for the pipeline.

#include <cstdint>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fmt/format.h>

#include "tesseract/cuda/detail/MoeForward.hpp"
#include "tesseract/utils/Logging.hpp"

#include "KernelUtils.cuh"

namespace tesseract::cuda::detail {

namespace {

void cck(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw DeviceError(fmt::format("[tesseract] MoE: {} failed: {}", what,
                                  cudaGetErrorString(err)));
  }
}

void ck(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    throw DeviceError(fmt::format("[tesseract] MoE grouped-GEMM: {} failed "
                                  "(cublas status {})", what,
                                  static_cast<int>(st)));
  }
}

// Per-device legacy cuBLAS handle (grouped-batched lives on the v2 API).
cublasHandle_t moe_handle(int device_index) {
  static cublasHandle_t handles[16] = {nullptr};
  TESSERACT_CHECK(device_index >= 0 && device_index < 16,
                  "MoE: device index {} out of range", device_index);
  if (handles[device_index] == nullptr) {
    DeviceGuard g(device_index);
    ck(cublasCreate(&handles[device_index]), "cublasCreate");
  }
  return handles[device_index];
}

// --- routing → permutation kernels -----------------------------------------

__global__ void histogram_kernel(int64_t T, int E, const float* __restrict__ mask,
                                 int* __restrict__ counts) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * E) return;
  if (mask[idx] != 0.0f) {
    const int e = static_cast<int>(idx % E);
    atomicAdd(&counts[e], 1);
  }
}

// One thread per token: walk its experts in ascending order so within an
// expert the rows stay token-ascending (matches the host path's stable order;
// not required for correctness since each row is independent, but keeps the
// permutation deterministic).
__global__ void scatter_kernel(int64_t T, int E,
                               const float* __restrict__ mask,
                               const float* __restrict__ gates,
                               const int* __restrict__ offsets,
                               int* __restrict__ cursor,
                               int* __restrict__ perm_token,
                               float* __restrict__ perm_gate) {
  const int64_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  for (int e = 0; e < E; ++e) {
    if (mask[t * E + e] != 0.0f) {
      const int pos = offsets[e] + atomicAdd(&cursor[e], 1);
      perm_token[pos] = static_cast<int>(t);
      perm_gate[pos] = gates[t * E + e];
    }
  }
}

__global__ void gather_rows_kernel(int64_t Ntot, int64_t D,
                                   const int* __restrict__ perm_token,
                                   const float* __restrict__ x,
                                   float* __restrict__ x_perm) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Ntot * D) return;
  const int64_t row = i / D, col = i % D;
  x_perm[i] = x[static_cast<int64_t>(perm_token[row]) * D + col];
}

__device__ __forceinline__ float siluf(float z) {
  return z / (1.0f + __expf(-z));
}

__global__ void silu_mul_kernel(int64_t n, const float* __restrict__ gate,
                                const float* __restrict__ up,
                                float* __restrict__ out) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = siluf(gate[i]) * up[i];
}

__global__ void combine_kernel(int64_t Ntot, int64_t D,
                               const int* __restrict__ perm_token,
                               const float* __restrict__ perm_gate,
                               const float* __restrict__ y_perm,
                               float* __restrict__ y) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Ntot * D) return;
  const int64_t row = i / D, col = i % D;
  const int t = perm_token[row];
  atomicAdd(&y[static_cast<int64_t>(t) * D + col], perm_gate[row] * y_perm[i]);
}

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Per-device grow-on-demand scratch (avoids cudaMalloc on the hot path).
struct Scratch {
  void* buf = nullptr;
  size_t cap = 0;
  void ensure(size_t bytes, int dev) {
    if (bytes <= cap) return;
    DeviceGuard g(dev);
    if (buf) (void)cudaFree(buf);
    cck(cudaMalloc(&buf, bytes), "MoE scratch malloc");
    cap = bytes;
  }
};

Scratch& scratch_for(int dev) {
  static Scratch s[16];
  return s[dev];
}

// Run one grouped GEMM: per (non-empty) expert g, C_g[m_g, N] = A_g[m_g, K] @ W_g[N, K]^T,
// all row-major. `m_g` = rows of expert g. cuBLAS is column-major, so we
// compute Ccm[N, m_g] = Wcm^T @ Acm: A_param = weight (op=T), B_param =
// activation (op=N). One group per expert (group_size=1) with its own m.
void grouped_gemm(cublasHandle_t h, int g_count,
                  const std::vector<int>& m,   // rows per group (= n_e)
                  int N, int K,
                  const std::vector<const float*>& A_act,   // [n_e, K] row-major
                  const std::vector<const float*>& W,       // [N, K] row-major
                  const std::vector<float*>& C_out,         // [n_e, N] row-major
                  int dev) {
  if (g_count == 0) return;
  std::vector<cublasOperation_t> ta(g_count, CUBLAS_OP_T);  // on weight
  std::vector<cublasOperation_t> tb(g_count, CUBLAS_OP_N);  // on activation
  std::vector<int> mm(g_count), nn(g_count), kk(g_count);
  std::vector<int> lda(g_count), ldb(g_count), ldc(g_count), gsize(g_count, 1);
  std::vector<float> alpha(g_count, 1.0f), beta(g_count, 0.0f);
  std::vector<const void*> Aptr(g_count), Bptr(g_count);
  std::vector<void*> Cptr(g_count);
  for (int e = 0; e < g_count; ++e) {
    mm[e] = N;        // cublas m = N (weight rows)
    nn[e] = m[e];     // cublas n = n_e (activation rows)
    kk[e] = K;
    lda[e] = K;       // weight col-major ld = K
    ldb[e] = K;       // activation col-major ld = K
    ldc[e] = N;       // output col-major ld = N
    Aptr[e] = W[e];   // A_param = weight
    Bptr[e] = A_act[e];
    Cptr[e] = C_out[e];
  }
  // Pointer arrays must live in device memory for the batched API.
  Scratch& sp = scratch_for(dev);
  const size_t ptr_bytes = static_cast<size_t>(g_count) * sizeof(void*) * 3;
  sp.ensure(ptr_bytes, dev);
  void** dA = static_cast<void**>(sp.buf);
  void** dB = dA + g_count;
  void** dC = dB + g_count;
  cck(cudaMemcpy(dA, Aptr.data(), g_count * sizeof(void*),
                        cudaMemcpyHostToDevice), "MoE Aptr");
  cck(cudaMemcpy(dB, Bptr.data(), g_count * sizeof(void*),
                        cudaMemcpyHostToDevice), "MoE Bptr");
  cck(cudaMemcpy(dC, Cptr.data(), g_count * sizeof(void*),
                        cudaMemcpyHostToDevice), "MoE Cptr");
  ck(cublasGemmGroupedBatchedEx(
         h, ta.data(), tb.data(), mm.data(), nn.data(), kk.data(),
         alpha.data(), const_cast<const void* const*>(dA), CUDA_R_32F, lda.data(),
         const_cast<const void* const*>(dB), CUDA_R_32F, ldb.data(), beta.data(),
         dC, CUDA_R_32F, ldc.data(), g_count, gsize.data(), CUBLAS_COMPUTE_32F),
     "GemmGroupedBatchedEx");
}

}  // namespace

bool launch_moe_grouped_ffn(int device_index, int64_t T, int64_t D,
                            int64_t dff, int64_t E, int64_t k,
                            const float* x, const float* gates,
                            const float* mask, const float* const* Wg,
                            const float* const* Wu, const float* const* Wd,
                            float* y, void* stream_handle) {
  if (T == 0) return true;
  if (E > 16) return false;  // legacy-handle path; large-E falls back

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  cublasHandle_t h = moe_handle(device_index);
  ck(cublasSetStream(h, stream), "cublasSetStream");

  const int64_t Ntot = T * k;

  // Workspace layout (single grow-on-demand block, separate from the pointer
  // scratch used inside grouped_gemm). Sizes in floats/ints.
  //   counts[E], offsets[E+1], cursor[E]              (ints)
  //   perm_token[Ntot]                                (ints)
  //   perm_gate[Ntot]                                 (floats)
  //   x_perm[Ntot*D], gate_buf[Ntot*dff],
  //   up_buf[Ntot*dff], y_perm[Ntot*D]                (floats)
  static Scratch work[16];
  Scratch& w = work[device_index];
  const size_t n_int = (E) + (E + 1) + (E) + Ntot;            // ints
  const size_t n_flt = Ntot + 2 * (Ntot * dff) + 2 * (Ntot * D);  // floats
  w.ensure(n_int * sizeof(int) + n_flt * sizeof(float), device_index);

  auto* base_i = static_cast<int*>(w.buf);
  int* d_counts = base_i;
  int* d_offsets = d_counts + E;
  int* d_cursor = d_offsets + (E + 1);
  int* d_perm_token = d_cursor + E;
  auto* base_f = reinterpret_cast<float*>(d_perm_token + Ntot);
  float* d_perm_gate = base_f;
  float* d_x_perm = d_perm_gate + Ntot;
  float* d_gate_buf = d_x_perm + Ntot * D;
  float* d_up_buf = d_gate_buf + Ntot * dff;
  float* d_y_perm = d_up_buf + Ntot * dff;

  cck(cudaMemsetAsync(d_counts, 0, E * sizeof(int), stream), "zero counts");
  cck(cudaMemsetAsync(d_cursor, 0, E * sizeof(int), stream), "zero cursor");
  cck(cudaMemsetAsync(y, 0, T * D * sizeof(float), stream), "zero y");

  constexpr int TPB = 256;
  histogram_kernel<<<static_cast<unsigned>(ceil_div(T * E, TPB)), TPB, 0, stream>>>(
      T, static_cast<int>(E), mask, d_counts);
  check_launch("moe_histogram");

  // Exclusive prefix sum on host (E is tiny). Need counts on host anyway for
  // the grouped-GEMM group sizes / per-expert base pointers.
  std::vector<int> counts(static_cast<size_t>(E));
  cck(cudaMemcpyAsync(counts.data(), d_counts, E * sizeof(int),
                             cudaMemcpyDeviceToHost, stream), "counts D2H");
  cck(cudaStreamSynchronize(stream), "counts sync");
  std::vector<int> offsets(static_cast<size_t>(E + 1), 0);
  for (int e = 0; e < E; ++e) offsets[e + 1] = offsets[e] + counts[e];
  cck(cudaMemcpyAsync(d_offsets, offsets.data(), (E + 1) * sizeof(int),
                             cudaMemcpyHostToDevice, stream), "offsets H2D");

  scatter_kernel<<<static_cast<unsigned>(ceil_div(T, TPB)), TPB, 0, stream>>>(
      T, static_cast<int>(E), mask, gates, d_offsets, d_cursor, d_perm_token,
      d_perm_gate);
  check_launch("moe_scatter");

  gather_rows_kernel<<<static_cast<unsigned>(ceil_div(Ntot * D, TPB)), TPB, 0,
                       stream>>>(Ntot, D, d_perm_token, x, d_x_perm);
  check_launch("moe_gather");

  // Build per-expert (non-empty) pointer/size lists for the grouped GEMMs.
  std::vector<int> m;
  std::vector<const float*> A_gate, A_up, A_down, W_gate, W_up, W_down;
  std::vector<float*> C_gate, C_up, C_down;
  for (int e = 0; e < E; ++e) {
    const int n_e = counts[e];
    if (n_e == 0) continue;
    const int64_t off = offsets[e];
    m.push_back(n_e);
    A_gate.push_back(d_x_perm + off * D);
    A_up.push_back(d_x_perm + off * D);
    W_gate.push_back(Wg[e]);
    W_up.push_back(Wu[e]);
    C_gate.push_back(d_gate_buf + off * dff);
    C_up.push_back(d_up_buf + off * dff);
    A_down.push_back(d_gate_buf + off * dff);  // h written back into gate_buf
    W_down.push_back(Wd[e]);
    C_down.push_back(d_y_perm + off * D);
  }
  const int gc = static_cast<int>(m.size());

  // gate = x_perm @ Wg^T  ;  up = x_perm @ Wu^T   (N=dff, K=D)
  grouped_gemm(h, gc, m, static_cast<int>(dff), static_cast<int>(D),
               A_gate, W_gate, C_gate, device_index);
  grouped_gemm(h, gc, m, static_cast<int>(dff), static_cast<int>(D),
               A_up, W_up, C_up, device_index);

  // h = silu(gate) * up  (in-place into gate_buf, which down reads as A)
  silu_mul_kernel<<<static_cast<unsigned>(ceil_div(Ntot * dff, TPB)), TPB, 0,
                    stream>>>(Ntot * dff, d_gate_buf, d_up_buf, d_gate_buf);
  check_launch("moe_silu");

  // y_perm = h @ Wd^T   (N=D, K=dff)
  grouped_gemm(h, gc, m, static_cast<int>(D), static_cast<int>(dff),
               A_down, W_down, C_down, device_index);

  combine_kernel<<<static_cast<unsigned>(ceil_div(Ntot * D, TPB)), TPB, 0,
                   stream>>>(Ntot, D, d_perm_token, d_perm_gate, d_y_perm, y);
  check_launch("moe_combine");
  return true;
}

}  // namespace tesseract::cuda::detail

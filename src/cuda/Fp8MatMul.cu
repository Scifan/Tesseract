// M4 perf-closeout Phase 3 — FP8 (E4M3) tensor-core GEMM (cuBLASLt).
//
// See include/tesseract/cuda/detail/Fp8MatMul.hpp for the layout mapping.
// The whole point of this TU is to win the dense GEMM / linear-layer
// battleline against PyTorch eager (which runs FP16 tensor cores on Ada):
// FP8 doubles the tensor-core math throughput, so a correctly-set-up FP8
// GEMM beats an FP16 GEMM of the same logical M·N·K shape.

#include "KernelUtils.cuh"

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <fmt/format.h>

#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/cuda/detail/Fp8MatMul.hpp"

namespace tesseract::cuda::detail {

namespace {

[[noreturn]] void throw_cublas(cublasStatus_t st, const char* what) {
  throw tesseract::DeviceError(fmt::format(
      "[tesseract] cuBLASLt FP8 {} failed: status {}", what,
      static_cast<int>(st)));
}
void ck(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS) throw_cublas(st, what);
}

__global__ void f32_to_e4m3_kernel(const float* __restrict__ src,
                                   __nv_fp8_storage_t* __restrict__ dst,
                                   int64_t n) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    dst[i] = __nv_cvt_float_to_fp8(src[i], __NV_SATFINITE, __NV_E4M3);
  }
}

cublasLtHandle_t fp8_handle(int device_index) {
  static cublasLtHandle_t handles[16] = {nullptr};
  const int slot = device_index & 15;
  if (!handles[slot]) {
    int prev = -1;
    cudaGetDevice(&prev);
    cudaSetDevice(device_index);
    ck(cublasLtCreate(&handles[slot]), "Create");
    if (prev >= 0) cudaSetDevice(prev);
  }
  return handles[slot];
}

constexpr std::size_t kFp8WorkspaceBytes = 4 * 1024 * 1024;

void* fp8_workspace(int device_index) {
  static void* ws[16] = {nullptr};
  const int slot = device_index & 15;
  if (!ws[slot]) cudaMalloc(&ws[slot], kFp8WorkspaceBytes);
  return ws[slot];
}

// Two device floats per device, reused for the A/B per-tensor scales.
float* fp8_scale_buf(int device_index) {
  static float* sc[16] = {nullptr};
  const int slot = device_index & 15;
  if (!sc[slot]) cudaMalloc(&sc[slot], 2 * sizeof(float));
  return sc[slot];
}

}  // namespace

bool fp8_gemm_supported(int device_index) {
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_index);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_index);
  return (major * 10 + minor) >= 89;
}

void quantize_to_fp8_e4m3(const float* src, void* dst, int64_t n,
                          void* stream_handle) {
  if (n <= 0) return;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  f32_to_e4m3_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
      src, static_cast<__nv_fp8_storage_t*>(dst), n);
  check_launch("f32_to_e4m3_kernel");
}

void launch_fp8_linear(int device_index, int64_t M, int64_t N, int64_t K,
                       const void* X, const void* W, void* Y,
                       float x_scale, float w_scale, void* stream_handle) {
  DeviceGuard guard(device_index);
  cublasLtHandle_t h = fp8_handle(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  // Column-major view (cuBLASLt default order). We compute
  //   Y_cm[N,M] = op(A)·op(B)  with op(A)=T, op(B)=N
  // where A = W (col-major [K,N] ld=K ≡ row-major [N,K]) and
  //       B = X (col-major [K,M] ld=K ≡ row-major [M,K]).
  // Y_cm[N,M] col-major (ld=N) is bit-identical to row-major Y[M,N].
  cublasLtMatmulDesc_t desc = nullptr;
  ck(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "DescCreate");
  const cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
  ck(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)), "TRANSA");
  ck(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)), "TRANSB");

  // Per-tensor scales live in device memory for FP8 matmul. A=W → A_scale
  // is w_scale; B=X → B_scale is x_scale.
  float* d_scales = fp8_scale_buf(device_index);
  float* d_ws = d_scales;      // A (=W) scale
  float* d_xs = d_scales + 1;  // B (=X) scale
  cudaMemcpyAsync(d_ws, &w_scale, sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_xs, &x_scale, sizeof(float), cudaMemcpyHostToDevice, stream);
  ck(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_ws, sizeof(d_ws)), "A_SCALE");
  ck(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_xs, sizeof(d_xs)), "B_SCALE");

  cublasLtMatrixLayout_t aL = nullptr, bL = nullptr, cL = nullptr;
  // A = W: col-major [K, N], ld = K, E4M3.
  ck(cublasLtMatrixLayoutCreate(&aL, CUDA_R_8F_E4M3, K, N, K), "aL");
  // B = X: col-major [K, M], ld = K, E4M3.
  ck(cublasLtMatrixLayoutCreate(&bL, CUDA_R_8F_E4M3, K, M, K), "bL");
  // C/D = Y_cm: col-major [N, M], ld = N, BF16.
  ck(cublasLtMatrixLayoutCreate(&cL, CUDA_R_16BF, N, M, N), "cL");

  void* ws = fp8_workspace(device_index);
  const std::size_t ws_bytes = kFp8WorkspaceBytes;

  cublasLtMatmulPreference_t pref = nullptr;
  ck(cublasLtMatmulPreferenceCreate(&pref), "PrefCreate");
  ck(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &ws_bytes, sizeof(ws_bytes)), "PrefWS");
  cublasLtMatmulHeuristicResult_t hr{};
  int returned = 0;
  ck(cublasLtMatmulAlgoGetHeuristic(h, desc, aL, bL, cL, cL, pref, 1, &hr, &returned),
     "Heuristic");
  if (returned == 0) {
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(aL); cublasLtMatrixLayoutDestroy(bL);
    cublasLtMatrixLayoutDestroy(cL); cublasLtMatmulDescDestroy(desc);
    throw tesseract::DeviceError("[tesseract] cuBLASLt FP8: no algorithm for shape");
  }

  const float alpha = 1.0f, beta = 0.0f;
  ck(cublasLtMatmul(h, desc, &alpha,
                    W, aL,
                    X, bL,
                    &beta, Y, cL, Y, cL,
                    &hr.algo, ws, ws_bytes, stream), "Matmul");

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(aL); cublasLtMatrixLayoutDestroy(bL);
  cublasLtMatrixLayoutDestroy(cL); cublasLtMatmulDescDestroy(desc);
}

}  // namespace tesseract::cuda::detail

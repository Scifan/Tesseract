// M4 Track A2 (B-039): selective state-space scan (Mamba / S6) forward kernel.
//
// One thread per `(batch b, inner channel d)`. The thread carries the `N`-wide
// hidden state in a per-thread register array and runs the sequential
// `t`-recurrence:
//
//   for t in 0..L-1:
//     dA = exp(delta[b,t,d] * A[d,n])
//     h[n] = dA * h[n] + (delta[b,t,d] * B[b,t,n]) * u[b,t,d]
//     y[b,t,d] = Σ_n C[b,t,n] * h[n] + D[d] * u[b,t,d]
//
// Threads are independent (no cross-thread communication), so the launch is
// embarrassingly parallel across `B·D`. FP32 interior math regardless of
// storage dtype — matches the CPU reference's FP32 path within tolerance. The
// register state array caps `N <= 32` (validated host-side in the op layer).

#include "KernelUtils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/SelectiveScan.hpp"

namespace tesseract::cuda::detail {

namespace {

constexpr int kMaxState = 32;

__device__ __forceinline__ float ss_to_float(float v)         { return v; }
__device__ __forceinline__ float ss_to_float(double v)        { return static_cast<float>(v); }
__device__ __forceinline__ float ss_to_float(__half v)        { return __half2float(v); }
__device__ __forceinline__ float ss_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T ss_from_float(float v);
template <> __device__ __forceinline__ float    ss_from_float<float>(float v)  { return v; }
template <> __device__ __forceinline__ double   ss_from_float<double>(float v) { return static_cast<double>(v); }
template <> __device__ __forceinline__ __half   ss_from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 ss_from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__global__ void selective_scan_kernel(
    const T* __restrict__ u, const T* __restrict__ delta,
    const T* __restrict__ A, const T* __restrict__ Bm, const T* __restrict__ Cm,
    const T* __restrict__ Dskip, const T* __restrict__ state_in,
    T* __restrict__ y, T* __restrict__ state_out,
    int64_t B, int64_t L, int64_t D, int64_t N) {
  const int64_t bd = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (bd >= B * D) return;
  const int64_t b = bd / D;
  const int64_t d = bd % D;

  float h[kMaxState];
  for (int64_t n = 0; n < N; ++n)
    h[n] = state_in ? ss_to_float(state_in[(b * D + d) * N + n]) : 0.0f;

  for (int64_t t = 0; t < L; ++t) {
    const int64_t bt = b * L + t;
    const float dt    = ss_to_float(delta[bt * D + d]);
    const float u_btd = ss_to_float(u[bt * D + d]);
    float y_acc = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
      const float a  = ss_to_float(A[d * N + n]);
      const float dA = __expf(dt * a);
      const float Bn = ss_to_float(Bm[bt * N + n]);
      h[n] = dA * h[n] + (dt * Bn) * u_btd;
      const float Cn = ss_to_float(Cm[bt * N + n]);
      y_acc += Cn * h[n];
    }
    y_acc += ss_to_float(Dskip[d]) * u_btd;
    y[bt * D + d] = ss_from_float<T>(y_acc);
  }
  for (int64_t n = 0; n < N; ++n)
    state_out[(b * D + d) * N + n] = ss_from_float<T>(h[n]);
}

// ===========================================================================
// Chunkwise parallel scan (FP32 prefill, state_in == null).
//
// The per-(b,d,n) recurrence h_t = dA_t·h_{t-1} + dBu_t is a first-order
// linear recurrence, so it composes: over a chunk the map is
// (P, s) where P = Π dA_t and s = the chunk's local final state from a
// zero entry. We split L into `C` chunks and:
//   pass 1 (B·D·C threads): each chunk computes its (P[n], s[n]) from zero;
//   combine (B·D threads):  sequentially fold chunk maps into per-chunk entry
//                           states E[c][n] (C is small, so this is cheap);
//   pass 2 (B·D·C threads): each chunk re-scans from E[c] and writes y.
// Parallelism jumps from B·D to B·D·C and the serial length drops from L to
// L/C — the key to long-context prefill throughput. Numerically identical to
// the sequential scan up to fp reassociation (the per-chunk products are the
// same factors in the same order).
// ===========================================================================

__global__ void scan_chunk_local_kernel(
    const float* __restrict__ u, const float* __restrict__ delta,
    const float* __restrict__ A, const float* __restrict__ Bm,
    int64_t B, int64_t L, int64_t D, int64_t N, int64_t chunk, int64_t C,
    float* __restrict__ Pbuf, float* __restrict__ Sbuf) {
  const int64_t bdc = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (bdc >= B * D * C) return;
  const int64_t c = bdc % C;
  const int64_t d = (bdc / C) % D;
  const int64_t b = bdc / (C * D);
  const int64_t t0 = c * chunk;
  const int64_t t1 = min(t0 + chunk, L);

  float P[kMaxState], s[kMaxState];
  for (int64_t n = 0; n < N; ++n) { P[n] = 1.0f; s[n] = 0.0f; }
  for (int64_t t = t0; t < t1; ++t) {
    const int64_t bt = b * L + t;
    const float dtv = delta[bt * D + d];
    const float u_btd = u[bt * D + d];
    for (int64_t n = 0; n < N; ++n) {
      const float dA = __expf(dtv * A[d * N + n]);
      const float Bn = Bm[bt * N + n];
      s[n] = dA * s[n] + (dtv * Bn) * u_btd;
      P[n] = dA * P[n];
    }
  }
  for (int64_t n = 0; n < N; ++n) {
    Pbuf[bdc * N + n] = P[n];
    Sbuf[bdc * N + n] = s[n];
  }
}

__global__ void scan_combine_kernel(
    int64_t B, int64_t D, int64_t N, int64_t C,
    const float* __restrict__ Pbuf, const float* __restrict__ Sbuf,
    float* __restrict__ Ebuf) {
  const int64_t bd = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (bd >= B * D) return;
  float carry[kMaxState];
  for (int64_t n = 0; n < N; ++n) carry[n] = 0.0f;
  for (int64_t c = 0; c < C; ++c) {
    const int64_t bdc = bd * C + c;
    for (int64_t n = 0; n < N; ++n) {
      Ebuf[bdc * N + n] = carry[n];  // entry state for chunk c
      carry[n] = Pbuf[bdc * N + n] * carry[n] + Sbuf[bdc * N + n];
    }
  }
}

__global__ void scan_chunk_output_kernel(
    const float* __restrict__ u, const float* __restrict__ delta,
    const float* __restrict__ A, const float* __restrict__ Bm,
    const float* __restrict__ Cm, const float* __restrict__ Dskip,
    const float* __restrict__ Ebuf,
    int64_t B, int64_t L, int64_t D, int64_t N, int64_t chunk, int64_t C,
    float* __restrict__ y, float* __restrict__ state_out) {
  const int64_t bdc = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (bdc >= B * D * C) return;
  const int64_t c = bdc % C;
  const int64_t d = (bdc / C) % D;
  const int64_t b = bdc / (C * D);
  const int64_t t0 = c * chunk;
  const int64_t t1 = min(t0 + chunk, L);

  float h[kMaxState];
  for (int64_t n = 0; n < N; ++n) h[n] = Ebuf[bdc * N + n];
  for (int64_t t = t0; t < t1; ++t) {
    const int64_t bt = b * L + t;
    const float dtv = delta[bt * D + d];
    const float u_btd = u[bt * D + d];
    float y_acc = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
      const float dA = __expf(dtv * A[d * N + n]);
      const float Bn = Bm[bt * N + n];
      h[n] = dA * h[n] + (dtv * Bn) * u_btd;
      y_acc += Cm[bt * N + n] * h[n];
    }
    y_acc += Dskip[d] * u_btd;
    y[bt * D + d] = y_acc;
  }
  // The final chunk holds each (b,d)'s end-of-sequence state.
  if (c == C - 1 && state_out) {
    for (int64_t n = 0; n < N; ++n)
      state_out[(b * D + d) * N + n] = h[n];
  }
}

struct ScanScratch {
  void* buf = nullptr;
  size_t cap = 0;
  void ensure(size_t bytes) {
    if (bytes <= cap) return;
    if (buf) (void)cudaFree(buf);
    if (cudaMalloc(&buf, bytes) != cudaSuccess)
      throw DeviceError("[tesseract] selective_scan chunk scratch malloc failed");
    cap = bytes;
  }
};

bool launch_chunked_f32(int64_t B, int64_t L, int64_t D, int64_t N,
                        const void* u, const void* delta, const void* A,
                        const void* Bm, const void* Cm, const void* Dskip,
                        void* y, void* state_out, cudaStream_t stream) {
  // Pick chunk count so each chunk is ~128 steps; only worth it for long L.
  constexpr int64_t kChunkLen = 128;
  const int64_t C = (L + kChunkLen - 1) / kChunkLen;
  if (C < 2) return false;  // short sequence: caller uses the sequential path
  const int64_t chunk = (L + C - 1) / C;

  static ScanScratch sc;
  const size_t per = static_cast<size_t>(B * D * C * N);
  sc.ensure(3 * per * sizeof(float));
  float* P = static_cast<float*>(sc.buf);
  float* S = P + per;
  float* E = S + per;

  const auto* uf = static_cast<const float*>(u);
  const auto* df = static_cast<const float*>(delta);
  const auto* af = static_cast<const float*>(A);
  const auto* bf = static_cast<const float*>(Bm);
  const auto* cf = static_cast<const float*>(Cm);
  const auto* dskf = static_cast<const float*>(Dskip);

  const int tpb = kBlockSize;
  const unsigned g_bdc = static_cast<unsigned>((B * D * C + tpb - 1) / tpb);
  const unsigned g_bd = static_cast<unsigned>((B * D + tpb - 1) / tpb);
  scan_chunk_local_kernel<<<g_bdc, tpb, 0, stream>>>(
      uf, df, af, bf, B, L, D, N, chunk, C, P, S);
  scan_combine_kernel<<<g_bd, tpb, 0, stream>>>(B, D, N, C, P, S, E);
  scan_chunk_output_kernel<<<g_bdc, tpb, 0, stream>>>(
      uf, df, af, bf, cf, dskf, E, B, L, D, N, chunk, C,
      static_cast<float*>(y), static_cast<float*>(state_out));
  return true;
}

template <typename T>
void launch_typed(int64_t B, int64_t L, int64_t D, int64_t N,
                  const void* u, const void* delta, const void* A,
                  const void* Bm, const void* Cm, const void* Dskip,
                  const void* state_in, void* y, void* state_out,
                  cudaStream_t stream) {
  const dim3 grid(launch_grid(B * D));
  const dim3 block(kBlockSize);
  selective_scan_kernel<T><<<grid, block, 0, stream>>>(
      static_cast<const T*>(u), static_cast<const T*>(delta),
      static_cast<const T*>(A), static_cast<const T*>(Bm),
      static_cast<const T*>(Cm), static_cast<const T*>(Dskip),
      static_cast<const T*>(state_in), static_cast<T*>(y),
      static_cast<T*>(state_out), B, L, D, N);
}

}  // namespace

void launch_selective_scan(DType dtype, int device_index,
                           int64_t B, int64_t L, int64_t D, int64_t N,
                           const void* u, const void* delta, const void* A,
                           const void* Bm, const void* Cm, const void* Dskip,
                           const void* state_in,
                           void* y, void* state_out, void* stream_handle) {
  if (B * L * D == 0) return;
  if (N > kMaxState) {
    throw DeviceError(fmt::format(
        "[tesseract] CUDA selective_scan d_state N={} exceeds kernel cap {}",
        N, kMaxState));
  }
  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  switch (dtype) {
    case DType::Float32:
      // Prefill (no entry state) over a long sequence: the chunkwise parallel
      // scan lifts parallelism from B·D to B·D·C and cuts the serial length to
      // L/C. Short / decode (L small) and stateful steps keep the sequential
      // per-thread path.
      if (state_in == nullptr &&
          launch_chunked_f32(B, L, D, N, u, delta, A, Bm, Cm, Dskip, y,
                             state_out, stream)) {
        break;
      }
      launch_typed<float>(B, L, D, N, u, delta, A, Bm, Cm, Dskip, state_in,
                          y, state_out, stream);
      break;
    case DType::Float64:
      launch_typed<double>(B, L, D, N, u, delta, A, Bm, Cm, Dskip, state_in,
                           y, state_out, stream);
      break;
    case DType::Float16:
      launch_typed<__half>(B, L, D, N, u, delta, A, Bm, Cm, Dskip, state_in,
                           y, state_out, stream);
      break;
    case DType::BFloat16:
      launch_typed<__nv_bfloat16>(B, L, D, N, u, delta, A, Bm, Cm, Dskip,
                                  state_in, y, state_out, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA selective_scan on dtype {} is not implemented "
          "(Float32 / Float64 / Float16 / BFloat16 only).",
          dtype_name(dtype)));
  }
  check_launch("selective_scan_kernel");
}

}  // namespace tesseract::cuda::detail

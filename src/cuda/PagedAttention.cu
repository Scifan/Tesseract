// Wave 11 (B-032+): fused ragged paged decode-attention kernel.
//
// One warp (32 threads) per `(request r, query-head h)` pair computes one
// decode query's attention output by streaming request r's KV prefix
// straight out of the shared physical pool `[num_blocks, Hkv, block_size,
// D]` via its block table — no gather, no `repeat_kv` materialization, no
// `[1, S_k]` score row written to HBM. GQA: head h reads KV head h/group.
//
// Per-lane Q fragment (scale folded), online (FlashAttention-2) softmax:
//   for each key j in [0, len):
//     s_j     = warp_reduce( Σ_d q[d]·K[block(j), h/group, slot(j), d] )
//     m_new   = max(m, s_j);  α = exp(m - m_new);  p = exp(s_j - m_new)
//     l       = l·α + p
//     o[d]    = o[d]·α + p·V[block(j), h/group, slot(j), d]
//     m       = m_new
//   O[d] = o[d] / l           (len == 0 ⇒ zero row)
//
// FP32 interior math regardless of storage dtype (FP32/FP16/BF16), matching
// the `ops::attention` / Wave 4.2 fused-attention numerical contract.
//
// Single instantiation at D_PER_LANE = D_MAX/32 = 4 (head_dim ≤ 128),
// guarding `d < D`; same structural choice as FusedAttention.cu.

#include "KernelUtils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/PagedAttention.hpp"

namespace tesseract::cuda::detail {

namespace {

__device__ __forceinline__ float pa_to_float(float v)         { return v; }
__device__ __forceinline__ float pa_to_float(__half v)        { return __half2float(v); }
__device__ __forceinline__ float pa_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T pa_from_float(float v);
template <> __device__ __forceinline__ float pa_from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ __half pa_from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__
__nv_bfloat16 pa_from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, offset);
  return v;
}

// Merge WARPS per-warp online-softmax states (held in shared memory) into a
// single normalized output row. Called by warp 0 after a __syncthreads().
// `sm_m[w]`, `sm_l[w]`, `sm_o[w][d]` hold each warp's running max / denom /
// numerator. Writes `O[rh, :]`.
template <typename T, int D_PER_LANE, int WARPS, int D_MAX>
__device__ __forceinline__ void pa_merge_and_write(
    const float (&sm_m)[WARPS], const float (&sm_l)[WARPS],
    const float (&sm_o)[WARPS][D_MAX], T* o_row, int lane, int Di) {
  float M = -INFINITY;
  #pragma unroll
  for (int w = 0; w < WARPS; ++w) M = fmaxf(M, sm_m[w]);
  float l_tot = 0.0f;
  if (M != -INFINITY) {
    #pragma unroll
    for (int w = 0; w < WARPS; ++w)
      if (sm_m[w] != -INFINITY) l_tot += __expf(sm_m[w] - M) * sm_l[w];
  }
  const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    if (d < Di) {
      float acc = 0.0f;
      if (M != -INFINITY) {
        #pragma unroll
        for (int w = 0; w < WARPS; ++w)
          if (sm_m[w] != -INFINITY) acc += __expf(sm_m[w] - M) * sm_o[w][d];
      }
      o_row[d] = pa_from_float<T>(acc * inv_l);
    }
  }
}

template <typename T, int D_PER_LANE, int WARPS, int D_MAX>
__global__ void paged_decode_attention_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_pool,
    const T* __restrict__ v_pool,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ lens,
    T* __restrict__ o,
    int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale) {
  const int rh      = blockIdx.x;          // r * H + h
  const int r       = static_cast<int>(rh / H);
  const int h       = static_cast<int>(rh % H);
  const int hkv     = h / group;
  const int warp_id = threadIdx.x >> 5;
  const int lane    = threadIdx.x & 31;
  const int Di      = static_cast<int>(D);
  const int len     = lens[r];

  // Per-lane Q fragment, scale folded in (so the score dot skips it).
  const T* q_row = q + static_cast<int64_t>(rh) * D;
  float q_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    q_frag[dd] = (d < Di) ? pa_to_float(q_row[d]) * scale : 0.0f;
  }

  float m_running = -INFINITY;
  float l_running = 0.0f;
  float o_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) o_frag[dd] = 0.0f;

  const int32_t* table = block_tables + static_cast<int64_t>(r) * max_logical;

  // Warp w strides the KV range: keeps all WARPS warps busy on one request.
  for (int j = warp_id; j < len; j += WARPS) {
    const int logical = static_cast<int>(j / block_size);
    const int slot    = static_cast<int>(j % block_size);
    int p = table[logical];
    if (p < 0 || p >= num_blocks) p = 0;  // defense in depth
    const int64_t row =
        ((static_cast<int64_t>(p) * Hkv + hkv) * block_size + slot) * D;
    const T* k_row = k_pool + row;
    const T* v_row = v_pool + row;

    float partial = 0.0f;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < Di) partial += q_frag[dd] * pa_to_float(k_row[d]);
    }
    const float s = warp_reduce_sum(partial);  // replicated across lanes

    const float new_m = fmaxf(m_running, s);
    const float alpha = (m_running == -INFINITY) ? 0.0f
                                                 : __expf(m_running - new_m);
    const float p_j   = __expf(s - new_m);
    l_running = l_running * alpha + p_j;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < Di) o_frag[dd] = o_frag[dd] * alpha + p_j * pa_to_float(v_row[d]);
    }
    m_running = new_m;
  }

  __shared__ float sm_m[WARPS];
  __shared__ float sm_l[WARPS];
  __shared__ float sm_o[WARPS][D_MAX];
  if (lane == 0) { sm_m[warp_id] = m_running; sm_l[warp_id] = l_running; }
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    if (d < Di) sm_o[warp_id][d] = o_frag[dd];
  }
  __syncthreads();
  if (warp_id == 0) {
    T* o_row = o + static_cast<int64_t>(rh) * D;
    pa_merge_and_write<T, D_PER_LANE, WARPS, D_MAX>(sm_m, sm_l, sm_o, o_row,
                                                    lane, Di);
  }
}

// INT8-direct variant: K/V are int8 payloads with a per-(block,head,slot)
// FP32 scale; dequantize `int8 * scale` inline. Q / O stay FP (template T).
template <typename T, int D_PER_LANE, int WARPS, int D_MAX>
__global__ void paged_decode_attention_int8_kernel(
    const T* __restrict__ q,
    const int8_t* __restrict__ k_pool,
    const float* __restrict__ k_scale,
    const int8_t* __restrict__ v_pool,
    const float* __restrict__ v_scale,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ lens,
    T* __restrict__ o,
    int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale) {
  const int rh      = blockIdx.x;
  const int r       = static_cast<int>(rh / H);
  const int h       = static_cast<int>(rh % H);
  const int hkv     = h / group;
  const int warp_id = threadIdx.x >> 5;
  const int lane    = threadIdx.x & 31;
  const int Di      = static_cast<int>(D);
  const int len     = lens[r];

  const T* q_row = q + static_cast<int64_t>(rh) * D;
  float q_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    q_frag[dd] = (d < Di) ? pa_to_float(q_row[d]) * scale : 0.0f;
  }

  float m_running = -INFINITY;
  float l_running = 0.0f;
  float o_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) o_frag[dd] = 0.0f;

  const int32_t* table = block_tables + static_cast<int64_t>(r) * max_logical;

  for (int j = warp_id; j < len; j += WARPS) {
    const int logical = static_cast<int>(j / block_size);
    const int slot    = static_cast<int>(j % block_size);
    int p = table[logical];
    if (p < 0 || p >= num_blocks) p = 0;
    const int64_t hbs = (static_cast<int64_t>(p) * Hkv + hkv) * block_size + slot;
    const int64_t row = hbs * D;
    const int8_t* k_row = k_pool + row;
    const int8_t* v_row = v_pool + row;
    const float ks = k_scale[hbs];
    const float vs = v_scale[hbs];

    float partial = 0.0f;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < Di) partial += q_frag[dd] * (static_cast<float>(k_row[d]) * ks);
    }
    const float s = warp_reduce_sum(partial);

    const float new_m = fmaxf(m_running, s);
    const float alpha = (m_running == -INFINITY) ? 0.0f
                                                 : __expf(m_running - new_m);
    const float p_j   = __expf(s - new_m);
    l_running = l_running * alpha + p_j;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < Di)
        o_frag[dd] = o_frag[dd] * alpha + p_j * (static_cast<float>(v_row[d]) * vs);
    }
    m_running = new_m;
  }

  __shared__ float sm_m[WARPS];
  __shared__ float sm_l[WARPS];
  __shared__ float sm_o[WARPS][D_MAX];
  if (lane == 0) { sm_m[warp_id] = m_running; sm_l[warp_id] = l_running; }
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    if (d < Di) sm_o[warp_id][d] = o_frag[dd];
  }
  __syncthreads();
  if (warp_id == 0) {
    T* o_row = o + static_cast<int64_t>(rh) * D;
    pa_merge_and_write<T, D_PER_LANE, WARPS, D_MAX>(sm_m, sm_l, sm_o, o_row,
                                                    lane, Di);
  }
}

// ---- Wave 14 (B-032++++): fused paged PREFILL kernels (S_new > 1) ----
//
// Same online-softmax structure as the decode kernels, but one warp per
// `(request r, new-query-index s, head h)` triple and a causal key bound:
// query s of request r sits at global position `pos_r + s` where
// `pos_r = kv_len[r] - S` (the context length *before* the S new tokens),
// so it attends to keys `[0, pos_r + s]` — i.e. `key_len = kv_len[r] - S +
// s + 1`. Collapses the `Sn > 1` per-sequence chunked-prefill loop into one
// launch over the whole active set. K/V read in place from the paged pool.

template <typename T, int D_PER_LANE>
__global__ void paged_prefill_attention_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_pool,
    const T* __restrict__ v_pool,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ kv_lens,
    T* __restrict__ o,
    int64_t S, int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale) {
  const int idx  = blockIdx.x;                 // r*(S*H) + s*H + h
  const int SH   = static_cast<int>(S * H);
  const int r    = idx / SH;
  const int rem  = idx % SH;
  const int s    = rem / static_cast<int>(H);
  const int h    = rem % static_cast<int>(H);
  const int hkv  = h / group;
  const int lane = threadIdx.x;
  const int Di   = static_cast<int>(D);
  const int kvl  = kv_lens[r];
  const int len  = kvl - static_cast<int>(S) + s + 1;  // causal bound

  const int64_t qo_row =
      ((static_cast<int64_t>(r) * S + s) * H + h) * D;
  const T* q_row = q + qo_row;
  float q_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    q_frag[dd] = (d < Di) ? pa_to_float(q_row[d]) * scale : 0.0f;
  }

  float m_running = -INFINITY;
  float l_running = 0.0f;
  float o_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) o_frag[dd] = 0.0f;

  const int32_t* table = block_tables + static_cast<int64_t>(r) * max_logical;

  for (int j = 0; j < len; ++j) {
    const int logical = static_cast<int>(j / block_size);
    const int slot    = static_cast<int>(j % block_size);
    int p = table[logical];
    if (p < 0 || p >= num_blocks) p = 0;
    const int64_t row =
        ((static_cast<int64_t>(p) * Hkv + hkv) * block_size + slot) * D;
    const T* k_row = k_pool + row;
    const T* v_row = v_pool + row;

    float partial = 0.0f;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < Di) partial += q_frag[dd] * pa_to_float(k_row[d]);
    }
    const float sc = warp_reduce_sum(partial);

    const float new_m = fmaxf(m_running, sc);
    const float alpha = (m_running == -INFINITY) ? 0.0f
                                                 : __expf(m_running - new_m);
    const float p_j   = __expf(sc - new_m);
    l_running = l_running * alpha + p_j;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < Di) o_frag[dd] = o_frag[dd] * alpha + p_j * pa_to_float(v_row[d]);
    }
    m_running = new_m;
  }

  T* o_row = o + qo_row;
  const float inv_l = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    if (d < Di) o_row[d] = pa_from_float<T>(o_frag[dd] * inv_l);
  }
}

template <typename T, int D_PER_LANE>
__global__ void paged_prefill_attention_int8_kernel(
    const T* __restrict__ q,
    const int8_t* __restrict__ k_pool,
    const float* __restrict__ k_scale,
    const int8_t* __restrict__ v_pool,
    const float* __restrict__ v_scale,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ kv_lens,
    T* __restrict__ o,
    int64_t S, int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale) {
  const int idx  = blockIdx.x;
  const int SH   = static_cast<int>(S * H);
  const int r    = idx / SH;
  const int rem  = idx % SH;
  const int s    = rem / static_cast<int>(H);
  const int h    = rem % static_cast<int>(H);
  const int hkv  = h / group;
  const int lane = threadIdx.x;
  const int Di   = static_cast<int>(D);
  const int kvl  = kv_lens[r];
  const int len  = kvl - static_cast<int>(S) + s + 1;

  const int64_t qo_row =
      ((static_cast<int64_t>(r) * S + s) * H + h) * D;
  const T* q_row = q + qo_row;
  float q_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    q_frag[dd] = (d < Di) ? pa_to_float(q_row[d]) * scale : 0.0f;
  }

  float m_running = -INFINITY;
  float l_running = 0.0f;
  float o_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) o_frag[dd] = 0.0f;

  const int32_t* table = block_tables + static_cast<int64_t>(r) * max_logical;

  for (int j = 0; j < len; ++j) {
    const int logical = static_cast<int>(j / block_size);
    const int slot    = static_cast<int>(j % block_size);
    int p = table[logical];
    if (p < 0 || p >= num_blocks) p = 0;
    const int64_t hbs = (static_cast<int64_t>(p) * Hkv + hkv) * block_size + slot;
    const int64_t row = hbs * D;
    const int8_t* k_row = k_pool + row;
    const int8_t* v_row = v_pool + row;
    const float ks = k_scale[hbs];
    const float vs = v_scale[hbs];

    float partial = 0.0f;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < Di) partial += q_frag[dd] * (static_cast<float>(k_row[d]) * ks);
    }
    const float sc = warp_reduce_sum(partial);

    const float new_m = fmaxf(m_running, sc);
    const float alpha = (m_running == -INFINITY) ? 0.0f
                                                 : __expf(m_running - new_m);
    const float p_j   = __expf(sc - new_m);
    l_running = l_running * alpha + p_j;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < Di)
        o_frag[dd] = o_frag[dd] * alpha + p_j * (static_cast<float>(v_row[d]) * vs);
    }
    m_running = new_m;
  }

  T* o_row = o + qo_row;
  const float inv_l = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    if (d < Di) o_row[d] = pa_from_float<T>(o_frag[dd] * inv_l);
  }
}

template <typename T>
void dispatch_prefill(int64_t A, int64_t S, int64_t H, int64_t Hkv, int64_t D,
                      int64_t block_size, int64_t num_blocks,
                      int64_t max_logical, int group, float scale,
                      const void* q, const void* k_pool, const void* v_pool,
                      const int32_t* block_tables, const int32_t* kv_lens,
                      void* o, cudaStream_t stream) {
  constexpr int D_PER_LANE = 4;
  const dim3 grid(static_cast<unsigned>(A * S * H));
  const dim3 block(32);
  paged_prefill_attention_kernel<T, D_PER_LANE><<<grid, block, 0, stream>>>(
      static_cast<const T*>(q), static_cast<const T*>(k_pool),
      static_cast<const T*>(v_pool), block_tables, kv_lens,
      static_cast<T*>(o), S, H, Hkv, D, block_size, num_blocks, max_logical,
      group, scale);
}

template <typename T>
void dispatch_prefill_int8(int64_t A, int64_t S, int64_t H, int64_t Hkv,
                           int64_t D, int64_t block_size, int64_t num_blocks,
                           int64_t max_logical, int group, float scale,
                           const void* q, const int8_t* k_pool,
                           const float* k_scale, const int8_t* v_pool,
                           const float* v_scale, const int32_t* block_tables,
                           const int32_t* kv_lens, void* o,
                           cudaStream_t stream) {
  constexpr int D_PER_LANE = 4;
  const dim3 grid(static_cast<unsigned>(A * S * H));
  const dim3 block(32);
  paged_prefill_attention_int8_kernel<T, D_PER_LANE><<<grid, block, 0, stream>>>(
      static_cast<const T*>(q), k_pool, k_scale, v_pool, v_scale,
      block_tables, kv_lens, static_cast<T*>(o), S, H, Hkv, D, block_size,
      num_blocks, max_logical, group, scale);
}

template <typename T>
void dispatch(int64_t A, int64_t H, int64_t Hkv, int64_t D,
              int64_t block_size, int64_t num_blocks, int64_t max_logical,
              int group, float scale, const void* q, const void* k_pool,
              const void* v_pool, const int32_t* block_tables,
              const int32_t* lens, void* o, cudaStream_t stream) {
  constexpr int D_PER_LANE = 4;  // D_MAX = 128
  constexpr int WARPS = 8;       // 256 threads: split each request's KV range
  constexpr int D_MAX = 128;
  const dim3 grid(static_cast<unsigned>(A * H));
  const dim3 block(WARPS * 32);
  paged_decode_attention_kernel<T, D_PER_LANE, WARPS, D_MAX><<<grid, block, 0, stream>>>(
      static_cast<const T*>(q), static_cast<const T*>(k_pool),
      static_cast<const T*>(v_pool), block_tables, lens,
      static_cast<T*>(o), H, Hkv, D, block_size, num_blocks, max_logical,
      group, scale);
}

template <typename T>
void dispatch_int8(int64_t A, int64_t H, int64_t Hkv, int64_t D,
                   int64_t block_size, int64_t num_blocks, int64_t max_logical,
                   int group, float scale, const void* q, const int8_t* k_pool,
                   const float* k_scale, const int8_t* v_pool,
                   const float* v_scale, const int32_t* block_tables,
                   const int32_t* lens, void* o, cudaStream_t stream) {
  constexpr int D_PER_LANE = 4;
  constexpr int WARPS = 8;
  constexpr int D_MAX = 128;
  const dim3 grid(static_cast<unsigned>(A * H));
  const dim3 block(WARPS * 32);
  paged_decode_attention_int8_kernel<T, D_PER_LANE, WARPS, D_MAX><<<grid, block, 0, stream>>>(
      static_cast<const T*>(q), k_pool, k_scale, v_pool, v_scale,
      block_tables, lens, static_cast<T*>(o), H, Hkv, D, block_size,
      num_blocks, max_logical, group, scale);
}

}  // namespace

void launch_paged_decode_attention(
    DType dtype, int device_index,
    int64_t A, int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale,
    const void* q, const void* k_pool, const void* v_pool,
    const int32_t* block_tables, const int32_t* lens,
    void* o, void* stream_handle) {
  if (A == 0 || H == 0 || D == 0) return;

  DeviceGuard guard(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  switch (dtype) {
    case DType::Float32:
      dispatch<float>(A, H, Hkv, D, block_size, num_blocks, max_logical,
                      group, scale, q, k_pool, v_pool, block_tables, lens,
                      o, stream);
      break;
    case DType::Float16:
      dispatch<__half>(A, H, Hkv, D, block_size, num_blocks, max_logical,
                       group, scale, q, k_pool, v_pool, block_tables, lens,
                       o, stream);
      break;
    case DType::BFloat16:
      dispatch<__nv_bfloat16>(A, H, Hkv, D, block_size, num_blocks,
                              max_logical, group, scale, q, k_pool, v_pool,
                              block_tables, lens, o, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA paged_decode_attention on dtype {} is not "
          "implemented (Float32 / Float16 / BFloat16 only).",
          dtype_name(dtype)));
  }
  check_launch("paged_decode_attention_kernel");
}

void launch_paged_decode_attention_int8(
    DType dtype, int device_index,
    int64_t A, int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale,
    const void* q, const int8_t* k_pool, const float* k_scale,
    const int8_t* v_pool, const float* v_scale,
    const int32_t* block_tables, const int32_t* lens,
    void* o, void* stream_handle) {
  if (A == 0 || H == 0 || D == 0) return;

  DeviceGuard guard(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  switch (dtype) {
    case DType::Float32:
      dispatch_int8<float>(A, H, Hkv, D, block_size, num_blocks, max_logical,
                           group, scale, q, k_pool, k_scale, v_pool, v_scale,
                           block_tables, lens, o, stream);
      break;
    case DType::Float16:
      dispatch_int8<__half>(A, H, Hkv, D, block_size, num_blocks, max_logical,
                            group, scale, q, k_pool, k_scale, v_pool, v_scale,
                            block_tables, lens, o, stream);
      break;
    case DType::BFloat16:
      dispatch_int8<__nv_bfloat16>(A, H, Hkv, D, block_size, num_blocks,
                                   max_logical, group, scale, q, k_pool,
                                   k_scale, v_pool, v_scale, block_tables,
                                   lens, o, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA paged_decode_attention_int8 on dtype {} is not "
          "implemented (Float32 / Float16 / BFloat16 only).",
          dtype_name(dtype)));
  }
  check_launch("paged_decode_attention_int8_kernel");
}

void launch_paged_prefill_attention(
    DType dtype, int device_index,
    int64_t A, int64_t S, int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale,
    const void* q, const void* k_pool, const void* v_pool,
    const int32_t* block_tables, const int32_t* kv_lens,
    void* o, void* stream_handle) {
  if (A == 0 || S == 0 || H == 0 || D == 0) return;

  DeviceGuard guard(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  switch (dtype) {
    case DType::Float32:
      dispatch_prefill<float>(A, S, H, Hkv, D, block_size, num_blocks,
                              max_logical, group, scale, q, k_pool, v_pool,
                              block_tables, kv_lens, o, stream);
      break;
    case DType::Float16:
      dispatch_prefill<__half>(A, S, H, Hkv, D, block_size, num_blocks,
                               max_logical, group, scale, q, k_pool, v_pool,
                               block_tables, kv_lens, o, stream);
      break;
    case DType::BFloat16:
      dispatch_prefill<__nv_bfloat16>(A, S, H, Hkv, D, block_size, num_blocks,
                                      max_logical, group, scale, q, k_pool,
                                      v_pool, block_tables, kv_lens, o, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA paged_prefill_attention on dtype {} is not "
          "implemented (Float32 / Float16 / BFloat16 only).",
          dtype_name(dtype)));
  }
  check_launch("paged_prefill_attention_kernel");
}

void launch_paged_prefill_attention_int8(
    DType dtype, int device_index,
    int64_t A, int64_t S, int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale,
    const void* q, const int8_t* k_pool, const float* k_scale,
    const int8_t* v_pool, const float* v_scale,
    const int32_t* block_tables, const int32_t* kv_lens,
    void* o, void* stream_handle) {
  if (A == 0 || S == 0 || H == 0 || D == 0) return;

  DeviceGuard guard(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  switch (dtype) {
    case DType::Float32:
      dispatch_prefill_int8<float>(A, S, H, Hkv, D, block_size, num_blocks,
                                   max_logical, group, scale, q, k_pool,
                                   k_scale, v_pool, v_scale, block_tables,
                                   kv_lens, o, stream);
      break;
    case DType::Float16:
      dispatch_prefill_int8<__half>(A, S, H, Hkv, D, block_size, num_blocks,
                                    max_logical, group, scale, q, k_pool,
                                    k_scale, v_pool, v_scale, block_tables,
                                    kv_lens, o, stream);
      break;
    case DType::BFloat16:
      dispatch_prefill_int8<__nv_bfloat16>(A, S, H, Hkv, D, block_size,
                                           num_blocks, max_logical, group,
                                           scale, q, k_pool, k_scale, v_pool,
                                           v_scale, block_tables, kv_lens, o,
                                           stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA paged_prefill_attention_int8 on dtype {} is not "
          "implemented (Float32 / Float16 / BFloat16 only).",
          dtype_name(dtype)));
  }
  check_launch("paged_prefill_attention_int8_kernel");
}

}  // namespace tesseract::cuda::detail

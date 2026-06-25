// Wave 4.2 (B-024): fused FlashAttention-2-style SDPA forward kernel.
//
// `ops::attention` composite is four GPU launches + one S_q×S_k
// intermediate tensor for scores/probs. The score matrix is the
// single biggest heap write on the attention critical path — for a
// Llama-7B prefill at S=4K that's 4K × 4K × 4 bytes × B*H = 1-4 GB
// of HBM traffic per layer just to materialize and re-read `scores`.
//
// FA2 collapses the chain into a single kernel that never
// materializes the full `[S_q, S_k]` matrix: each block processes
// one `(b, h, q)` triple, streams K/V in tiles, and uses an online
// softmax to accumulate O incrementally without holding all of
// `exp(scores)` in memory.
//
//     m_new = max(m_old, max_j tile_scores[j])
//     α     = exp(m_old - m_new)                    // rescale previous
//     s_j   = exp(tile_scores[j] - m_new)           // per-tile exp
//     l_new = l_old * α + Σ_j s_j                   // running denom
//     O_new = O_old * α + Σ_j s_j * V[j, :]         // running numer
//     ...
//     O_final = O / l_final                         // final normalize
//
// Grid:   (B*H, ceil(S_q / BLOCK_Q), 1)  with BLOCK_Q=4.
// Block:  128 threads = 4 warps, each warp owns one query row within
//         the block. All 4 warps share the `K_tile` / `V_tile`
//         shared-memory stage, so a tile of K/V is pulled from HBM
//         once per block (4 query rows) instead of once per query
//         row — which is the dominant HBM cost for long-S prefill.
//   * Q row for warp `w` (row `q = blockIdx.y * BLOCK_Q + w`) is held
//     in registers as `q_frag[d_per_lane]` across the 32 lanes, 4
//     FP32 values per lane for D=128 (2 for D=64, 1 for D=32).
//   * Score pass: each warp, on its own query, dots 32 registers
//     against the 32 K rows of the tile (one dot per key per lane),
//     folded with a warp-shuffle sum reduce, yielding
//     `scores[0..BLOCK_K)` replicated in every lane of that warp.
//   * Softmax state (`m_running`, `l_running`, `o_frag`) is per-warp
//     register state; no cross-warp synchronization except for the
//     `__syncthreads` around cooperative K/V tile loads.
//   * Partial trailing blocks (when S_q is not a multiple of
//     BLOCK_Q) have the out-of-range warps participate in the
//     cooperative tile loads but skip compute and output writes.
//
// Tile sizing: BLOCK_Q=8, BLOCK_K=32. K_tile/V_tile sit in shared
// memory with a padded stride of D_MAX+1 to avoid the 32-way bank
// conflict the natural D_MAX=128 stride would hit. Shared-mem
// budget at D=128:
//     K_tile : 32 × 129 × 4 =  16.5 KB
//     V_tile : 32 × 129 × 4 =  16.5 KB
//     total  ≈  33.0 KB  (fits 48 KB default)
// Per-thread register footprint: q_frag[4] + o_frag[4] + scores[32]
// ≈ 40 floats of algorithmic state; the rest are control/temps.
// At 256 threads/block with ~33 KB smem we land near 2–3 blocks/SM
// on Ada, the structural ceiling for this algorithm without
// warp-specialization / WGMMA (Hopper-only) or WMMA tensor-core
// rewrites (the Wave 4.2+ WMMA follow-up).
//
// Throughput ceiling on Ada: this kernel runs the score and O-
// accumulation matmuls on FP32 CUDA cores, not on tensor cores. On
// prefill shapes the composite `matmul → softmax → matmul` path
// uses cuBLASLt FP16 tensor cores and therefore has ~4× the
// effective FLOP throughput. The fused kernel wins unconditionally
// on (i) decode (S_q=1) once B·H saturates the SM count, where
// composite's three GEMV launches + score-tensor round trip
// dominate, and (ii) any shape where avoiding the S_q×S_k score
// materialization matters more than raw throughput. See
// `benchmarks/bench_cuda_fused_attention.cpp` for the hard-bar
// policy and `docs/backlog.md` (B-024) for the WMMA follow-up.
//
// Numerical contract:
//   * All accumulation (scores, exp/log, weighted V sum, normalize)
//     runs in FP32 regardless of storage dtype. Matches the M2F
//     softmax policy, so CPU↔CUDA parity holds to the TF32 / FP16
//     tolerance the `ops::attention` composite already exposes.
//   * `__expf` (MUFU.EX2 intrinsic) is used for the softmax
//     exp. FlashAttention's reference implementation does the same;
//     the ~2 ulp error is dwarfed by the accumulation error.
//   * When a tile is entirely masked (`causal` with `j_start > q`),
//     `tile_max == -inf`; the rescale then divides by zero which
//     `__expf` handles as NaN. We guard against this by skipping
//     the update when both `m_running` and `tile_max` are -inf.
//
// Forward-only. Backward flows through the composite path in
// `src/ops/cpu/Attention.cpp` (`matmul → softmax → matmul`) which
// stitches together existing autograd nodes. Same convention as
// `launch_rms_norm` / `launch_swiglu_silu_gate`.

#include "KernelUtils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <type_traits>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/FusedAttention.hpp"

namespace tesseract::cuda::detail {

namespace {

// Small device-side min for `(int, int64_t)` pairs; avoids pulling
// in `<algorithm>` on the device side where the std::min template
// overload resolution gets hairy with CUDA's C++17 configuration.
__device__ __forceinline__ int64_t dev_min_i64(int64_t a, int64_t b) {
  return (a < b) ? a : b;
}

// FP32 promotion helpers (same convention as RMSNorm / SwiGLU).
__device__ __forceinline__ float fa_to_float(float v)         { return v; }
__device__ __forceinline__ float fa_to_float(__half v)        { return __half2float(v); }
__device__ __forceinline__ float fa_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T fa_from_float(float v);
template <> __device__ __forceinline__
float fa_from_float<float>(float v) { return v; }
template <> __device__ __forceinline__
__half fa_from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__
__nv_bfloat16 fa_from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// Warp-shuffle sum. `0xffffffff` mask assumes a full warp, which
// is always true inside this kernel's launch (BLOCK_Q * 32 threads,
// warp-sized slices per query). Don't reuse for partial warps.
__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, offset);
  }
  return v;
}

// FP32-accumulated kernel for FP32 / FP16 / BFloat16 storage. The
// storage type drives how Q/K/V are loaded and how O is stored; all
// interior math is FP32. `D_MAX` is the compile-time upper bound on
// head dim (always 128 today) and sizes the shared-memory tiles.
//
// Layout: 4 warps per block, each warp owns one query row so a
// single K/V tile read is amortized across BLOCK_Q=4 queries.
template <typename Tstorage, int BLOCK_Q, int BLOCK_K,
          int D_MAX, int D_PER_LANE, int THREADS>
__global__ void fused_attention_fp32_kernel(
    const Tstorage* __restrict__ Q,
    const Tstorage* __restrict__ K,
    const Tstorage* __restrict__ V,
    Tstorage* __restrict__ O,
    int64_t B, int64_t H, int64_t H_kv, int64_t S_q, int64_t S_k, int64_t D,
    float scale, bool causal,
    int64_t q_b, int64_t q_h, int64_t q_s,
    int64_t k_b, int64_t k_h, int64_t k_s,
    int64_t o_b, int64_t o_h, int64_t o_s) {
  static_assert(BLOCK_Q * 32 == THREADS,
                "BLOCK_Q warps (one per query row) must fill the block");
  static_assert(D_MAX == D_PER_LANE * 32,
                "D_PER_LANE must be the per-lane fragment of D_MAX");
  static_assert(BLOCK_K > 0, "BLOCK_K must be positive");

  const int bh      = blockIdx.x;          // b * H + h  (query head)
  const int tid     = threadIdx.x;
  const int warp_id = tid >> 5;   // 0..BLOCK_Q-1 = query index within block
  const int lane    = tid & 31;

  (void)B;
  // GQA: query head `h` reads KV head `h / (H / H_kv)`. With H_kv == H
  // (plain MHA) this collapses to `kv_bh == bh`, so the non-GQA caller
  // pays nothing. `K`/`V` are laid out as `[B, H_kv, S_k, D]`.
  const int b_idx  = static_cast<int>(bh / H);
  const int h_idx  = static_cast<int>(bh % H);
  const int group  = static_cast<int>(H / (H_kv > 0 ? H_kv : H));
  const int kv_h   = group > 0 ? h_idx / group : h_idx;
  const int kv_bh  = b_idx * static_cast<int>(H_kv) + kv_h;

  // Padded stride to break the 32-way shared-memory bank conflict
  // the score-phase access pattern otherwise hits. Score dot reads
  // `K_tile[j_local * STRIDE + d]` for a fixed `d` across lanes of
  // a warp; with STRIDE == D_MAX == 128 every lane lands in the
  // same bank (128 · 4 B = exact multiple of 32-bank × 4 B = 128 B
  // bank width). Padding STRIDE to 129 floats makes the per-thread
  // bank offset vary as `lane mod 32`, collapsing a 32-way conflict
  // to 0 and recovering full shared-memory throughput.
  constexpr int KV_STRIDE = D_MAX + 1;

  (void)kv_bh;
  const int q        = blockIdx.y * BLOCK_Q + warp_id;
  const bool q_valid = q < S_q;

  // Stride-aware bases (B-024c): read Q / K / V from their in-place layouts
  // (permuted Q view, KV-cache narrows) without a contiguous() copy.
  const Tstorage* q_row =
      q_valid ? Q + b_idx * q_b + h_idx * q_h + static_cast<int64_t>(q) * q_s
              : Q;  // dummy, never read when q_valid == false
  const Tstorage* k_base = K + b_idx * k_b + kv_h * k_h;
  const Tstorage* v_base = V + b_idx * k_b + kv_h * k_h;

  // Shared memory — K/V tiles are shared across all 4 warps.
  __shared__ float K_tile [BLOCK_K * KV_STRIDE];
  __shared__ float V_tile [BLOCK_K * KV_STRIDE];

  const int D_int = static_cast<int>(D);

  // Per-lane Q fragment, held in registers. For D=128, lane `l` owns
  // Q indices {4l, 4l+1, 4l+2, 4l+3}; generalized by D_PER_LANE.
  float q_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    q_frag[dd] = (q_valid && d < D_int) ? fa_to_float(q_row[d]) * scale
                                        : 0.0f;
    // scale folded into Q so the score dot doesn't need a
    // per-element multiply afterwards.
  }

  // Per-warp running state. `o_frag[dd]` is the running numerator
  // for output slot `lane * D_PER_LANE + dd`.
  float m_running = -INFINITY;
  float l_running = 0.0f;
  float o_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) o_frag[dd] = 0.0f;

  const int n_tiles = static_cast<int>((S_k + BLOCK_K - 1) / BLOCK_K);

  for (int tile = 0; tile < n_tiles; ++tile) {
    const int j_start  = tile * BLOCK_K;
    const int tile_len = static_cast<int>(
        dev_min_i64(static_cast<int64_t>(BLOCK_K), S_k - j_start));

    // Cooperative K/V tile load — all 128 threads participate
    // regardless of `q_valid`, so the required `__syncthreads()`
    // below is reached uniformly.
    #pragma unroll 1
    for (int linear = tid; linear < tile_len * D_int; linear += THREADS) {
      const int j_local = linear / D_int;
      const int d       = linear - j_local * D_int;
      const int j       = j_start + j_local;
      K_tile[j_local * KV_STRIDE + d] = fa_to_float(k_base[static_cast<int64_t>(j) * k_s + d]);
      V_tile[j_local * KV_STRIDE + d] = fa_to_float(v_base[static_cast<int64_t>(j) * k_s + d]);
    }
    __syncthreads();

    // All the per-warp math below updates only register state; no
    // cross-warp communication, and no __syncthreads on any control-
    // flow branch. That keeps the single end-of-tile barrier below
    // reachable by every warp, avoiding the classic divergent-sync
    // hazard when warps disagree on "tile fully masked" flags.
    if (q_valid) {
      // Per-warp score computation: dot `q_frag` against each K row
      // in the tile, fold via warp-shuffle sum. `scores[j_local]`
      // ends up replicated across all 32 lanes.
      float scores[BLOCK_K];
      #pragma unroll
      for (int j_local = 0; j_local < BLOCK_K; ++j_local) {
        float partial = 0.0f;
        #pragma unroll
        for (int dd = 0; dd < D_PER_LANE; ++dd) {
          const int d = lane * D_PER_LANE + dd;
          if (d < D_int) {
            partial += q_frag[dd] * K_tile[j_local * KV_STRIDE + d];
          }
        }
        scores[j_local] = warp_reduce_sum(partial);
      }

      // Mask invalid positions (tile tail + causal) and reduce max.
      float tile_max = -INFINITY;
      #pragma unroll
      for (int j_local = 0; j_local < BLOCK_K; ++j_local) {
        const int j = j_start + j_local;
        float s = scores[j_local];
        if (j_local >= tile_len)     s = -INFINITY;
        else if (causal && j > q)    s = -INFINITY;
        scores[j_local] = s;
        tile_max = fmaxf(tile_max, s);
      }

      const float new_max = fmaxf(m_running, tile_max);

      // Fully-masked tile (causal pre-row or an empty tail); skip
      // the running-state update but keep every warp executing the
      // same control flow below (the tile-end `__syncthreads` is
      // placed outside this `if` block).
      if (new_max != -INFINITY) {
        float tile_sum = 0.0f;
        #pragma unroll
        for (int j_local = 0; j_local < BLOCK_K; ++j_local) {
          scores[j_local] = (scores[j_local] == -INFINITY)
                                ? 0.0f
                                : __expf(scores[j_local] - new_max);
          tile_sum += scores[j_local];
        }

        const float alpha = (m_running == -INFINITY)
                                ? 0.0f
                                : __expf(m_running - new_max);
        l_running = l_running * alpha + tile_sum;

        #pragma unroll
        for (int dd = 0; dd < D_PER_LANE; ++dd) {
          const int d = lane * D_PER_LANE + dd;
          if (d >= D_int) continue;
          float acc = 0.0f;
          #pragma unroll
          for (int j_local = 0; j_local < BLOCK_K; ++j_local) {
            acc += scores[j_local] * V_tile[j_local * KV_STRIDE + d];
          }
          o_frag[dd] = o_frag[dd] * alpha + acc;
        }

        m_running = new_max;
      }
    }

    __syncthreads();  // before next tile overwrites K_tile / V_tile
  }

  // Normalize and write. Fully-masked rows (l_running == 0) produce
  // zeros — same fallback as PyTorch / FA's reference.
  if (q_valid) {
    Tstorage* o_row =
        O + b_idx * o_b + h_idx * o_h + static_cast<int64_t>(q) * o_s;
    const float inv_l = (l_running > 0.0f) ? (1.0f / l_running) : 0.0f;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < D_int) {
        o_row[d] = fa_from_float<Tstorage>(o_frag[dd] * inv_l);
      }
    }
  }
}

// ===========================================================================
// WMMA tensor-core fused FlashAttention-2 prefill kernel (B-024+).
//
// The FP32 kernel above runs both matmuls (Q·Kᵀ and P·V) on CUDA cores, which
// caps prefill throughput at ~1/4 of Ada's FP16 tensor-core peak — the reason
// the composite cuBLASLt path (and vLLM's FlashAttention) win TTFT. This
// kernel does the same online-softmax FA2 recipe but runs both matmuls on the
// **FP16/BF16 tensor cores** via `nvcuda::wmma` (16×16×16 fragments), while
// never materializing the [S_q, S_k] score matrix in HBM. So it gets tensor-
// core FLOPs *and* avoids the score round-trip the composite pays.
//
//   Tile:  BLOCK_M queries × BLOCK_N keys, head dim D (≤128, multiple of 16).
//   Block: WARPS warps = BLOCK_M/16 m-tiles, one warp per 16-row band.
//   Shared (FP16 Q/K/V tiles + FP16 P + FP32 S + FP32 running O + m/l):
//     at BM=32, BN=32, D=128 ≈ 46 KB → fits the 48 KB default carveout.
//
// Per KV tile, per warp (its 16 query rows):
//   1. S = Q·Kᵀ  via WMMA (accumulate over D in 16-wide k-chunks) → S_sh (FP32).
//   2. Online softmax over the warp's rows: row max (+ causal/tail mask),
//      m_new, α = exp(m_old−m_new), rescale running O_sh row by α, P = exp(S·
//      scale − m_new) → P_sh (FP16), update l.
//   3. O_sh += P·V via WMMA (load O_sh fragment, add fresh P·V fragment, store
//      back — layout-agnostic, no assumption about accumulator element order).
// Finally O_sh /= l and narrow back to storage dtype.
//
// O is kept in shared memory (not in registers) so the per-row α rescale is
// addressed explicitly by (row, d); the only thing read back from a WMMA
// accumulator is via store_matrix_sync, so the kernel makes no undocumented
// fragment-layout assumption.
//
// FP32 inputs stay on the CUDA-core kernel (no FP16 TC path); FP64 on the
// composite. Forward-only, same autograd policy as the FP32 kernel.

template <typename Tstorage, int BM, int BN, int DMAX, int WARPS>
__global__ void fused_attention_wmma_kernel(
    const Tstorage* __restrict__ Q,
    const Tstorage* __restrict__ K,
    const Tstorage* __restrict__ V,
    Tstorage* __restrict__ O,
    int64_t B, int64_t H, int64_t H_kv, int64_t S_q, int64_t S_k, int64_t D,
    float scale, bool causal,
    int64_t q_b, int64_t q_h, int64_t q_s,
    int64_t k_b, int64_t k_h, int64_t k_s,
    int64_t o_b, int64_t o_h, int64_t o_s) {
  using namespace nvcuda;
  constexpr int WM = 16, WN = 16, WK = 16;
  static_assert(BM % WM == 0 && BN % WN == 0, "tile must be a multiple of 16");
  static_assert(BM / WM == WARPS, "one warp per 16-row m-tile");
  constexpr int THREADS = WARPS * 32;
  constexpr int N_NTILE = BN / WN;  // key sub-tiles per KV block

  const int bh   = blockIdx.x;
  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  const int b_idx = static_cast<int>(bh / H);
  const int h_idx = static_cast<int>(bh % H);
  const int group = static_cast<int>(H / (H_kv > 0 ? H_kv : H));
  const int kv_h  = group > 0 ? h_idx / group : h_idx;
  const int kv_bh = b_idx * static_cast<int>(H_kv) + kv_h;

  const int Dint = static_cast<int>(D);
  const int nDt  = Dint / WK;  // D sub-tiles
  const int q0   = blockIdx.y * BM;

  // Stride-aware bases (B-024c): Q is the [B,H,S,D] permuted view, K/V are
  // the [B,Hkv,S,D] KV-cache narrows, O is written in the [B,S,H,D] layout —
  // all read/written in place, no contiguous() copy. The head-dim stride is 1.
  const Tstorage* qbase = Q + b_idx * q_b + h_idx * q_h;
  const Tstorage* kbase = K + b_idx * k_b + kv_h * k_h;
  const Tstorage* vbase = V + b_idx * k_b + kv_h * k_h;

  // Shared-memory partition (dynamic). Half region first, then float region
  // (the half count is even so the float region stays 4-byte aligned).
  extern __shared__ unsigned char smem_raw[];
  Tstorage* Qsh = reinterpret_cast<Tstorage*>(smem_raw);  // [BM][DMAX]
  Tstorage* Ksh = Qsh + BM * DMAX;                        // [BN][DMAX]
  Tstorage* Vsh = Ksh + BN * DMAX;                        // [BN][DMAX]
  Tstorage* Psh = Vsh + BN * DMAX;                        // [BM][BN]
  float* Ssh = reinterpret_cast<float*>(Psh + BM * BN);   // [BM][BN]
  float* Osh = Ssh + BM * BN;                             // [BM][DMAX]
  float* msh = Osh + BM * DMAX;                           // [BM]
  float* lsh = msh + BM;                                  // [BM]

  // Stage Q into shared once, zero-padding rows beyond S_q.
  for (int idx = tid; idx < BM * Dint; idx += THREADS) {
    const int r = idx / Dint, c = idx - r * Dint;
    const int gq = q0 + r;
    Qsh[r * DMAX + c] =
        (gq < S_q) ? qbase[static_cast<int64_t>(gq) * q_s + c] : Tstorage(0);
  }
  for (int i = tid; i < BM * Dint; i += THREADS) Osh[i] = 0.0f;
  for (int i = tid; i < BM; i += THREADS) { msh[i] = -INFINITY; lsh[i] = 0.0f; }
  __syncthreads();

  const int m_tile = warp;              // this warp's 16-row band
  const int row0   = q0 + m_tile * WM;  // global first query row of the band

  const int n_kv_tiles = static_cast<int>((S_k + BN - 1) / BN);
  for (int kvt = 0; kvt < n_kv_tiles; ++kvt) {
    const int j0 = kvt * BN;
    // Square-causal: once a KV tile starts past the block's last query row,
    // every later key is masked for every row here — stop (skips the upper
    // triangle, the main causal-prefill speedup).
    if (causal && j0 > q0 + BM - 1) break;
    const int tile_len =
        static_cast<int>(dev_min_i64(static_cast<int64_t>(BN), S_k - j0));

    // Cooperative K/V load (FP16/BF16, no FP32 conversion). Rows past the
    // tail are zeroed so masked keys contribute exactly 0 to S and P·V.
    for (int idx = tid; idx < BN * Dint; idx += THREADS) {
      const int r = idx / Dint, c = idx - r * Dint;
      const bool ok = r < tile_len;
      Ksh[r * DMAX + c] =
          ok ? kbase[static_cast<int64_t>(j0 + r) * k_s + c] : Tstorage(0);
      Vsh[r * DMAX + c] =
          ok ? vbase[static_cast<int64_t>(j0 + r) * k_s + c] : Tstorage(0);
    }
    __syncthreads();

    // ---- 1. S = Q · Kᵀ  (this warp's [16, BN]) -------------------------- //
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> cS[N_NTILE];
    #pragma unroll
    for (int n = 0; n < N_NTILE; ++n) wmma::fill_fragment(cS[n], 0.0f);
    for (int kk = 0; kk < nDt; ++kk) {
      wmma::fragment<wmma::matrix_a, WM, WN, WK, Tstorage, wmma::row_major> aQ;
      wmma::load_matrix_sync(aQ, Qsh + (m_tile * WM) * DMAX + kk * WK, DMAX);
      #pragma unroll
      for (int n = 0; n < N_NTILE; ++n) {
        // B[d, j] = K[j, d] = Ksh[j*DMAX + d] → col-major with ldm = DMAX.
        wmma::fragment<wmma::matrix_b, WM, WN, WK, Tstorage, wmma::col_major> bK;
        wmma::load_matrix_sync(bK, Ksh + kk * WK + (n * WN) * DMAX, DMAX);
        wmma::mma_sync(cS[n], aQ, bK, cS[n]);
      }
    }
    #pragma unroll
    for (int n = 0; n < N_NTILE; ++n)
      wmma::store_matrix_sync(Ssh + (m_tile * WM) * BN + n * WN, cS[n], BN,
                              wmma::mem_row_major);
    __syncwarp();

    // ---- 2. Online softmax over this warp's 16 rows -------------------- //
    // BN columns spread across the 32 lanes, COLS_PER_LANE each (1 at BN=32,
    // 2 at BN=64), so a single warp handles wide key tiles.
    constexpr int COLS_PER_LANE = BN / 32;
    for (int r = 0; r < WM; ++r) {
      const int lrow = m_tile * WM + r;
      const int grow = q0 + lrow;

      float sv[COLS_PER_LANE];
      float local_max = -INFINITY;
      #pragma unroll
      for (int cc = 0; cc < COLS_PER_LANE; ++cc) {
        const int col = lane + cc * 32;
        const int j = j0 + col;
        float v = Ssh[lrow * BN + col] * scale;
        if (col >= tile_len)         v = -INFINITY;
        else if (causal && j > grow) v = -INFINITY;
        sv[cc] = v;
        local_max = fmaxf(local_max, v);
      }
      float row_max = local_max;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));

      const float m_old = msh[lrow];
      const float m_new = fmaxf(m_old, row_max);
      const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);

      float local_sum = 0.0f;
      #pragma unroll
      for (int cc = 0; cc < COLS_PER_LANE; ++cc) {
        const int col = lane + cc * 32;
        float p = (sv[cc] != -INFINITY && m_new != -INFINITY)
                      ? __expf(sv[cc] - m_new)
                      : 0.0f;
        Psh[lrow * BN + col] = fa_from_float<Tstorage>(p);
        local_sum += p;
      }
      float row_sum = local_sum;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);

      // Rescale running O row by α and update m/l (lane 0 owns scalars; all
      // lanes cooperate on the D-wide O rescale).
      for (int d = lane; d < Dint; d += 32) Osh[lrow * DMAX + d] *= alpha;
      if (lane == 0) {
        lsh[lrow] = lsh[lrow] * alpha + row_sum;
        msh[lrow] = m_new;
      }
    }
    __syncwarp();

    // ---- 3. O_sh += P · V  (this warp's [16, D]) ---------------------- //
    #pragma unroll 1
    for (int dt = 0; dt < nDt; ++dt) {
      wmma::fragment<wmma::accumulator, WM, WN, WK, float> cO;
      wmma::fill_fragment(cO, 0.0f);
      #pragma unroll
      for (int n = 0; n < N_NTILE; ++n) {
        wmma::fragment<wmma::matrix_a, WM, WN, WK, Tstorage, wmma::row_major> aP;
        wmma::load_matrix_sync(aP, Psh + (m_tile * WM) * BN + n * WN, BN);
        wmma::fragment<wmma::matrix_b, WM, WN, WK, Tstorage, wmma::row_major> bV;
        wmma::load_matrix_sync(bV, Vsh + (n * WN) * DMAX + dt * WK, DMAX);
        wmma::mma_sync(cO, aP, bV, cO);
      }
      // O_sh[band, dt*16 ..] += cO  (load running, add, store — same layout).
      wmma::fragment<wmma::accumulator, WM, WN, WK, float> cAcc;
      wmma::load_matrix_sync(cAcc, Osh + (m_tile * WM) * DMAX + dt * WK, DMAX,
                             wmma::mem_row_major);
      #pragma unroll
      for (int e = 0; e < cAcc.num_elements; ++e) cAcc.x[e] += cO.x[e];
      wmma::store_matrix_sync(Osh + (m_tile * WM) * DMAX + dt * WK, cAcc, DMAX,
                              wmma::mem_row_major);
    }
    __syncthreads();  // before the next tile overwrites K/V/P/S
  }

  // ---- normalize and write -------------------------------------------- //
  for (int r = 0; r < WM; ++r) {
    const int lrow = m_tile * WM + r;
    const int grow = q0 + lrow;
    if (grow >= S_q) continue;
    const float l = lsh[lrow];
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    Tstorage* o_row =
        O + b_idx * o_b + h_idx * o_h + static_cast<int64_t>(grow) * o_s;
    for (int d = lane; d < Dint; d += 32)
      o_row[d] = fa_from_float<Tstorage>(Osh[lrow * DMAX + d] * inv_l);
  }
}

template <typename Tstorage, int BM, int BN, int DMAX, int WARPS>
size_t wmma_smem_bytes(int Dint) {
  // Q/K/V/P are half; S/O/m/l are float. D columns only (DMAX stride).
  const size_t halfs = static_cast<size_t>(BM) * DMAX +    // Q
                       static_cast<size_t>(BN) * DMAX * 2 +  // K, V
                       static_cast<size_t>(BM) * BN;         // P
  const size_t floats = static_cast<size_t>(BM) * BN +      // S
                        static_cast<size_t>(BM) * DMAX +    // O
                        static_cast<size_t>(BM) * 2;        // m, l
  (void)Dint;
  return halfs * sizeof(Tstorage) + floats * sizeof(float);
}

template <typename Tstorage>
bool dispatch_wmma_prefill(int64_t B, int64_t H, int64_t H_kv,
                           int64_t S_q, int64_t S_k, int64_t D,
                           float scale, bool causal,
                           const void* q, const void* k, const void* v,
                           void* o, cudaStream_t stream,
                           int64_t q_b, int64_t q_h, int64_t q_s,
                           int64_t k_b, int64_t k_h, int64_t k_s,
                           int64_t o_b, int64_t o_h, int64_t o_s) {
  if (D > 128 || D % 16 != 0) return false;
  // Size the shared-memory tiles to the smallest power-of-two stride ≥ D so a
  // D=64 model (e.g. TinyLlama) does not pay for a 128-wide footprint — that
  // halves the smem bill and roughly doubles resident blocks/SM.
  constexpr int BM = 32, BN = 32, WARPS = BM / 16;  // 2 warps, 2 m-tiles
  constexpr int THREADS = WARPS * 32;
  const unsigned grid_y = static_cast<unsigned>((S_q + BM - 1) / BM);
  const dim3 grid(static_cast<unsigned>(B * H), grid_y, 1);

  auto launch = [&](auto dmax_tag) {
    constexpr int DMAX = decltype(dmax_tag)::value;
    const size_t smem = wmma_smem_bytes<Tstorage, BM, BN, DMAX, WARPS>(0);
    auto kern = fused_attention_wmma_kernel<Tstorage, BM, BN, DMAX, WARPS>;
    static bool attr_set = false;  // per-instantiation, one-shot
    if (!attr_set) {
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem));
      attr_set = true;
    }
    kern<<<grid, THREADS, smem, stream>>>(
        static_cast<const Tstorage*>(q), static_cast<const Tstorage*>(k),
        static_cast<const Tstorage*>(v), static_cast<Tstorage*>(o),
        B, H, H_kv, S_q, S_k, D, scale, causal,
        q_b, q_h, q_s, k_b, k_h, k_s, o_b, o_h, o_s);
  };

  if (D <= 64) launch(std::integral_constant<int, 64>{});
  else         launch(std::integral_constant<int, 128>{});
  return true;
}

// Note: FP64 is intentionally not supported on the fused path.
//
// The static-shared-memory budget at `BLOCK_K=32, D_MAX=128` with
// 8-byte elements would be ≥ 65 KB, which exceeds Ada's 48 KB
// default per-block limit and would force either `cudaFuncSetAttribute`
// opt-in at 96 KB (halving occupancy) or `BLOCK_K=16` with a
// parallel-structure rewrite. Neither is worth shipping on a path
// no inference workload exercises — FP64 attention stays on the
// `matmul → softmax → matmul` composite in `ops::attention`.

// ===========================================================================
// Split-K decode kernel (S_q == 1).
//
// Decode attention is fundamentally HBM-bandwidth bound: a single query row
// must read all of K and V exactly once (2·S_k·D·sizeof(T) bytes) and the
// Q·K is a GEMV with no data reuse — Tensor Cores (WMMA) give nothing on a
// bandwidth-bound GEMV, they accelerate compute-bound GEMMs. The real lever
// is *parallelism*: the original FA2 kernel runs one block per (b,h) and
// streams the entire KV range serially inside it, so when B·H < SM count the
// GPU sits mostly idle while a handful of blocks crawl through a long cache.
//
// Split-K fixes this: partition the KV range into `num_splits` chunks, each
// handled by its own block (warp). Every block runs an online softmax over
// its chunk and writes an *unnormalized* partial (local max m, local denom
// l, local numerator o[D]). A second tiny kernel combines the partials with
// the standard FlashDecoding reduction:
//     M       = max_s m[s]
//     l_tot   = Σ_s exp(m[s]-M) · l[s]
//     o_tot[d]= Σ_s exp(m[s]-M) · o[s][d]
//     O[d]    = o_tot[d] / l_tot
// This both fills the SMs (B·H·num_splits blocks) and keeps each K/V byte
// read exactly once. One warp per block → coalesced K/V reads, no smem.
// ===========================================================================

// Multi-warp split-K partial kernel. Each block owns one (bh, split) and
// runs `WARPS` warps over the split's KV range (warp `w` strides the range
// at step WARPS), then merges the per-warp online-softmax states in shared
// memory into a single block partial. Multiple warps per block keep
// occupancy high (8 warps/block → up to 6 blocks/SM on Ada) while the grid
// (BH·num_splits blocks) covers every SM even when B·H is small.
template <typename Tstorage, int D_MAX, int D_PER_LANE, int WARPS>
__global__ void decode_splitk_partial_kernel(
    const Tstorage* __restrict__ Q,
    const Tstorage* __restrict__ K,
    const Tstorage* __restrict__ V,
    float* __restrict__ partial_o,   // [BH, num_splits, D]
    float* __restrict__ partial_m,   // [BH, num_splits]
    float* __restrict__ partial_l,   // [BH, num_splits]
    int64_t BH, int64_t H, int64_t H_kv,
    int64_t S_k, int64_t D, int num_splits, int split_len,
    float scale, bool causal, int64_t q_pos) {
  const int bh      = blockIdx.x;
  const int split   = blockIdx.y;
  const int warp_id = threadIdx.x >> 5;
  const int lane    = threadIdx.x & 31;
  if (bh >= BH) return;

  const int64_t j0 = static_cast<int64_t>(split) * split_len;
  int64_t j1 = j0 + split_len;
  if (j1 > S_k) j1 = S_k;

  const int D_int = static_cast<int>(D);
  // GQA: query head `bh % H` reads KV head `(bh % H) / (H / H_kv)`. For
  // standard MHA (H_kv == H) this collapses to `kv_bh == bh`, so the
  // mapping is free. This lets the model skip the `repeat_kv` materialization
  // (which copies the whole KV cache H/H_kv× every layer/step) entirely.
  const int64_t b     = static_cast<int64_t>(bh) / H;
  const int64_t h     = static_cast<int64_t>(bh) % H;
  const int64_t group = H / H_kv;
  const int64_t kv_bh = b * H_kv + (h / group);
  const Tstorage* q_row = Q + static_cast<int64_t>(bh) * D;  // S_q == 1
  const Tstorage* k_base = K + kv_bh * S_k * D;
  const Tstorage* v_base = V + kv_bh * S_k * D;

  float q_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    q_frag[dd] = (d < D_int) ? fa_to_float(q_row[d]) * scale : 0.0f;
  }

  float m_running = -INFINITY;
  float l_running = 0.0f;
  float o_frag[D_PER_LANE];
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) o_frag[dd] = 0.0f;

  // Warp w processes j = j0 + w, j0 + w + WARPS, ...
  for (int64_t j = j0 + warp_id; j < j1; j += WARPS) {
    if (causal && j > q_pos) break;
    const Tstorage* k_row = k_base + j * D;
    float partial = 0.0f;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < D_int) partial += q_frag[dd] * fa_to_float(k_row[d]);
    }
    const float score = warp_reduce_sum(partial);

    const float new_max = fmaxf(m_running, score);
    const float alpha = (m_running == -INFINITY) ? 0.0f
                                                 : __expf(m_running - new_max);
    const float p = __expf(score - new_max);
    l_running = l_running * alpha + p;
    const Tstorage* v_row = v_base + j * D;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < D_int) o_frag[dd] = o_frag[dd] * alpha + p * fa_to_float(v_row[d]);
    }
    m_running = new_max;
  }

  // Merge the WARPS per-warp states in shared memory.
  __shared__ float sm_m[WARPS];
  __shared__ float sm_l[WARPS];
  __shared__ float sm_o[WARPS][D_MAX];
  if (lane == 0) {
    sm_m[warp_id] = m_running;
    sm_l[warp_id] = l_running;
  }
  #pragma unroll
  for (int dd = 0; dd < D_PER_LANE; ++dd) {
    const int d = lane * D_PER_LANE + dd;
    if (d < D_int) sm_o[warp_id][d] = o_frag[dd];
  }
  __syncthreads();

  // Warp 0 combines the warp partials into the block partial.
  if (warp_id == 0) {
    float M = -INFINITY;
    #pragma unroll
    for (int w = 0; w < WARPS; ++w) M = fmaxf(M, sm_m[w]);

    float l_tot = 0.0f;
    if (M != -INFINITY) {
      #pragma unroll
      for (int w = 0; w < WARPS; ++w) {
        if (sm_m[w] != -INFINITY) l_tot += __expf(sm_m[w] - M) * sm_l[w];
      }
    }
    const int64_t base = (static_cast<int64_t>(bh) * num_splits + split);
    if (lane == 0) {
      partial_m[base] = M;
      partial_l[base] = l_tot;
    }
    float* o_out = partial_o + base * D;
    #pragma unroll
    for (int dd = 0; dd < D_PER_LANE; ++dd) {
      const int d = lane * D_PER_LANE + dd;
      if (d < D_int) {
        float acc = 0.0f;
        if (M != -INFINITY) {
          #pragma unroll
          for (int w = 0; w < WARPS; ++w) {
            if (sm_m[w] != -INFINITY) acc += __expf(sm_m[w] - M) * sm_o[w][d];
          }
        }
        o_out[d] = acc;
      }
    }
  }
}

template <typename Tstorage>
__global__ void decode_splitk_reduce_kernel(
    const float* __restrict__ partial_o,  // [BH, num_splits, D]
    const float* __restrict__ partial_m,  // [BH, num_splits]
    const float* __restrict__ partial_l,  // [BH, num_splits]
    Tstorage* __restrict__ O,             // [BH, D]
    int64_t BH, int64_t D, int num_splits) {
  const int bh = blockIdx.x;
  if (bh >= BH) return;
  const int d = threadIdx.x;
  const int D_int = static_cast<int>(D);

  // Global max over splits (every thread recomputes; num_splits is small).
  float M = -INFINITY;
  for (int s = 0; s < num_splits; ++s) {
    M = fmaxf(M, partial_m[static_cast<int64_t>(bh) * num_splits + s]);
  }
  if (M == -INFINITY) {  // all splits empty
    if (d < D_int) O[static_cast<int64_t>(bh) * D + d] = fa_from_float<Tstorage>(0.0f);
    return;
  }

  // Denominator is split-independent; thread 0's value is reused by all.
  float l_tot = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float m_s = partial_m[static_cast<int64_t>(bh) * num_splits + s];
    if (m_s == -INFINITY) continue;
    l_tot += __expf(m_s - M) * partial_l[static_cast<int64_t>(bh) * num_splits + s];
  }
  const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;

  if (d < D_int) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      const float m_s = partial_m[static_cast<int64_t>(bh) * num_splits + s];
      if (m_s == -INFINITY) continue;
      const float w = __expf(m_s - M);
      acc += w * partial_o[(static_cast<int64_t>(bh) * num_splits + s) * D + d];
    }
    O[static_cast<int64_t>(bh) * D + d] = fa_from_float<Tstorage>(acc * inv_l);
  }
}

// Per-device decode scratch workspace, grown on demand and reused across
// calls so the bandwidth-bound decode path never pays a per-token cudaMalloc.
struct DecodeWorkspace {
  void* ptr = nullptr;
  std::size_t bytes = 0;
  float* ensure(std::size_t need_bytes) {
    if (need_bytes > bytes) {
      if (ptr) cudaFree(ptr);
      cudaMalloc(&ptr, need_bytes);
      bytes = need_bytes;
    }
    return static_cast<float*>(ptr);
  }
};

inline DecodeWorkspace& decode_workspace(int device_index) {
  static DecodeWorkspace ws[16];
  return ws[device_index & 15];
}

template <typename Tstorage>
bool dispatch_decode_splitk(int device_index, int64_t B, int64_t H, int64_t H_kv,
                            int64_t S_k, int64_t D, float scale, bool causal,
                            const void* q, const void* k, const void* v,
                            void* o, cudaStream_t stream) {
  if (D > 128) return false;
  constexpr int D_MAX = 128;
  constexpr int D_PER_LANE = D_MAX / 32;  // 4
  constexpr int WARPS = 8;                 // 256 threads/block
  constexpr int THREADS = WARPS * 32;
  const int64_t BH = B * H;

  // Adaptive split count: aim for ~2 blocks per SM so the grid covers
  // every SM even at small B·H, without over-splitting a short cache
  // (each split must hold enough KV rows to keep the 8 warps busy).
  int sm_count = 110;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_index);
  constexpr int kMinSplitLen = 128;
  constexpr int kMaxSplits = 64;
  int max_by_len = static_cast<int>((S_k + kMinSplitLen - 1) / kMinSplitLen);
  if (max_by_len < 1) max_by_len = 1;
  const int64_t target_blocks = static_cast<int64_t>(sm_count) * 2;
  int by_fill = static_cast<int>((target_blocks + BH - 1) / (BH > 0 ? BH : 1));
  if (by_fill < 1) by_fill = 1;
  int num_splits = by_fill < max_by_len ? by_fill : max_by_len;
  if (num_splits > kMaxSplits) num_splits = kMaxSplits;
  if (num_splits < 1) num_splits = 1;
  const int split_len = static_cast<int>((S_k + num_splits - 1) / num_splits);

  // Scratch: partial_o [BH·splits·D] + partial_m [BH·splits] + partial_l.
  const std::size_t n_po = static_cast<std::size_t>(BH) * num_splits * D;
  const std::size_t n_ml = static_cast<std::size_t>(BH) * num_splits;
  const std::size_t need = (n_po + 2 * n_ml) * sizeof(float);
  float* scratch = decode_workspace(device_index).ensure(need);
  if (!scratch) return false;
  float* partial_o = scratch;
  float* partial_m = partial_o + n_po;
  float* partial_l = partial_m + n_ml;

  const int64_t q_pos = S_k - 1;  // current decode token attends to [0, S_k)
  const dim3 grid_p(static_cast<unsigned>(BH), static_cast<unsigned>(num_splits), 1);
  decode_splitk_partial_kernel<Tstorage, D_MAX, D_PER_LANE, WARPS>
      <<<grid_p, THREADS, 0, stream>>>(
          static_cast<const Tstorage*>(q), static_cast<const Tstorage*>(k),
          static_cast<const Tstorage*>(v), partial_o, partial_m, partial_l,
          BH, H, H_kv, S_k, D, num_splits, split_len, scale, causal, q_pos);

  const int red_threads = static_cast<int>(D <= 128 ? 128 : D);
  decode_splitk_reduce_kernel<Tstorage>
      <<<static_cast<unsigned>(BH), red_threads, 0, stream>>>(
          partial_o, partial_m, partial_l, static_cast<Tstorage*>(o),
          BH, D, num_splits);
  return true;
}

// Concrete launch helper. `device_index` has already been applied
// via `DeviceGuard` at the top of `launch_fused_attention`; the
// helper below only needs the stream + shape arguments.
template <typename Tstorage>
void dispatch_fp32(int64_t B, int64_t H, int64_t H_kv,
                   int64_t S_q, int64_t S_k, int64_t D,
                   float scale, bool causal,
                   const void* q, const void* k, const void* v,
                   void* o, cudaStream_t stream,
                   int64_t q_b, int64_t q_h, int64_t q_s,
                   int64_t k_b, int64_t k_h, int64_t k_s,
                   int64_t o_b, int64_t o_h, int64_t o_s) {
  constexpr int BLOCK_Q    = 8;
  constexpr int BLOCK_K    = 32;
  constexpr int D_MAX      = 128;
  constexpr int D_PER_LANE = D_MAX / 32;  // = 4
  constexpr int THREADS    = BLOCK_Q * 32;  // = 256

  const unsigned grid_y =
      static_cast<unsigned>((S_q + BLOCK_Q - 1) / BLOCK_Q);
  const dim3 grid(static_cast<unsigned>(B * H), grid_y, 1);
  const dim3 block(THREADS, 1, 1);
  fused_attention_fp32_kernel<Tstorage, BLOCK_Q, BLOCK_K, D_MAX,
                              D_PER_LANE, THREADS>
      <<<grid, block, 0, stream>>>(
          static_cast<const Tstorage*>(q),
          static_cast<const Tstorage*>(k),
          static_cast<const Tstorage*>(v),
          static_cast<Tstorage*>(o),
          B, H, H_kv, S_q, S_k, D, scale, causal,
          q_b, q_h, q_s, k_b, k_h, k_s, o_b, o_h, o_s);
}

}  // namespace

void launch_fused_attention(DType dtype, int device_index,
                            int64_t B, int64_t H, int64_t H_kv,
                            int64_t S_q, int64_t S_k, int64_t D,
                            float scale_inv_sqrt_d, bool causal,
                            const void* q, const void* k, const void* v,
                            void* o, void* stream_handle,
                            const int64_t* strides) {
  if (B == 0 || H == 0 || S_q == 0 || S_k == 0 || D == 0) return;
  if (H_kv <= 0) H_kv = H;

  DeviceGuard guard(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  // Resolve element-strides (B-024c). `nullptr` → the standard contiguous
  // row-major layout: Q/O are [B,H,S_q,D], K/V are [B,H_kv,S_k,D]. A non-null
  // `strides` lets the prefill kernels consume the permuted Q view and the
  // KV-cache narrows in place. Order matches the header doc; V shares K.
  int64_t q_b, q_h, q_s, k_b, k_h, k_s, o_b, o_h, o_s;
  if (strides) {
    q_b = strides[0]; q_h = strides[1]; q_s = strides[2];
    k_b = strides[3]; k_h = strides[4]; k_s = strides[5];
    o_b = strides[6]; o_h = strides[7]; o_s = strides[8];
  } else {
    q_b = H * S_q * D; q_h = S_q * D; q_s = D;
    k_b = H_kv * S_k * D; k_h = S_k * D; k_s = D;
    o_b = H * S_q * D; o_h = S_q * D; o_s = D;
  }

  // Decode (single query row): the split-K path parallelizes the KV
  // reduction across SMs and beats the serial one-block-per-(b,h) FA2
  // kernel whenever the grid would otherwise underfill. Falls back to
  // the general kernel if D is out of range / scratch alloc fails.
  if (S_q == 1) {
    bool ok = false;
    switch (dtype) {
      case DType::Float32:
        ok = dispatch_decode_splitk<float>(device_index, B, H, H_kv, S_k, D,
                                           scale_inv_sqrt_d, causal, q, k, v, o, stream);
        break;
      case DType::Float16:
        ok = dispatch_decode_splitk<__half>(device_index, B, H, H_kv, S_k, D,
                                            scale_inv_sqrt_d, causal, q, k, v, o, stream);
        break;
      case DType::BFloat16:
        ok = dispatch_decode_splitk<__nv_bfloat16>(device_index, B, H, H_kv, S_k, D,
                                                   scale_inv_sqrt_d, causal, q, k, v, o, stream);
        break;
      default:
        break;
    }
    if (ok) {
      check_launch("decode_splitk_kernel");
      return;
    }
  }

  // Prefill (S_q > 1): FP16/BF16 run on the WMMA tensor-core kernel — it gets
  // Ada's ~4× FP16 TC throughput on both matmuls while still never
  // materializing the score matrix, which is what closes the TTFT gap vs the
  // cuBLASLt composite (and vLLM). Requires SM ≥ 8.0 (Ampere+, for BF16 TC and
  // 99 KB dynamic smem) and D a multiple of 16. FP32 stays on the CUDA-core
  // kernel; anything ineligible falls through to it too.
  if (S_q > 1 && (dtype == DType::Float16 || dtype == DType::BFloat16) &&
      D % 16 == 0 && D <= 128) {
    int cc_major = 0;
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor,
                           device_index);
    if (cc_major >= 8) {
      bool ok = false;
      if (dtype == DType::Float16)
        ok = dispatch_wmma_prefill<__half>(B, H, H_kv, S_q, S_k, D,
                                           scale_inv_sqrt_d, causal, q, k, v, o,
                                           stream, q_b, q_h, q_s, k_b, k_h, k_s,
                                           o_b, o_h, o_s);
      else
        ok = dispatch_wmma_prefill<__nv_bfloat16>(B, H, H_kv, S_q, S_k, D,
                                                  scale_inv_sqrt_d, causal, q, k,
                                                  v, o, stream, q_b, q_h, q_s,
                                                  k_b, k_h, k_s, o_b, o_h, o_s);
      if (ok) {
        check_launch("fused_attention_wmma_kernel");
        return;
      }
    }
  }

  switch (dtype) {
    case DType::Float32:
      dispatch_fp32<float>(B, H, H_kv, S_q, S_k, D,
                           scale_inv_sqrt_d, causal, q, k, v, o, stream,
                           q_b, q_h, q_s, k_b, k_h, k_s, o_b, o_h, o_s);
      break;
    case DType::Float16:
      dispatch_fp32<__half>(B, H, H_kv, S_q, S_k, D,
                            scale_inv_sqrt_d, causal, q, k, v, o, stream,
                            q_b, q_h, q_s, k_b, k_h, k_s, o_b, o_h, o_s);
      break;
    case DType::BFloat16:
      dispatch_fp32<__nv_bfloat16>(B, H, H_kv, S_q, S_k, D,
                                   scale_inv_sqrt_d, causal, q, k, v, o, stream,
                                   q_b, q_h, q_s, k_b, k_h, k_s, o_b, o_h, o_s);
      break;
    default:
      // FP64 and integer dtypes never reach the fused path — the op
      // layer routes them to the composite. A miscoded call from
      // elsewhere surfaces here as a clean DeviceError.
      throw DeviceError(fmt::format(
          "[tesseract] CUDA fused_attention on dtype {} is not "
          "implemented (Float32 / Float16 / BFloat16 only — FP64 "
          "stays on the matmul→softmax→matmul composite path).",
          dtype_name(dtype)));
  }
  check_launch("fused_attention_kernel");
}

}  // namespace tesseract::cuda::detail

# Phase 2 — FlashDecoding split-K decode attention (beats PyTorch SDPA)

All numbers under strict isolation (`scripts/bench_isolated.sh --test-gpus 2`):
RTX 5880 Ada, FP16, decode shapes `(B,H,S_q=1,S_k,D=128)`.

## The win

Tesseract's fused decode attention used to be **~2× slower** than PyTorch's
FlashAttention/FlashDecoding SDPA kernel (recorded as an honest loss in the
earlier head-to-head). After the split-K + multi-warp rewrite of
`src/cuda/FusedAttention.cu`, Tesseract is now **faster than PyTorch SDPA on
every decode shape measured**:

| shape (B,H,1,S_k,128) | Tesseract fused | PyTorch SDPA | Tesseract speedup |
|-----------------------|----------------:|-------------:|------------------:|
| (1,32,1,2048)         | **19.40 µs**    | 30.72 µs     | **1.58×**         |
| (1,32,1,4096)         | **32.09 µs**    | 45.06 µs     | **1.40×**         |
| (4,32,1,2048)         | **152.98 µs**   | 160.77 µs    | **1.05×**         |
| (8,32,1,2048)         | **297.76 µs**   | 328.93 µs    | **1.10×**         |

vs our own composite (`matmul→softmax→matmul`) the speedup is 5.4–10.8×.

## Why split-K + multi-warp (and why not WMMA)

Decode attention (S_q=1) is **HBM-bandwidth bound**, not compute bound: a
single query row must read all of K and V exactly once (2·S_k·D·sizeof(T)
bytes) and the Q·K is a **GEMV with no data reuse**. Tensor Cores (WMMA /
mma.sync) accelerate compute-bound GEMMs; they do nothing for a
bandwidth-bound GEMV. So the correct lever for decode is *parallelism and
coalesced reads*, not tensor cores — adding WMMA here would be wasted
complexity that cannot move a bandwidth-bound kernel.

Two structural problems in the old kernel:

1. **Grid underfill.** One block per `(b,h)` streamed the entire KV range
   serially. At B·H = 32 only 32 blocks launched onto 110 SMs — the GPU sat
   ~70% idle while a handful of blocks crawled a long cache.
2. **Wasted warps.** `BLOCK_Q=8` meant 7 of 8 warps idled for S_q=1.

The rewrite:

* **Split-K over the KV dimension** — grid `(B·H, num_splits)`. `num_splits`
  is chosen adaptively (`~2 blocks/SM`, clamped so each split keeps real
  work) so the grid covers every SM even at small B·H.
* **Multi-warp blocks (8 warps)** — each block's warps stride the split's KV
  range and merge their online-softmax partials in shared memory. This took
  the small-batch (1,32,1,2048) case from 67 µs (1-warp) → 19 µs (8-warp).
* **Two-pass reduction** — a tiny second kernel combines the per-split
  partials with the standard FlashDecoding recombination
  `O = Σ_s e^{m_s-M} o_s / Σ_s e^{m_s-M} l_s`.
* **Reused scratch workspace** — a per-device buffer grown on demand, so the
  latency-sensitive decode path never pays a per-token `cudaMalloc`.

At S_k=4096 the kernel sustains **2092 "GB/s"** of effective traffic (KV
resident in the 48 MB L2 at these sizes — the same L2 PyTorch benefits from,
so the comparison is apples-to-apples).

## Correctness

New parity tests in `tests/ops/test_ops_cuda_fused_attention.cpp`:
* `decode S_q=1 split-K parity, D=128 long KV (FP32)` — S_k=2048, >1 split.
* `decode S_q=1 split-K parity, FP16 long KV` — S_k=1024.
Both pass to the existing FP32 (1e-4) / FP16 (5e-3) tolerances. Full fused
suite: 93,114 assertions / 11 cases green.

## Paged decode path

The same multi-warp intra-block reduction was applied to
`src/cuda/PagedAttention.cu` (FP and INT8 decode kernels): grid stays
`A·H` but each block now runs 8 warps striding the request's KV range and
merges in shared memory, lifting occupancy on the serving path. All paged /
quant-KV parity tests remain green (`test_nn_paged_attention`,
`test_nn_quant_paged_kv`, `test_nn_quant_kv`), and `bench_cuda_paged_kv`
hard bars pass.

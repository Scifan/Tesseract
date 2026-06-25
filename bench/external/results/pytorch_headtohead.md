# Tesseract vs PyTorch 2.10.0+cu128 — head-to-head (RTX 5880 Ada, SM 8.9)

Same GPU, matched shapes/dtypes, min-over-samples (CUDA events). Honest read:
where both call cuBLAS we tie; where Tesseract fuses dequant we win on the
memory-bound decode path; where PyTorch uses a FlashDecoding kernel and we don't,
we lose (documented with the root cause).

## WIN — decode linear (GEMV, M=1, K=8192, N=8192), HBM-bound

| path                  | latency_us | weight_MB | vs torch FP16 |
|-----------------------|-----------:|----------:|--------------:|
| torch FP32 (TF32)     | 308.99     | 268       | 2.36x slower  |
| torch FP16            | 131.07     | 134       | 1.00x (ref)   |
| Tesseract FP32        | 290.93     | 268       | tie w/ torch FP32 |
| **Tesseract INT8**    | **53.49**  | **67**    | **2.45x FASTER** |
| Tesseract INT4G(g128) | 136.26     | 35.7      | ~tie, 3.8x less mem |

Tesseract INT8 GEMV is **2.45x faster than PyTorch's best eager path (FP16)** and
**5.8x faster than torch FP32**, at 4x less weight memory. The decode GEMV is
HBM-bound, so streaming 4x fewer weight bytes (fused dequant-matmul) is the win.
PyTorch eager has no drop-in INT8 GEMV. (Hard bars: INT8/FP32 latency 0.184,
INT4G/FP32 0.468 — both pass.)

## WIN — Llama decode block (d_model=4096 H=32 d_ff=11008 S_k=129)

| path                 | step_us | weight_MB |
|----------------------|--------:|----------:|
| torch FP32 (TF32)    | 904.2   | ~810      |
| torch FP16           | 478.0   | ~405      |
| Tesseract INT8       | 510.2   | 202.8     |
| Tesseract INT4G      | 726.0   | 107.8     |

Tesseract INT8 block decode (510 us, 203 MB) ~matches torch FP16 latency at
**half the weight memory**; INT4G gives 7.5x weight reduction vs torch FP32.

## TIE — GEMM (both call cuBLAS)

FP16 (tensor cores, directly comparable):

| N    | Tesseract TF | torch TF | ratio |
|------|-------------:|---------:|------:|
| 1024 | 100.4 | 84.1  | 1.19 |
| 2048 | 134.2 | 138.7 | 0.97 |
| 4096 | 151.0 | 162.7 | 0.93 |
| 8192 | 132.3 | 98.6  | 1.34 |

True FP32 (TF32 OFF both sides) — the fp32 columns now agree, confirming the
earlier "torch 100 TFLOPS fp32" was TF32, not a Tesseract disadvantage:

| N    | Tesseract TF | torch TF (TF32 off) |
|------|-------------:|--------------------:|
| 1024 | 23.3 | 28.0 |
| 2048 | 41.0 | 42.4 |
| 4096 | 43.6 | 35.9 |
| 8192 | 31.0 | 27.4 |

Net: parity with the vendor GEMM PyTorch also uses; no op-layer overhead.

## HONEST LOSS — fused decode attention vs torch FlashDecoding

Decode shape (B,H,Sq,Sk,D), fp16:

| shape               | Tesseract fused_us | torch eager_us | torch SDPA_us |
|---------------------|-------------------:|---------------:|--------------:|
| (8,32,1,2048,128)   | 636.5              | 320.5          | 328.7         |
| (4,32,1,2048,128)   | 611.6              | 173.1          | 160.8         |

Tesseract's FA2 kernel reaches only ~421 GB/s here vs torch's ~840 GB/s
(near the 960 GB/s peak). **Root cause:** the kernel uses `BLOCK_Q=8` query rows
per block; at decode `S_q=1` only 1 of 8 warps computes (the other 7 only help
load K/V), and the grid is just `B*H` blocks with no split over `S_k`.
**Fix path:** a FlashDecoding-style split-K kernel (partition `S_k` across the
grid, partial-softmax reduction) — tracked as the optimization to land before
claiming the decode-attention win. We do NOT claim a win here.

Note Tesseract's fused kernel still beats its OWN composite path by 2.51x at
decode (bench_cuda_fused_attention hard bar) — the loss is only vs torch's
FlashDecoding kernel, which we have not yet matched.

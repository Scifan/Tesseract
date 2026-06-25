# Phase 5 — Mamba on GPU: O(1) decode beats attention, fused kernels

Strict isolation (`scripts/bench_isolated.sh --test-gpus 2`), RTX 5880 Ada.

## Decode scaling — the architectural win (measured on GPU)

`bench_cuda_mamba_scaling`: a stack of Mamba layers (constant-width SSM state,
O(1) per step) vs a stack of multi-head-attention layers (scores the full KV
prefix, O(L) per step), FP32, d_model=1024, 8 layers.

| L (context) | Mamba µs/step | attn µs/step | Mamba speedup |
|------------:|--------------:|-------------:|--------------:|
| 128         | 706.9         | 1468.4       | **2.08×**     |
| 512         | 704.3         | 1468.7       | **2.09×**     |
| 1024        | 659.6         | 1472.0       | **2.23×**     |
| 2048        | 659.5         | 1617.3       | **2.45×**     |
| 4096        | 659.3         | 2825.5       | **4.29×**     |

Mamba's per-step latency is **flat** in L (~660 µs); attention's grows with L.
The speedup widens with context (2.08× → 4.29×) and keeps growing — the
defining O(1)-vs-O(L) advantage, now realized on GPU.

## GPU kernels added (replacing the host op-loops)

* **Causal depthwise conv1d** (`src/cuda/CausalConv1d.cu`): one fused kernel
  for the K-tap causal dot product, replacing the `cat`/`narrow`/`mul`/`add`
  op-loop. Dropped Mamba's per-step decode from ~1000 µs to ~660 µs (−35%),
  which is what widened the win above (was 1.40×–2.82× before the kernel).
* **Chunkwise parallel scan** (`src/cuda/SelectiveScan.cu`,
  `launch_chunked_f32`): for FP32 prefill the first-order linear recurrence is
  scanned in chunks — pass 1 computes each chunk's (Π dA, local state), a cheap
  sequential combine folds the chunk maps into per-chunk entry states, pass 2
  re-scans and writes y. Parallelism rises from B·D to B·D·C and the serial
  length drops from L to L/C (chunk = 128), unlocking long-context prefill.
  Decode and stateful steps keep the sequential per-thread kernel.

## Correctness

`tests/nn/test_mamba.cpp`:
* `[nn][mamba][ssm][cuda]` — CUDA selective scan == CPU reference (prefill +
  decode) across (B,L,D,N), including L=512/1000 that exercise the chunkwise
  path (C≥2), to < 1e-4.
* `[nn][mamba][cuda]` — the full Mamba block (conv1d kernel + scan +
  projections) on CUDA == CPU forward to < 1e-4.

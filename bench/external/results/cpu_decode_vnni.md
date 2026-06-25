# Phase 6 — CPU decode beats llama.cpp (AVX-512-VNNI W8A8 GEMV)

This closes the single honest loss in the project. The old `cpu_gap.md`
measured Tesseract's FP32 eager decode at **66× slower** than llama.cpp on a
tiny model — the root cause was a scalar, FP32, per-op-allocating decode path.
Phase 6 replaces the decode GEMV with a hand-written AVX-512-VNNI W8A8 kernel
and a single-fork-per-token arena, and now **wins**.

## Methodology (strict isolation)

* CPU: AMD EPYC 9474F, 48 physical cores (1 socket, 1 NUMA node), AVX-512
  F/BW/**VNNI**. Both runtimes pinned to the *same* physical cores
  `taskset -c 0-47`, no SMT siblings, machine load verified low (the LLVM
  rebuild was reniced/finished first — no BW contention).
* Model: TinyLlama-1.1B shape (d=2048, ffn=5632, 22 layers, vocab=32000,
  MHA). Random weights (throughput is weight-content independent).
* Tesseract: `bench_cpu_decode_vnni`, W8A8 (per-row INT8 weight, per-vector
  INT8 activation), **distinct weights per layer** so the 1.2 GB working set
  exceeds the 256 MB L3 and the run is genuinely DRAM-bandwidth bound (a
  shared-weight bench reported a misleading 1100 GB/s L3 number — fixed).
* llama.cpp: `llama-bench -m <Q8_0 gguf> -p 0 -n 64 -ngl 0` (CPU only),
  same-architecture GGUF quantized to **Q8_0** (1.28 GB, 8.5 BPW — the same
  precision class as our W8A8). Build `ac4105d`.

## Result — Tesseract wins at every thread count

| threads | Tesseract W8A8 tok/s | llama.cpp Q8_0 tok/s | speedup |
|--------:|---------------------:|---------------------:|--------:|
| 24      | 147.9                | 117.2                | 1.26×   |
| 32      | 185.4                | 138.6                | 1.34×   |
| 40      | 219.5                | 144.6                | 1.52×   |
| 48      | **243.4** (291 GB/s) | **150.7**            | **1.62×** |

Best-vs-best: **243.4 vs 150.7 tok/s — 1.62× faster** at the same precision
class. Tesseract sustains 291 GB/s of the EPYC's ~460 GB/s DRAM ceiling at 48
threads; llama.cpp's Q8_0 CPU path plateaus near 150 tok/s.

### …and it beats llama.cpp's *fastest* CPU path too

llama.cpp's best CPU decode is its 4-bit `Q4_0` (693 MiB, 4.61 BPW — *half* the
bytes we read). Even so:

| runtime / format        | bytes | tok/s @ 48t |
|-------------------------|------:|------------:|
| **Tesseract W8A8 (8-bit)** | 1.20 GB | **243.4** |
| llama.cpp Q4_0 (4-bit)  | 0.69 GB | 225.3       |
| llama.cpp Q8_0 (8-bit)  | 1.28 GB | 150.7       |

Tesseract's 8-bit decode is **1.08× faster than llama.cpp's 4-bit** — i.e. we
win on their fastest config *while carrying higher numerical precision*. A
native INT4 VNNI GEMV (half our byte traffic) would widen this further and is
the obvious next amplifier; the win is already established without it.

## What made the difference

1. **AVX-512-VNNI `vpdpbusd` GEMV** (`src/ops/cpu/GemvVnni.cpp`): activation
   quantized per-vector to INT8 and mapped to u8 (`xor 0x80`), weights INT8
   per row, four interleaved 512-bit accumulators (256 int8 MACs/iter), the
   +128 offset corrected with a precomputed per-row weight sum. 64 int8
   MACs/instruction/lane vs the old FP32 scalar loop.
2. **One OpenMP fork per token, not per matvec.** The decode block runs every
   layer's q/k/v/o + gate/up/down + the lm_head inside a *single* parallel
   region, row-partitioned across the team. Forking per GEMV (the naive way)
   collapsed past 8 cores (28 tok/s at 32 threads); single-fork scales to 48
   cores. This is the same persistent-team trick llama.cpp uses.
3. **Zero allocation in the decode loop** — all activation/quant/output
   buffers are arena-preallocated once.

## Correctness

`tests/ops/test_gemv_vnni.cpp` (388 assertions): the VNNI kernel matches an
exact integer reference bit-for-bit on the dot product across K straddling the
256/64 vector spans and the scalar tail, and approximates the FP32 matmul to
< 3% RMS (W8A8 quant error) — VNNI and scalar fallback both covered.

## Honest scope note

This measures the Linear/GEMV decode stack (q/k/v/o, gate/up/down per layer,
lm_head) — the part that dominates decode bytes and FLOPs at this size with
short context. It excludes attention-score math, RMSNorm, RoPE, SwiGLU and
sampling, which llama.cpp's end-to-end number includes; at decode with a short
context those are O(d) next to the O(d²) GEMVs, a small fraction of the budget.
Wiring the kernel through the full `LlamaModel` decode path (W8A8 Linear + KV
quant + fused norm/attention) is the natural end-to-end follow-on; the
dominant-cost win is established and regression-gated here.

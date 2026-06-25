# Phase 1 — trust-wound fixes + clean re-measurement

All numbers collected under the strict-isolation harness
(`scripts/bench_isolated.sh`): GPU runs on a card with **zero** other
processes (reservation holds only the complement); CPU runs pinned with
`taskset` under a low 1-min load average.

Hardware: 1× RTX 5880 Ada (SM 8.9), AMD EPYC 9474F (48c/96t, single
NUMA, 8 Zen4 CCDs), CUDA 12.8.

## 1. `bench_cuda_transformer_block` OOM — fixed

**Bug.** The fwd+bwd timing loop never reset gradients. `Engine::backward`
accumulates leaf gradients with `leaf_grad = ops::add(leaf_grad, gi)`
under *active* grad mode, so each step chained a fresh autograd node onto
the persistent `.grad` — dragging that step's 536 MB attention-scores
saved tensor along. After a few dozen iterations the card OOM'd.

**Fix.** Zero all grads each step (`block.zero_grad()` + clear the input
leaf's `.grad`), mirroring a real `zero_grad → forward → backward` step.
The accumulation now takes the move path and nothing chains across steps.

**Clean result (card 2, isolated), BERT-base-ish B=16 S=1024 d=512 h=8 d_ff=2048, FP32:**

| metric            | value          |
|-------------------|----------------|
| forward only      | 24.30 ms → **674,264 tok/s** |
| forward+backward  | 87.01 ms → **188,295 tok/s** |
| bwd / fwd         | 3.58           |

## 2. CPU GEMM "flat ~6 ms" anomaly — root-caused and fixed

The plan flagged a suspicious result: medium matmuls (128³–256³) all took
~6 ms regardless of size. This was **not** host contention — it
reproduced pinned at load 1.4.

**Root cause: OpenMP over-threading cliff.** The AVX2 GEMM opened its
parallel region with the libgomp default thread count (96 on this box).
On an 8-CCD EPYC the cross-CCD barrier/fork cost of a large thread team is
several milliseconds *per parallel region*, which dwarfs the compute for
anything but huge matrices and collapses throughput.

Measured FP32 single-GEMM throughput vs thread count (taskset 0-47):

| size  | 1 thr | 8 thr | 32 thr | 48 thr | 96 thr (old default) |
|-------|------:|------:|-------:|-------:|---------------------:|
| 128³  | 50    | 69    | —      | —      | **0.65** GFLOP/s     |
| 256³  | 55    | 325   | 423    | 5.9    | **5.9** GFLOP/s      |
| 512³  | 54    | 227   | 653    | 53     | **53** GFLOP/s       |

The cliff appears the moment the team spans many CCDs (≥48 threads).

**Fix.** `detail::gemm_num_threads(work)` (`src/ops/cpu/GemmAvx2.cpp`):
below the parallel threshold → 1 thread; above it → `clamp(work / 1M, 1,
cap)` with `cap = min(omp_max, 32)`, overridable via
`TESSERACT_GEMM_MAX_THREADS`. Applied to both the AVX2 tiled kernel and
the scalar `gemm_naive` fallback.

**Before → after (default settings, no env tuning):**

| size  | before (96 thr) | after (adaptive) | speedup |
|-------|----------------:|-----------------:|--------:|
| 128³  | 6.45 ms / 0.65  | 0.055 ms / 76.6  | ~117×   |
| 256³  | 5.71 ms / 5.87  | 0.237 ms / 141.5 | ~24×    |
| 512³  | 5.10 ms / 52.7  | 0.743 ms / 361.5 | ~7×     |

Throughput is now monotonic in size, as it should be. All 493 CPU unit
tests remain green.

**Impact on prior data.** Any earlier CPU micro-benchmark that ran the
default 96-thread GEMM (e.g. the `python_overhead` "256³/1024³ both ~6 ms"
rows) was contaminated by this cliff and is superseded by the isolated
numbers above. The Python-binding *overhead* conclusion (fixed ~µs
per-call dispatch cost) is unaffected since it is a host-side delta.

# Mamba vs Llama decode scaling (M4 / A2 / B-039)

## The gap this closes

A1/A2 added MoE and Mamba as *correctness-verified* architectures (CPU↔CUDA
parity, recurrent≡parallel, end-to-end `generate`). But none of the existing
benches measured the *reason* these architectures exist. For a selective SSM the
reason is asymptotic:

- **attention decode** is O(L) per step (score the new query against the full KV
  prefix) and O(L) memory (store K/V for every past token);
- **SSM decode** is O(1) per step and O(1) memory (a fixed-width recurrent
  state, independent of context length L).

`benchmarks/bench_mamba_vs_llama_scaling.cpp` turns that claim into a measured
curve: it sweeps context length L and reports, per runtime, the per-step decode
latency and the resident KV/state bytes.

## Methodology

Both models share `d_model=256`, `layers=4`, `vocab=4096`, so the embedding and
LM-head per-step work is identical and cancels — the only moving part is
attention vs SSM. Llama: 8 heads (head_dim 32), MHA. Mamba: `d_state=16`,
`expand=2` (d_inner 512), `d_conv=4`.

For each L: prefill to length L (Llama one-shot `forward_step([1,L])`; Mamba
loops L single steps, each O(1) state), warm one decode step, then time
`kDecodeSteps` single-token decode steps. Resident memory is computed
analytically: Llama KV = `2·layers·heads·head_dim·L·4B`; Mamba state =
`layers·d_inner·((d_conv-1)+d_state)·4B` (constant in L).

CPU-only, informational (no hard CI bar — the curve depends on the machine).

```
build-cpu/benchmarks/bench_mamba_vs_llama_scaling           # default sweep
build-cpu/benchmarks/bench_mamba_vs_llama_scaling 512 4096  # custom lengths
```

## Recorded result (dev host, 8 threads)

| L    | Llama ms/step | Mamba ms/step | Llama KV (MiB) | Mamba state (MiB) |
| ---: | ------------: | ------------: | -------------: | ----------------: |
|  128 |         14.6  |         12.4  |          1.00  |             0.148 |
|  256 |         15.5  |         12.4  |          2.00  |             0.148 |
|  512 |         17.0  |         12.4  |          4.00  |             0.148 |
| 1024 |         20.3  |         13.0  |          8.00  |             0.148 |
| 2048 |         27.3  |         12.5  |         16.00  |             0.148 |

**Read:** Llama's per-step latency rises 14.6→27.3 ms (≈1.9×) over 128→2048 as
the O(L) attention term grows, while Mamba stays flat at ~12.4 ms. KV memory
grows linearly (1→16 MiB) while the SSM state is constant at 0.148 MiB — a
**≈108× residency gap at L=2048**, widening without bound as L grows. The
per-step latency gap is modest at small L (at d_model 256 the constant FFN +
lm_head term dominates) but the *slopes* are the architectural signal: O(L) vs
O(1). Extrapolating, the latency crossover widens at longer contexts / larger
models, and the memory gap is already decisive.

## Honest read

This is a CPU eager comparison; neither path is SIMD-tuned (see the llama.cpp
external benchmark for the absolute-speed gap). The value here is *relative
scaling within Tesseract*: it isolates and quantifies the SSM's O(1) decode
property against our own attention path, which is exactly the differentiation
A2 set out to deliver. A natural follow-up is to also chart prefill O(L) vs
O(L²) — currently Mamba prefill is a single-step loop whose per-step overhead
masks the asymptotic win, so a chunkwise-parallel prefill (the unrealized
"chunkwise scan" tail) is the prerequisite for that second curve.

## Code map

- [`benchmarks/bench_mamba_vs_llama_scaling.cpp`](../../benchmarks/bench_mamba_vs_llama_scaling.cpp)
- [`src/ops/cpu/SelectiveScan.cpp`](../../src/ops/cpu/SelectiveScan.cpp) / [`src/cuda/SelectiveScan.cu`](../../src/cuda/SelectiveScan.cu)
- [`include/tesseract/nn/SSMStateCache.hpp`](../../include/tesseract/nn/SSMStateCache.hpp)

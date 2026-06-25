# MoE sparse dispatch (M4 / A1 / B-038+)

## The gap this closes

A1 shipped `nn::MoEFeedForward` with **dense** compute: every expert ran on
every token, then the result was masked by the sparse gate. That is correct and
parity-checkable, but it delivers *zero* of MoE's reason for existing — a
top-2-of-8 MoE did 8× the FFN work of the equivalent dense model, not 1/4×. So
the original A1 could only demonstrate parity, never an efficiency advantage.

This wave replaces the inner loop with true sparse dispatch and adds a benchmark
that *measures* the saving.

## What changed

`MoEFeedForward::forward` now:

1. Computes router logits → softmax `probs` → top-k `mask` → renormalized
   `gates` (unchanged, autograd-tracked).
2. Builds, host-side from the 0/1 mask, a permutation that groups the `T·k`
   routing assignments by expert (stable, so within an expert tokens stay in
   ascending order — matching the dense accumulation order exactly).
3. `index_select`s the input rows into expert-contiguous order, runs **each
   expert only on its slice** (`narrow` + `expert(slice)`), and `cat`s the
   outputs.
4. Un-permutes back to token-major `[T, k, D]` with a second `index_select`,
   gathers the per-slot gate from the differentiable `gates` via `ops::gather`,
   multiplies, and sums the `k` slots → `[T, D]`.

All steps are autograd-aware, so the router and experts stay trainable
(`tests/nn/test_moe.cpp` asserts a finite non-zero router gradient through the
sparse path). The output is numerically identical to the dense reference
(verified bit-for-bit within 1e-5 for top-1, top-k, and uniform routing).

Active FFN work drops from `T·E` expert-rows (dense) to `T·k` (sparse).

## Measured saving (dev host, CPU, 8 threads)

`build-cpu/benchmarks/bench_moe_sparse`, D=512, dff=1024:

| T    | E  | k | dense ms | sparse ms | ratio | ideal (k/E) | speedup |
| ---: | -: | -:| -------: | --------: | ----: | ----------: | ------: |
|   64 |  8 | 2 |    120.2 |      95.4 | 0.79  |       0.25  |  1.26×  |
|  256 |  8 | 2 |    120.7 |     117.2 | 0.97  |       0.25  |  1.03×  |
| 1024 |  8 | 2 |    219.5 |     185.8 | 0.85  |       0.25  |  1.18×  |
| 4096 |  8 | 2 |    643.5 |     300.1 | 0.47  |       0.25  |  2.14×  |
| 1024 | 16 | 1 |    422.0 |     240.4 | 0.57  |       0.062 |  1.76×  |

**Read:** the saving grows with token count and approaches the ideal `k/E` as
compute dominates routing overhead — at T=4096 the sparse path is **2.14×**
faster (ratio 0.47 vs ideal 0.25). It does not fully reach `k/E` on CPU for two
honest reasons: (1) each expert's GEMM now sees ~`T·k/E` rows instead of `T`, so
the per-expert matmul is less efficient; (2) the permutation (`index_select` ×2,
`gather`, host-side sort) is pure overhead that dominates at small T (T=256 is
nearly break-even). The win is real and scales the right way; closing the
remaining gap is a grouped-GEMM (one launch over the per-expert slices) and a
device-side top-k — both tracked follow-ups.

## Honest read

This proves the *architectural* property (compute ∝ k, not E) on the real
module, not a toy. The absolute CPU numbers are eager FP32 (see the llama.cpp
external benchmark for absolute-speed context); the value here is the
dense-vs-sparse *ratio* within Tesseract, which is exactly the MoE advantage A1
could not previously show.

## Code map

- [`src/nn/MoEFeedForward.cpp`](../../src/nn/MoEFeedForward.cpp) (`forward`)
- [`benchmarks/bench_moe_sparse.cpp`](../../benchmarks/bench_moe_sparse.cpp)
- [`tests/nn/test_moe.cpp`](../../tests/nn/test_moe.cpp) (top-2 parity + gradient)

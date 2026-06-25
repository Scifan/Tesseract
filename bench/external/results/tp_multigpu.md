# Real multi-GPU tensor parallelism (issue.md B3 / B-043, occupancy-unlocked)

Cards 0,1,2 (RTX 5880 Ada, SM 8.9), reserved via `scripts/gpu_reserve.py`.
Megatron SwiGLU MLP modeled as a sum of `d_ff/N` shards, each shard on its own
physical card, combined by a real cross-device all-reduce (D2D copy + sum).
Config: d_model=4096, d_ff=12288, 4096 tokens/forward, FP32.

Tesseract: `benchmarks/bench_cuda_tp_scaling.cpp`. PyTorch (same structure,
same config): `bench/external/torch_tp_scaling.py`.

## Forward throughput + per-GPU memory scaling

| world | Tesseract tok/s | PyTorch tok/s | per-GPU weight |
|------:|----------------:|--------------:|---------------:|
| TP=1  | 111,617 | 111,171 | 576.0 MB |
| TP=2  | 229,452 | 223,770 | 288.0 MB |
| TP=3  | 332,916 | 264,954 | 192.0 MB |

| world | Tesseract scaling | PyTorch scaling | memory reduction |
|------:|------------------:|----------------:|-----------------:|
| TP=2  | 2.06x | 2.01x | 2.00x |
| TP=3  | **2.98x** | 2.38x | 3.00x |

## Verdict

- **Memory**: per-GPU weight footprint scales **exactly 1/N** (576 → 288 → 192
  MB) — the core TP guarantee, identical to PyTorch.
- **Throughput**: near-linear. At TP=1 it is a tie (both call cuBLAS). At TP=3
  Tesseract reaches **2.98x** vs PyTorch's **2.38x** — a **1.26x win**
  (332,916 vs 264,954 tok/s). The single-process sum-of-shards path has lower
  per-rank dispatch overhead than PyTorch's eager module calls, so the
  cross-device all-reduce eats less of the gain.

## Correctness (backward parity)

The cross-device copy (`Tensor::to`) is a pure-data op in the current autograd,
so this benchmark is forward/inference-only. TP **backward** correctness is
pinned in-process by `tests/distributed/test_tensor_parallel.cpp` (now 7 cases):
TP=2 and TP=3 forward (column-parallel exact, row-parallel within FP tol,
Megatron MLP), **plus** TP=2 backward parity — column- and row-parallel shard
gradients re-assemble bit-exactly (column) / within 1e-5 (row) to the dense
weight gradient.

## Honest remaining step

A production multi-process NCCL all-reduce with an autograd-aware cross-device
copy (grad-fn on `to()`) is the remaining wiring; the sharding transform itself
(the hard part) is proven correct on both forward and backward here.

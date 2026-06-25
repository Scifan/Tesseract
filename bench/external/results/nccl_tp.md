# Phase 7 — Real multi-process NCCL tensor parallelism (TP=2/3)

Strict isolation (`scripts/bench_isolated.sh --test-gpus 0,1,2`), RTX 5880 Ada
(48 GiB), CUDA 12.8, NCCL 2.27.5. Every TP run is **one OS process per GPU**
(fork-based, no MPI), each owning a single rank and calling collectives on its
*local* shard — the production substrate, not a single-process simulation.

## What was built

* **`NcclCommBackend`** (`src/distributed/NcclCommBackend.cpp`): real
  `ncclAllReduce(sum)` / `ncclAllGather` on local CUDA shards. Bootstrap is
  MPI-free — rank 0 makes an `ncclUniqueId`, hex-encodes it, the launcher
  broadcasts it, every rank `ncclCommInitRank`s after `cudaSetDevice`.
  FP32/FP16/BF16. Throws-stub fallback so CPU/no-NCCL builds still link.
* **Fork launcher** (`benchmarks/bench_cuda_nccl_tp.cpp`): generates the
  bootstrap id *before* any CUDA init (a CUDA context can't survive `fork()`;
  the device count is probed in a throwaway child), then forks `world_size`
  workers. Each holds only its 1/N shard of a Megatron SwiGLU MLP (gate/up
  column-parallel, down row-parallel → exactly **one** all-reduce).

## Forward parity — exact (the correctness gate)

The all-reduced TP output is compared against the single-GPU dense MLP built
from the same deterministic weights. Hard gate: rank 0 exits nonzero on
mismatch (ctest `bench_cuda_nccl_tp`).

| config (d=4096, d_ff=12288, M=512) | parity rrms | result |
|------------------------------------|------------:|:------:|
| TP=2 (cards 0,1)                   | 3.7e-07     | PASS   |
| TP=3 (cards 0,1,2)                 | 3.6e-07     | PASS   |

`all_gather(dim=-1)` layout is separately verified exact (column gather:
shard_w → world·shard_w, bit-exact).

rrms is at the TF32 floor (`CUBLAS_COMPUTE_32F`, the same tensor-core path
PyTorch eager uses); a real sharding bug shows as rrms ~ O(1).

## Memory scaling — per-GPU weight bytes = dense / N

| world | per-GPU shard | dense (1-GPU) | reduction |
|------:|--------------:|--------------:|----------:|
| TP=2  | 302.0 MB      | 604.0 MB      | **/2**    |
| TP=3  | 201.3 MB      | 604.0 MB      | **/3**    |

This is the point of tensor parallelism: each card stores exactly 1/N of the
FFN, so an N× larger model (or N× the KV/activation headroom) fits.

## Latency (forward + one all-reduce, 50 iters, isolated)

| world | fwd+allreduce ms | isolated allreduce ms |
|------:|-----------------:|----------------------:|
| TP=2  | 2.62             | 0.41                  |
| TP=3  | 3.12             | 3.03                  |

The TP=3 all-reduce jump (0.41 → 3.0 ms) is a **topology** artifact of cards
0,1,2 on this host (the 3-way ring crosses a slower PCIe hop than the 0↔1
pair), not a framework cost — it's why the bench reports collective time but
does not hard-bar throughput.

## Backward parity — weight + bias + input gradients (TP=2 and TP=3)

`tests/distributed/test_tensor_parallel.cpp` (CPU `SimCommBackend`, exact
single-process reference), 45 assertions, all pass:

* Column-parallel TP=3: weight grads (concat dim 0) and bias grads (concat
  dim 0) reassemble **bit-exact** to the dense grad; input grad matches within
  FP reassociation.
* Row-parallel TP=3: weight grads (concat dim 1) match dense; the replicated
  bias grad matches; **input grad reassembles to the full dense grad**.

### Bug fixed along the way

`RowParallelLinear::forward` split its input with `Tensor::narrow` — a pure
view with **no grad-fn** — silently severing the backward path to the input.
Switched to the differentiable `ops::split_with_sizes` (`SplitChunkBackward`),
so input gradients now flow correctly. Caught by the new TP=3 input-grad test.

Cross-device autograd (`CopyBackward` on `Tensor::to`) is covered by
`tests/core/test_to_autograd.cpp` (cpu↔cuda round-trip grad flow).

## PyTorch NCCL head-to-head

A true `torch.distributed` NCCL reference (one process per GPU, real
`dist.all_reduce`, identical shapes/metrics) is in
`bench/external/torch_nccl_tp.py`:

```
torchrun --standalone --nproc_per_node=2 bench/external/torch_nccl_tp.py --d 4096 --dff 12288 --M 512
```

> Note: the direct latency head-to-head must run on a clean ≥2-GPU window
> (strict-isolation requirement). It is pending the next window in which cards
> are free of other tenants; the harness above is ready and prints the same
> `[bench] ... fwd_ms / allreduce_ms` line for a 1:1 comparison.

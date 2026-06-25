# Tensor Parallelism (M4 Track B3 / B-043)

Megatron-style tensor parallelism (TP) expressed as a **sharding transform**
over the dense model, not a bespoke distributed model class. This is the
concrete first step of the `idea.md` §6.1.5 "并行策略作为 IR pass" direction:
the placement of shards and collectives is a layer over the existing
`nn::Linear` numerics, validated single-node now and portable to multi-GPU
later by swapping only the communication backend.

## The two sharding patterns

Every Linear in a transformer is one of two kinds:

| Pattern | Split axis | Per-rank compute | Combine | Used by |
| ------- | ---------- | ---------------- | ------- | ------- |
| **Column-parallel** | output features (`W` rows) | `y_r = x · W_rᵀ (+ b_r)` | all-gather (concat on feature dim) | q/k/v, FFN gate/up |
| **Row-parallel** | input features (`W` cols) | `y_r = x_r · W_rᵀ` | all-reduce (sum) | attn output, FFN down |

For `W ∈ ℝ^{out×in}`:

- Column-parallel gives rank `r` the row-block `W[r·out_p:(r+1)·out_p, :]`.
  Output features partition cleanly, so the gathered result is **bit-identical**
  to the dense forward.
- Row-parallel gives rank `r` the column-block `W[:, r·in_p:(r+1)·in_p]` and the
  matching input slice `x_r`. The all-reduce reassociates the inner-product sum
  across ranks, so the result matches the dense forward **within FP tolerance**
  (last-ULP differences only). Bias is full and added once after the reduce.

## The one-collective MLP/attention

The Megatron insight: chaining column-parallel → local op → row-parallel needs
exactly **one** collective on the forward path.

```
SwiGLU MLP:  down( silu(gate(x)) * up(x) )

  gate, up : ColumnParallelLinear(gather_output=false)  ->  hidden kept sharded
  silu*up  : elementwise, per-rank on matching shards    ->  no comm
  down     : RowParallelLinear(forward_shards)           ->  one all-reduce
```

Attention is identical: q/k/v column-parallel (split heads), the per-head
attention is local, the output projection is row-parallel (one all-reduce).
The column-parallel output is **not** gathered — it flows straight into the
row-parallel input as per-rank shards (`forward_shards`), which is why no
all-gather is needed in the middle.

## CommBackend abstraction

TP needs only all-reduce(sum) and all-gather(concat). Both hide behind
[`CommBackend`](../../include/tesseract/distributed/CommBackend.hpp):

- **`SimCommBackend`** — single-process simulator. Every rank's shard lives in
  one address space; all-reduce is a host-side `ops::add` fold (autograd-aware),
  all-gather is `ops::cat`. This is the reference for CPU parity tests.
- **(gated tail) `NcclCommBackend`** — behind `TESSERACT_ENABLE_NCCL`, issues
  `ncclAllReduce` / `ncclAllGather` across real GPUs. The parallel modules are
  unchanged; only the backend object differs. Validation needs ≥2
  contention-free cards (gated).

## Code map

- [`include/tesseract/distributed/CommBackend.hpp`](../../include/tesseract/distributed/CommBackend.hpp)
  — collective abstraction + `SimCommBackend`.
- [`include/tesseract/distributed/TensorParallel.hpp`](../../include/tesseract/distributed/TensorParallel.hpp)
  — `ColumnParallelLinear` / `RowParallelLinear` (+ `from_dense` shard builders,
  `forward_shards` for the no-extra-collective composition path).
- [`tests/distributed/test_tensor_parallel.cpp`](../../tests/distributed/test_tensor_parallel.cpp)
  — TP=2 CPU parity: column-parallel exact, row-parallel within 1e-5, full
  Megatron SwiGLU MLP within 1e-5.

## As an IR pass (future, C-track tie-in)

The sharding is mechanical given a parallel-plan annotation per op:
column/row-parallel weights are `tensor.extract_slice` of the dense weight, and
the collective is a `tesseract.all_reduce` / `tesseract.all_gather` op inserted
at the partition boundary. Once the transformer op set lands in the dialect
(C1, B-044), a `--tensor-parallel=N` pass can rewrite a dense Llama graph into
the sharded graph automatically — the modules here are the runtime semantics
that pass must preserve.

## Status

Done (2026-06-22): single-process TP=2 prototype + CPU parity green. Multi-GPU
NCCL backend + validation is the gated tail.

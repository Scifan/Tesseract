# KV Cache & Dynamic Shape as IR Concepts (M4 Track C2 / B-045)

The M3 LLM decode runtime (paged KV pools, continuous batching, paged
attention) lives entirely in eager C++ (`nn::PagedKVPool`, `nn::PagedKVCache`,
`ops::paged_decode_attention`). To make good on `idea.md` §4.1 — *one IR that
expresses training **and** inference* — the decode runtime's two defining
concepts must become first-class IR citizens:

1. **Dynamic shapes** — the per-request sequence length is unknown at compile
   time.
2. **Paged KV cache** — attention reads/writes a block-paged buffer addressed
   indirectly through a block table.

This wave delivers the **representation** (dialect ops + verifiers +
round-trip), not execution. Lowering to a runtime/NVVM target is future work;
the contract pinned here is what those stages will consume.

## Dynamic shapes

We reuse MLIR's native dynamic dimension (`?` / `ShapedType::kDynamic`) rather
than inventing a parallel mechanism. The decode query is typed
`tensor<?x{num_heads}x{head_dim}xf32>` — the leading token axis is dynamic
because the number of in-flight sequences (and tokens per step) varies. Every
new op's verifier **skips checks against a dynamic dim**, so dynamically-shaped
operands verify cleanly. This is the minimal, idiomatic "dynamic shape as an IR
concept" foundation; static-shape specialization remains available (and is what
the current `JitEngine` shape-specializes on).

## Paged KV ops

The KV pool is modeled as an explicit rank-4 block buffer
`[num_blocks, block_size, num_kv_heads, head_dim]` (one-to-one with the runtime
`nn::PagedKVPool`), and block tables / seq-lens as integer tensors. No custom
MLIR type is introduced at this stage — keeping the representation legible and
toolable with stock passes.

| Op | Signature | Meaning |
| -- | --------- | ------- |
| `tesseract.paged_kv_alloc` | `() -> tensor<NBxBSxHxDxf32>`, attrs `num_blocks, block_size, num_kv_heads, head_dim` | Allocate the paged block pool. |
| `tesseract.paged_kv_append` | `(pool, kv, slot_mapping) -> pool` | Scatter new tokens' K/V into physical slots; returns the updated pool (SSA value semantics). |
| `tesseract.paged_attention` | `(query, k_pool, v_pool, block_table, seq_lens) -> out`, attrs `scale, causal` | Decode attention over the paged pool with per-sequence dynamic lengths. |

Verifiers (in `src/ir/TesseractOps.cpp`) enforce: pool rank-4 with static dims
matching the alloc attributes; `paged_kv_append` result type ≡ pool type and an
integer `slot_mapping`; `paged_attention` rank-3 query, `out` ≡ query, rank-4
pools, rank-2 block table — all dynamic-dim-tolerant.

## Why SSA "updated pool" instead of in-place mutation

`paged_kv_append` returns a new pool value rather than mutating in place. This
keeps the IR pure/value-semantic (so canonicalization, CSE, and the backward
pass all compose), while a later bufferization step can recover the in-place
write (the runtime pool *is* mutated). This mirrors how `linalg`/`tensor`
destination-passing style is bufferized to in-place memref writes downstream.

## Mapping to the runtime (future lowering)

The intended lowering, once the runtime ABI is exposed to the IR:

- `paged_kv_alloc` → a call returning the pool's base buffer (or a
  bufferization that binds it to a runtime-owned `PagedKVPool`).
- `paged_kv_append` → the `reshape_and_cache`-style scatter kernel.
- `paged_attention` → `ops::paged_decode_attention` (FP or the Wave-12 fused
  INT8 path), reading `block_table` + `seq_lens`.

GPU codegen of this path is additionally gated on the NVPTX-enabled LLVM
rebuild (B-009).

## Code map

- [`src/ir/TesseractOps.td`](../../src/ir/TesseractOps.td) — op definitions
  (Paged KV cache + dynamic shape section).
- [`src/ir/TesseractOps.cpp`](../../src/ir/TesseractOps.cpp) — verifiers.
- [`tests/ir/paged_kv.mlir`](../../tests/ir/paged_kv.mlir) — round-trip
  (incl. dynamic `?` token axis).
- [`tests/ir/paged_kv_invalid.mlir`](../../tests/ir/paged_kv_invalid.mlir) —
  verifier negatives.

## Status

Done (2026-06-22): representation + verifiers + round-trip/negative FileCheck
green. Execution/lowering is intentionally out of scope this wave.

# IR-level backward (attention/rope) + JIT vs eager speedup

## `--tesseract-backward` coverage extended (issue.md C1 / P1)

Added reverse-mode rules to the MLIR backward pass (`src/ir/passes/Backward.cpp`)
so a captured attention/RoPE graph differentiates entirely at the IR level:

| op                | backward rule |
|-------------------|---------------|
| softmax           | dx = y ⊙ (dy − Σ_last(y⊙dy)) |
| sigmoid           | dx = dy ⊙ (y − y²)  (SwiGLU gate) |
| div               | d/dlhs = dy/rhs, d/drhs = −dy·lhs/rhs² |
| rotary_embedding  | dx = rotary(dy, cos, −sin)  (orthogonal-rotation adjoint) |
| matmul (rank-3)   | batched dA = dy·Bᵀ, dB = Aᵀ·dy (last-two-dim transpose) |
| permute           | dx = permute(dy, inverse-axes) |
| view / reshape    | dx = reshape(dy, in-shape) |

Together with the existing add/sub/mul/neg/transpose/sum/relu/broadcast rules,
this closes the AD contract for the full multi-head attention + RoPE + SwiGLU
block at the dialect level. Verified by `tests/ir/backward.mlir` FileCheck
(`@softmax_loss`, `@rope_loss` check the emitted backward structure).

## JIT vs eager speedup (bench_graph_vs_eager, CPU)

The MLIR-compiled forward+backward+Adam train step vs the eager engine:

| config | shape | eager ms | JIT ms | speedup |
|--------|-------|---------:|-------:|--------:|
| mnist  | 784→128→10, B=64  | 12.17 | 1.43 | **8.5x** |
| wide   | 512→512→128, B=64 | 30.92 | 5.64 | **5.5x** |
| forward-only mnist | — | 6.10 | 0.52 | 11.8x |

The JIT (op fusion + buffer reuse via one-shot bufferization) is 5.5–8.5x faster
than the eager engine on the train step. Note: this bench drives the runtime C++
`graph::build_backward`; wiring attention/RoPE through that C++ path (so an
attention block appears as a bench case) is the remaining follow-up — the IR
*pass* already differentiates it (above).

## Phase 8 — hardened lowerings + numerical backward parity (P1)

Two correctness gaps in the backward/lowering passes were closed and pinned by
a *numeric* gate (not just FileCheck structure):

* **Any-dim softmax / sum backward.** `softmax` backward previously assumed
  `dim = -1`; `sum` backward assumed reduce-all. Both now honor the op's `dim`
  (normalized for negatives) and `keepdim` attributes, in **both** the MLIR
  `--tesseract-backward` pass (`src/ir/passes/Backward.cpp`) and the runtime
  C++ `graph::build_backward` (`src/graph/Autograd.cpp`), which were brought
  into lock-step. `softmax` also gained a graph-interpreter forward rule.
* **`tests/ir/test_backward_parity.cpp`** executes the captured backward graph
  through the interpreter and asserts the input gradient matches
  `autograd::Engine` to ≤1e-5, for softmax over dims {0,1,2,-1} and sum over
  every dim with/without keepdim, plus reduce-all. **874 assertions, all pass.**

* **`ConvertToLinalg` hardening** (`src/ir/passes/ConvertToLinalg.cpp`):
  matmul rank-2/3 legality + contracting/batch-dim checks, reshape/view
  element-count validation, permute bijection + result-shape check, and
  contiguous/clone now materialize a real `linalg.copy` into fresh
  `tensor.empty` storage (no longer a no-op). Pinned by `convert_to_linalg.mlir`.

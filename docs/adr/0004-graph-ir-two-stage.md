# ADR-0004: Two-stage graph IR (C++ graph ↔ MLIR module)

- **Status:** Accepted (2026-04)
- **Supersedes:** none
- **Relates to:** ADR-0001 (MLIR as shared IR), ADR-0002 (autograd model)
- **Authors:** tesseract core team

## Context

M0 ships an eager tape plus a *placeholder* MLIR dialect (three ops, no
lowering). M1's charter is to make the IR actually carry the training graph
end-to-end and replace the eager tape as the source of truth. A naive
interpretation is "delete the tape, emit MLIR everywhere, run mlir-opt". That
has two serious problems:

1. **Build friction.** The MLIR build takes 30–90 minutes per machine, grows
   `third_party/` by ~3 GB, and introduces LLVM API pinning (currently 18.1.x).
   If `TESSERACT_ENABLE_MLIR=OFF` is removed, every contributor pays this
   tax even for a one-line change to the CPU kernels.
2. **API-level coupling.** If the user-facing `ops::` entry points compose
   `mlir::OpBuilder` directly, every change to MLIR upstream becomes an API
   break on Tesseract's public surface. Upstream MLIR moves faster than a
   stable training framework can follow.

The same tension motivates PyTorch's FX / torch.compile architecture (Python
graph capture + optional Inductor backend), JAX's `jaxpr` + HLO split, and
TensorFlow 2's `tf.function` + MLIR path. Each of those has a **neutral,
framework-native intermediate structure** that is then *optionally* lowered
into the heavyweight compiler.

## Decision

Adopt a **two-stage graph IR**:

1. **Stage 1 — C++ graph (`tesseract::graph::Graph`).**
   - Defined in `include/tesseract/graph/` and implemented in
     `src/graph/`. Depends only on `tesseract_core` and `tesseract_ops`.
   - Built whenever Tesseract is built (no CMake option toggles it off).
   - Holds an SSA list of `graph::Op` entries, each carrying
     `(op_name, std::vector<Value>, std::vector<Value>, attr_map,
     std::vector<Tensor> saved_for_backward)`.
   - `Value` wraps a `(shape, dtype, device)` triple; no storage at record
     time. Materialized tensors are produced by `graph::run(g, inputs)`.
   - Serializable to and from a compact textual form so that graphs
     produced on a CPU-only developer machine can be replayed on an MLIR
     developer machine.

2. **Stage 2 — MLIR module (`tesseract` dialect).**
   - Built only under `TESSERACT_ENABLE_MLIR=ON`.
   - Produced by `graph::emit_mlir(Graph&)`, a straight one-to-one
     translation that never reshapes or reorders ops.
   - Consumed by the `tesseract → linalg → scf → arith` pass pipeline and
     the `tesseract-opt` tool for offline experimentation.

This split makes Stage 1 the **source of truth** and Stage 2 a **view**
onto it. Users who just want graph-mode training get it without MLIR;
users who want compiler-level optimizations flip the option on.

## Consequences

### Positive

- **Zero-dependency graph mode.** `TESSERACT_ENABLE_MLIR=OFF` continues to
  be the default; nobody is forced to build LLVM to use graph capture.
- **Testable in isolation.** Each stage has its own test suite. Stage 1
  uses Catch2 as today; Stage 2 uses `tesseract-opt` + FileCheck.
- **API stability decoupled from upstream MLIR.** `tesseract::graph::*` is
  the versioned surface; MLIR moves beneath it.
- **Reusable by M2 / M3.** CUDA backend (M2) and LLM inference (M3) both
  consume `tesseract::graph::Graph` as their entry point. The MLIR path is
  one possible backend, not the only one.
- **Diffability.** A textual Stage-1 dump is human-readable and survives
  MLIR version bumps unchanged.

### Negative

- **Two data structures to keep in sync.** Every new op must be added to
  both Stage 1 (`graph::OpKind`) and Stage 2 (`tesseract.<op>`). Mitigated
  by a registry in `src/graph/OpTable.cpp` that drives both declarations
  from a single list.
- **Emitter is an extra moving part.** Bugs in `emit_mlir` show up only
  when MLIR is enabled. Mitigated by a parity test: every graph produced
  during ctest is, under `ENABLE_MLIR=ON`, round-tripped through
  `tesseract-opt --verify-each`.
- **Duplicated type system.** `graph::Value` (shape + dtype) overlaps with
  `mlir::RankedTensorType`. We keep them intentionally separate; Stage 1
  must not import any MLIR headers.

### Mitigations

- Define the canonical op list (name → arity → attrs) in one C++ table.
  Both `graph::Op` construction and TableGen `*.td` files read from this
  table; a small CI check warns if the two drift.
- Parity test: if `TESSERACT_ENABLE_MLIR=ON`, every `graph::Graph`
  constructed in ctest also exercises `emit_mlir` + `verify()`.

## Alternatives considered

- **Stage-1 only (no MLIR).** Rejected because it gives up the lowering /
  fusion / codegen machinery that makes MLIR worth depending on — the
  whole point of ADR-0001.
- **Stage-2 only (MLIR everywhere).** Rejected because of the build
  friction and API coupling described above.
- **JAX-style tracing from Python.** Rejected at M1 because Tesseract is
  still C++-first; Python bindings are an M4 task. We revisit this when
  pybind11 bindings land.

## References

- PyTorch FX & TorchDynamo (2022–2024): <https://pytorch.org/docs/stable/fx.html>
- JAX `jaxpr`: <https://jax.readthedocs.io/en/latest/jaxpr.html>
- TensorFlow 2 `tf.function` + MLIR: <https://www.tensorflow.org/mlir>
- `docs/m1-plan.md` for the concrete M1 track breakdown.

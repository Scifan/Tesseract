# ADR-0001: Use MLIR as Tesseract's shared IR

- **Status:** Accepted (2026-04)
- **Supersedes:** none
- **Authors:** tesseract core team

## Context

The single most important architectural decision for Tesseract is what to use
as its shared compilation substrate — the representation through which
training graphs, inference graphs, autograd, distributed-parallelism plans,
and backend code generators all speak to each other. Every mainstream deep
learning framework has made this choice:

- PyTorch uses TorchScript / torch.fx / TorchDynamo + PrimTorch — three
  overlapping IRs glued by Python conventions.
- JAX uses HLO (StableHLO), compiled through XLA / OpenXLA / IREE.
- TensorFlow uses MLIR (TFLite / XLA HLO / TFRT) and increasingly
  StableHLO.
- Modular Mojo/MAX targets MLIR natively.
- IREE, Triton, OpenXLA, ExecuTorch, TVM Relay — every major new project in
  2022–2026 is MLIR-first.

Tesseract's central value proposition (see [`idea.md`](../../idea.md)) is
*training–inference unification*: the same IR carries the same tensor
operations all the way from user-facing eager-mode code to deployed inference
kernels. Recreating that substrate from scratch would be a multi-year
undertaking duplicating the work of the upstream MLIR community; it would
also isolate Tesseract from the growing pool of vendor dialects (ROCm,
Qualcomm QNN, Apple Metal Performance Shaders, Groq, etc.) that already
target MLIR.

## Decision

Adopt **MLIR** as Tesseract's canonical IR. Specifically:

1. Every op at the user-visible layer (`tesseract::Tensor` + `nn`) gains a
   corresponding op in the `tesseract` dialect defined under `src/ir/`.
2. The autograd engine, after M1, emits and consumes MLIR directly — the
   eager tape is preserved as a fast path but is no longer the source of
   truth for backward pass shapes.
3. The backend stack (CPU kernels in M0, CUDA kernels in M2, accelerators
   later) takes `tesseract` dialect as input and lowers through MLIR passes
   to its own executable form (linalg / LLVM IR / Triton IR / CUTLASS
   templates as appropriate).
4. Runtime features that today ship as bespoke C++ (paged KV cache,
   continuous batching, quantization) are modeled as IR attributes + IR
   passes wherever possible, so that *both* training and inference benefit
   from the same implementation.

## Consequences

### Positive

- **Immediate ecosystem benefit.** StableHLO, Linalg, SCF, Arith, Tensor,
  Bufferization, Async, LLVM — all upstream dialects become usable without
  re-invention. FlashAttention-3, CuTe DSL, OpenXLA all speak compatible
  dialects.
- **Training/inference fusion is natural.** Graphs produced during training
  can be replayed verbatim for inference after a simple pass pipeline swap.
- **Compiler-first UX.** Autotuning, fusion, memory planning can all be
  phrased as IR passes and composed by users.
- **Hardware portability.** New backends (NPU / WebGPU / Metal / Blackwell)
  can author dialects rather than rewriting the framework internals.

### Negative

- **Build complexity.** LLVM + MLIR source builds are non-trivial;
  `scripts/build_llvm.sh` exists precisely to isolate this from the CPU-only
  development workflow.
- **Opinionated abstractions.** We inherit MLIR's idioms (regions, traits,
  ODS, TableGen). New contributors must learn them.
- **API stability risk.** MLIR upstream is still moving fast; Tesseract
  pins to a specific LLVM tag (currently 18.1.x) and upgrades deliberately.

### Mitigations

- The IR layer is **opt-in** via `TESSERACT_ENABLE_MLIR` and builds to a
  separate static library (`tesseract_ir`) plus binary (`tesseract-opt`).
  The CPU training stack (`tesseract_core`) builds and tests without any
  LLVM artifact.
- We ship a user-space LLVM builder (`scripts/build_llvm.sh`) so that
  contributors without root access can still exercise the MLIR path.
- We check in the authoritative TableGen files (`.td`) and the generated
  `.h.inc` / `.cpp.inc` are produced at build time (never committed), so the
  single source of truth is version-controlled.

## Alternatives considered

- **Own IR from scratch.** Rejected: replicating MLIR's passes, dialect
  infrastructure, region model, and community ecosystem is multi-year work
  with unclear payoff.
- **StableHLO as a native layer without MLIR infrastructure.** Rejected:
  StableHLO is *expressed* in MLIR, and we eventually need the pass manager
  regardless.
- **PyTorch's FX / torch.fx-only approach.** Rejected: FX is a Python graph
  abstraction; we are explicitly C++-first.

## References

- `idea.md` §2 "Pain points re-evaluated" and §4 "Truly unresolved
  problems" — the core motivation.
- MLIR upstream: <https://mlir.llvm.org>
- StableHLO: <https://github.com/openxla/stablehlo>
- OpenXLA IREE: <https://iree.dev>

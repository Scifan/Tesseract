# ADR-0006: M4 scope — parallel A/B/C tracks, and three recorded deviations

- **Status:** Accepted (2026-06-22)
- **Supersedes:** none
- **Relates to:** ADR-0001 (MLIR as shared IR), ADR-0004 (two-stage graph IR),
  ADR-0005 (CUDA HAL); `idea.md` §4.1/§4.2/§5/§6.2
- **Authors:** tesseract core team

## Context

A pre-M4 review of the codebase against `idea.md` surfaced one structural
scope deviation plus two unrealized theses that were never written down. This
ADR records them, and records the decision on how M4 proceeds.

### Deviation 1 — M4 was silently redefined

`idea.md` §6.2's milestone table defines:

| Milestone | `idea.md` definition |
| --------- | -------------------- |
| **M4 (months 16–19)** | **SSM / MoE / DiT native support** (Mamba-3, Mixtral, DiT) |
| **M5 (months 20–24)** | Distributed training + edge runtime |

The `docs/roadmap.md` M4 section had been rewritten to "Distributed training +
Python frontend", and M5 to "Edge + open source release" — i.e. `idea.md`'s
entire M4 (the non-Transformer architectures milestone) was dropped from the
roadmap, with no SSM/Mamba/MoE/DiT code anywhere in `src/`. This matters
because `idea.md` §4.2 and §5 (direction B) name **non-Transformer
architectures as a top differentiator** ("SSM/DiT 是 2026–2028 的主流增量"),
whereas `idea.md` §5 explicitly states that "做一个更好的分布式" is *not* a
differentiation point. The redefinition therefore spent the differentiation
budget on a non-differentiator and shelved the headline differentiator —
without a record.

### Deviation 2 — "one IR, train+infer" is unrealized on the LLM path

`idea.md` §4.1/§6.1's #1 thesis is "一份 Graph IR, 训练与推理是同一张图的不同
执行模式". In reality, the entire M3 LLM runtime (`src/models/`,
`src/nn/`) is a hand-written **eager** stack that never touches the
`tesseract` dialect or `JitEngine`. The IR path only covers the MNIST-scale
op set on CPU. Wave 15 began bridging this (tesseract → GPU dialect) but is
toolchain-gated (LLVM built `host`-only, no NVPTX).

### Deviation 3 — no external-framework benchmark

`idea.md` §6.2/§8.5 demand reproducible benchmarks vs PyTorch / vLLM / SGLang
/ llama.cpp at every milestone ("拒绝 hand-wave 的更快"). All 16 framework
benchmarks are internal (vs composite / vs eager / vs memcpy roofline). The
M3 exit bar literally reads "serve Llama-3 8B at parity with vLLM throughput"
yet no head-to-head with any external engine has ever been run.

## Decision

### 1. M4 runs three tracks in parallel, not one of them

Rather than choose A (idea.md's SSM/MoE/DiT) *or* B (the roadmap's
distributed+Python) *or* C (close the one-IR gap), M4 advances all three in
parallel, taking from each the slice that is **currently unblocked and
highest-leverage**, sequenced by dependency and risk. Full wave breakdown in
[`docs/m4-plan.md`](../m4-plan.md).

- **Track A — architecture differentiation (primary).** MoE (A1, reuses the
  existing `FeedForward`/`Linear` stack), then SSM/Mamba (A2, the headline
  prize — `selective_scan` + decode-state cache). DiT (A3) is design-only this
  milestone; its diffusion-scheduler/VAE runtime is deferred to M5 because the
  modality is far from the current autoregressive stack.
- **Track B — adoption + scale.** Python frontend (B1, pybind11 — unblocked,
  the single biggest adoption lever per `idea.md` §8.2). Single-GPU LLM
  training loop (B2 — fills the hidden prerequisite that "distributed
  *training*" presupposes but which never existed: the LLM has only ever run
  inference). Tensor parallelism as an IR/runtime transform (B3 — design +
  single-process multi-rank CPU parity now, real multi-GPU NCCL validation
  gated on free cards).
- **Track C — one-IR coherence.** Bring the transformer op set into the
  dialect + tesseract→linalg so a single Llama block round-trips and runs
  through the CPU `JitEngine` at eager parity (C1, unblocked, CPU-only). KV
  cache / dynamic shape as first-class IR concepts (C2 — design + initial
  ops). The LLM GPU-codegen tail stays gated on the NVPTX-enabled LLVM
  rebuild (B-009).

### 2. The three deviations are recorded, not silently carried

This ADR is the record. Additionally:

- `idea.md` §6.2 gains a footnote noting M4's actual (ABC-parallel) shape and
  that DiT-runtime + distributed-at-scale slip toward M5.
- `docs/roadmap.md`'s M4 section is replaced with the ABC-parallel wave table.
- Deviation 2 is the explicit motivation for Track C; Deviation 3 is addressed
  by B-046 (a vs-`llama.cpp` CPU decode-throughput harness — the easiest fair
  external alignment; vLLM/Hopper alignment stays gated).

### 3. Gated work is labeled, never quietly dropped

Anything blocked by SM 9.0+ (FA3, WGMMA INT8), the NVPTX-less LLVM (LLM GPU
codegen), or scarce free cards (multi-GPU NCCL) is tagged "gated tail" in the
wave docs and tracked in the backlog, not removed.

## Consequences

### Positive

- The headline differentiator (SSM/MoE) is back on the critical path and is
  buildable on the current SM 8.9 box.
- The "one IR" thesis stops being aspirational: a real Llama block will
  execute through the IR (CPU) by end of M4.
- Python bindings unblock any external adoption / comparison at all.
- The single-GPU LLM training loop (B2) de-risks every later distributed-
  training claim by making the prerequisite real and measured.

### Negative

- Three parallel tracks widen the surface area and the review/test load within
  one milestone. We accept this because each track's first wave is independent
  (A1/B1/B2/C1 have no cross-dependencies) and individually low-risk.
- DiT and at-scale distributed slip toward M5; M4 delivers their design and
  (for distributed) a single-node prototype, not a production runtime.

### Deferred / gated

- **LLM GPU codegen (B-009 tail):** needs LLVM rebuilt with NVPTX. Deferred by
  user decision; the Wave-15 `gpu.module` IR is the foundation it plugs onto.
- **Multi-GPU NCCL validation of TP (B3):** needs ≥2 contention-free cards.
- **FA3 / WGMMA INT8:** SM 9.0+ only (box is RTX 5880 Ada, SM 8.9).
- **DiT runtime:** M5.

# M4 — non-Transformer architectures + adoption + one-IR coherence

**Status:** **closeout — done (perf-closeout 2026-06-24; vLLM serving win
2026-06-25).** All three tracks landed; the 7-point combined exit bar is met
(see bottom of doc); the GPU benches are un-gated on reserved cards and the
external scoreboard is complete. The last open axis — full-model single-stream
serving vs vLLM — flipped from an honest loss to a **decode/end-to-end win** at
matched FP16. Scope decision in [ADR-0006](adr/0006-m4-parallel-abc-scope.md).

M4 advances three tracks in parallel (see ADR-0006 for the why):

- **Track A — architecture differentiation.** MoE → SSM/Mamba native; DiT
  design-only. This is `idea.md`'s original M4 and the headline differentiator.
- **Track B — adoption + scale.** Python frontend; single-GPU LLM training
  loop; tensor parallelism (single-node prototype + gated multi-GPU).
- **Track C — one-IR coherence.** Bring the transformer op set into the
  `tesseract` dialect so a real Llama block executes through the IR; KV/dynamic
  shape as IR concepts.

Each wave keeps the M3 discipline: a CPU reference, a CUDA fast path where it
helps, a parity test, and a backlog "Resolved:" entry. Constraints carried
from M3: box GPUs are RTX 5880 Ada (SM 8.9, no Hopper); local LLVM is built
`host`-only (no NVPTX); multi-GPU has external contention. Blocked slices are
labeled "gated tail" and not put on the critical path.

Dependency / ordering:

- Phase 0 (docs) first.
- A1 / B1 / B2 / C1 are independent — parallel start.
- A2 after A1 (reuses the block plumbing); A3 design-only.
- B3 after B2 (needs a training path) + design-first.
- C2 after C1.

---

## Phase 0 — record first (DONE, 2026-06-22)

- [ADR-0006](adr/0006-m4-parallel-abc-scope.md) — M4 scope + three recorded
  deviations.
- This plan doc.
- `idea.md` §6.2 footnote + `docs/roadmap.md` M4 section rewritten to the
  ABC-parallel table.
- Backlog placeholders **B-038 … B-046**.

---

## Track A — architecture differentiation

### A1 — MoE native (B-038) — DONE (2026-06-22)

**Goal.** Sparse mixture-of-experts as a first-class FFN, reusing the existing
dense stack so the routing is the only new concept.

- `nn::MoEFeedForward` (`include/tesseract/nn/MoEFeedForward.hpp` +
  `src/nn/MoEFeedForward.cpp`): a `router` `Linear` (`d_model → num_experts`)
  producing top-k gating weights (softmax over the selected logits), `N`
  expert `FeedForward` modules, and token dispatch/combine. CPU reference is a
  straightforward per-token gather over the chosen experts; CUDA reuses the
  existing op stack (no new kernel in A1 — a fused grouped-GEMM dispatch is a
  B-038+ follow-up).
- `nn::TransformerBlock` gains an optional MoE-FFN branch (a `num_experts > 0`
  ctor arg selects `MoEFeedForward` in the FFN slot; `0` keeps dense — fully
  backward compatible).
- `LlamaConfig` gains `num_experts` / `num_experts_per_tok`; `from_json` parses
  the Mixtral-style fields.
- Optional load-balance aux loss exposed for the training path.

**Exit bar.** With `top_k == 1` and one expert, MoE output is bit-identical to
the equivalent dense `FeedForward`; with `top_k == num_experts` and equal
gates it equals the mean of the experts. Parity tests on CPU (exact) + CUDA
(tolerance). End-to-end: a 2-layer MoE Llama generates deterministically.

### A2 — SSM / Mamba native (B-039) — DONE (2026-06-22, CUDA gated tail)

**Goal.** The headline non-Transformer architecture: a selective-state-space
(Mamba) block that is O(L) in sequence length, with a decode-state cache so it
plugs into the existing generation loop.

- `ops::selective_scan` (`include/tesseract/ops/SelectiveScan.hpp` +
  `src/ops/cpu/SelectiveScan.cpp` + `src/cuda/SelectiveScan.cu`): the
  discretized SSM recurrence `h_t = exp(Δ_t·A)·h_{t-1} + (Δ_t·B_t)·x_t`,
  `y_t = C_t·h_t + D·x_t`, computed chunkwise (parallel prefix within a chunk,
  sequential across chunks) so prefill is parallel and decode is recurrent.
- `nn::Mamba` block (`include/tesseract/nn/Mamba.hpp` + `src/nn/Mamba.cpp`):
  input projection → causal depthwise conv1d → SiLU → `selective_scan` →
  output projection, the Mamba-1 recipe.
- `nn::SSMStateCache` (a `KVCacheBase`-analog for SSM): holds the per-channel
  recurrent state `h` (+ the conv1d ring buffer) so a decode step advances the
  state without recomputing the prefix.

**Exit bar.** Recurrent step-by-step decode == parallel chunkwise prefill
(CPU ≤ 1e-5, CUDA tolerance). A minimal Mamba model runs end-to-end `generate`
reusing the Wave-6 `Sampler` and the Wave-7 scheduler.

### A3 — DiT (B-040, design only this milestone) — DONE (design, 2026-06-22)

DiT (diffusion transformer) module + IR-representation **design** captured here
and in the placeholder interface
[`include/tesseract/nn/DiTBlock.hpp`](../include/tesseract/nn/DiTBlock.hpp)
(construct-allowed, `forward` throws — runtime is M5). The diffusion scheduler
(timestep loop) + VAE/UNet runtime are deferred to M5 — the modality
(non-autoregressive, iterative denoising) is far enough from the current LLM
stack that a runtime now would be premature. Design:

- **adaLN-Zero block (Peebles & Xie 2023).** A `DiTBlock` is a pre-norm
  transformer block whose LayerNorms are replaced by *adaptive* LN conditioned
  on `c = timestep_embedding(t) + class_embedding(y)` `[B, d_cond]`:
  the conditioning MLP emits `(shift1, scale1, gate1, shift2, scale2, gate2)`
  (6·D), `modulate(z, shift, scale) = z·(1+scale) + shift`, and the gates are
  zero-initialized so each block starts as the identity (stable training). The
  only structural delta from `nn::TransformerBlock` is this norm/conditioning
  path; attention + FFN are the existing modules.
- **IR alignment (Track C).** `modulate` is two fused affine ops over `[B,S,D]`
  with `[B,1,D]` broadcast operands — expressible via the existing `mul`/`add`
  lowerings plus a `dit.modulate` marker, so a DiT block round-trips through the
  `tesseract` dialect on the same footing as a Llama block (B-044) once the A3
  runtime lands.
- **Diffusion loop.** A host-side scheduler over `t = T..0` calling the model
  with timestep conditioning — orthogonal to the autoregressive KV-cache loop;
  it will live in a `models::DiT` driver in M5.

---

## Track B — adoption + scale

### B1 — Python frontend (B-041)

**Goal.** `import tesseract` from Python with zero-friction access to the
training and inference stack — the single biggest adoption lever
(`idea.md` §8.2).

- New CMake option `TESSERACT_BUILD_PYTHON` (default OFF; pulls pybind11 via
  `FetchContent`). A `python/` directory holds the binding TU(s) building a
  `tesseract._core` extension + a thin `tesseract/` Python package.
- Bind: `Tensor` (creation, dtype/device, `to`, numpy buffer-protocol
  in/out, autograd `.backward()` / `.grad`), the `nn` modules
  (`Linear`/`Sequential`/`LlamaModel`/…), `LlamaModel.generate`, the tokenizer,
  and the sampler params.

**Exit bar.** `import tesseract` works; a Python script trains the MNIST MLP
and runs a Llama inference smoke; `pytest python/tests` is green. CPU-only is
sufficient (CUDA bindings ride the same `Device` arg).

**Status: done (2026-06-22).** `python/_core.cpp` binds Tensor (NumPy buffer
protocol both directions, autograd, arithmetic + matmul), `ops`, `nn`
(Module/Linear/Embedding/activations/Sequential), `optim` (SGD/Adam), `models`
(Llama incl. MoE + Mamba `generate`), `io` (tokenizers). The package is staged
into the build tree (`python/CMakeLists.txt`); `PYTHONPATH=<build>/python pytest
python/tests` → 9 green incl. an MNIST-style MLP overfit (loss → < 0.5).

### B2 — single-GPU LLM training loop (B-042)

**Goal.** Close the hidden prerequisite for distributed *training*: the LLM has
only ever run inference. Demonstrate that `LlamaModel` trains.

- `examples/llama_train.cpp`: next-token cross-entropy training on a small
  synthetic/text corpus — forward over `LlamaModel`, `ops::cross_entropy`,
  `backward`, `optim::Adam::step`, on `--device {cpu,cuda}`.
- Whatever autograd gaps surface (the transformer forward ops all have
  backward from M2, but the generation-only paths bypass them) are filled here.

**Exit bar.** Training loss decreases monotonically over N steps on a tiny
overfit set (memorize a fixed batch → loss → ~0), recorded in the backlog and
a smoke test.

**Status: done (2026-06-22).** `examples/llama_train.cpp` (next-token CE + Adam,
`--device cpu|cuda`) drives a tiny Llama's fixed-batch loss **4.26 → 0.0065** in
80 CPU steps. No autograd gaps surfaced — `LlamaModel::forward` already records a
full backward graph. Smoke: `tests/models/test_llama_train.cpp` asserts the loss
drops below 0.1.

### B3 — tensor parallelism as a transform (B-043)

**Goal.** Express tensor parallelism (Megatron-style) as a partition transform,
validated single-node now, multi-GPU later.

- Column-parallel (`q/k/v/gate/up` projections) + row-parallel (`o/down`
  projections) sharding helpers, with an all-reduce at the row-parallel output.
- A `CommBackend` abstraction: a single-process **multi-rank simulator**
  (shards held in one process, all-reduce = sum) for deterministic CPU parity
  now; a NCCL backend behind `TESSERACT_ENABLE_NCCL` for real multi-GPU
  (validation gated on free cards).
- Design doc (here) on how this becomes an IR pass over the graph (the
  `idea.md` §6.1.5 "并行策略作为 IR pass" direction).

**Exit bar.** A TP=2 sharded Llama block produces output bit-identical to the
unsharded block under the single-process simulator (CPU). Multi-GPU NCCL run
is a gated tail.

**Status: done (2026-06-22).** New `tesseract_distributed` lib:
`distributed::CommBackend` + `SimCommBackend` (all-reduce-sum / all-gather),
`ColumnParallelLinear` + `RowParallelLinear` (`from_dense` shard builders,
`forward_shards` for the one-all-reduce column→row composition). Parity
(`tests/distributed/test_tensor_parallel.cpp`): column-parallel bit-identical,
row-parallel + Megatron SwiGLU MLP within 1e-5. Design + IR-pass formulation in
[`docs/design/tensor-parallel.md`](design/tensor-parallel.md). NCCL backend +
multi-GPU validation = gated tail.

---

## Track C — one-IR coherence

### C1 — transformer op set in the dialect (B-044)

**Goal.** Make a real Llama block executable through the IR — the concrete
down payment on the "one IR" thesis (ADR-0006 deviation 2).

- Extend the `tesseract` dialect + `--convert-tesseract-to-linalg`
  (`src/ir/passes/ConvertToLinalg.cpp`) to cover the transformer op set:
  `rms_norm`, `silu`/`swiglu`, `rope`, and `attention` — either via direct
  linalg lowering or as marker ops decomposed into the already-lowerable
  primitives.
- A single Llama block captured through `GraphScope`, emitted to MLIR, lowered,
  and run through the CPU `JitEngine`.

**Exit bar.** Round-trip + FileCheck for the new lowerings; the IR-executed
Llama block matches the eager block within FP32 tolerance on CPU.

**Status: FFN + normalization slice done (2026-06-22); attention/rope gated.**
`--convert-tesseract-to-linalg` now lowers `sigmoid`/`exp`/`log`/`sqrt`/`tanh`
(→ `math.*` / arith) and `mean` (→ `linalg.reduce` + scale), plus the new
`tesseract.sqrt` op — i.e. every primitive `ops::rms_norm` and
`ops::swiglu_silu_gate` record. FileCheck (`tests/ir/convert_transformer.mlir`)
pins the lowerings; JIT parity (`tests/ir/test_jit_transformer.cpp`) runs
sigmoid, the SwiGLU activation, and the **full RMSNorm primitive chain** through
`ir::JitEngine` matching eager within 1e-4 (29/29 IR ctests green). The
remaining C1 work — a real `rope` lowering (it's a monolithic kernel, not a
recorded primitive chain) and the `attention` path (head reshape/permute +
`softmax` lowering) needed to run a *whole* Llama block through the IR — is the
documented gated tail.

### C2 — KV cache / dynamic shape as IR concepts (B-045)

**Goal.** Lay the IR groundwork `idea.md` §4.1 calls for: KV cache and dynamic
shapes as first-class IR citizens.

- Design doc + initial ops/attributes (a paged-buffer / dynamic-dim
  representation). Execution is **not** required this wave — the goal is a
  representation the later NVVM/runtime stages can target.

**Status: done (2026-06-22).** Added `tesseract.paged_kv_alloc` /
`paged_kv_append` / `paged_attention` (dialect + verifiers) representing the
paged KV pool, the SSA-valued slot scatter, and decode attention with
`block_table`/`seq_lens`; dynamic shapes use MLIR's native `?` dim (verifiers
are dynamic-dim-tolerant). Round-trip + verifier-negative FileCheck green.
Design: [`docs/design/kv-cache-ir.md`](design/kv-cache-ir.md). Lowering is
deferred by design.

### Gated tail (B-009 cont.)

LLM GPU codegen (PTX/cubin JIT of the IR'd block) stays blocked on the
NVPTX-enabled LLVM rebuild. The Wave-15 `gpu.module` pipeline + C1's lowerings
are the foundation it plugs onto.

---

## Cross-cutting — external benchmark (B-046)

`idea.md` §6.2/§8.5 require external-framework alignment; M3 had none. M4 adds
a `benchmarks/bench_vs_llama_cpp.*` harness comparing end-to-end CPU decode
tok/s against `llama.cpp` on a matched small model — the easiest fair
alignment. vLLM / Hopper throughput alignment stays gated on hardware.

**Status: scaffold done (2026-06-22).** `benchmarks/bench_llama_decode_cpu.cpp`
emits Tesseract's CPU decode/prefill tok/s (`llama-bench`-comparable);
`scripts/bench_vs_llama_cpp.sh` drives the comparison and prints the matching
`llama-bench` command. Methodology + recorded baseline + honest read in
[`docs/design/external-benchmark.md`](design/external-benchmark.md).

**Resolved (perf-closeout 2026-06-24).** llama.cpp row filled (local llama.cpp +
matched GGUF) and the GPU comparison un-gated on reserved cards: PyTorch GPU
head-to-head + a 14-row external scoreboard with honest win/tie/loss verdicts
([`docs/design/external-benchmark.md`](design/external-benchmark.md)).

**Resolved (2026-06-25) — vLLM serving flipped to a WIN.** The one honest loss
(full-model single-stream serving vs vLLM) is now a decode-throughput +
end-to-end-latency win at matched FP16, via FP16 model loading (B-022 loader
cast), capture-safe `Storage`, whole-model CUDA-graph capture/replay (decode AND
prefill), and GQA-native fused decode/prefill attention. Matched-FP16
TinyLlama-1.1B (prompt/gen 128, clean RTX 5880 Ada): **decode 321.1 vs vLLM
305.8 tok/s (+5.0%), TPOT 3.115 vs 3.275 ms, end-to-end ~403 vs 420 ms — all
wins**; only TTFT/prefill is still behind. Two prefill optimizations landed
2026-06-25: B-024+ WMMA tensor-core FlashAttention (TTFT 7.28 → 6.59 ms) and
B-024c stride-aware / BSHD attention layout (removes the per-layer
`contiguous`/transpose copies; `strided_copy` 25.8 % → 15.9 % of prefill GPU
time; TTFT 6.59 → **5.86 ms**, vLLM 5.47 → gap **1.07×**). Reproduce in
[`bench/external/results/vllm_serving.md`](../bench/external/results/vllm_serving.md).
**Remaining follow-up: B-024e** — a BSHD-native RoPE (kills the last big
`strided_copy`) + norm/RoPE/residual epilogue fusion to clear the final ~0.4 ms
(shared cuBLAS GEMM floor + un-fused elementwise launches) and *win* TTFT outright.

---

## M4 combined exit bar — all met (closeout 2026-06-24 / 2026-06-25)

1. [x] MoE + Mamba models run end-to-end `generate` at parity with their
   reference semantics (CPU exact / CUDA tolerance). — A1/A2 done.
2. [x] `import tesseract` trains MNIST + runs a Llama inference smoke; `pytest`
   green. — B1 done (9 green).
3. [x] Single-GPU LLM training loss converges (recorded). — B2 done.
4. [x] A single Llama block executes through the IR (CPU JIT) at eager parity. —
   C1 done (full multi-head block).
5. [x] TP=2 single-process parity (multi-GPU NCCL gated). — B3 done, plus real
   multi-GPU TP=3 forward+backward parity (2026-06-24).
6. [x] At least one vs-`llama.cpp` external decode-throughput data point (CPU). —
   B-046 done; extended to a 14-row scoreboard + a vLLM serving win.
7. [x] CPU ctest green + runnable CUDA paths green throughout; every wave's docs
   (backlog + roadmap + this plan) kept in sync. — CUDA 529/529, CPU 552/552.

**Net:** M4 is complete. The sole sub-metric still behind any external baseline
is vLLM TTFT/prefill, now within 7 % (5.86 vs 5.47 ms) after the B-024+ WMMA
(7.28 → 6.59) and B-024c stride-aware/BSHD attention-layout (6.59 → 5.86)
follow-ups both landed. The remaining closer is B-024e (BSHD-native RoPE +
norm/RoPE/residual epilogue fusion) to clear the final ~0.4 ms — the shared
cuBLAS GEMM floor plus Tesseract's un-fused elementwise launches.

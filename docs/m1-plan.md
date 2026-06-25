# M1 — Graph IR + lowering (plan)

**Milestone:** M1. **Status:** in progress (2026-04-18). **Owner:** tesseract core team.
**Precondition:** M0 complete (eager CPU stack, 81/81 ctests, MNIST 94/96/97 at epochs 1/2/3).

---

## 1. Scope statement

M1 replaces the eager tape with a proper **Graph IR** that:

1. **captures** every op from the M0 public surface (`ops::*` + `nn::*`) into an SSA form,
2. **canonicalizes** and **optimizes** it through MLIR upstream passes,
3. **lowers** to `linalg` + `scf` + `arith` and then to CPU code,
4. **matches M0's numerical output** on MNIST to within single-precision tolerances.

This is a four-month task. The plan below breaks it into nine concrete code
tracks (M1C–M1I) plus a benchmark (M1J). The order is picked so that every
track ends with a testable artifact — nothing is "done later"; every merge
must keep `ctest` green.

## 2. Design decisions locked today

### 2.1 Two-stage IR ("C++ graph → MLIR module")

The `GraphScope` frontend records ops into a **C++-level SSA structure**
(`tesseract::graph::Graph`) regardless of whether MLIR is built. This lets
the graph mode be used, tested, and even optimized in the `TESSERACT_ENABLE_MLIR=OFF`
configuration — the default for everyone who has not run `build_llvm.sh`.

When `TESSERACT_ENABLE_MLIR=ON`, an emitter lowers the C++ graph into a
`mlir::ModuleOp` that uses the `tesseract` dialect, which we then pass to
upstream passes. This is a strict translation, not a duplicate representation:
the emitter is a ~200-LOC function, and the C++ graph is the source of truth.

Rationale: keeping the two stages decouples the hard-to-build MLIR path from
the CPU development workflow. It also sets us up for M2's CUDA path — once
the graph is in MLIR form, the backend choice is a pass-pipeline selection.

See [ADR-0004](adr/0004-graph-ir-two-stage.md) for the full reasoning and
consequences.

### 2.2 `GraphScope` semantics

```cpp
{
  graph::GraphScope scope;     // RAII: installs active graph on this thread
  Tensor y = nn::Linear(...)->forward(x);   // recorded, not executed
  Tensor loss = ops::cross_entropy_with_logits(y, targets);
  auto& g = scope.graph();     // inspect / serialize / lower
}
// Scope ends -> active graph cleared.
```

Key properties:

- **Thread-local.** Multiple threads can record independent graphs.
- **Transparent.** No change to user code above the `ops::` layer: every
  autograd-aware op checks `graph::is_recording()` *before* the eager
  fallback and routes to the recorder when true.
- **Composable.** Nested `GraphScope`s error out (the engine does not
  support sub-graphs at M1; we defer that to M2).
- **No implicit materialization.** If you call `data_ptr<T>()` on a
  recording-mode tensor, we throw. This forces users to an explicit
  `graph::run(g, inputs)` entry point.

### 2.3 Autograd as a graph transform

Reverse-mode AD is a **standalone MLIR pass** (`--tesseract-backward`) that
runs on `tesseract.function`. It extends the function in-place from
    `(inputs…, params…) -> (outputs…)`
to
    `(inputs…, params…, grad_outputs…) -> (outputs…, dparams…)`,
so the caller explicitly supplies the cotangent of each output and receives
the gradient of each `tesseract.param` in declaration order. This is the
standard vjp lowering used by JAX / XLA.

The pass seeds `grad_map[output_i] = %grad_output_i` (the new block
argument), then walks ops in reverse and materializes per-op rules using
the same `tesseract` vocabulary. After M1I the rule table covers
`add / sub / mul / neg / matmul / transpose / broadcast_to / sum /
relu / cross_entropy_with_logits` plus the structural `param`
pass-through — enough for the full MNIST MLP training step.

For the two composite forward ops (`relu` and
`cross_entropy_with_logits`) we added **fused backward ops to the
dialect** — `tesseract.relu_backward` and
`tesseract.cross_entropy_with_logits_backward`. This keeps the backward
IR closed under `tesseract.*`, lets `--tesseract-backward` stay
purely syntactic, and defers the decomposition (compare+select,
softmax+scatter) to a later pattern-rewrite pass. The trade-off is one
more opaque op in the dialect surface per forward op; the alternative
(open-coding the decomposition inside the AD pass) would couple AD to
the lowering plan and block higher-order gradients. See
`src/ir/passes/Backward.cpp` for the current rule list.

Because the backward is expressed inside the same `tesseract` dialect, it
composes cleanly with `--convert-tesseract-to-linalg` (M1G): running the
two passes in sequence produces a single Linalg function that computes
both forward outputs and parameter gradients, with MLIR's greedy DCE
pruning unused gradient paths (e.g. `dx` when `x` is a non-trainable
input).

### 2.4 Operator coverage for M1

Must-have, locked today:

- **Binary elementwise (with broadcast):** `add`, `sub`, `mul`, `div`.
- **Unary:** `neg`, `relu`, `sigmoid`, `tanh`, `exp`, `log`.
- **Reductions:** `sum`, `mean`, `max` (all-reduce + dim-specific with `keepdim`).
- **Linear algebra:** `matmul` (rank-2 only at M1; batched ⇒ M2).
- **Shape:** `view`, `reshape`, `permute`, `transpose`, `contiguous`, `clone`.
- **Composite:** `softmax`, `log_softmax`, `cross_entropy_with_logits`.
- **Module boundary:** `tesseract.graph`, `tesseract.function`, `tesseract.param`,
  `tesseract.return`.

Nice-to-have (tracked in backlog if they slip): `split`, `cat`, `index_select`,
`gather`, batched matmul (items B-003, B-004).

## 3. Track breakdown

| Track | Description                                                                       | Artifact                                                 |
|-------|-----------------------------------------------------------------------------------|----------------------------------------------------------|
| M1A   | Plan doc + ADR-0004 (this file)                                                   | `docs/m1-plan.md` + `docs/adr/0004-graph-ir-two-stage.md` |
| M1B   | Provision LLVM 18.1.8 via `build_llvm.sh` (non-blocking; runs in background)      | `third_party/llvm-install/`                              |
| M1C   | Expand `TesseractOps.td` to cover §2.4 op set; add shape verifiers                | `src/ir/TesseractOps.td`, updated `roundtrip.mlir`       |
| M1D   | Add structural ops: `tesseract.graph`, `tesseract.function`, `tesseract.param`     | Same files as M1C                                        |
| M1E   | `graph::Graph` / `graph::Value` / `graph::Op` in `include/tesseract/graph/`, `GraphScope` RAII, tape hooks in every `ops::*` entry point | `src/graph/*`                                            |
| M1F   | `graph::emit_mlir(Graph&)` → `mlir::ModuleOp`; gated on `TESSERACT_ENABLE_MLIR=ON` | `src/graph/Emit.cpp`                                     |
| M1G   | `tesseract → linalg` conversion pass; registered in `tesseract-opt`               | `src/ir/passes/ConvertToLinalg.cpp`                      |
| M1H   | `Engine::backward` extended to recognize recorded tensors and run graph AD        | `src/graph/Autograd.cpp`                                 |
| M1I   | `examples/mnist.cpp` gains `--mode graph` flag; loss curve logged                 | Updated example + new `tests/graph/test_mnist_parity.cpp`|
| M1J   | `benchmarks/bench_graph_vs_eager.cpp` for a 2-layer MLP forward pass              | New benchmark                                            |

## 4. Verification bar

- **Unit.** Every `graph::*` header has gradcheck coverage through the graph
  path. `ctest` passes in both `ENABLE_MLIR=OFF` and `ENABLE_MLIR=ON`.
- **Integration.** Training MNIST through `--mode graph` for 3 epochs reaches
  the same (± 0.5 %) test-set accuracy the eager path reached in M0.
- **IR roundtrip.** `tesseract-opt` round-trips every graph produced by a
  forward pass through MNIST (CI gated behind ENABLE_MLIR).
- **Benchmark.** `bench_graph_vs_eager` shows graph mode ≥ 1.3 × the eager
  throughput for the MLP forward pass on CPU — the minimum justification
  for the added complexity.

## 5. Out of scope for M1

- CUDA / GPU anything — tracked by M2.
- Batched matmul and multi-dim reductions beyond a single axis (B-004).
- `split` / `cat` / `index` / `gather` backward (B-003).
- Quantization / mixed-precision dialects — tracked by M3.
- Python bindings — tracked by M4.

## 6. Incremental milestones (intra-M1)

Because the track list is long, we ship in three incremental merges:

- **M1.α (done).** M1A + M1E. The graph frontend works; every op in §2.4
  records into `graph::Graph` while still executing eagerly.
  **Exit bar hit:** `test_graph_smoke` green in `ENABLE_MLIR=OFF`.
- **M1.β (done).** M1B + M1C + M1D + M1F. LLVM/MLIR built from source,
  `tesseract` dialect covers the full §2.4 op set plus structural ops,
  `graph::emit_mlir` rewrites any recorded `Graph` into a verified
  `mlir::ModuleOp`.
  **Exit bar hit:** `ir_dialect_roundtrip` + `test_ir_emit_mlir` both
  green (92/92 ctests).
- **M1.γ (in progress).** M1G + M1H + M1I + M1J.
  * M1G — `--convert-tesseract-to-linalg` rewrites add/sub/mul/div →
    `linalg.*`, matmul → `linalg.fill` + `linalg.matmul`, sum →
    `linalg.fill` + `linalg.reduce` (+ `tensor.expand_shape` when
    `keepdim = true`).
  * M1H — `--tesseract-backward` applies reverse-mode AD in-place on
    `tesseract.function`: appends one cotangent per output and one
    `dparam` per `tesseract.param` to the signature, seeds `grad_map`
    from the new block args, walks ops in reverse to materialize per-op
    rules (add/sub/mul/neg/matmul/sum(-1)/param). Composes cleanly with
    M1G.
  * M1I.1 (done) — **Graph-mode IR parity for the MNIST MLP.** The
    AD rule table grew to cover the rest of the MNIST op surface:
    `relu` (via fused `tesseract.relu_backward`),
    `cross_entropy_with_logits` (via fused
    `tesseract.cross_entropy_with_logits_backward`),
    `transpose` (self-inverse along the same axis pair), and
    `broadcast_to` (sum-over-broadcast-axes, enough for
    `[C] → [B, C]` bias broadcasts). `examples/mnist.cpp` gained a
    `--mode graph [--dump-ir]` flag: one mini-batch is captured
    through `graph::GraphScope`, lowered through
    `ir::emit_mlir → --tesseract-backward →
    --convert-tesseract-to-linalg`, and asserted to pass
    `mlir::verify`. A new `test_graph_mnist_mode` ctest pins this
    pipeline so regressions surface at CI time.
  * M1I.2.a (done) — **C++ graph interpreter + autograd as the
    execution source-of-truth.** Rather than gate end-to-end execution
    on the MLIR JIT, we added a native C++ path that shares the
    `graph::Graph` data structure with the IR emitter:
    `graph::build_backward` (`src/graph/Autograd.cpp`) is the exact
    mirror of the `--tesseract-backward` pass but operates directly on
    `graph::Graph`, and `graph::run`
    (`src/graph/Interpreter.cpp`, linked into the new
    `tesseract_graph_runtime` library to break the `ops↔graph`
    cycle) walks the backward-extended graph in declaration order and
    dispatches each op back to the eager `ops::*` kernel. Two new
    fused eager kernels — `ops::relu_backward` and
    `ops::cross_entropy_with_logits_backward` — give the interpreter
    the same op surface the MLIR pass uses. `examples/mnist.cpp
    --mode graph` is now a real training loop (capture → build_backward
    → interpret → Adam step), and `--mode graph --dump-ir` still
    prints the lowered IR. Exit bar: `test_graph_autograd` (gradient
    parity on two graphs) + `test_graph_train_parity` (40-step loss
    curve matches eager to < 1e-5) + a 3-epoch
    `examples/tesseract_mnist --mode graph` run reproduces the eager
    loss curve bit-for-bit and reaches 96.71 % test accuracy on
    MNIST.
  * M1I.2.b-Phase-1 (done) — `mlir::ExecutionEngine` JIT path. New
    `tesseract::ir::JitEngine` takes a `graph::Graph`, emits a
    `func.func @tesseract_entry` module via `emit_func_mlir`, and runs
    the full CPU lowering pipeline programmatically:

        --convert-tesseract-to-linalg
        --one-shot-bufferize{bufferize-function-boundaries=true,
                             layout=IdentityLayoutMap}
        --buffer-results-to-out-params
        --convert-linalg-to-loops
        --lower-affine
        --convert-scf-to-cf
        --expand-strided-metadata
        --finalize-memref-to-llvm
        --convert-func-to-llvm
        --arith-to-llvm
        --convert-cf-to-llvm
        --reconcile-unrealized-casts

    `JitEngine::invoke(bindings)` marshals each caller `Tensor` through
    the `_mlir_ciface_*` StridedMemRef ABI (output buffers pre-
    allocated host-side so ownership is trivial). New
    `tests/ir/test_jit_forward.cpp` exercises six graph shapes
    (pure-elementwise, matmul+sum, dual outputs, input+param mix,
    stacked matmul chain, invocation idempotence) and asserts parity
    with the C++ interpreter — every test passes, full ctest now 103
    cases green.

    Scope note: Phase-1 only covers graphs whose ops are already
    lowered by `--convert-tesseract-to-linalg` (add/sub/mul/div,
    matmul, sum). That's enough to prove the pipeline end-to-end but
    short of MNIST, which also uses relu, transpose, broadcast_to and
    the fused cross-entropy/backward ops.
  * M1I.2.b-Phase-2 (done) — Full-coverage linalg lowering. New
    patterns in `--convert-tesseract-to-linalg`: `relu`, `neg`,
    `relu_backward` (single `linalg.generic` with `arith.cmpf` +
    `arith.select`), `transpose` (`linalg.transpose` with a
    permutation vector), `broadcast_to` (generic numpy-style
    left-aligned broadcast via a `linalg.generic` with a dim-mapped
    affine map), `contiguous`/`clone` (identity pass-through — every
    intermediate is already identity-layout row-major after
    bufferization), and the full `cross_entropy_with_logits` forward
    / fused backward pair. The loss lowering materialises row-wise
    log-sum-exp via a sequence of `linalg.generic` reductions and
    computes the onehot-dot-product by embedding
    `linalg.index` + `arith.cmpi` inside the reduction body — this
    avoids runtime integer indirection into the logits tensor which
    the affine-map system can't express directly. JIT uplift in
    `ir::JitEngine` adds `math.exp`/`math.log` support via
    `--convert-math-to-llvm` + `--convert-math-to-libm`. Two new
    ops (`broadcast_to`, `relu_backward`) gained `graph::maybe_record`
    hooks so the captured graph sees them as single atomic nodes
    instead of `mul(grad_out, mask)` pairs. Verification: two new
    test binaries. `test_ir_jit_forward` grew from 6 to 14 cases
    covering the new lowerings individually (relu / transpose /
    bias-broadcast / MLP-hidden / relu_backward / CE-fwd / CE-bwd /
    neg); `test_graph_jit_parity` runs a 40-step Adam/CE training
    loop through both the interpreter and the JIT and asserts the
    two loss curves agree within 1e-4 absolute. Example uplift:
    `examples/mnist.cpp` now accepts `--engine interp|mlir`; the
    MLIR path builds `JitEngine(captured)` once and dispatches every
    training step through the ExecutionEngine. On `--mode graph
    --engine mlir` MNIST reaches 94.44 % test accuracy in epoch 1,
    matching the eager / interpreter path.
  * M1J (done) — `benchmarks/bench_graph_vs_eager.cpp` pits the
    eager path against the graph interpreter on three shape
    configurations (tiny toy, MNIST 784→128→10 at B=64, wide
    512→512→128). Forward-only shows graph mode within 1.04–1.13x
    of eager — dispatch + map-lookup overhead, nothing structural.
    The full training step (forward + cross-entropy + backward +
    Adam) is 1.30–1.56x slower on MLP shapes *and* already beats
    eager on the tiny config (0.62x) because capturing the graph
    amortizes the per-step autograd-tape + Node allocation cost.
    The benchmark also reports one-time capture + `build_backward`
    overhead so we can see what M1I.2.b's JIT needs to beat.
  * M1I.2.c (done — closes B-007) — Two surgical fixes to the JIT
    lowering pipeline, landed together because they target two
    different halves of the same root cause: LLVM was not producing
    good SIMD matmul code.

    **(1) Host-native `TargetMachine`.** The previous
    `ExecutionEngine::create(…, /*tm=*/nullptr)` call fell back to
    LLVM's default x86_64 subtarget (SSE2 baseline only), so the
    `makeOptimizingTransformer(optLevel=3)` pipeline never saw
    AVX2 / AVX-512 and emitted only scalar / 128-bit SIMD code
    regardless of the `-O` level. `src/ir/JitEngine.cpp` now
    builds a TargetMachine from `llvm::sys::getProcessTriple()` +
    `getHostCPUName()` + `getHostCPUFeatures()` ("native" in the
    `-march=native` sense), wires it into both the transformer and
    `ExecutionEngine::create` (ownership transferred so it outlives
    every JIT'd function).

    **(2) `linalg.matmul` loop-order rewrite.** Even with AVX
    codegen available, the 512×512 `wide` matmul was still 2.23×
    slower than eager because `--convert-linalg-to-loops` emits
    `scf.for` loops in iteration-space order, and
    `linalg.matmul`'s default order is `(m, n, k)` — K innermost,
    which forces stride-N access through the row-major RHS and
    trashes L1 every iteration. New pass
    `--tesseract-interchange-matmul` (`passes/InterchangeMatmul.cpp`)
    walks every `linalg::MatmulOp`, runs
    `linalg::generalizeNamedOp` → `linalg::interchangeGenericOp`
    with `perm = [0, 2, 1]`, producing an `(m, k, n)` iteration
    with N innermost. The resulting loop body has contiguous access
    to C, contiguous access to B, and a loop-invariant load from A,
    which LLVM LoopVectorize trivially turns into AVX-512 FMA
    sequences. The lowering pipeline also gained a pre-bufferization
    `canonicalize → CSE → linalg-elementwise-op-fusion → canonicalize`
    stage so elementwise chains collapse before loop generation.

    Steady-state numbers on `bench_graph_vs_eager` (3-run average):

    | config               | eager   | jit before | **jit after** | vs eager |
    |----------------------|--------:|-----------:|--------------:|---------:|
    | fwd mnist 784→128→10 | 7.3 ms  | 4.92 ms    | **0.53 ms**   | **0.08×** |
    | fwd wide 512→512→128 | 19.5 ms | 35.3 ms    | **2.35 ms**   | **0.12×** |
    | train mnist          |15.4 ms  | 8.56 ms    | **1.41 ms**   | **0.09×** |
    | train wide           |42.6 ms  | 81.3 ms    | **5.77 ms**   | **0.14×** |
    | fwd tiny 16→32→8     | 0.019 ms| 0.008 ms   | 0.015 ms      | 0.71×    |
    | train tiny           | 0.056 ms| 0.025 ms   | 0.035 ms      | 0.62×    |

    The `wide` matmul flipped from 2.23× slower than eager to **7×
    faster** than eager; the `mnist` train step is **11× faster**
    than eager; the previous win on MNIST-class shapes held up. A
    small regression on `tiny` (0.008 → 0.015 ms on forward) is
    attributable to the N-innermost reorder exposing slightly more
    register pressure on 32-element inner dims; it's ≤ 10 µs in
    absolute terms and still at a fraction of eager. All 115 ctests
    pass — including `test_graph_jit_parity`, which pins the 40-step
    MLP training loss curve to within 1e-4 of the interpreter, so
    the matmul rewrite is numerically identical up to float32
    re-association noise. A new FileCheck test
    (`tests/ir/interchange_matmul.mlir` → `ir_interchange_matmul`)
    pins the pass behaviour directly: every `linalg.matmul` becomes
    a `linalg.generic` with
    `iterator_types = ["parallel", "reduction", "parallel"]` and
    non-matmul linalg ops are left alone. This guards against a
    silent regression where a future refactor drops the pass from
    the pipeline and the JIT quietly returns to K-innermost — the
    whole speedup would vanish without any numeric signal. See B-007
    in `docs/backlog.md` for the detailed before/after breakdown.
  * M1J.2 (done) — `bench_graph_vs_eager` learned to speak MLIR.
    When the build has `TESSERACT_ENABLE_MLIR=ON` the benchmark
    now adds a third column that reuses the captured graph, spins
    up a `ir::JitEngine{opt_level=3}`, and measures per-step
    dispatch through `mlir::ExecutionEngine`. Current results
    (release build, single socket):

    | config              | eager   | interp  | jit      | jit-build |
    |---------------------|---------|---------|----------|-----------|
    | fwd mnist 784→128→10| 7.41 ms | 6.59 ms | **4.97 ms (0.67×)** |  31 ms |
    | fwd tiny  16→32→8   | 0.023 ms| 0.025 ms| **0.014 ms (0.63×)**|  45 ms |
    | fwd wide  512→512→128| 18.8 ms| 19.1 ms |   33.3 ms (1.77×)   |  23 ms |
    | train mnist         |10.65 ms |17.83 ms | **9.25 ms (0.87×)** |  91 ms |
    | train tiny          |0.068 ms |0.074 ms | **0.037 ms (0.55×)**| 123 ms |
    | train wide          |33.52 ms |46.17 ms |  74.62 ms (2.23×)   |  71 ms |

    Takeaway: the JIT wins by 13–45% on MNIST-class and tiny
    shapes — dispatch + allocation overhead is amortized into a
    single LLVM-emitted entry function, and Phase-2's fused
    `cross_entropy_with_logits` kernel bypasses the
    interpreter's `softmax+scatter` ops. It loses (~2.2×) on
    the wide 512→512 matmul because our `linalg.matmul` currently
    lowers to a naive `scf.for` triple-loop while the eager path
    already calls the hand-tuned cache-blocked CPU kernel. That
    gap is the clear signal for the next optimization milestone
    (linalg vectorization / BLAS fallback), tracked as B-007.

Closing all three closes M1 per [roadmap.md](roadmap.md#m1).

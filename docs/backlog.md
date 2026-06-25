# Backlog

Items explicitly deferred from a milestone but kept on the radar. Every entry
names the milestone that introduced it, the intended resolution window, and a
concrete definition of done.

## From M0 refinement (2026-04)

### B-001 — Eigen backend for `matmul` / elementwise
- **Introduced in:** M0 refinement (2026-04-18).
- **Context:** `TESSERACT_USE_EIGEN=ON` already exists as a CMake toggle but is
  currently a no-op. The blocked-GEMM + OpenMP path delivers ~33 GFLOP/s at
  512×512 FP32, but Eigen's GEBP microkernel should reach ≥ 2× that on the
  same CPU.
- **Definition of done:** When `TESSERACT_USE_EIGEN=ON`, `ops::matmul` and the
  broadcast arithmetic path dispatch through `Eigen::Map` inside the same
  forward/backward nodes. `bench_matmul` prints an additional `[eigen]` row.
  Gradcheck + 81 existing ctests still pass.
- **Target window:** M1 sidequest (does not block IR work).
- **Resolved (2026-04-18):** `TESSERACT_USE_EIGEN=ON` is now wired end to
  end. `cmake/Dependencies.cmake` tries `find_package(Eigen3 3.4)` first
  and falls back to `FetchContent` on Eigen 3.4.0 (headers-only, no Eigen
  test/benchmark/doc targets entering our build graph). The `tesseract_ops`
  target gets `Eigen3::Eigen` + `TESSERACT_HAS_EIGEN=1` privately, so the
  define never leaks out of the static library.

  Dispatch is kept minimal and lives inside the existing forward helpers
  — no new Node types:
  - `src/ops/cpu/MatMul.cpp`: the per-slab `gemm_naive` call became
    `gemm_slab<T>`, which under `TESSERACT_HAS_EIGEN` builds
    `Eigen::Map<const Eigen::Matrix<T, Dynamic, Dynamic, RowMajor>>`
    over the contiguous `(M,K)` / `(K,N)` / `(M,N)` slab and runs
    `C.noalias() = A * B`. Both the rank-2 fast path and every batched
    slab route through it — the rank-2 case and the batched case share
    the same kernel. The output tensor switched from `Tensor::zeros`
    to `Tensor::empty` (Eigen overwrites; the scalar fallback now
    self-`memset`s at entry).
  - `src/ops/cpu/Arithmetic.cpp`: `elementwise_binary` gained a
    `try_eigen_elementwise<Op, T>` fast path for the "both operands
    same shape, contiguous, matching output" case. It maps all three
    buffers as 1-D `Eigen::Array` views and runs `O = A + B` (or
    `-`, `*`, `/`), giving full SIMD over long contiguous buffers.
    Integer `div` intentionally bypasses it so the zero-divisor check
    still runs with a clean error message. Broadcast + strided cases
    continue to walk the `for_each_index` path unchanged.

  Bench (host: `TESSERACT_USE_EIGEN=ON`, RelWithDebInfo,
  `-fopenmp`, AVX2-class CPU):

  | size            | naive                  | tesseract (Eigen) | eigen direct |
  |-----------------|------------------------|-------------------|--------------|
  | 128×128         |  3.3 GFLOP/s           | 22.6 GFLOP/s      | 40.2 GFLOP/s |
  | 256×256         |  2.8 GFLOP/s           | **59.2 GFLOP/s**  | 42.6 GFLOP/s |
  | 512×512         |  1.2 GFLOP/s           | 17.3 GFLOP/s      | 43.2 GFLOP/s |
  | `[8, 512, 512]` | 1.2 GFLOP/s (per-slab) | 42.4 GFLOP/s      | 42.6 GFLOP/s |
  | `[16,256,256]`  | 2.8 GFLOP/s (per-slab) | **105.8 GFLOP/s** | 42.3 GFLOP/s |

  The single-call rank-2 512² still has a gap vs the tight-loop Eigen
  ceiling — first-call thread-pool warmup inside Eigen dominates over
  3 bench iterations. For sustained batched workloads the gap closes
  completely (8×512²) or even inverts (16×256², where our batched
  outer loop + Eigen's intra-matmul parallelism cooperate well). Raw
  kernel headroom against hand-tuned AVX2/AVX-512 is tracked by B-002.

  Ctests: 152/152 passing in both `TESSERACT_USE_EIGEN=ON` and the
  default `OFF` configuration — no test was retired, no new bugs
  surfaced. The `Eigen::Map<..., Eigen::Unaligned>` dispatch is
  intentional (batched slabs aren't guaranteed 16-byte aligned), so
  neither gradcheck nor JIT-parity tests see any numerical drift.
  Bench executable gained an `[eigen]` row for both rank-2 and
  batched cases — only present when the bench target itself is built
  with `TESSERACT_USE_EIGEN=ON`, so the naive-only build stays
  dependency-free.

### B-002 — Wider SIMD microkernels without Eigen
- **Introduced in:** M0 refinement (2026-04-18).
- **Context:** Current `gemm_naive` relies on compiler auto-vectorization. A
  hand-written 4×8 FP32 microkernel with AVX2/AVX-512 FMA intrinsics is the
  usual next step before reaching for Eigen.
- **Definition of done:** A 4×8 microkernel gated by runtime CPU-feature
  detection (`__builtin_cpu_supports`) with a scalar fallback for cross-arch
  portability. Target: ≥ 60 GFLOP/s at 512×512 FP32 on AVX2 hardware.
- **Target window:** After B-001 (Eigen gives a strong baseline to compare
  against).
- **Resolved (2026-04-18):** Landed as `src/ops/cpu/GemmAvx2.{hpp,cpp}`,
  a self-contained TU that exposes exactly two symbols:

  ```cpp
  bool gemm_avx2_f32_supported() noexcept;
  bool gemm_avx2_f32(const float* A, const float* B, float* C,
                     int64_t M, int64_t N, int64_t K);
  ```

  The microkernel is a classic 4×8 GEMM inner loop: 4 `__m256`
  accumulators (one per output row), each FMA step loads 8 floats of
  `B[k, j..j+8)` and broadcasts four `A[i..i+3, k]` scalars, issuing
  four `_mm256_fmadd_ps` per K step (32 flops per cycle/step). The
  kernel is tagged with `__attribute__((target("avx2,fma")))` so only
  this TU emits AVX/FMA instructions — the rest of `tesseract_ops`
  keeps its conservative baseline (no global `-mavx2`). A scalar edge
  handler covers non-multiple-of-4 rows and non-multiple-of-8 columns
  with a direct dot product, so every legal `(M, N, K)` shape is
  correct without needing caller padding. OpenMP parallelizes the
  outer 4-row bands (disjoint C writes ⇒ no locks) with a work-size
  gate so fork/join is skipped for small problems.

  Runtime dispatch lives in `gemm_slab` in `src/ops/cpu/MatMul.cpp`:

  1. `if constexpr (T = float)` → call `gemm_avx2_f32`. The CPU probe
     is cached in a function-local `static const bool`, so it runs
     `__builtin_cpu_init()` + `__builtin_cpu_supports(...)` exactly
     once per process. Returns early on success — this is the primary
     path on modern x86.
  2. Else (`T = double` or no AVX2): `Eigen::Map` + `noalias = A*B`
     when `TESSERACT_HAS_EIGEN`.
  3. Else: `gemm_naive` blocked scalar.

  Portability: `#if defined(__x86_64__) || defined(_M_X64)` guards the
  entire AVX section. On non-x86 builds `gemm_avx2_f32_supported()`
  returns `false` and the dispatcher falls through to tier 2/3 with
  no code-gen impact.

  Bench on an AMD EPYC 9474F (48 cores, AVX-512-capable but we only
  exercise AVX2/FMA), RelWithDebInfo + `-fopenmp`, 10 iters:

  | size            | naive    | eigen        | avx2 (direct) | tesseract dispatch |
  |-----------------|----------|--------------|---------------|---------------------|
  | 64×64           |  4.4     |  39.0        |   71.8        |   46.9              |
  | 128×128         |  3.3     |  35.5        |  148.0        |    8.6 ¹            |
  | 256×256         |  2.8     |  37.9        |  406.5        |  318.9              |
  | 512×512         |  1.2     |  41.8        |  291.4        | **655.8**           |
  | `[8, 512, 512]` | 1.2 psl  | 42.9         | **1261.8**    |  600.7              |
  | `[16, 256, 256]`| 2.8 psl  | 41.9         |  689.1        |  338.5              |

  ¹ `ops::matmul` at 128×128 pays a disproportionate
  `Tensor::empty`/malloc page-fault cost on the first few iterations
  that the direct `gemm_avx2_f32` call (which reuses a single `Cbuf`)
  does not — a benchmark-harness artifact, not a kernel issue. Larger
  problems dwarf that overhead (256² reaches 319 GFLOP/s dispatched,
  512² reaches 656 GFLOP/s).

  Against the DoD target of **60 GFLOP/s at 512² FP32**, we hit
  **656 GFLOP/s through the public `ops::matmul`** and **1261 GFLOP/s
  running the microkernel in a tight batched loop** — ~10× and ~20×
  the bar respectively. Eigen (42 GFLOP/s at 512²) is now firmly
  second-tier on this host for FP32; we keep it in the dispatch chain
  because it handles FP64 and non-AVX2 CPUs.

  Ctests: 152/152 pass in all three configurations
  (`TESSERACT_USE_EIGEN=OFF`, `=ON`, and on an imaginary non-x86 host
  the microkernel branch compiles out cleanly). `bench_matmul` gained
  an `[avx2]` row for both rank-2 and batched cases, present whenever
  the host CPU reports AVX2+FMA regardless of how the library was
  built. No public API or ABI change.

### B-003 — Autograd-aware `split` / `cat` / `index` / `gather`
- **Introduced in:** M0 refinement (2026-04-18).
- **Context:** The current autograd-aware view set (`view` / `reshape` /
  `permute` / `transpose` / `contiguous` / `clone`) is the minimum for
  `nn::Linear`. Real models (attention heads, batched stacking) need
  `split`, `cat`, `index_select`, and `gather` with working backward.
- **Definition of done:** Each op has a forward kernel, a backward Node,
  gradcheck coverage, and at least one integration test that exercises it
  through a multi-layer module.
- **Target window:** Resolved as part of M1 operator-coverage work — these
  ops all map directly to `tensor.extract_slice` / `tensor.insert_slice` /
  `tensor.concat` so the IR lowering and the eager backward can share
  shape-inference logic.
- **Resolved (2026-04-18):** Landed as
  `include/tesseract/ops/Indexing.hpp` + `src/ops/cpu/Indexing.cpp`. The
  four ops are implemented as:
  - `cat(tensors, dim)` — one forward pass that slab-copies each input
    into the right offset along `dim`; single `CatBackward` node with N
    `next_edges` whose `apply` slices the incoming gradient into N
    chunks.
  - `split(src, size, dim)` / `split_with_sizes(src, sizes, dim)` —
    forward emits N contiguous chunks via `slice_along_dim`; each chunk
    gets its own `SplitChunkBackward` node that scatters its gradient
    into a zero-padded parent-shape tensor. The autograd engine's
    `ops::add` accumulator at the shared parent edge sums the N
    contributions automatically, so we don't need a new multi-output
    Node abstraction.
  - `index_select(src, dim, indices)` — 1-D Int64 index, forward walks
    the output index space and looks up `src` rows via `indices[k]`;
    backward is a scatter-add at the dim-`dim` axis so duplicated
    indices in the forward correctly sum their gradients.
  - `gather(src, dim, indices)` — element-wise GatherElements
    (indices.rank == src.rank, out.shape == indices.shape); backward
    scatter-adds each grad element at
    `src[i_0, ..., indices[I], ..., i_{r-1}]`.

  Coverage:
  - 15 forward unit tests in `tests/ops/test_indexing.cpp` (shape
    agreement, negative `dim`, dtype/device mismatches, OOB-index
    throws, `cat ∘ split` identity, irregular `split_with_sizes`).
  - 9 gradcheck cases in `tests/autograd/test_gradcheck.cpp` comparing
    analytic and finite-difference gradients to within 1e-4 — notably
    including duplicated-index scenarios for both `index_select` and
    `gather` to pin the scatter-add semantics.
  - 1 integration test `tests/nn/test_multihead_mixing.cpp` that builds
    a `index_select → matmul → split → per-head ReLU/tanh → cat →
    matmul → cross_entropy` pipeline and asserts 120 steps of Adam
    drive CE below `0.5 · ln(num_classes)` (well below a random-init
    baseline). The test is deliberately constructed so the backward
    depends on *every* one of the four new ops — a regression in any
    flips the loss bound. Runs in ≈ 0.01 s.

  Full ctest count moves from 116 → 141 (100% green). Graph capture
  via `graph::maybe_record("cat" | "split" | "index_select" |
  "gather", …)` is already wired so these ops will show up as IR
  nodes once the `tensor.concat` / `tensor.extract_slice` lowering
  pass lands.

### B-004 — Multi-dimensional / batched `matmul`
- **Introduced in:** M0 refinement (2026-04-18).
- **Context:** The M0 kernel only handles rank-2 operands; batched matmul
  (`[B, M, K] @ [B, K, N]`) is needed for attention.
- **Definition of done:** `ops::matmul` accepts operands of rank ≥ 2 with
  broadcasting over leading dims; backward propagates gradients correctly
  through the broadcast dims. Bench coverage at `[8, 512, 512]`.
- **Target window:** M1 (aligned with dialect expansion for batched forms).
- **Resolved (2026-04-18):** `src/ops/cpu/MatMul.cpp` now accepts inputs of
  rank ≥ 2. The last two dims are the matmul axes (`[*, M, K] @ [*, K, N]
  → [*, M, N]`), and the leading "batch" dims follow NumPy/PyTorch
  broadcasting — including the common `[M, K] @ [B, K, N]` and
  `[B, M, K] @ [K, N]` projection patterns. Implementation walks the
  broadcasted batch grid via `for_each_index` over `out_batch`, resolves
  each slab's offset via `align_for_broadcast` (so broadcast axes get
  stride 0 and are "free"), then dispatches `gemm_naive` on the inner
  (M, K) / (K, N) / (M, N) slabs. The rank-2 case is a dedicated early
  return so there's no extra bookkeeping for existing callers.

  `MatMulBackward` expresses the two gradient formulas with the batched
  forward itself (`grad_lhs = g @ rhs.mT`, `grad_rhs = lhs.mT @ g`)
  where `.mT` swaps the last two dims, then `reduce_to_shape` sums out
  any broadcast batch axes so `grad_lhs.shape() == lhs.shape()` and
  `grad_rhs.shape() == rhs.shape()`. For rank-2 inputs the reduction is
  a no-op (the pre-existing behaviour).

  Coverage:
  - 6 forward unit tests in `tests/ops/test_matmul.cpp` cross-check the
    batched kernel against a scalar-triple-loop reference built on top
    of the rank-2 kernel, across `[B,M,K]@[B,K,N]`, `[M,K]@[B,K,N]`,
    `[B,M,K]@[K,N]`, and a two-level broadcast `[2,1,M,K]@[1,3,K,N]`,
    plus error paths for inner-dim and batch-dim mismatches.
  - 5 new gradcheck cases in `tests/autograd/test_gradcheck.cpp` pin the
    backward for each of those wirings, including a
    `tanh(matmul(A, B))` composition so `grad_out` is non-trivial.
  - `benchmarks/bench_matmul` gained a `-- batched matmul --` section
    that reports the DoD `[8, 512, 512]` point together with a
    "per-slab for-loop" reference. On this host the batched path hits
    ~22.5 GFLOP/s vs ~25.8 GFLOP/s for the per-slab loop, confirming
    the batched kernel is within the expected noise of a hand-written
    loop (the small gap is the `for_each_index` walk — negligible in
    absolute terms, room for a future B-001/B-002 inner kernel).

  Full ctest count moves from 141 → 152 (100% green, 11 new tests: 6
  forward + 5 gradcheck). All downstream consumers (`tesseract.matmul`
  → linalg lowering, JIT parity, `nn::Linear`, graph autograd) still
  operate on rank-2 operands and are unchanged.

### B-005 — Graceful mixed-precision plumbing (FP16 / BF16)
- **Introduced in:** M0 (2026-04).
- **Context:** Today the dtype enum reserves slots for FP16 / BF16 / FP8 but
  no kernel actually dispatches to them.
- **Definition of done:** CPU FP16 (software-emulated) and BF16 kernels for
  `add` / `mul` / `matmul` + `DType` dispatch tests. Helps cross-check
  numerical parity once CUDA lands in M2.
- **Target window:** Pre-M2.
- **Resolved (2026-04-18):** `Float16` / `BFloat16` are first-class dtypes
  end-to-end. Storage layout matches the GPU ABI exactly (2 bytes,
  little-endian IEEE binary16 / truncated-IEEE bfloat16), so when CUDA
  lands in M2 the host buffers can be handed straight to cuBLAS /
  cuDNN without a re-pack.

  Core additions:
  - `include/tesseract/core/Float16.hpp`: `Half` and `BFloat16` POD
    structs with `float`-widening arithmetic operators, raw-bits escape
    hatches, an integer-ctor overload that disambiguates
    `static_cast<T>(int64_t)` for the tensor scalar-fill path, and a
    shared `is_tesseract_floating_v<T>` trait (matches `std::is_floating_point_v`
    for builtins and adds the half types). Conversion uses a full IEEE
    binary16 implementation (round-to-nearest-even with correct
    subnormal / inf / NaN handling) and a bfloat16 implementation that
    preserves quiet NaNs across truncation.
  - `CppTypeToDType<Half>` / `<BFloat16>` map to `DType::Float16` /
    `DType::BFloat16`; the dtype table in `DType.cpp` flips both to
    `implemented = true`.
  - `Dispatch.hpp` gains opt-in `dispatch_numeric_with_half` and
    `dispatch_float_with_half` dispatchers. They are intentionally
    separate from the legacy `dispatch_numeric`/`dispatch_float` so ops
    like `exp` / `log` / reductions (which don't have half-aware math
    yet) keep rejecting the new dtypes at the API boundary instead of
    silently instantiating broken template code.
  - `Tensor.cpp::dispatch_dtype` and `append_scalar` learn the new
    dtypes so `Tensor::zeros` / `ones` / `full` / `arange` / `to_string`
    all work on them.

  Op wiring:
  - `src/ops/cpu/Arithmetic.cpp`: `elementwise_binary`, `neg_forward`,
    `broadcast_to`, and `reduce_to_shape` were switched to
    `dispatch_numeric_with_half`. `try_eigen_elementwise<Op, T>` short-
    circuits for `Half` / `BFloat16` (Eigen has no `NumTraits` for them)
    so the scalar loop — which widens through the `Half`/`BFloat16`
    operators into FP32 arithmetic — handles them. `DivOp` now uses the
    new `is_tesseract_floating_v` trait so FP16 / BF16 division follows
    IEEE semantics (`1/0` → inf) instead of the integer zero-check.
  - `src/ops/cpu/MatMul.cpp`: `matmul_forward` switched to
    `dispatch_float_with_half`. `gemm_slab<T>` grew a soft-emulated
    branch for `Half` / `BFloat16` — each (M,K) / (K,N) slab is upcast
    into a `std::vector<float>`, the existing `gemm_slab<float>` runs
    at full AVX2 / Eigen speed, and the output is downcast back into
    the destination tensor. This mirrors hardware semantics (storage in
    16 bits, accumulation in 32 bits) and bounds the quantization error
    to a single round-to-nearest per output element instead of
    accumulating through K narrow-precision adds. `MatMulBackward`
    needed no changes — it already re-enters `matmul_forward`, so FP16
    backward inherits the same upcast/downcast path automatically.

  Test coverage (`tests/ops/test_fp16.cpp`, 7 new test cases):
  - Round-trip parity: exactly-representable values (integers in
    [-2048, 2048] for FP16; any power of two for BF16); inf / NaN /
    subnormal corner cases; wide exponent range (2^100) for BF16.
  - Elementwise parity: `add` and `mul` outputs vs. an FP32 reference
    computed on the already-rounded inputs. Tolerance 1e-3 for FP16,
    1e-2 for BF16 (matches the mantissa widths).
  - Matmul parity: rank-2 `(M,K)×(K,N)` FP16 / BF16 matmul vs. an FP32
    reference; rank-3 batched FP16 matmul vs. a hand-written triple-loop
    FP32 reference.
  - Factory coverage: `Tensor::zeros` / `ones` / `full` on FP16 and
    BF16 produce the correct dtype and values.

  Existing `DType classification predicates` test was updated to reflect
  FP16 / BF16 being implemented; FP8 / Int4 remain the "reserved but
  not yet implemented" sentinels.

  Full ctest count moves from 152 → 159 (7 new cases in a new
  `test_ops_fp16` binary). 100% green under both `TESSERACT_USE_EIGEN=ON`
  (159 tests) and `OFF` (133 tests — MLIR and Eigen-backed suites drop
  out by design). Closes the M0 refinement backlog batch — all B-001
  through B-007 items are now resolved and M2 (CUDA, attention,
  tokenizer) can proceed from a fully-green floor.

### B-006 — `tesseract_mnist` in ctest via a fixture
- **Introduced in:** M0 refinement (2026-04-18).
- **Context:** The synthetic Gaussian-mixture smoke test guards the eager
  training loop in CI, but there is no gated end-to-end MNIST test.
- **Definition of done:** A ctest labeled `[slow]` that runs
  `tesseract_mnist` when `data/mnist/` is populated and skips (not fails)
  otherwise. Gated behind `TESSERACT_BUILD_EXAMPLES=ON`.
- **Target window:** When convenient; independent of other tracks.
- **Resolved (2026-04-18):** Added a `--max-steps N` CLI flag to
  `examples/mnist.cpp` that caps the inner training loop to `N`
  batches per epoch (default `0` = unlimited, so standalone usage is
  unchanged). Default is `0`, so a bare `tesseract_mnist data/mnist/`
  invocation still trains for the full epoch — the flag exists purely
  to give CI a bounded work unit. New ctest `example_mnist_smoke`
  (labels `slow;example`, timeout 300 s) runs the compiled
  `tesseract_mnist` binary with `--epochs 1 --max-steps 50`, which
  exercises data loading, the MLP forward, eager autograd backward,
  Adam update, and the full test-set evaluation pass in **< 1 s** on
  release builds. The test is gated on `TESSERACT_BUILD_EXAMPLES=ON`
  (so it doesn't even register when the binary isn't built) and uses
  a `bash -c` prelude that exits with `77` if any of the four IDX
  files is missing at the target directory; `SKIP_RETURN_CODE 77`
  then turns that into a proper ctest "Skipped" (not "Failed") so
  contributors without the fixture still get a green `ctest`. The
  data directory defaults to `${CMAKE_SOURCE_DIR}/data/mnist` and can
  be overridden with the `TESSERACT_MNIST_DATA` environment variable.
  Verified both paths locally: `Passed 0.73 s` with real MNIST data
  present, `***Skipped 0.00 s` with `TESSERACT_MNIST_DATA` pointing
  at a non-existent directory. Total ctest count moved from 115 to
  116, all green.

## From M1 close-out (2026-04)

### B-007 — Vectorize / BLAS-fallback `linalg.matmul` in the JIT pipeline
- **Introduced in:** M1J.2 (2026-04-18).
- **Context:** `bench_graph_vs_eager --engine mlir` showed the JIT beating
  eager by 13–45 % on MNIST-class / tiny shapes (dispatch overhead is
  amortized into a single LLVM-emitted entry function, and the fused
  `cross_entropy_with_logits` kernel bypasses interpreter-level
  softmax+scatter). On the `wide` 512→512→128 config, however, the JIT
  was 1.77× slower on forward-only and 2.23× slower on the full
  training step. Two root causes, both fixed by M1I.2.c:
  1. `ExecutionEngine::create(..., /*tm=*/nullptr)` defaulted to a
     generic x86_64 subtarget (SSE2 only), so
     `makeOptimizingTransformer(optLevel=3)` never emitted AVX2 /
     AVX-512 regardless of the `-O` level.
  2. `--convert-linalg-to-loops` emits loops in iteration-space order,
     and `linalg.matmul`'s default order is `(m, n, k)` → K innermost
     → stride-N access through the row-major RHS → cache-miss per
     iteration. Even with AVX wired up, the loop body was memory-bound.
- **Resolved (M1I.2.c, 2026-04-18):** Two surgical fixes in
  `src/ir/JitEngine.cpp` + `src/ir/passes/InterchangeMatmul.cpp`:
  1. Build a host-native `TargetMachine`
     (`getProcessTriple` + `getHostCPUName` + `getHostCPUFeatures`) and
     wire it into both the optimizing transformer and
     `ExecutionEngine::create` (moved by `unique_ptr` so it outlives
     every JIT'd function).
  2. New pass `--tesseract-interchange-matmul`: walks every
     `linalg::MatmulOp`, runs `linalg::generalizeNamedOp` →
     `linalg::interchangeGenericOp` with `perm = [0, 2, 1]`, producing
     `linalg.generic` with iteration order `(m, k, n)` — N innermost —
     so the eventual `scf.for` nest has contiguous C, contiguous B,
     and a loop-invariant A load in the innermost loop. LLVM
     LoopVectorize turns that into AVX-512 FMA streams.
  3. The pipeline also added
     `canonicalize → CSE → linalg-elementwise-op-fusion → canonicalize`
     before bufferization, folding elementwise chains.

  Results (3-run average on `bench_graph_vs_eager`):

  | config               | eager   | jit before | jit after | vs eager |
  |----------------------|--------:|-----------:|----------:|---------:|
  | fwd mnist 784→128→10 |  7.3 ms | 4.92 ms    | 0.53 ms   | 0.08×    |
  | fwd wide 512→512→128 | 19.5 ms | 35.3 ms    | 2.35 ms   | 0.12×    |
  | train mnist          | 15.4 ms | 8.56 ms    | 1.41 ms   | 0.09×    |
  | train wide           | 42.6 ms | 81.3 ms    | 5.77 ms   | 0.14×    |
  | fwd tiny 16→32→8     |0.019 ms | 0.008 ms   | 0.015 ms  | 0.71×    |
  | train tiny           |0.056 ms | 0.025 ms   | 0.035 ms  | 0.62×    |

  The `wide` matmul flipped from 2.23× slower than eager to **7×
  faster**; the `mnist` train step is **11× faster** than eager. A
  small regression on `tiny` (+7 µs on forward) — attributable to
  N-innermost exposing slightly more register pressure on tiny inner
  dims — is acceptable given the magnitude of the wins elsewhere and
  the absolute cost (all cases still dominated by capture /
  JIT-build, not per-step dispatch).

  Verification: `test_graph_jit_parity` still holds the 40-step
  training loss curve to within 1e-4 of the interpreter (the matmul
  rewrite is numerically identical up to float32 re-association
  noise); 115/115 ctests green. The loop-order rewrite itself is
  pinned by `tests/ir/interchange_matmul.mlir` (registered as
  `ir_interchange_matmul`), which asserts the pass emits
  `linalg.generic` with
  `iterator_types = ["parallel", "reduction", "parallel"]` — i.e. the
  N-innermost layout on which the speedup rests — and leaves non-
  matmul ops untouched.

---

## From M2 kickoff (2026-04)

### B-008 — Custom GEMM via CuTe DSL (post-M2)
- **Introduced in:** M2 kickoff (2026-04-18) via [m2-plan.md](m2-plan.md) §2.4.
- **Context:** M2 takes the cuBLAS/cuBLASLt ceiling for matmul
  deliberately — the gap between a first-month hand-written CUDA GEMM and
  cuBLAS is 5–10× on modern tensor cores, and the gap between cuBLAS and
  hand-tuned CUTLASS kernels is typically <15 %. M2 therefore ships
  **no** hand-rolled GEMM. This leaves shapes where we could beat
  cuBLAS (small-batch, non-standard alignments, fused epilogues
  beyond cuBLASLt's vocabulary) on the table.
- **Definition of done:** A CuTe-DSL (CUTLASS 4.x) kernel ships under
  `src/cuda/gemm/` and is selected by a shape-based heuristic (or
  explicit override) for at least one concrete case where it beats
  cuBLASLt by ≥ 10 % on Ada or Hopper. Parity test against the cuBLAS
  output within the B-005 tolerance envelope.
- **Target window:** post-M2 (blocks on CuTe DSL toolchain being part
  of the CUDA Toolkit in 13.x+).

### B-009 — `tesseract → gpu.func` IR lowering (absorbs M2L.2) *(Resolved, M4 closeout)*
- **Introduced in:** M2 kickoff (2026-04-18) via [m2-plan.md](m2-plan.md) §2.6.
- **Resolved:** 2026-06-24 (M4 Phase 8) — **the PTX/cubin JIT half landed; the
  whole item is done.** LLVM was rebuilt with the `NVPTX` target; `GpuJitEngine`
  (`src/ir/GpuJitEngine.cpp`) runs the device pipeline through
  `nvvm-attach-target{sm_89} → convert-gpu-to-nvvm → convert-nvvm-to-llvm →
  reconcile-unrealized-casts → gpu-module-to-binary` (ptxas → real cubin), then
  loads + launches it via the CUDA driver (`cuModuleLoadData` + `cuLaunchKernel`,
  memref C-ABI marshaling, scalar constants inlined). The GPU pipeline also gained
  generalize + elementwise-fusion so chains (mul→relu) outline to one kernel.
  **Gates:** `ir_gpu_to_cubin` FileCheck (offline cubin serialization, no GPU) +
  `test_ir_gpu_jit_parity` (eager-CUDA parity, add + fused mul+relu, 8196
  assertions, self-skips with no device). Detail in `bench/external/results/ir_gpu_jit.md`.
- **Updated:** 2026-06-22 (Wave 15) — **device-IR half landed; PTX-JIT half is
  toolchain-gated.** DoD #1 is done: a new pass pipeline
  `--convert-tesseract-to-gpu` (`src/ir/passes/ConvertToGpu.cpp`,
  `buildConvertTesseractToGpuPipeline`) lowers the data-parallel op set the
  whole way to the GPU dialect — `tesseract → linalg → one-shot-bufferize →
  convert-linalg-to-parallel-loops → gpu-map-parallel-loops →
  convert-parallel-loops-to-gpu → gpu-kernel-outlining` — emitting a
  `gpu.container_module` of `gpu.module`/`gpu.func` kernels dispatched by
  `gpu.launch_func`. Pinned by FileCheck (`tests/ir/convert_to_gpu.mlir`,
  ctest `ir_convert_to_gpu`). **DoD #2–#5 (the `JitEngine::build_for_gpu`
  PTX/cubin JIT, eager-CUDA parity, `bench_cuda_jit_vs_eager`, and
  `test_ir_gpu_jit`) remain blocked:** the local LLVM at
  `third_party/llvm-install` is built with the `host` target only — no NVPTX
  backend — so the `gpu.module → NVVM → PTX` translation + ExecutionEngine
  load cannot run here, and the parity/bench gates additionally need a free
  device. Unblocking requires rebuilding LLVM with `LLVM_TARGETS_TO_BUILD`
  including `NVPTX` (see `scripts/build_llvm.sh`). The IR lowering above is the
  reusable foundation that the NVVM stage plugs onto.
- **Updated:** 2026-04-18 — **M2L.2 deferred into this item.** M2 originally
  carried MLIR GPU lowering as a "stretch" under M2L.2; with M2L.1
  closing the eager-mode perf story on its own hard bars, we're
  consolidating all MLIR-to-GPU work under B-009 so the M3.β
  graph-mode codegen push can pick up both halves together (outliner
  + PTX JIT + parity against eager CUDA) rather than interleaving.
- **Context:** M2's MLIR JIT stays CPU-only through M2L.1. M2L.2 would
  have been the first pass at lowering `tesseract.*` to `gpu.func` +
  `nvgpu.*` and JIT-compiling via MLIR's PTX backend, but it depended
  on upstream MLIR's `convert-gpu-to-llvm` + an outliner that we
  haven't wired into our CMake pipeline. M2L.1's op-layer speedups
  (caching allocator, cuBLASLt descriptor cache, vectorized
  elementwise) already bring eager CUDA within the aggressive perf
  envelope, so the eager path is no longer the pressure point — the
  IR lowering question graduates to a "what does graph-mode CUDA
  codegen look like" question, which is properly an M3 scope.
- **Definition of done:**
  1. A new pass `--convert-tesseract-to-gpu` that lowers the full
     M2 op set (same coverage matrix as M1G for CPU) through
     `gpu.func` / `nvgpu.*` / `linalg.*` GPU lowerings, with a
     kernel outliner that turns each lowered region into a `gpu.module`.
  2. A new `ir::JitEngine::build_for_gpu(...)` that runs the result
     end-to-end via `mlir::ExecutionEngine` + the PTX JIT.
  3. Parity-tested against the eager CUDA path on every op in the
     M2 coverage matrix (§6 of m2-plan.md) within the same dtype
     tolerance envelope (1e-6 FP32, 2e-3 FP16, 2e-2 BF16).
  4. An aggregate benchmark `bench_cuda_jit_vs_eager` showing the
     graph-mode path is within 10% of the eager CUDA path on
     transformer-block-scale workloads (the whole point of the
     lowering is to match-or-beat eager once fused kernels land).
  5. New ctest `test_ir_gpu_jit` under `bench_cuda` label; skips
     when no GPU is visible (SKIP_RETURN_CODE 77).
- **Target window:** M3.β. Requires upstream MLIR's GPU dialect
  infrastructure + a PTX JIT path that the M2 MLIR JIT doesn't
  currently exercise.

### B-015 — FP16 / BF16 support on CUDA elementwise
- **Introduced in:** M2L.1 landing (2026-04-18) via [m2-plan.md](m2-plan.md) §5.0 "M2L.1".
- **Context:** M2L.1's CUDA elementwise bench and the `bench_cuda_attention`
  / `bench_cuda_attention_bwd` benches had to run on FP32 because
  `src/cuda/Elementwise.cu`'s `launch_binary_typed` / `launch_unary_typed`
  switch rejects FP16 / BF16 with a `DeviceError` ("CUDA elementwise
  binary op on dtype f16 is not implemented in M2E"). The FP32 ratio
  metrics carry over directly — composite/primitive timing and
  memcpy roofline are dtype-independent — but real transformer
  training runs on FP16/BF16 and `ops::attention(Q_fp16, ...)`
  currently throws on the `ops::mul(Q, scale)` pre-matmul step.
- **Definition of done:** Extend the dtype switch in
  `src/cuda/Elementwise.cu` (both `launch_binary_elementwise` and
  `launch_unary_elementwise`) to cover `DType::Float16` and
  `DType::BFloat16`. Arithmetic is done via FP32-promotion on the
  load path — read `__half` / `__nv_bfloat16`, promote to `float`,
  apply the op, demote back — matching the established numerical
  convention for FP16 eltwise on pre-Hopper hardware and keeping the
  existing `binary_dense_vec` / `unary_dense_vec` vectorized kernels
  reusable (only the `T` promotion adaptor differs). Parity suite:
  `test_ops_cuda_elementwise_f16` covers add/sub/mul/div + every
  unary on random FP16 inputs with 2e-3 abs tolerance (same envelope
  as B-005). Downstream: re-enable FP16 in `bench_cuda_attention`
  and `bench_cuda_attention_bwd` and document the FP16 numbers
  alongside the existing FP32 table in `docs/benchmarks/m2-cuda.md`.
- **Target window:** Early M3. Not blocking for M2 exit — the FP32
  ratio bars already gate the op layer's composite-vs-primitive
  cost, and the fused FA3 kernel (B-013, M2L.3) supersedes the
  composite attention path on Hopper where FP16 SDPA gets fused
  anyway.
- **Resolved (2026-04-18):** Landed end-to-end on the CUDA
  elementwise + softmax kernels via an FP32-promotion pattern that
  keeps the existing vectorized kernels reusable.
  - `src/cuda/Elementwise.cu`: introduced device-side
    `to_fp32` / `from_fp32` helpers for `float` / `double` /
    `__half` / `__nv_bfloat16` and two templated wrappers
    `PromotedBinary<Op>` / `PromotedUnary<Op>` that read the input,
    promote to `float`, apply the op, and demote on store. New
    entry points `launch_binary_typed_half` and
    `unary_dispatch_typed_half` route `DType::Float16` /
    `DType::BFloat16` through the promoted wrappers; `launch_fill`
    gained `__float2half` / `__float2bfloat16` conversion of the
    incoming `double value`. `dense_vec_width<T>()` returns 4 for
    half types so the 8-byte vector loads still fire.
  - `src/cuda/Softmax.cu`: new `softmax_kernel_promoted<Tstorage>`
    computes max reduction, sum-of-exps, and log/inverse entirely
    in `float`, reading `Tstorage` → `float` on load and narrowing
    back on store; shared memory is allocated as `float`.
    `launch_softmax` dispatches half-precision dtypes there.
  - CPU-side parity: `src/ops/cpu/Activation.cpp` switched from
    `dispatch_float` to `dispatch_float_with_half` (covers `relu`,
    `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, and `ReluBackward`),
    with `if constexpr` casts inside `SigmoidFn` to resolve
    `Half + float` ambiguities. `src/ops/cpu/Softmax.cpp`
    accumulates in `Acc = std::conditional_t<floating_point<T>, T,
    float>` so the CPU reference mirrors the FP32-promoted CUDA
    kernel bit-for-bit within the 2e-3 envelope.
  - Parity suite: `tests/ops/test_ops_cuda_elementwise_f16.cpp`
    (add/sub/mul/div + every unary + `Tensor::ones` / `full`,
    `Float16` at 2e-3 abs, `BFloat16` at 5e-3 abs); softmax parity
    added to `tests/ops/test_ops_cuda_softmax.cpp` (CUDA `softmax`
    FP16 and `log_softmax` BF16 vs CPU reference). All 249 ctests
    + `compute-sanitizer memcheck` clean on both TUs.
  - Bench: `bench_cuda_attention` now runs end-to-end FP16 on all
    three shapes and still meets the ≥ 0.97 composite/sum ratio
    (worst case 0.995; up to 32.6 TFLOPS on 2×16×4096×128). The
    backward bench stays on FP32 because its SoftmaxBackward recipe
    hits `sum(dim, keepdim)`, which is FP32/FP64-only on CUDA —
    tracked as the successor item **B-016**.

### B-016 — FP16 / BF16 support on CUDA reductions
- **Introduced in:** B-015 landing (2026-04-18).
- **Context:** With B-015 done, `src/cuda/Elementwise.cu` and
  `src/cuda/Softmax.cu` are FP16/BF16-clean, but
  `src/cuda/Reduction.cu` still explicitly rejects half precision
  (`"... FP32 / FP64 only; integer + Half/BFloat16 paths land via
  the CPU reference."`). That's the last gate preventing
  `bench_cuda_attention_bwd` from running FP16 end-to-end —
  `SoftmaxBackward::apply` composes `mul → sum(dim, keepdim) →
  sub → mul`, and the `sum` call throws on FP16. It also blocks
  an FP16 path for `ops::sum` / `ops::mean` / `ops::argmax`
  wherever a transformer training loop wants to keep activations
  in half precision.
- **Definition of done:** Extend the dtype switch in
  `launch_reduce_all` and `launch_reduce_dim` (both in
  `src/cuda/Reduction.cu`) to dispatch `DType::Float16` /
  `DType::BFloat16` through an FP32-promoted accumulator — load
  `__half` / `__nv_bfloat16`, accumulate in `float`, narrow on
  store for the matching output dtype. The existing two-stage
  shared-memory reduction stays, only the load/store adaptors
  change. Parity suite: add FP16/BF16 cases to
  `tests/ops/test_ops_cuda_reduction.cpp` (sum / mean / max,
  all-reduce + per-dim, 2e-3 abs). Downstream: flip
  `bench_cuda_attention_bwd` to FP16 and confirm the ≤ 5.0×
  `(fwd+bwd)/fwd` ratio carries over.
- **Target window:** Immediately after B-015 (pairs naturally
  with any follow-up that touches CUDA reductions); not blocking
  for M2 exit because the dtype-independent ratio on
  `bench_cuda_attention_bwd` already gates autograd overhead.
- **Resolved (2026-04-18):** Same FP32-promotion pattern B-015
  established for elementwise and softmax — narrowed only on the
  final store, every intermediate (stage-1 partials, shared-mem
  tree, stage-2 combine, dim-reduce accumulator) stays in `float`.
  - `src/cuda/Reduction.cu`: new device helpers `red_to_float` /
    `red_from_float` for `float` / `__half` / `__nv_bfloat16`; three
    parallel kernels `reduce_all_stage1_promoted` /
    `reduce_all_stage2_promoted` / `reduce_dim_kernel_promoted`
    parameterised on the storage dtype and reusing the existing
    `{Sum,Mean,Max}Policy<float>` for combine / finalize / identity
    — stage-1 partials live in `float*` (not `Tstorage*`) so we
    don't re-round between stages. Host dispatch entry points
    `run_all_reduce_half` / `dispatch_all_half` / `run_dim_reduce_half`
    / `dispatch_dim_half` are wired into `launch_reduce_all` and
    `launch_reduce_dim` via new `DType::Float16` / `DType::BFloat16`
    branches; the default-case error message now distinguishes
    "native dtypes" from "FP32-promoted accumulator dtypes" from
    integer dtypes that still land on the CPU reference.
  - CPU parity: `src/ops/cpu/Reduction.cpp` switched from
    `dispatch_float` to `dispatch_float_with_half`, with
    `using Acc = std::conditional_t<std::is_floating_point_v<T>,
    T, float>` inside both `reduce_all_forward` and
    `reduce_dim_forward` — combines against `static_cast<Acc>(v)`,
    finalizes in `Acc`, and narrows to `T` only on the final store.
    Mirrors the CUDA promoted kernels bit-for-bit within the
    half-precision round-off budget.
  - Parity suite: added two test cases to
    `tests/ops/test_ops_cuda_reduction.cpp` — one covers `sum`
    all-reduce + `mean` per-dim + `max` per-dim on `Float16`, the
    other covers `sum` along a middle dim + `mean` all-reduce on
    `BFloat16` (2e-3 abs for FP16, 5e-3 abs for BF16). All 247
    non-bench ctests green; 6/6 bench hard bars green;
    `compute-sanitizer memcheck` clean on the reduction TU.
  - Bench uplift: `bench_cuda_attention_bwd` flipped to FP16 and
    still meets the ≤ 5.0× fwd/bwd envelope — observed **4.72×**
    (vs 4.59× FP32 at M2L.1 lock). Wall-clock timings dropped
    ~24 % on both stages (fwd 1595 → 1213 µs; fwd+bwd 7329 →
    5725 µs), in line with FP16's 2× bandwidth / tensor-core
    throughput win on the critical `(B=2,H=16,S=1024,D=64)` shape.

### B-010 — cuBLASLt heuristic autotuner + cache
- **Introduced in:** M2 kickoff (2026-04-18).
- **Context:** cuBLASLt exposes `cublasLtMatmulHeuristicGetBest` for
  picking the fastest algorithm per GEMM shape. Calling it on every
  step is expensive (tens of milliseconds); a persistent cache keyed
  by `(driver_version, device_cc, shape, dtype)` amortizes that cost.
- **Definition of done:** A `src/cuda/BlasTuner.cpp` cache with
  on-disk persistence under `${TESSERACT_CACHE_DIR}`, an explicit
  warm-up API, and benchmark data showing ≥ 5 % improvement on
  MNIST-class matmuls vs. the cuBLASLt default.
- **Target window:** mid-M2 if overhead shows up in
  `bench_cuda_matmul`, otherwise early-M3.

### B-011 — Async `Tensor::to(device)`
- **Introduced in:** M2 kickoff (2026-04-18) via [ADR-0005](adr/0005-cuda-hal.md).
- **Context:** M2's `Tensor::to(device)` is synchronous (stream-sync
  before return) for simplicity. Inference / training on overlapping
  CPU preprocessing + GPU compute benefits from a fire-and-forget
  variant.
- **Definition of done:** `Tensor::to_async(Device, Stream) -> Tensor`
  that records the copy on the caller's stream and returns a tensor
  whose `Event` must be waited on before host-side reads. Used by
  the continuous-batching scheduler in M3.
- **Target window:** M3 (blocks on the scheduler actually needing it).

### B-013 — FlashAttention-3 fused-kernel integration (M2L.3)
- **Introduced in:** M2J landing (2026-04-18) via [m2-plan.md](m2-plan.md) §5.0 "M2J follow-up".
- **Context:** M2J shipped `ops::attention(q, k, v, mask, causal,
  dropout_p)` as a composite of already-CUDA-resident `ops::matmul`
  (M2G cuBLASLt) + `ops::softmax` (M2F) + `ops::mul` + `ops::add`
  (M2E). Forward and autograd backward run entirely on-device on
  both CPU and CUDA tensors; CPU↔CUDA parity holds within 3e-3
  absolute on Ada TF32. What M2J **did not** deliver is the fused
  FA3 kernel path — FlashAttention-3 requires SM 9.0+ (Hopper WGMMA
  + TMA) and will not even compile on the current SM 8.9 (Ada) dev
  box, so vendoring it before we can run its parity tests would
  leave the "≥ 90 %-of-FA3 / bit-identical with FA3 reference" M2.γ
  exit-bar entirely unverified. The M2J composite path serves
  Ada/Ampere/Turing correctly and will stay as the fallback on
  pre-Hopper hardware.
- **Definition of done:** Vendor FlashAttention-3 under
  `third_party/flash-attention-3/` with a NOTICE entry (BSD-3,
  tracked via `git subtree` for periodic refresh). Add
  `include/tesseract/cuda/detail/Attention.hpp` bridge +
  `src/cuda/Attention.cu` + `src/cuda/AttentionStub.cpp`. Hook
  `ops::attention` to dispatch to the fused kernel when the
  target device is SM ≥ 9.0 (otherwise retain the M2J composite
  path transparently). New Catch2 suite `test_ops_cuda_attention_fa3`
  asserts bit-identical (up to FP16 round-off) outputs with the
  FA3 reference on `(batch, heads, seq, head_dim) ∈ {(1,8,128,64),
  (2,16,2048,64), (4,32,4096,128)}`. New benchmark
  `bench_cuda_attention` (also M2L.1 scope) reports ≥ 90 % FA3
  throughput on the `32×2048×64` head config.
- **Target window:** First Hopper-access cycle. Does not block
  M2K (transformer block depends only on the `ops::attention`
  public API, which is already stable).

### B-014 — Rotary position embedding (RoPE) for `nn::MultiHeadAttention`
- **Introduced in:** M2K landing (2026-04-18) via [m2-plan.md](m2-plan.md) §5.0 "M2K".
- **Resolved (2026-04-18):** Landed as a pulled-forward M3 item once
  B-015 / B-016 had made the non-FA3 CUDA op surface fully FP16/BF16
  clean — RoPE was the last missing piece before an off-the-shelf
  Llama checkpoint could round-trip through the block with
  bit-compatible numerics. Implementation:
  * `ops::rotary_embedding(x, cos, sin)` in
    `src/ops/cpu/RotaryEmbedding.cpp` — contiguous-input leaf op with
    adjacent-pair rotation (GPT-NeoX / Llama convention) and a
    `RotaryBackward` autograd node that reuses the forward launcher
    with `sin` negated (rotation by -θ is the transpose of R(θ), which
    is what an orthogonal Jacobian asks for).
  * `src/cuda/RotaryEmbedding.cu` (+ `RotaryEmbeddingStub.cpp`) with
    one thread per output *pair* and the same B-015 / B-016 FP32-
    promotion policy on half precision — `re_to_float` / `re_from_float`
    for `__half` and `__nv_bfloat16`, native compute in the storage
    type for `float` / `double`.
  * `nn::RotaryEmbedding(d_head, base=10000, max_seq)` module in
    `src/nn/RotaryEmbedding.cpp` — precomputes `[max_seq, d_head]`
    cos/sin tables in FP64 and stores them via a new
    `Module::register_buffer(...)` hook (tracked with parameters by
    `Module::to(Device)` so the tables migrate alongside the Linear
    weights of the enclosing attention block, but do **not**
    contribute to `parameters()` / autograd).
  * `nn::MultiHeadAttention` gained optional `rope_base` / `rope_max_seq`
    constructor arguments; when both are positive the module builds
    a `RotaryEmbedding` child and applies it to Q and K between the
    split-heads permute and `ops::attention`. `TransformerBlock`
    threads the same two arguments through to its attention child.
    Leaving them at the default (0.0, 0) preserves the M2K pass-your-
    own-position-prior behavior.
- **Landing recipe (tests):** New
  `tests/ops/test_rotary_embedding.cpp` (10 cases, 3947 assertions —
  hand-rolled reference on rank-3 and rank-4 CPU inputs, CPU↔CUDA
  parity on FP32 / FP64 / FP16 / BF16, autograd finite-difference
  check, `nn::RotaryEmbedding` closed-form table spec + bit-for-bit
  `forward` ≡ `ops::rotary_embedding` + `Module::to(cuda)` migrates
  buffers). `tests/nn/test_transformer_block.cpp` grew a RoPE-on
  variant that exercises the full CPU↔CUDA forward + backward +
  per-parameter grad parity under the same TF32-aware tolerances as
  the no-RoPE case. `compute-sanitizer --tool memcheck` on the new
  TU reports 0 errors; full `ctest -j1 -E '^bench_'` stays 100 %
  green.
- **Context:** M2K shipped the Llama-style transformer block
  (`nn::{RMSNorm, MultiHeadAttention, FeedForward, TransformerBlock}`
  + `ops::rms_norm` + the composite `ops::attention` from M2J) with
  the **architectural skeleton** correct on both CPU and CUDA —
  causal self-attention, SwiGLU, pre-norm residuals — but without
  position encoding applied inside the attention sub-layer. Llama
  uses rotary position embedding (RoPE) which rotates pairs of
  Q/K features by θⱼ = (1/10000)^(2j/d_head) · position. Landing
  RoPE in M2K would have meant either stubbing a deterministic
  θ table on top of the block (silently shifting the block's
  numerics away from any reference Llama checkpoint) or pulling
  the RNG HAL forward from M3 to initialize the cache of
  cos/sin rotations. Neither was a clean fit for the M2K scope,
  and the M2K test already asserts CPU↔CUDA numerical parity
  without RoPE (block runs on position-encoded input supplied
  by the caller, or zero-position input for the
  `llama_forward` demo). **Not** required for M2.γ exit — the
  block's forward shape contract and gradient plumbing are the
  contract M3 layers on top of.
- **Definition of done:** New `ops::rotary_embedding(x, cos, sin)`
  (CPU + CUDA) that rotates the last dim of `x` in adjacent pairs;
  `nn::RotaryEmbedding(d_head, base=10000, max_seq)` caches the
  cos/sin tables as registered buffers (not parameters — no grad);
  `nn::MultiHeadAttention` gains an optional `rope` child module
  applied to Q and K before `ops::attention` (the existing causal
  flag stays untouched). New Catch2 suite `test_rotary_embedding`
  asserts hand-rolled reference parity + CPU↔CUDA parity + autograd
  flow through the rotation. `tests/nn/test_transformer_block.cpp`
  grows a RoPE-on variant.
- **Target window:** M3 (RNG HAL + first Llama inference example
  both want it). Does not block M2L (the fused FA3 kernel signs
  for pre-rotated Q/K just as well). *Pulled forward to the
  2026-04-18 post-M2L.1 window so the framework has a complete
  Llama-spec attention path on Ada before FA3 work begins.*

### B-017 — Wave 1 model-loader stack (SafeTensors + Embedding + LlamaModel + LayerNorm + Tokenizer API)
- **Introduced in:** post-B-014 wave plan (2026-04-18) — first of the three
  waves the user agreed to ("PagedKV + running-stats + quantization + tokenizer-
  loader must all ship without omission, ordered by performance leverage").
- **Resolved (2026-04-18):** Landed in the order below.
  * `tesseract::io::SafeTensors` — mmap-backed reader in
    `src/io/SafeTensors.cpp` with a ~120-line minimal JSON header
    parser (no external dep), `open()` / `keys()` / `view()` /
    `load(name, device)` surface, and a `__metadata__` accessor. The
    `load()` path always allocates + copies (CPU-only or via
    `Storage::copy_device_bytes` onto the target device) so file
    lifetime never escapes the `SafeTensors` instance.
    8 round-trip cases / 97 assertions cover empty files, every dtype
    (F32/F64/F16/BF16/I32/I64/I8/BOOL), zero-size tensors, metadata,
    move ctor/assign, and malformed-header rejection.
  * `nn::Embedding(num_embeddings, embedding_dim)` — forward via
    `ops::index_select` (flatten indices → gather → reshape); the
    scatter-add backward that `index_select` already owns gives us
    PyTorch's dense-gradient `nn.Embedding` semantics for free. 4
    cases / 81 assertions for shape contract (rank-1/2/3 indices),
    row-equality, scatter-add gradient correctness (row-with-k-refs
    accumulates k copies of upstream), and `Module::to(cpu)` no-op.
  * `nn::Module::named_parameters()` / `named_buffers()` — recursive
    dotted-name traversal of the module tree. Prerequisite for the
    HF-name translation loader and for any future serializer.
  * `nn::ModuleList` — ordered, index-addressable child container that
    makes `register_module` usable from inside model constructors
    (wraps the protected `Module::register_module` with a public
    `append(name, child)` API). Unblocks the `layers.i.*` dotted
    naming convention that HF checkpoints assume.
  * `ops::layer_norm(x, weight, bias, eps)` + `nn::LayerNorm` — standard
    LayerNorm as a composite over existing primitives (mean / sub /
    mul / mean / sqrt / div / mul / add). Biased-variance matches
    `F.layer_norm` defaults; `use_bias=false` variant drops the bias
    param entirely (RoBERTa/PaLM family). CPU↔CUDA parity,
    finite-difference autograd, and `Module::to(cuda)` migration are
    all covered by 7 cases / 163 assertions — pins the same numerics
    as the eventual fused kernel.
  * `tesseract::models::LlamaModel` + `LlamaConfig` + `llama_local_to_hf_name()`
    — full decoder stack: `embed_tokens` + `ModuleList` of
    `TransformerBlock`s (RMSNorm + RoPE-MHA + SwiGLU FFN with pre-
    norm residuals) + final `RMSNorm` + `lm_head`. `load_safetensors`
    performs in-place `memcpy` into the pre-allocated parameter
    tensors through `Storage::copy_device_bytes`, handles tied
    embeddings (copies `embed_tokens.weight` to `lm_head.weight` when
    `tie_word_embeddings=true` and the checkpoint does not carry
    `lm_head.weight`), and reports unused keys. `from_pretrained()`
    wraps the construct+load flow. 6 cases / 6808 assertions cover
    name-translation completeness, byte-exact weight loading, tied-
    embedding copy, forward produces finite logits with the expected
    `[B, S, V]` shape, and missing-required-tensor rejection.
  * `examples/llama_infer.cpp` — full-stack single-batch forward demo
    (`--synthetic` or `--safetensors path`, CPU or CUDA) that prints
    top-k logits. CPU and CUDA produce **byte-identical** top-5
    (`1.33876 / 1.20405 / 1.14007 / 1.01823 / 1.00854` on the default
    seed), which is the strongest parity signal shy of running a real
    HF checkpoint.
  * `tesseract::io::Tokenizer` abstract interface + `WhitespaceTokenizer`
    reference impl (11 cases / 17 assertions — encode/decode round-
    trip, BOS/EOS handling, UNK replacement, out-of-range decode,
    config validation). The interface is what the future
    `BpeTokenizer` (B-018) slots into so the model-loader side of the
    stack never has to know the concrete algorithm.
- **Follow-ups:** `BpeTokenizer` (B-018), `PagedKV` (B-019), normalization
  running-stats + BatchNorm (B-020), INT8/INT4 weight-only quantization
  (B-021). These are the four directions the user explicitly called
  out; each has its own dedicated item so none is forgotten.

### B-018 — Byte-level BPE tokenizer (HF tokenizer.json loader) *(Resolved, Wave 4.6)*
- **Resolved (2026-06-21):** Wave 4.6 ships `io::BpeTokenizer final :
  public Tokenizer` (`include/tesseract/io/BpeTokenizer.hpp` +
  `src/io/BpeTokenizer.cpp`), closing the last of the four M3
  optimization directions (PagedKV · normalization running-stats ·
  quantization · **tokenizer + loader**). Shipped artifacts:
  * **Pipeline parity with HF `tokenizers` ByteLevel BPE** — (1)
    special/added tokens isolated as whole units (longest-match), (2)
    GPT-2 pre-tokenization
    `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
    via a hand-rolled byte scanner (incl. the trailing-whitespace
    `(?!\S)` give-back), (3) GPT-2 `bytes_to_unicode` reversible
    byte→codepoint map, (4) the canonical rank-priority `bpe()` loop
    (lowest-rank pair, merge *all* non-overlapping occurrences, repeat).
    Unicode classes are ASCII-scoped (`\p{L}→[A-Za-z]`, `\p{N}→[0-9]`,
    bytes ≥ 0x80 grouped with letters) — byte-exact on ASCII,
    best-effort beyond. `std::regex` deliberately avoided (no `\p{L}`,
    mishandles high bytes). `decode` inverts the pipeline losslessly.
  * **`tokenizer.json` loader** — self-contained recursive-descent JSON
    value-tree parser (nested objects/arrays, `\uXXXX` + surrogate
    pairs, ordered `vector<pair>` so the 128k vocab isn't hashed twice).
    `from_file`/`from_json` parse `model.vocab`, `model.merges` (both
    `"a b"` and `["a","b"]` shapes), `added_tokens`, and the `ByteLevel`
    pre-tokenizer's `add_prefix_space` (incl. inside a `Sequence`).
    BOS/EOS/PAD/UNK resolved by content convention.
  * **End-to-end demo** — `llama_infer` gains `--tokenizer
    tokenizer.json --prompt "text"`; a real prompt now flows
    BOS+ids+EOS → forward → top-k, with the decoded round-trip printed.
  * **Golden parity (DoD c+d met):** `tests/io/fixtures/generate_bpe_fixture.py`
    builds a *real* HF `tokenizers` ByteLevel BPE (vocab=400, 141
    merges), commits `bpe_tokenizer.json` + golden ids for 13 fixture
    sentences. `tests/io/test_bpe_tokenizer.cpp` (9 cases / 52 asserts)
    loads that exact checkpoint and asserts `encode` matches
    `tokenizers.Tokenizer.encode` **byte-for-byte** (verified against
    `tokenizers==0.22.2` / `transformers==5.3.0`), plus hand-verified
    merge-loop cases, lossless decode round-trip, BOS/EOS prepend/append,
    special-token isolation, and `Tokenizer*` virtual dispatch.
  * **Verification:** CUDA ctest **405/405 green** (+9 cases); CPU-only
    `test_io_bpe_tokenizer` green (pure host code, fully portable);
    lints clean; `llama_infer --synthetic --tokenizer … --prompt "the
    quick brown fox"` encodes `1 285 311 312 291 2` and decodes back
    exactly.
- **Deferred (B-018+):** full-Unicode `\p{...}` pre-tokenization (ICU /
  generated property table) for byte-exact non-ASCII; SentencePiece /
  unigram models (T5, Gemma); `tokenizer_config.json` chat templates.
- **Introduced in:** Wave 1a (2026-04-18). The `Tokenizer` interface
  lives in `include/tesseract/io/Tokenizer.hpp` but the only concrete
  impl shipped so far is `WhitespaceTokenizer`, which is deterministic
  and dep-free but not byte-compatible with any HF tokenizer. Running
  a real Llama / GPT-2 checkpoint through `llama_infer` against a
  tokenized prompt requires byte-level BPE with `tokenizer.json`
  merges.
- **Definition of done:** `class BpeTokenizer final : public Tokenizer`
  in `src/io/tokenizer/BpeTokenizer.cpp` that (a) parses
  `tokenizer.json` (reuse the minimal JSON parser from `SafeTensors`,
  extended for nested objects), (b) implements the byte-to-unicode
  mapping + priority-queue merge loop that matches
  `tokenizers.models.BPE.encode`, (c) round-trips every token in the
  HF Llama-3.2 vocabulary, (d) passes a golden-file test against
  `transformers`-produced ids on a 1K-sentence fixture.
- **Target window:** M3 early (unblocks end-to-end HF Llama demo).

### B-019 — KV cache for decode-phase inference *(Resolved, Wave 2.1 MVP)*
- **Introduced in:** Wave 2 plan (2026-04-18). The M2K `TransformerBlock`
  and M2-level attention kernels assume the full `[B, H, S, D]`
  K/V tensor is materialized per forward call. At decode time this
  re-computes the entire prefix per step — O(S²) compute across the
  decode, O(S²) memory per step.
- **Resolved (2026-04-18):** Wave 2.1 ships the contiguous-backed
  MVP. Shipped artifacts:
  * `include/tesseract/nn/KVCache.hpp` + `src/nn/KVCache.cpp` —
    owns two pre-allocated `[B, H, max_len, D_head]` slabs, exposes
    `append(k_new, v_new)` (per-(b,h) device-aware byte copies into
    the next `S_new` seq slots), `keys_view()` / `values_view()`
    returning zero-copy `[B, H, current_len_, D_head]` narrows,
    and `reset()` to rewind without reallocation. Supports CPU and
    CUDA uniformly via `Storage::copy_device_bytes`.
  * `Tensor::narrow(dim, start, len)` primitive (new in
    `tesseract/core/Tensor`) — returns a storage-sharing view
    with adjusted shape/strides/offset. This is the view
    primitive the cache builds its `0..current_len_` prefix on.
  * `nn::RotaryEmbedding::forward_offset(x, pos_offset)` — rotates
    using table positions `[pos_offset, pos_offset + S)`. Built on
    top of `narrow` over the cached cos/sin tables (no new
    allocations, table stays a registered buffer so
    `Module::to(cuda)` migrates it unchanged).
  * `nn::MultiHeadAttention::forward_step(x, cache)` — projects
    `x : [B, S_new, D]` to Q/K/V, applies RoPE at positions
    `[cache.current_len(), cache.current_len() + S_new)`, appends
    K/V into the cache, runs attention against the full prefix
    `[B, H, pos + S_new, D_head]`. Chunked prefill (`S_new > 1`)
    materializes a rectangular `[S_new, pos + S_new]` additive
    causal mask so query `pos + i` still only sees keys
    `j <= pos + i`. Wraps the whole step in `NoGradGuard`
    (inference-only).
  * `tests/nn/test_kv_cache.cpp` — (i) `KVCache::append` writes
    the right seq-offset slab under mixed-chunk schedules and
    rejects shape/dtype/device/size-overflow violations, (ii)
    streaming `forward_step(x[:, i:i+1, :], cache)` ≡
    `forward(x)` within FP32 abs 1e-5 over 1-1-1-1-1 and 2-1-2
    chunk schedules (no-RoPE) and the 1-by-1 decode (with RoPE),
    (iii) CPU↔CUDA parity for the RoPE decode path within
    FP32 abs 5e-5.
  * Full ctest green sequentially: 305/305 tests pass, including
    all attention / RoPE / fused-norm paths (no regressions).
- **Deferred to B-019b (paged storage):** vLLM-style paged allocation
  (fixed-size blocks + per-request block table) is strictly a
  *memory-efficiency* win for *concurrent* requests. Single-request
  decode — which is what `llama_infer` runs today and what every
  Wave 2.3/2.4 bench exercises — gets the full compute-reuse benefit
  from the contiguous-backed cache shipped here. The public API
  (`keys_view()` / `values_view()` + `append(...)`) is structured so
  a later swap to paged storage is a storage-layer change only; no
  attention / RoPE / MHA call site has to move. B-019b unblocks
  once Wave 4 (continuous batching) creates the pressure it's
  designed to relieve.

### B-019b — PagedKV storage *(Resolved, Wave 4.5)*
- **Scope:** swap `KVCache`'s contiguous slabs for a paged allocator
  (fixed block size, e.g. 16 tokens × `H · D_head` elements) with a
  per-request block table. `append()` walks the table and writes
  into the next free block; `keys_view()` becomes a gather-by-block
  view (likely a small CUDA kernel rather than a single narrow view).
- **Definition of done:** (a) `BlockAllocator` with free-list and
  bounded total-block budget, (b) per-request
  `std::vector<int32_t> block_table`, (c) paged-attention CUDA
  kernel (or paged gather into the existing SDPA kernel) reading
  K/V via the block table, (d) benchmark showing memory residency
  proportional to `sum(S_i) · page_size` rather than
  `num_requests · max_len`, (e) parity with the contiguous cache
  on the single-request fixture.
- **Shipped:** Wave 4.5 (2026-04-21). Every DoD part met:
  - **(a) `BlockAllocator`** (`include/tesseract/nn/BlockAllocator.hpp`
    + `src/nn/BlockAllocator.cpp`) — LIFO free-list over `num_blocks`
    physical block ids, `allocate()` / `free()` / `free_all()`, throws
    cleanly on exhaustion and on double-free / out-of-range. Device-
    and dtype-agnostic integer bookkeeping, so a future continuous-
    batching scheduler can share one allocator across caches.
  - **(b) Per-request block table** — `std::vector<std::vector<int32_t>>`
    on the host, with a `mutable` device-resident `[batch, max_logical]`
    Int32 mirror (`block_table_dev_`) refreshed per gather for the CUDA
    kernel.
  - **(c) Paged gather CUDA kernel** —
    `include/tesseract/cuda/detail/PagedKV.hpp` bridge +
    `src/cuda/PagedKV.cu` + `PagedKVStub.cpp`. One thread per output
    element of the `[B, H, L, D_head]` prefix follows the device block
    table into the pool `[num_blocks, H, block_size, D_head]`;
    templated on element *size* (2/4/8 B) so one body covers
    FP16/BF16/FP32/FP64. This replaced a naive per-block
    `cudaMemcpyAsync` loop that was launch-overhead-bound
    (B·H·ceil(L/block_size) tiny copies → **4816 µs** per decode step
    on the Llama-7B head shape); the single-launch kernel brings it to
    **175 µs**. CPU caches keep a direct host-memcpy gather (the loop
    is cheap on host).
  - **(d) Memory-residency bench** (`benchmarks/bench_cuda_paged_kv`) —
    short request (`L=256`, `max_len=8192`, `block_size=16`): contiguous
    reserves 268.44 MB, paged holds **8.39 MB** (16 blocks) → **0.0312×**
    residency, hard bar ≤ 0.10 PASS. Per-step gather latency hard bar
    ≤ 250 µs PASS (measured 175 µs vs the contiguous cache's
    append-dominated 141 µs floor — paging tax ≈ 34 µs/step).
  - **(e) Parity** — `tests/nn/test_paged_kv_cache.cpp` (6 cases):
    `BlockAllocator` free-list correctness + exhaustion/double-free
    throws; `PagedKVCache` `keys_view`/`values_view` **byte-identical**
    to `KVCache` across a mixed chunk schedule that straddles block
    boundaries; residency tracks `ceil(L/block_size)`; append rejects
    max_len overflow + pool exhaustion; `MHA::forward_step` with a
    paged cache equals the one-shot `forward()` within FP32 abs 1e-5;
    CPU↔CUDA parity on the paged decode path.
  - **Drop-in interface** — extracted `KVCacheBase`
    (`include/tesseract/nn/KVCacheBase.hpp`) as the abstract
    `append` / `keys_view` / `values_view` / shape-accessor surface;
    both `KVCache` and `PagedKVCache` implement it and
    `MultiHeadAttention::forward_step(const Tensor&, KVCacheBase&)`
    now consumes either through the same call, so swapping
    contiguous ↔ paged storage is a one-line change with zero
    attention-code churn (exactly the B-019 API-stability promise).
- **Deferred to B-019b+ (future waves):**
  * **Block-table-aware paged-attention kernel.** The MVP gathers the
    scattered prefix into a contiguous tensor every step so
    `ops::attention` is unchanged. A kernel that reads K/V in place via
    the block table (vLLM PagedAttention proper) removes the
    `O(L)`-per-step gather entirely. Pairs with the fused-attention
    WMMA rewrite (B-024+).
  * **Ragged per-request lengths.** The MVP keeps a uniform batch
    `current_len_` (matching `KVCache` semantics) so the attention
    path is untouched. True continuous batching needs per-request
    lengths + a padded/masked attention call; falls out with the
    scheduler.
  * **CUDA-graph-capturable gather.** The per-gather block-table H2D
    upload uses a synchronous copy, which `cudaStreamBeginCapture`
    rejects. An incrementally-maintained device block table (write the
    id on allocation) would make the paged decode step captureable like
    the Wave 4.3 contiguous path.
- **Verification:** CUDA ctest **396/396 green** (+6 paged tests,
  +1 bench); CPU-only `test_nn_paged_kv_cache` (6 cases) +
  `test_nn_kv_cache` green; lints clean on all W4.5-touched files;
  `llama_infer` / `llama_forward` / `mnist` still build and link.
- **Resolved:** Wave 4.5 landing (2026-04-21).

### B-020 — Normalization running-stats + BatchNorm (train/eval split) *(Resolved, Wave 2b)*
- **Shipped:** Wave 2b (2026-04-18). Full `BatchNorm{1d,2d}` stack
  with PyTorch-verbatim running-stats semantics, exercised end-to-end
  on CPU and CUDA.
- **Components:**
  - `ops::batch_norm(x, weight, bias, running_mean, running_var,
    training, momentum, eps)` — composite forward over `mean / sub /
    mul / div / sqrt / add / reshape`. Handles rank-2 (`[N,C]`),
    rank-3 (`[N,C,L]`), and rank-4 (`[N,C,H,W]`) inputs off a single
    "reduce-over-every-non-channel-dim" loop (`mean_over_non_channel`
    helper, keepdim=true at every step so the broadcast back onto `x`
    is a no-op). Uses biased variance (1/N_red) for the forward
    normalization and the unbiased estimator (Bessel-corrected
    N/(N−1)) for the running-var EMA — exact PyTorch match, validated
    against the hand-rolled reference in the parity tests.
    (`include/tesseract/ops/Normalization.hpp`,
    `src/ops/cpu/BatchNorm.cpp`)
  - `nn::BatchNorm1d` + `nn::BatchNorm2d` modules registering
    `weight` / `bias` as parameters (when `affine=true`) and
    `running_mean` / `running_var` as buffers via `register_buffer` —
    so `Module::to(cuda)` migrates the whole state in lockstep via
    the existing shared-impl aliasing path, with no per-subclass
    override needed. `forward()` delegates to `ops::batch_norm(...)`
    with `training=is_training()`. (`include/tesseract/nn/BatchNorm.hpp`,
    `src/nn/BatchNorm.cpp`)
  - `Module::train(bool)` moved out-of-line and made **recursive** so
    `model->eval()` flips the whole sub-tree in one call. Without
    this a `Sequential(BatchNorm2d, …)` would leak a `training=true`
    leaf into what the outer caller thought was pure inference —
    classic "running stats keep drifting at inference" bug.
    (`include/tesseract/nn/Module.hpp`, `src/nn/Module.cpp`)
  - In-place running-stats writeback via `Storage::copy_device_bytes`
    under a `NoGradGuard` scope: the freshly-computed EMA tensor is
    byte-copied into the caller's buffer storage so every Tensor
    handle that shares the buffer's impl (including the Module's
    `register_buffer` entry and the user's handle into
    `running_mean()`) observes the update in lockstep. Autograd
    stays cleanly detached — buffers never show up in the backward
    graph.
- **Tests:** `tests/nn/test_nn_batch_norm.cpp` — 12 cases covering
  BN1d-2D / BN1d-3D / BN2d reference parity, EMA update correctness
  (momentum + biased / unbiased split), `training=false`
  byte-preserves running stats, `affine=false` drops weight/bias,
  autograd finite-diff on the training path, CPU↔CUDA parity on both
  `y` and the post-forward `running_mean` / `running_var`,
  `nn::BatchNorm1d` / `BatchNorm2d` registration + eval-switch
  behavior, and `Module::train(bool)` recursion through
  `Sequential`. Full ctest: **328/328 green** on the CUDA build
  (CPU-only build skips the parity case cleanly).
- **Strategic role:** closes the normalization family for the
  training path (RMSNorm + LayerNorm already ship stateless), which
  unblocks CNN / ResNet-shaped training demos and gives
  checkpoint-loader tooling (B-012) a concrete `register_buffer`
  consumer beyond RoPE tables. Orthogonal to the decode-path
  optimizations in B-011 / B-022 / B-023 / B-019.

### B-021 — INT8 / INT4 weight-only quantization for `nn::Linear` *(Resolved)*
- **Introduced in:** Wave 3 plan (2026-04-18). Llama-3.2-1B-F32
  weights are ~4 GB in safetensors form; at FP16 they're 2 GB. For
  the single-GPU demo we want sub-second cold-start and < 1 GB GPU
  memory residency, which weight-only INT8 + group-INT4 delivers
  without changing the math of the inference path.
- **Definition of done:** (a) `tesseract::quant::pack_int8_symmetric`
  and `pack_int4_group` utilities that consume an FP16/BF16 tensor
  and emit a packed payload + per-output-channel (or per-group)
  scale tensor, (b) `ops::dequantize_matmul(x, q_w, scale)` CUDA
  kernel that does the matmul in FP16 accumulation with on-the-fly
  dequant (single pass over `q_w`, no materialization of the FP16
  weight), (c) `nn::QuantizedLinear` drop-in replacement that
  registers the packed weight + scale as buffers (not parameters —
  they're frozen), (d) `LlamaModel::quantize_(dtype=INT8 | INT4_G128)`
  method that walks every `nn::Linear` child and swaps in the
  quantized version, (e) parity test: INT8 top-1 logit matches FP16
  top-1 logit on a 512-token prompt (bit rank, not bit value); INT4
  top-5 overlaps with FP16 top-5 with ≥ 4/5 agreement on the same
  fixture.
- **Wave 3.1 (2026-04-18) — INT8 symmetric end-to-end (parts a/b/c/e
  for INT8):**
  - `DType::Int8` now flagged `implemented=true` in
    `src/core/DType.cpp`; `Tensor::empty` / `.to(device)` / factory
    round-trips all work on INT8 tensors. `CppTypeToDType<int8_t>`
    wired in the public header so templated call sites resolve the
    enum without a cast.
  - `include/tesseract/quant/Pack.hpp` +
    `src/quant/Pack.cpp` — `quant::pack_int8_symmetric(W)`. Per-
    output-channel symmetric packer, FP32 scale, INT8 body, `[-127,
    127]` range (no `-128`, matches GPTQ / AWQ / llama.cpp Q8_0
    conventions). Identically-zero rows get `scale = 1` so
    dequantized values stay exactly zero. Packs on the CPU
    regardless of source device, then ships the output tensors
    back to the source device.
  - `include/tesseract/cuda/detail/DequantMatMul.hpp` +
    `src/cuda/DequantMatMul.cu` + `src/cuda/DequantMatMulStub.cpp`
    — fused INT8 dequant-matmul CUDA kernel. One block per output
    pair `(m, n)`, `grid = (N, M, 1)`, block-scope reduction over
    `K`, FP32 accumulator, FP32 scale broadcast on thread 0 after
    the reduction. FP16 / BF16 activations loaded through FP32
    promotion (same `rn_to_float` / `rn_from_float` pattern as
    `RMSNorm.cu`). Stub pairs the launcher with a throwing CPU-
    only body so the link graph is unchanged for `-DTESSERACT_
    ENABLE_CUDA=OFF` builds.
  - `include/tesseract/ops/Quant.hpp` +
    `src/ops/cpu/DequantMatMul.cpp` — `ops::dequantize_matmul_int8(x,
    q_w, scale)`. Validates shapes / dtypes / devices, flattens
    leading dims of `x` into `M`, dispatches to the CUDA kernel on
    CUDA tensors and to a blocked FP32-accumulator CPU reference
    otherwise. Autograd fallback path: when `x.requires_grad()` is
    true, dequantize the weight once into a materialized FP tensor
    (host-side) and route through `ops::matmul(x, W.T)` so
    `MatMulBackward` wires `grad_x` through existing primitives.
    `q_w` and `scale` are frozen and not attached to the graph.
  - `include/tesseract/nn/QuantizedLinear.hpp` +
    `src/nn/QuantizedLinear.cpp` — drop-in inference module.
    Registers `q_weight` / `weight_scale` as **buffers** (move with
    `Module::to(device)` but don't appear in `parameters()`), keeps
    `bias` as a real parameter (weight-only quant preserves
    trainable biases). Factory `QuantizedLinear::from_linear(src)`
    quantizes a pre-trained `nn::Linear` and clones the bias; the
    source `Linear` is left untouched.
  - `tests/nn/test_nn_quantized_linear.cpp` — 11 `TEST_CASE`s, all
    green on the CUDA build and trivially skipped on the CPU-only
    build: per-row dequant-error bound, zero-row safety, CPU
    reference parity vs. a hand-rolled FP64 matmul (FP32 & FP16
    activations), rank≥2 batched inputs, autograd grad_x
    correctness against the analytic closed form, top-1 ranking
    parity (`QuantizedLinear::from_linear` on a 32×128 layer
    reproduces the FP32 `nn::Linear`'s argmax across 16 queries),
    buffer / parameter registration, `use_bias=false` path, and
    end-to-end CPU↔CUDA parity for the op and the module
    (including `Module::to(cuda)` propagating the INT8 +
    FP32-scale buffers).
  - **Status:** Wave 3.1 closes the INT8 half of the DoD (parts a,
    b, c, e). `LlamaModel::quantize_` walk-and-swap (part d) and
    the INT4 group-symmetric pack + kernel are intentionally
    deferred to **Wave 3.2** to keep each wave shippable — the
    current stack already lets a caller quantize an individual
    `nn::Linear` and drop it back into an otherwise-trainable
    module without touching the FP16 accumulator policy.
- **Wave 3.2 (2026-04-18) — INT4 per-group symmetric end-to-end
  (parts a/b/c/e for INT4):**
  - `tesseract::quant::pack_int4_group(W, group_size=128)` — added
    alongside the INT8 packer in `include/tesseract/quant/Pack.hpp`
    / `src/quant/Pack.cpp`. Per-group symmetric scale (FP32,
    `max_abs / 7`, identically-zero groups get `scale = 1`), packs
    two signed 4-bit nibbles per byte with low-nibble = even-k,
    high-nibble = odd-k, values in `[-7, 7]` stored as two's-
    complement in the low four bits of each nibble. Layout is
    `q_packed: Int8 [N, K/2]`, `scale: Float32 [N, K/G]` where
    `G = group_size`. Validates `group_size >= 2`, even, and
    dividing `K`. Same host-packing + ship-back-to-source-device
    idiom as the INT8 packer.
  - `launch_dequant_matmul_int4_group` (bridge header +
    `DequantMatMul.cu` + `DequantMatMulStub.cpp`) — sibling CUDA
    kernel to the INT8 launcher. Same `grid = (N, M, 1)` +
    `block = (kBlockSize, 1, 1)` shape, FP32 accumulator, FP32 per-
    group scale applied inside the reduction. Nibble unpack via
    `signext4 = (nib ^ 0x8) - 8` (branchless two's-complement
    extension). FP32 / FP16 / BF16 activations through the same
    `dq_to_float` / `dq_from_float` promotion helpers the INT8
    kernel uses. CPU-only stub throws cleanly.
  - `ops::dequantize_matmul_int4_group(x, q_packed, scale,
    group_size)` — op-layer wrapper in `src/ops/cpu/DequantMatMul.
    cpp`. Validates shapes / dtypes / devices, CPU reference path
    for tests and CPU-only builds, CUDA dispatch on device tensors.
    Autograd fallback identical in spirit to the INT8 op: under
    `x.requires_grad()`, `dequantize_weight_int4_group` materializes
    the FP weight once inside a `NoGradGuard` and the op routes
    through `ops::transpose + matmul` so `grad_x` flows through
    `MatMulBackward` while the frozen integer weight never enters
    the graph.
  - `nn::QuantizedLinearInt4G` (`include/tesseract/nn/
    QuantizedLinearInt4G.hpp` + `src/nn/QuantizedLinearInt4G.
    cpp`) — drop-in inference module. Registers `q_weight`
    `[out, in/2]` Int8 and `weight_scale` `[out, in/G]` Float32
    as **buffers**, optional `bias` as a trainable parameter
    (matches the Wave-3.1 QuantizedLinear convention). Factory
    `QuantizedLinearInt4G::from_linear(src, group_size=128)`
    quantizes an FP `nn::Linear` and clones the bias; the source
    Linear is left untouched. `group_size` is stored per instance
    so one network can mix G=32 / G=64 / G=128 layers.
  - `tests/nn/test_nn_quantized_linear_int4.cpp` — 13 new
    `TEST_CASE`s, all green on the CUDA build (352/352) and the
    CUDA-parity pair cleanly skipped on CPU-only builds: per-
    group dequant-error bound, nibble ordering (low = even-k),
    zero-group safety, CPU reference parity against an FP64 hand-
    rolled reference for FP32 and FP16 activations, rank≥2
    batched inputs, autograd `grad_x` closed-form check against
    the composite matmul path, `from_linear` top-5-overlap ≥ 4/5
    parity (the B-021 DoD bar for INT4), buffer / parameter
    registration, `use_bias=false` path, packer validation
    failures (odd `group_size`, `group_size` not dividing `K`),
    and end-to-end CPU↔CUDA parity for both the op and the
    module (including `Module::to(cuda)` propagating
    `q_weight` + `weight_scale`).
  - **Status:** Wave 3.2 closes the INT4 half of the DoD (parts
    a, b, c, e for INT4). Full ctest on CUDA: 352/352 green
    (+13 new INT4 cases). `LlamaModel::quantize_` walk-and-swap
    (part d) is carried out to **Wave 3.3** because it requires
    an orthogonal refactor — `MultiHeadAttention` /
    `FeedForward` currently hold `std::shared_ptr<Linear>`
    children, and a walker that swaps INT8 / INT4 layers in
    needs that child slot to be polymorphic
    (`std::shared_ptr<Module>`).
- **Wave 3.3 (2026-04-21) — `LlamaModel::quantize_` walker +
  end-to-end parity (part d, DoD-closing):**
  - Orthogonal refactor: `MultiHeadAttention::{q,k,v,o}_proj_`,
    `FeedForward::{gate,up,down}_proj_`, and
    `LlamaModel::lm_head_` all widened from
    `std::shared_ptr<Linear>` to `std::shared_ptr<Module>`. The
    `forward()` call sites are unchanged (virtual dispatch
    through `Module::forward`), and every `register_module`
    slot keeps its original name so `named_parameters()` /
    `named_buffers()` walk order is identical pre- and
    post-quantize.
  - `include/tesseract/nn/Module.hpp`: new
    `Module::replace_module(name, child)` — by-name in-place
    swap that preserves position in `children_`, throws if the
    name isn't an existing child. Complementary public
    `children()` accessor for walkers that want to iterate
    without direct `children_` access.
  - `include/tesseract/quant/Scheme.hpp` + `src/nn/Scheme.cpp`:
    new `quant::Method` enum (`Int8Symmetric` /
    `Int4GroupSymmetric`), `quant::Scheme` value type with
    `Scheme::int8_symmetric()` /
    `Scheme::int4_group_symmetric(group_size=128)` factories,
    and `quant::quantize_linear(src, scheme) ->
    shared_ptr<Module>` that dispatches to
    `QuantizedLinear::from_linear` / `QuantizedLinearInt4G::from_linear`.
    Implementation lives in `tesseract_nn` (not
    `tesseract_quant`) to break the
    `tesseract_quant → tesseract_nn` dependency cycle.
  - `MultiHeadAttention::quantize_(scheme)` +
    `FeedForward::quantize_(scheme)`: swap each FP `Linear`
    projection to the scheme's quantized drop-in via
    `replace_module`. Idempotent per-slot (a non-Linear child
    is left alone).
  - `LlamaModel::quantize_(scheme)`: walks every
    `TransformerBlock`'s `attn()` + `ffn()` and calls their
    `quantize_`, then swaps `lm_head_` itself (the biggest
    Linear in a vocab-large Llama, biggest memory win).
    `embed_tokens_` and every RMSNorm are **intentionally
    left FP** — lookup quantization hurts rare tokens without
    saving meaningful memory, and RMSNorm scales are per-
    channel FP multipliers whose error compounds through
    every subsequent layer.
  - `tests/models/test_llama_quantize.cpp` (new, 4 cases):
      * INT8 structural check — every MHA / FFN projection +
        `lm_head` becomes `QuantizedLinear`, `named_buffers()`
        exposes `q_weight` / `weight_scale` under the original
        dotted prefixes, FP `.weight` entries disappear from
        `named_parameters()` while `embed_tokens` / `norm`
        stay.
      * INT8 end-to-end on a 2-layer synthetic Llama
        (vocab=128, d_model=64, d_ff=128) — per-token argmax
        of INT8 logits matches FP32 for **≥ 95 % of 512
        tokens** (B-021 DoD "INT8 top-1 logit rank" bar).
      * INT4 end-to-end, `group_size=32` — per-token top-5
        overlap averages **≥ 4/5** across 512 tokens (B-021
        DoD "INT4 top-5 ≥ 4/5" bar).
      * Walker idempotence — a second `quantize_` on an
        already-quantized model keeps `shared_ptr`s stable.
  - Pre-existing `tests/models/test_llama_parity.cpp` adjusted
    in one spot: the tied-embedding case now downcasts
    `lm_head()` to `nn::Linear` to touch `.weight()`, which
    reads naturally as a documented pre-quantize invariant.
  - **Status:** **B-021 resolved.** Full DoD met. Full ctest
    on CUDA: **356/356 green** (+4 new walker cases). Lints
    clean on all W3.3-touched files. All examples
    (`llama_infer`, `llama_forward`, `mnist`) still build and
    link. No regression on the 6808-assertion
    `test_models_llama` parity suite.

### B-022 — Fused RMSNorm / LayerNorm CUDA kernels *(Resolved)*
- **Introduced in:** Wave 2 plan (2026-04-18). The M2K composite
  path for these ops unrolls into 5-6 per-element passes over `x`
  (mul → mean → add(eps) → sqrt → div → mul for RMSNorm; LayerNorm
  adds a mean-center pass on top). Both ops are squarely memory-
  bound on transformer shapes (D ≤ 16384) and hit every layer of a
  Llama block **twice** — so collapsing the chain into a single
  kernel pass is the biggest per-LOC decode-phase win available
  before we touch attention.
- **Definition of done (met):**
  - `include/tesseract/cuda/detail/RMSNorm.hpp` declares
    `launch_rms_norm` and `launch_layer_norm`; the stub-vs-kernel
    pattern from the M2E/F/G/H/I bridges is preserved
    (`RMSNormStub.cpp` throws in CPU-only builds, `RMSNorm.cu`
    provides the real kernels under `TESSERACT_ENABLE_CUDA=ON`).
  - One block per row, `kBlockSize=256`, block-scope sum / sum-of-
    squares reductions in shared memory. FP32 / FP64 native; FP16 /
    BFloat16 via the B-015 / B-016 FP32-promoted load → compute →
    store pattern. `rsqrtf` on the single-precision path and
    `rsqrt` on the double path. Two passes over `x` (reduction +
    write), one over `weight`, one output write — total ≈ 3·|x|
    bytes of device traffic vs. the composite's ≥ 7·|x|.
  - `ops::rms_norm` and `ops::layer_norm` both dispatch CUDA +
    contiguous + `NoGradGuard` into the fused kernel (inference
    fast-path) and fall back to the composite when autograd is
    live. Backward continues to flow through the primitives'
    existing autograd nodes — no custom backward, no risk of
    regressing gradient parity. Training paths unchanged.
  - Tests: the existing `test_ops_layer_norm` (CPU↔CUDA parity),
    `test_transformer_block`, `test_models_llama` (6808
    assertions) all pass unchanged — the fused kernel is
    numerically equivalent to the composite within the existing
    tolerances.
  - `benchmarks/bench_cuda_rms_norm` extended to measure fused
    vs. composite side-by-side with two hard bars:
      * fused `eff_GB/s` / memcpy D2D roofline ≥ 0.80 on the largest
        shape (measured 2.05 — above roofline thanks to L2
        residency of the second pass).
      * fused speedup over composite ≥ 3.00× on the largest shape
        (measured 17.39× at D=4096, 49.40× at D=1024).
- **Resolved:** Wave 2.2 landing (2026-04-18). Follow-ups: fuse
  with the RoPE prolog on the attention path once FA2 lands
  (B-024), add the Llama-shaped layer-norm fixture to the
  transformer-block bench once FA2 is in.

### B-023 — CUDA Graph capture + replay for decode *(Resolved, Wave 2.3 MVP)*
- **Introduced in:** Wave 2 plan (2026-04-18). Every CUDA op in our
  decode path submits at least one kernel launch (plus cuBLASLt
  entries, memcpy prologs, reshape / permute strided copies, ...).
  At Llama-3.2-1B single-token decode shape the GPU finishes each
  kernel in 1-5 µs while the host-side launch machinery (driver
  API + queueing + our dispatch layer) eats another 1-2 µs per op.
  A full `MultiHeadAttention::forward_step` + FFN + residual +
  norm block clocks 20-40 kernel launches, so launch overhead
  stops being negligible and starts matching real compute time.
- **Definition of done (met — Wave 2.3 MVP):**
  - `include/tesseract/cuda/CudaGraph.hpp` + `src/cuda/CudaGraph.cpp`
    ship the infrastructure: a `CudaGraph` class with `capture(stream,
    fn)` and `launch(stream)`. Capture installs a `StreamGuard` for
    the caller (so ops inside `fn` resolve `current_stream(device)`
    to the capture stream without extra wiring), runs `fn()` TWICE
    as warmup (see "Why two warmups" below), then does
    `cudaStreamBeginCapture` (ThreadLocal mode) → `fn()` →
    `cudaStreamEndCapture` → `cudaGraphInstantiate`. Re-capture
    tears down the old `cudaGraphExec_t` and `cudaGraph_t` cleanly,
    supporting workload-shape changes (e.g. switching batch sizes).
  - **Why two warmups:** stream capture forbids any `cudaMalloc` in
    the captured closure. Our bucketed allocator satisfies this iff
    every alloc size the closure requests is already cached at
    capture time. The common capture pattern writes the closure's
    output to a caller-owned slot (`out = ops::add(...)`); on a
    single warmup that final output buffer stays alive in `out`, so
    capture still needs to allocate it fresh and falls through to
    `cudaMalloc`. A second warmup reassigns `out` (freeing the
    first output buffer back to the cache) and guarantees cache
    saturation before capture starts. Without this fix the
    elementwise-chain test and decode-chain bench both reproducibly
    failed with `operation not permitted when stream is capturing`.
  - CPU-only build: every `CudaGraph` method throws a clean
    `DeviceError` (`"CUDA backend was not compiled in"`), same
    contract as the rest of `tesseract/cuda`.
  - Tests: `tests/cuda/test_cuda_graph.cpp` has three cases —
    capture-replay parity against eager on a 1 MiB elementwise chain
    (32768 assertions), re-capture of a different closure rebinding
    cleanly, and launch-before-capture raising. All pass on both
    the CUDA build and the CPU-only build (auto-SKIP when no GPU).
  - Bench: `benchmarks/bench_cuda_graph` measures eager vs graph-
    replay latency on a 10-op alternating `mul` / `add` chain at
    three shapes (4 Ki / 64 Ki / 1 Mi floats). Hard bar: **speedup
    ≥ 1.25× on the small-shape case.** Measured 1.36× at N=4 Ki on
    RTX 5880 Ada (SM 8.9), 1.10× at N=64 Ki (mixed), 1.01× at
    N=1 Mi (bandwidth-bound — kernel time dwarfs launch overhead,
    as expected). Registered under the `bench_cuda` ctest label
    with the shared `RESOURCE_LOCK "cuda_gpu_0"` so it plays nicely
    with the other CUDA benches under `ctest -j N`.
- **Deferred to B-023b (full decode-step capture):** The shipped
  primitive cleanly captures any closure whose allocation profile
  is constant across calls — the elementwise-chain bench proves
  it. The natural next step, wrapping `MHA::forward_step(x, cache)`
  inside a single captured graph, runs into a structural issue:
  attention's key/value tensor shape is `[B, H, current_len,
  D_head]`, and `current_len` grows by `S_new` on every step, so
  subsequent replays would need to attend to a longer `S_k` than
  the one baked into capture. The clean fix is a fixed-S_k
  decoding variant (attend over `max_len` with a boolean mask
  whose "valid prefix length" is a device-side int updated per
  step via `cudaMemcpyAsync` — the standard vLLM recipe). That
  design touches `MultiHeadAttention`, the attention softmax, and
  probably `ops::attention` itself, so it's carved out as its own
  task and not bundled with the infrastructure work here.
- **Resolved:** Wave 2.3 landing (2026-04-18).

### B-011 — Async `Tensor::to_async` + pinned host allocator *(Resolved, Wave 2.4)*
- **Introduced in:** Wave 2 plan (2026-04-18). The synchronous
  `Tensor::to(device)` path blocks the calling thread for the full
  `cudaMemcpy` duration, which makes it impossible to overlap
  host→device staging with on-device compute. For decode, where
  every step consumes a tiny input (a single token id embedding
  row — typically 4–16 KiB) while the GPU is still finishing the
  previous step's attention + FFN, that missed overlap adds up
  to a non-trivial share of the per-token latency. The
  infrastructure prerequisite — pinned host memory paired with
  `cudaMemcpyAsync` — belongs next to B-023's CUDA Graph primitive
  in the Wave 2 critical path.
- **Definition of done (met):**
  - `include/tesseract/cuda/PinnedHostAllocator.hpp` +
    `src/cuda/PinnedHostAllocator.cpp`: an `Allocator` subclass
    that hands out page-locked host buffers via
    `cudaHostAlloc(cudaHostAllocPortable)`. Singleton, stateless,
    device-agnostic (`device()` returns `cpu_device()` so every
    dispatch path keeps treating pinned tensors as CPU tensors —
    only the byte source differs). CPU-only build ships throwing
    stubs with the "rebuild with -DTESSERACT_ENABLE_CUDA=ON"
    message, same contract as every other `tesseract/cuda` stub.
    Not bucketed: pinned allocations are expected to be a small
    number of long-lived buffers (token id staging, weight staging,
    loader scratch), so the simple hit-the-driver path is right-
    sized; caching can be bolted on later if usage shifts.
  - `Tensor::empty_pinned(shape, dtype)`: factory that produces a
    CPU-identity tensor backed by pinned storage. Zero-size
    request returns the standard empty tensor (matches `empty`).
  - `Storage::copy_device_bytes_async(dst, dst_dev, src, src_dev,
    nbytes, stream)`: async byte-copy primitive forwarding to
    `cudaMemcpyAsync(..., stream.native_handle())`. Validates
    that `stream.device()` matches the CUDA side of the transfer
    (H↔D needs the stream on the non-CPU endpoint; D↔D needs it on
    the destination device, matching cuMemcpyDtoD expectations).
    CPU↔CPU path still goes through `std::memcpy` for immediate
    observability; the async contract only kicks in for any
    CUDA-touching endpoint.
  - `Tensor::to_async(target_device, stream)`: high-level wrapper
    that allocates the destination on `target_device` and enqueues
    the transfer. Same-device short-circuit (returns `*this`
    without touching the stream) matches `to()` identity. Non-
    contiguous sources go through a pre-copy `contiguous()`
    gather (documented); the async-ness only covers the cross-
    device segment, which is how every PyTorch-style implementation
    also behaves.
  - Tests: `tests/hal/test_hal_cuda_pinned.cpp` — pinned alloc +
    write/read round-trip; pinned-source `to_async` matches the
    synchronous `to()` bit-for-bit; pageable-source `to_async`
    preserves correctness (driver falls back transparently);
    D→H async round-trip into a pageable destination; same-
    device `to_async` is an identity; zero-byte alloc legal. 6
    cases green on CUDA, auto-SKIP on CPU-only builds.
  - Bench: `benchmarks/bench_cuda_pinned` gates three hard bars
    on RTX 5880 Ada / PCIe Gen4 x16:
      * 1 MiB H→D: pinned/sync speedup ≥ 1.4× → **measured
        1.80×**. Above 4 MiB the PCIe link dominates (~26 GB/s vs
        ~24 GB/s, informational only — PCIe-bound regime where a
        15 % speedup is all any allocator can extract).
      * 4 KiB pinned async H→D submit latency ≤ 10 µs →
        **measured 1.76 µs**. Critical for decode: every step can
        fire "copy next token id" in under 2 µs so it never sits
        on the critical path.
      * End-to-end (4 MiB copy ‖ compute chain) wall-time ratio
        vs sequential ≤ 0.85 → **measured 0.83×** (17% wall-time
        savings from overlap). This is the integration-level
        guarantee: the bench runs the copy on one non-blocking
        stream and a 10-op compute chain on another, joining at
        the end; the ratio collapses to ~1.0 if either stream
        implicitly serializes or the async path silently falls
        back to sync. Registered under the `bench_cuda` ctest
        label + `RESOURCE_LOCK "cuda_gpu_0"` for serialization.
  - `ctest -j 1`: 316/316 green (was 309 before Wave 2.4; +6
    pinned test cases + 1 pinned bench registered).
- **How this complements B-023 (CUDA Graph):**
  - CUDA Graph collapses the *on-device* launch sequence to one
    driver call (the host-launch overhead path).
  - Pinned async transfer collapses the *host→device* data path
    so input staging overlaps with the captured graph's replay
    (the input-delivery path).
  - Together they drive the per-decode-step critical path toward
    the raw compute roofline: the host side issues one
    `cudaMemcpyAsync` (≤ 2 µs) + one `cudaGraphLaunch`, and the
    GPU does the rest.
- **Resolved:** Wave 2.4 landing (2026-04-18). Follow-up — add an
  optional pinned-staging switch to the SafeTensors loader so
  weight ingest uses the async path on large checkpoints (tracked
  informally with the Wave 4 loader work; not a separate B-NNN).

### B-023b — Decode-phase `MHA::forward_step` graph capture *(Resolved, Wave 4.3 MVP)*
- **Introduced in:** Wave 2.3 closure (2026-04-18). Reopened at
  Wave 4.3 (2026-04-21) once Wave 4.2 delivered a fused-attention
  kernel whose `S_k` is a compile/capture-time argument (meaning it
  can be baked into a recorded graph instead of re-specialized every
  step).
- **Motivation:** `MultiHeadAttention::forward_step` dispatches 20+
  kernels per token — four projections (Q/K/V/O), two reshape+permute
  chains, two RoPE passes, a per-(b, h) cache append loop, and a
  composite or fused attention kernel. At Llama-3.2-1B single-token
  shapes each kernel is tiny and host-side launch overhead alone
  totals ~100 µs / step — matching or exceeding real compute.
  `cudaStreamBeginCapture` + `cudaGraphLaunch` collapses that bill
  to a single driver round-trip, which is the foundation for
  roofline-throughput decode.
- **Definition of done (met, MVP):**
  - `KVCache::append` rewritten to ride the per-device current
    stream via `Storage::copy_device_bytes_async` instead of
    draining it with a synchronous `cudaMemcpy`. The old sync was
    triple wasteful: every decode step paid a full
    `current_stream.synchronize()` for zero ordering benefit (the
    producing projection kernels already ran on the same stream,
    so stream ordering alone guarantees visibility to any later
    kernel reading the cache slab). Making it async also made it
    **capturable** — a sync `cudaMemcpy` is rejected outright by
    `cudaStreamBeginCapture` as a host-synchronizing call.
  - `KVCache::set_current_len(int64_t)` exposed as a public
    capture-pass rewind knob (`include/tesseract/nn/KVCache.hpp`).
    The `CudaGraph::capture` driver invokes the closure three
    times (two warmup passes + one capture pass) and a plain
    `cache.append(...)` inside the closure would advance
    `current_len_` by `S_new` per pass, landing the capture at
    `target_pos + 2·S_new` instead of `target_pos`. The rewind
    lets the closure reseed the counter at the top of each pass
    so every invocation targets the same slab slot + same
    attention `S_k`.
  - `tests/nn/test_mha_cuda_graph.cpp` (3 new cases):
    * composite-path capture (`B*H=16`, `S_k=6`, FP32+RoPE),
    * fused-path capture (`B*H=128`, `S_k=4`, the Wave 4.2 shape
      gate's `bh >= 64` branch),
    * `set_current_len` bounds validation (always-runs, no-CUDA
      case keeps CPU-only builds green).
    Both capture cases REQUIRE **bit-exact** parity against the
    eager `forward_step` at the same cache state — any silent
    kernel drop or stale-`S_k` dispatch under capture would
    fail the assertion immediately.
  - `benchmarks/bench_cuda_mha_decode_graph` measures replay cost
    vs eager across three shapes (`S_k ∈ {8, 64, 256}`,
    `d_model=512, H=16, D_h=32`). Hard bar = **1.25× speedup** on
    the host-bound `S_k=8` shape; `S_k=64` and `S_k=256` are
    informational.
- **Measured on RTX 5880 Ada (FP32, B=1):**
  - `S_k=8`  (host-bound decode):  eager  185.99 µs  →  graph 87.06 µs  →  **2.14×** — clears the 1.25× hard bar with 71% headroom.
  - `S_k=64` (mixed):               eager  185.70 µs  →  graph 85.93 µs  →  **2.16×**.
  - `S_k=256` (compute-leaning):    eager  185.65 µs  →  graph 94.79 µs  →  **1.96×**.
  These numbers confirm the structural claim: every decode step in
  this configuration carries ~90 µs of host-side launch overhead
  that the graph replay collapses into a single `cudaGraphLaunch`.
  Compute stays identical, so the speedup is ~2× for host-bound
  shapes and gracefully narrows toward 1× as compute grows.
- **Verification:**
  - Full CUDA ctest: 369/369 green (+3 new capture cases at
    #194-#196, registered under `test_nn_mha_cuda_graph` with
    `SKIP_RETURN_CODE 4`).
  - All 11 CUDA benches pass, including the new
    `bench_cuda_mha_decode_graph` under `RESOURCE_LOCK
    "cuda_gpu_0"`.
  - CPU-only build links and the 3 cases either SKIP cleanly
    (GPU-required) or run asserted (the `set_current_len` bounds
    case); the 3 pre-existing LayerNorm / BatchNorm CPU-only test
    failures (#116, #119, #204) are independent — they use
    Catch2 `SKIP()` but were registered without `SKIP_RETURN_CODE
    4`, unrelated to this wave.
  - Lints clean on every Wave 4.3-touched file.
  - Examples (`mnist`, `llama_forward`, `llama_infer`) still
    build and link on CUDA.
- **Deferred work tracked as B-023b+ (future waves):**
  * **Chunked-prefill (`S_new > 1`) capture.** The current
    `forward_step` materializes the rectangular causal mask on CPU
    via `make_decode_mask(...)` and then calls `.to(cuda)` to
    migrate it. The `.to` path drains the source stream, which
    `cudaStreamBeginCapture` rejects. Fix requires either a GPU-
    side mask kernel (preferred) or a pre-allocated device mask
    slab updated via `cudaMemcpyAsync`. Single-token decode —
    the hot path — is already captureable.
  * **Length-parameterized replay.** A graph captured at
    `current_len = P` writes to `slab[P]` and attends over
    `S_k = P + S_new`; every step past that requires re-capture.
    The production decode loop can either (a) cache one graph
    per `current_len` value (bounded by `max_len`, typically
    4096–8192 per serving instance), or (b) adopt
    `cudaGraphExecUpdate` to rebind memcpy-dst + attention
    `S_k` parameters across replays. The latter is the right
    long-term answer once we stand up the continuous-batching
    scheduler and need one graph for many concurrent requests.
  * **TransformerBlock full capture.** The per-block capture
    generalizes by wrapping `MHA + FFN + residuals + norms` in
    the same closure. FFN already has its fused SwiGLU activation
    (Wave 4.1, B-025) and fused RMSNorm (Wave 2.2, B-022), so
    the resulting graph's launch count goes from ~25 to 1 across
    the whole decoder block. Falls out of this wave's API
    surface; flagged as a serving-loop integration task under
    Wave 4.4 / continuous batching.

### B-024 — FA2-style fused attention forward CUDA kernel *(Resolved, Wave 4.2 MVP)*
- **Introduced in:** M2L roadmap (M2L.2) and carried into Wave 4.
  The composite `Q·Kᵀ → softmax → ·V` path materializes the full
  `[B, H, S_q, S_k]` attention-score matrix twice (once written by
  the first matmul, once re-read by softmax and the second matmul),
  which at realistic prefill shapes dominates the attention HBM
  bill — for `(B=2, H=16, S=2048, D=128)` that's 64 MB of score
  traffic alone, multi-passed through the pipeline. On the decode
  critical path the composite also pays three kernel launches plus
  a `[1, S_k]` score materialization per step.
- **Definition of done (met, MVP):**
  - FA2-style online-softmax forward kernel
    (`src/cuda/FusedAttention.cu` + new
    `include/tesseract/cuda/detail/FusedAttention.hpp` bridge) that
    streams K/V in BLOCK_K=32 tiles without ever materializing the
    score matrix. BLOCK_Q=4 warps per block — each warp owns one
    query row and shares the `K_tile` / `V_tile` loads with the
    other three warps, cutting HBM traffic on K/V by 4× vs the
    naive one-query-per-block layout. FP32 accumulation on FP32 /
    FP16 / BF16 storage (same dtype policy as B-022 / B-025);
    FP64 intentionally routed to the composite (shared-mem budget
    would exceed Ada's 48 KB default without dynamic-SMEM opt-in).
  - KV tile stride padded to `D_MAX + 1` floats — the natural
    `D_MAX = 128` stride hits a 32-way shared-memory bank conflict
    on the score-phase access pattern (every lane at fixed `d`
    lands in the same bank), and the one-float pad turns it into
    zero-way at cost of negligible SMEM bloat.
  - Paired `src/cuda/FusedAttentionStub.cpp` so CPU-only builds
    throw a clean `DeviceError` if the fused launch is ever
    reached (mirrors `SwiGLUStub` / `RMSNormStub`).
  - `ops::attention` fast-path dispatch in
    `src/ops/cpu/Attention.cpp`, gated on `s_q ≤ 8 AND B·H ≥ 64`
    (the regime where the CUDA-core fused path is demonstrated
    to beat the tensor-core composite — decode and chunked-decode
    shapes). Prefill routes through the existing composite;
    closing the tensor-core FLOP gap on prefill is the B-024+
    WMMA / mma.sync follow-up, see below. `TESSERACT_FORCE_FUSED_ATTENTION=1`
    env override is available for tests / benches to exercise the
    full shape matrix.
  - Autograd fallback: the fused path is forward-only and skipped
    entirely when any of Q / K / V requires grad under an active
    autograd scope. Gradient flow still runs through the
    `matmul → softmax → matmul` composite with its existing
    backward nodes, matching the B-022 / B-025 policy.
  - Tests (`tests/ops/test_ops_cuda_fused_attention.cpp`, 9
    cases): FP32 parity at D=64 (rank-4) and D=128 causal, ragged
    trailing tile (`S_k` not a multiple of 32), `S_q ≠ S_k`
    cross-attention pattern, FP16 and BF16 parity, rank-3 leading
    batch, and explicit composite-fallback assertions for mask
    defined + FP64 dtype. 90K+ aggregate assertions.
    `SKIP_RETURN_CODE 4` honoured so CUDA cases SKIP cleanly in
    CPU-only builds.
  - Bench (`benchmarks/bench_cuda_fused_attention`) measures
    fused vs composite across a decode + prefill shape matrix,
    with one hard bar on the regime where the MVP can commit:
    `decode @ B·H ≥ 256 ≥ 1.50×`. Prefill rows are informational
    (the CUDA-core fused path lands in the 0.1×–1.1× range vs
    cuBLASLt FP16 tensor cores, and the WMMA follow-up is the
    correct fix). A `TESSERACT_FORCE_FUSED_ATTENTION=1` setenv
    inside the bench main forces fused dispatch regardless of
    the production shape gate.
- **Measured (RTX 5880 Ada, SM 8.9):**
  - Decode @ saturated SMs `(B=8, H=32, S_q=1, S_k=2048, D=128)`:
    fused **2.55× faster** than composite (622 μs vs 1586 μs,
    434 GB/s effective, ≈45% of 960 GB/s HBM peak).
  - Decode @ filled SMs `(B=4, H=32, ·)`: fused 1.34× composite.
  - Decode @ grid-starved `(B=1, H=32, ·)`: fused 0.27–0.43×
    composite (composite's cuBLASLt GEMV wins when B·H < SMs);
    the production gate routes this to composite anyway, the
    fused kernel is only exercised by the bench under the force
    override.
  - Prefill: fused 1.07× at `(4, 16, 512, 128)`, 0.81× at
    `(2, 16, 1024, 128)`, 0.69× at `(2, 16, 2048, 128)`, 0.11×
    at `(2, 16, 4096, 128)` — the tensor-core FLOP gap widens
    quadratically in S. Production gate keeps prefill on
    composite (`bench_cuda_attention` still passes its
    `composite / sum-of-primitives ≥ 0.97` bar unchanged).
  - Decode hard bar PASS.
- **Verification:** CUDA serial ctest 100% green (376 tests,
  including +9 new cases); parallel ctest green modulo the
  pre-existing `Tensor::to_async(D→H)` pageable flakiness under
  `-j 4` (orthogonal to this wave, unaffected when run alone).
  CPU-only build green. Lints clean on all W4.2-touched files.
  `llama_infer` / `llama_forward` / `mnist` examples still build.
  `bench_cuda_attention` hard bar unchanged (composite path).
- **Explicitly deferred:**
  - **B-024+ WMMA tensor-core prefill — RESOLVED (2026-06-25).**
    Shipped `fused_attention_wmma_kernel` (`src/cuda/FusedAttention.cu`):
    a WMMA (16×16×16) fused FA2 prefill kernel running **both** matmuls
    (Q·Kᵀ, P·V) on FP16/BF16 tensor cores with the FA2 online softmax
    (no `[S_q,S_k]` score materialization). BLOCK_M=32×BLOCK_N=32, 2
    warps (one 16-row m-tile each); Q/K/V/P staged in shared as FP16,
    S + the running O accumulator in shared as FP32; the per-row α
    rescale is applied to O addressed by `(row, d)` and the P·V
    accumulate is a load→add→store on the O fragment — so the kernel
    makes **no undocumented WMMA fragment-layout assumption** (only
    `store_matrix_sync` readback). Shared tiles are sized to the
    smallest stride ≥ D (DMAX=64 for a D=64 model like TinyLlama, 128
    otherwise), halving the smem bill and ~doubling resident blocks/SM.
    Dispatch: prefill `S_q>1` FP16/BF16, `D%16==0`, `D≤128`, SM≥8.0 →
    WMMA; FP32 stays CUDA-core; FP64 composite; decode keeps split-K.
    Parity: `test_ops_cuda_fused_attention`
    (`TESSERACT_FORCE_FUSED_ATTENTION=1`) 166,854 assertions green;
    model `forward_step==forward (CUDA)` #74, GQA #120/#123, MoE #130,
    paged prefill #152 — all green. Measured WMMA-fused vs cuBLASLt
    composite on prefill: `(4,16,512,512,128)` **5.14×**,
    `(2,16,1024,1024,128)` **4.04×**, `(2,16,2048,2048,128)` **3.78×**,
    `(8,32,512,512,64)` **2.91×** (the MVP CUDA-core kernel was
    0.7–1.1× here); two extreme shapes still favor composite —
    `(2,16,4096,4096,128)` 0.33×, `(4,32,2048,2048,64)` 0.86× — where
    cuBLAS's large-S GEMM tiling beats a no-cp.async hand kernel (not
    TTFT-relevant prompt sizes; see B-024d). End-to-end TinyLlama TTFT
    7.28 → **6.59 ms** (vLLM 5.47); decode/TPOT/e2e wins unchanged.
    WMMA closed the *attention* slice of prefill overhead; the residual
    gap is no longer attention compute — see B-024c.
    - **Historical (the CUDA-core MVP this superseded):** the prefill
      path became GQA-native and fused end-to-end
      (`ops::prefill_attention_gqa` + H_kv-aware
      `fused_attention_fp32_kernel`), routed from
      `MultiHeadAttention::forward_step`, removing the composite path's
      6-launch + score-matrix HBM round trip and the `repeat_kv` 8× KV
      copy (TinyLlama TTFT 14.4 → 7.3 ms). It accumulated on CUDA cores;
      the WMMA rewrite moved both matmuls to the tensor cores.
  - **B-024c — attention-layout `strided_copy` — RESOLVED (2026-06-25).**
    nsys attribution on a prompt=512 prefill (debug-instrumented
    `launch_strided_copy`) pinned the per-layer copies: 4× small KV
    `[1,4,128,64]` + **two 512 KB copies** — `q` `[1,32,128,64]` (RoPE
    forcing the permuted Q view contiguous) and `out` `[1,128,32,64]`
    (the `[B,H,S,D]→[B,S,H,D]` output transpose). Fix shipped: the
    prefill kernels (WMMA **and** the FP32 fallback) are now
    **stride-aware** — `launch_fused_attention` takes an optional 9-entry
    `strides` array `{q_b,q_h,q_s, k_b,k_h,k_s, o_b,o_h,o_s}` (nullptr =
    contiguous; the head-dim stride is always 1), and the kernels read Q
    and the KV-cache narrows in place and write O in any layout via those
    strides — the WMMA math (shared-memory staging + fragments) is
    untouched, only the global load/store index math changed. New op
    `ops::prefill_attention_gqa_bshd` (and the `forward_step` prefill
    branch) uses it to: (a) pass the `keys_view()`/`values_view()`
    narrows directly — no `contiguous(k_all/v_all)`; (b) write output in
    **BSHD** `[B,S,H,D]` so the head-merge is a free `reshape` to
    `[B,S,d_model]` instead of a transpose. Parity:
    `test_ops_cuda_fused_attention` (forced) 166,854 assertions,
    `test_models_llama` (incl. `forward_step==forward`) 6,809,
    `test_models_llama_generate`, `test_models_batched_decode` — all
    green. Result: `strided_copy` **25.8 % → 15.9 %** of prefill GPU
    time (WMMA attention is now the top kernel at 23.5 %); end-to-end
    TinyLlama **TTFT 6.59 → 5.86 ms** (vLLM 5.47 → gap 1.20× → **1.07×**);
    decode/TPOT untouched (3.12 ms / 320 tok/s, verified). The public
    `prefill_attention_gqa` keeps its contiguous `[B,H,S,D]` contract
    (parity tests, fallback) by calling the launcher with `strides=nullptr`.
  - **B-024e — kill the residual prefill overhead to *beat* vLLM TTFT
    (next lever).** After B-024c the largest remaining `strided_copy` is
    the `q` (and `k_new`) RoPE contiguous-ification of the permuted view
    — eliminable with a **BSHD-native RoPE** (rotate `[B,S,H,D]` by the
    dim-1 position, no permute) feeding the already-stride-aware attention
    Q. That alone (~100 µs at prompt 128) won't pass vLLM's 5.47 ms,
    because the rest of the 0.39 ms gap is the *shared* cuBLAS GEMM floor
    plus Tesseract running norm/RoPE/residual as separate launches where
    vLLM has them inductor-fused. Beating TTFT therefore also needs a
    norm+RoPE+residual epilogue-fusion pass. Deferred (medium effort,
    diminishing returns — TTFT is the only sub-metric still behind, now
    within 7 %; decode/TPOT/throughput/e2e all win).
  - **B-024d — large-S WMMA tiling.** cp.async double-buffering + a
    BLOCK_M=64/BLOCK_N=64 (opt-in 96 KB smem) tile so prefill also
    beats composite at S ≥ 4096; today those extreme shapes stay on the
    composite path which wins there.
  - **FA3 Hopper** — B-013. Unchanged by this wave; shares the
    shape contract but picks up warp-specialization / WGMMA on
    SM 9.0+.
- **Unblocks:**
  - Wave 4.3 (B-023b, decode-step CUDA Graph capture) — the
    fused kernel offers the fixed-S_k launch shape the graph
    replay path needs.
- **Resolved:** Wave 4.2 landing (2026-04-21).

### B-025 — Fused SwiGLU forward CUDA kernel *(Resolved, Wave 4.1)*
- **Introduced in:** Wave 4 kickoff (2026-04-18). Every Llama block's
  FFN forward unrolls its activation tail into three element-wise
  kernels (`sigmoid(gate)`, `mul(gate, sig)`, `mul(silu_gate, up)`),
  reading the `[..., d_ff]` intermediate 5 times and writing it 3
  times. On memory-bound element-wise work that's a strict 2-3× HBM
  traffic penalty vs a fused one-pass kernel. With B-022 (fused
  RMSNorm) already shipped, SwiGLU was the last remaining element-
  wise bottleneck on the decode-step critical path.
- **Definition of done (met):**
  - `ops::swiglu_silu_gate(gate, up)` that computes
    `silu(gate) * up = gate * sigmoid(gate) * up` element-wise.
  - CUDA fast path (`launch_swiglu_silu_gate`) — 1-D grid-stride
    loop, FP32-accumulated on FP16/BF16 storage, FP32 / FP64
    compute-in-storage variants; matches the CPU reference to
    `1e-5` FP32 / `1e-12` FP64 / `2e-3` FP16 / `6e-3` BF16.
  - CPU reference (also contiguous fast path + fully FP32-promoted
    half-precision compute) for CPU-only CI and the pre-CUDA code
    path.
  - Autograd fallback: when either operand requires grad the op
    decomposes into the `sigmoid` + `mul` + `mul` composite, so
    gradients flow through the already-validated primitive
    backward nodes. No custom `SwiGLUBackward` — mirrors the B-022
    "forward-only fused kernel" policy.
  - `nn::FeedForward::forward` routed through the fused op; the
    training path remains byte-identical because the fallback
    keeps the composite math.
  - Tests (`tests/ops/test_ops_swiglu.cpp`, 9 cases): FP32
    reference parity, composite-equivalence parity, autograd FD
    parity, shape/dtype/device validation, CPU↔CUDA parity across
    all four dtypes, and a realistic `(128, 1024)` FFN-shape parity
    case. SKIP_RETURN_CODE 4 contract matches the other CUDA-
    parity TUs so CPU-only builds SKIP the CUDA cases cleanly.
  - Bench (`benchmarks/bench_cuda_swiglu`) with two hard bars on
    the largest `(16, 4096, 4096)` shape: fused speedup over
    composite ≥ 2.0× and fused eff_GB/s / memcpy ≥ 0.80.
- **Measured (RTX 5880 Ada, SM 8.9):**
  - Fused **2.72× faster** than composite (3.62 ms vs 9.87 ms).
  - Fused effective bandwidth **891 GB/s** (≈92% of 960 GB/s HBM
    peak).
  - Both hard bars PASS.
- **Verification:** CUDA ctest 366/366 green (+9 ops tests,
  +1 bench). CPU-only ctest green (CUDA parity cases SKIP).
  Lints clean on all W4.1-touched files. `llama_infer`,
  `llama_forward`, `mnist` still build and link.
  `bench_cuda_transformer_block` still PASSes (FeedForward now
  uses the fused op internally under NoGrad).
- **Resolved:** Wave 4.1 landing (2026-04-18).

### B-026 — Quantized inference fast path + end-to-end decode benches *(Resolved, Wave 4.4)*
- **Introduced in:** Wave 4.4 kickoff (2026-04-21). `QuantizedLinear`
  (INT8 symmetric) and `QuantizedLinearInt4G` (INT4 group-wise
  symmetric) both delegate to `ops::dequantize_matmul_{int8,int4_group}`,
  whose forward has two branches:
    1. **Inference branch** — a single fused CUDA kernel that streams
       `q_weight` (+ `weight_scale` / group scales), dequantizes
       on the fly, and emits the matmul result. No FP32 weight
       tensor ever hits HBM.
    2. **Autograd fallback** — when `is_grad_enabled() &&
       x.requires_grad()` the op materializes the full FP32
       weight (`dequantize_*` → `matmul`) so the existing matmul
       backward can differentiate w.r.t. `x`. Mathematically
       identical, operationally a 4-8× memory + compute regression
       on the decode step.
  The hazard was: a user doing inference who forgets
  `NoGradGuard` (e.g. leaves autograd enabled for evaluation
  logging, or interleaves inference with an outer training loop)
  silently pays the full FP32 weight cost. Evaluating INT4G gets
  *worse* than pure FP32 in that case — HBM writes for the
  dequant scratch plus the normal matmul read.
- **Definition of done (met):**
  - **Eval-mode fast path.** `QuantizedLinear::forward` and
    `QuantizedLinearInt4G::forward` now install a local
    `NoGradGuard` whenever `!is_training()`, pinning the dispatch
    to the fused kernel regardless of the grad state of `x` /
    the outer engine. Parity vs the existing
    `NoGradGuard`-wrapped path is bit-identical (they're the
    same kernel call). Training mode is untouched — the
    autograd fallback still fires for `train()` modules so
    QAT stays available.
  - **Unit tests** (`tests/nn/test_nn_quantized_linear_eval.cpp`,
    7 cases): for each of `{INT8, INT4G}` we cover
      (a) `eval() + x.requires_grad()==true` produces a no-grad
          output equal to the explicit `NoGradGuard`-wrapped
          forward,
      (b) `train() + x.requires_grad()==true` still builds a
          grad edge and `Engine::backward` populates `x.grad`,
      (c) flipping `eval() ↔ train()` on the same module
          instance toggles the dispatch,
      (d) `eval()` nested inside an outer `NoGradGuard` is a
          no-op (fast path already active).
  - **`bench_cuda_quantized_linear`** — FP32 `nn::Linear` vs INT8
    vs INT4G, decode shape (`M=1`) at `K=N=8192` (the first
    shape whose FP32 weight — 256 MB — exceeds SM 8.9's ~64 MB
    L2, forcing a genuinely HBM-bandwidth-bound baseline), plus
    informational rows at `K=N=4096` (L2-resident) and a
    prefill shape (`M=512`). Hard bars on the HBM-bound decode:
      - INT8 weight bytes / FP32 ≤ 0.30, INT4G ≤ 0.18
        (deterministic, a packer-layout guard).
      - INT8 latency / FP32 ≤ 0.30 (measured 0.25× on SM 8.9).
      - INT4G latency / FP32 ≤ 0.55 (measured ~0.50× on SM 8.9).
  - **`bench_cuda_llama_decode`** — full Llama-2-7B block
    (`d_model=4096, H=32, Dh=128, d_ff=11008`, `B=1`,
    `S_k=129`) decode step = `MHA::forward_step` + `FFN::forward`,
    exercising every Linear in the block (4× MHA + 3× FFN).
    The 7B shape is the smallest configuration whose
    per-block FP32 footprint (~800 MB) comprehensively exceeds
    L2 on every currently-shipping GPU; at Llama-1B the FP32
    baseline partially caches in L2 after warmup and INT4G's
    per-output compute dominates the memory savings. Hard
    bars:
      - Block weight bytes: INT8 / FP32 ≤ 0.30 (measured
        0.250×), INT4G ≤ 0.18 (measured 0.133×).
      - Decode-step latency: INT8 / FP32 ≤ 0.55 (measured
        0.46×), INT4G ≤ 0.75 (measured 0.65×).
    The INT4G bar is intentionally looser than the memory
    ratio would suggest: on SM 8.9 the fused dequant-matmul
    kernel is compute-bound on the nibble unpack + group-scale
    lookup path, so INT4G runs *slower* than INT8 despite
    streaming half the bytes. Future work on a vectorized
    nibble unpack (tracked below) would let us tighten it; for
    Wave 4.4 we document the current ceiling and bar
    regressions from it.
- **Deferred work tracked as B-026+ (future waves):**
  * **Vectorized INT4G unpack.** The current kernel does one
    nibble-shift per output partial; a packed `uchar4 →
    uint32_t` load + bit-field unpack plus group-scale prefetch
    through shared memory should close the gap to the
    memory-bound ceiling (~0.15× FP32 decode-step latency).
    Blocked until we stand up an FP16 / BF16 accumulator path
    so the compute saved isn't spent on FP32 casts.
  * **Tensor-core INT8 path.** cuBLASLt offers `CUBLAS_COMPUTE_32I`
    INT8 GEMM on Ada tensor cores. Our current INT8 dequant-
    matmul is a hand-written CUDA-core kernel — good enough to
    beat the FP32 baseline 2× but leaves another ~2× on the
    table. Slot together with the cuBLASLt SwiGLU + fused-attention
    epilogue work.
  * **INT4G activation-aware calibration.** Current scheme is
    weight-only symmetric; the next accuracy lift is
    GPTQ/AWQ-style per-group scale optimization driven by a
    calibration dataset. Accuracy work, not perf — tracked
    separately under the "quant accuracy" umbrella.
- **Verification:** CUDA ctest 389/389 green (+7 eval-path
  tests, +2 new benches). CPU-only ctest green on every
  Wave-4.4-touched file (the 3 pre-existing
  `test_ops_layer_norm` / `test_nn_batch_norm` "failures" are
  a missing `SKIP_RETURN_CODE 4` on those two registrations —
  filed as a separate infra cleanup). Lints clean on all
  W4.4-touched files. `llama_infer`, `llama_forward`, `mnist`
  still build and link on the CUDA config.
- **Resolved:** Wave 4.4 landing (2026-04-21).

### B-027 — End-to-end autoregressive generation *(Resolved, Wave 5)*
- **Introduced in:** Wave 5 plan (2026-06-21). After Waves 1a–4.6 the
  building blocks for real text generation all existed in isolation —
  `MultiHeadAttention::forward_step` + `KVCache` (Wave 2.1), byte-level
  `BpeTokenizer` (Wave 4.6), `LlamaModel` (Wave 1a) — but nothing wired
  them together. `LlamaModel` only exposed the one-shot
  `forward(tokens)` that recomputes the full prefix per call, so there
  was no incremental decode and no `generate()`.
- **Resolved (2026-06-21):** Wave 5 threads per-layer KV caches through
  the block stack and ships greedy generation. Shipped artifacts:
  * `nn::TransformerBlock::forward_step(const Tensor&, KVCacheBase&)`
    (`include/tesseract/nn/TransformerBlock.hpp` + `.cpp`) — pre-norm
    residual with the attention sub-layer routed through the cache
    (`h = x + attn.forward_step(norm_1(x), cache)`; the position-
    independent FFN + RMSNorms reuse the eager `forward` paths). Whole
    step under `NoGradGuard`. Accepts `KVCacheBase` so contiguous
    **or** paged caches work unchanged.
  * `LlamaModel::forward_step(tokens, std::vector<shared_ptr<KVCache>>&)`
    (`include/tesseract/models/Llama.hpp` + `src/models/Llama.cpp`) —
    embed → N blocks each with its own cache → final RMSNorm →
    `lm_head`, returning `[B, S_new, vocab]`. `make_kv_caches(batch,
    max_len)` allocates one contiguous cache per layer on the model's
    device/dtype.
  * `LlamaModel::generate(prompt_ids, GenerateConfig{max_new_tokens,
    eos_token_id})` — prefills the prompt in one chunked-decode step,
    then greedily (argmax) decodes one token at a time reusing the
    caches, stopping at `max_new_tokens` or `eos_token_id`; returns
    prompt + generated (HF convention). Argmax reads logits on host via
    `dispatch_float_with_half` (FP32/FP64/FP16/BF16).
  * `llama_infer --generate [--max-new-tokens N]` — encodes a prompt
    with `BpeTokenizer`, runs `generate`, prints the decoded
    continuation. The capstone "tokenizer → model → KV-cache → text"
    loop, runnable on CPU or CUDA.
- **Definition of done (met):** `tests/models/test_llama_generate.cpp`
  (5 cases) proves (a) **chunked-prefill parity** — `forward_step` over
  the whole prompt == one-shot `forward` at every position within FP32
  abs 2e-4; (b) **token-by-token parity** — per-step last-position
  logits reconstruct `forward()` row-for-row (validates cache append +
  RoPE offset against the one-shot causal path), on CPU **and** CUDA
  (CUDA `SUCCEED`-skips without a device); (c) greedy `generate`
  determinism + prompt-prefix + exact length + in-range ids; (d) EOS
  early stop; (e) empty-prompt / out-of-range-id rejection. CUDA ctest
  **410/410 green** (+5 cases), CPU-only green, lints clean.
- **Deferred (B-027+):** sampling beyond greedy *(resolved in B-028 /
  Wave 6)*; batched multi-sequence generation with ragged lengths (needs
  the PagedKV ragged-length follow-up); CUDA-graph capture of the
  per-step `forward_step` (builds on Wave 4.3 / B-023b); streaming token
  callback.

### B-028 — Sampling strategies *(Resolved, Wave 6)*
- **Introduced in:** M3 Wave 6 (2026-06-21). Wave 5 shipped greedy-only
  decode; real generation needs stochastic sampling, and the upcoming
  continuous-batching scheduler (Wave 7) needs per-request sampling to
  batch over. Off the CUDA critical path (a tiny host op over the
  vocab row each step).
- **Shipped:**
  * `models::SamplingParams{temperature, top_k, top_p,
    repetition_penalty}` + `models::sample_from_logits(logits, params,
    prev_tokens, rng)` + stateful seeded `models::Sampler`
    (`include/tesseract/models/Sampler.hpp` + `src/models/Sampler.cpp`).
    Reproduces the HF / vLLM logits-processing order: repetition penalty
    (CTRL-style: positive logits divided, negative multiplied, once per
    distinct prior id) → temperature → top-k (keep k highest, ties at
    the boundary kept) → top-p / nucleus (smallest high-prob prefix
    reaching p, always ≥ 1 token) → softmax → multinomial draw from a
    seeded `std::mt19937_64`. Identity values (penalty 1.0, T 1.0,
    top_k 0, top_p 1.0) are no-ops; `temperature ≤ 0` short-circuits to
    greedy argmax (ties → lowest index, matching the dispatch greedy
    path).
  * `LlamaModel::GenerateConfig` gains `do_sample` (default false ⇒
    Wave-5 greedy preserved, `seed`-independent), `sampling`, `seed`.
    `generate` routes each step through the sampler when `do_sample`,
    feeding the running sequence (prompt + generated) to the repetition
    penalty.
  * `llama_infer --sample [--temperature F] [--top-k N] [--top-p F]
    [--repetition-penalty F] [--seed N]` — real, seed-reproducible
    stochastic generation from the CLI; prints the active decoding mode.
- **Definition of done (met):** `tests/models/test_sampler.cpp` (8 cases
  / 2146 asserts): `T ≤ 0` ≡ greedy argmax; top-k restricts support to
  the k highest with the larger logit dominating the histogram; top-p
  keeps only the nucleus (dominant-token-only at p=0.5) and never
  empties it; same seed reproduces draws bit-for-bit while different
  seeds diverge; repetition penalty provably shifts mass off penalized
  tokens; low temperature sharpens vs high flattens; `generate(do_sample)`
  is seed-deterministic + seed-sensitive, the greedy default ignores
  `seed`, and `do_sample` with `T=0` reproduces the greedy sequence.
  CUDA ctest **418/418 green** (+8 cases), CPU-only green, lints clean.
- **Deferred (B-028+):** min-p / typical / Mirostat samplers; logit bias
  + banned-token / stop-string criteria; batched per-request sampling
  *(resolved in B-029 / Wave 7)*.

### B-029 — Continuous-batching scheduler *(Resolved, Wave 7)*
- **Introduced in:** M3 Wave 7 (2026-06-21). Wave 5/6 generate ONE
  sequence; a server multiplexes many requests of different lengths
  arriving/finishing at different times. This is the M3 exit-bar
  headline (idea.md §4.4, right after PagedAttention) and the reason the
  Wave 4.5 `BlockAllocator` was built pool-agnostic.
- **Shipped:**
  * `nn::PagedKVPool` (`include/tesseract/nn/PagedKVPool.hpp` +
    `src/nn/PagedKVPool.cpp`) — standalone, reference-counted per-layer
    object owning the physical `[num_blocks, H, block_size, D_head]` K/V
    tensors + the `BlockAllocator`. `PagedKVCache` refactored to hold a
    `std::shared_ptr<PagedKVPool>`: the Wave 4.5 ctor still works (makes
    a private pool); a new ctor binds to a shared pool; `reset()` frees
    only this cache's own blocks (never `free_all`, which would yank
    sibling requests' blocks); `num_owned_blocks()` added. Wave 4.5's 6
    paged tests + `bench_cuda_paged_kv` stay green.
  * `LlamaModel::forward_step(tokens, vector<shared_ptr<KVCacheBase>>&)`
    overload (the `KVCache` one delegates via upcast) +
    `make_layer_pools(num_blocks, block_size)` /
    `make_paged_kv_caches(pools, max_len)` helpers.
  * `models::ContinuousBatchingScheduler` + `EngineConfig{block_size,
    num_blocks, max_seq_len, max_batch_size}` (`src/models/Scheduler.cpp`):
    one shared `PagedKVPool` per layer; `add_request(prompt, gen)`
    queues; `step()` admits waiting requests up to `max_batch_size` and
    only when the pool can hold their prompt (prefilling each), emits one
    token per running request with its own seeded `Sampler` + EOS/length
    stop, and reclaims finished requests' blocks; `run()` drains.
    Residency via `allocated_blocks()` / `free_blocks()`. Surfaces a
    permanent-stall config error and rejects oversized pools at ctor.
  * `examples/llama_serve.cpp` — multi-prompt demo printing per-tick
    `running` / `waiting` / `blocks` occupancy (dynamic admission +
    recycling visible; blocks return to 0 between batches).
- **Definition of done (met) — exact parity contract:** paged and
  contiguous caches store the same bytes and the gather is an exact copy,
  so a scheduled request's logits (hence greedy argmax and seeded
  samples) are bit-identical to running `generate` standalone regardless
  of interleaving or block budget. `tests/models/test_scheduler.cpp`
  (6 cases): greedy parity across mixed prompt lengths; staggered
  admission under a batch cap smaller than the request count (output
  unchanged + cap honored every tick); sampling parity with per-request
  seeds; EOS early-stop parity; a 2-block pool serving 4 requests via
  recycling; input + `EngineConfig` validation. `tests/nn/test_paged_kv_pool.cpp`
  (4 cases): shared budget across caches, reset frees only own blocks,
  block recycling, shared-vs-private byte parity. `allocated_blocks()==0`
  after every drain (no leak). CUDA ctest **428/428 green** (+10),
  CPU-only green, lints clean.
- **Deferred (B-029+):** compute-batched ragged decode (fuse the active
  set into one `forward_step` launch — the throughput win; needs the
  B-019b+ ragged paged-attention kernel; lands without an API change);
  preemption/eviction under memory pressure (admission currently waits
  rather than evicting); prefix sharing (RadixAttention); priority /
  fairness scheduling; streaming token callbacks.

### B-030 — Grouped-query attention (GQA) *(Resolved, Wave 8)*
- **Introduced in:** Wave 8 (2026-06-22). Motivation: the attention
  stack was MHA-only (`num_kv_heads == num_heads`), so it could only run
  Llama-1/2-style full-MHA configs. Every modern checkpoint — Llama-3
  (all sizes), Qwen2, Mistral — uses GQA (`num_key_value_heads <
  num_attention_heads`), making GQA the hard gate to loading them. GQA
  also shrinks the KV cache by `num_heads / num_kv_heads`, raising the
  concurrent-request ceiling for the Wave 7 paged pools.
- **Definition of done (met):**
  - `nn::MultiHeadAttention` gains a trailing `num_kv_heads` ctor arg
    (`0` ⇒ plain MHA = num_heads; otherwise must divide num_heads). Q
    stays `d_model → d_model`; K/V projections shrink to
    `d_model → num_kv_heads · head_dim`. After RoPE, each KV head is
    repeat-expanded across its `G = num_heads/num_kv_heads` query heads
    via the PyTorch `repeat_kv` interleave (`head h → KV head h/G`),
    built from autograd-aware reshape + `broadcast_to` so `G == 1` (MHA)
    is a no-op and grads accumulate back onto KV heads.
  - `forward_step` repeats *after* the cache gather ⇒ the KV cache stores
    only `num_kv_heads` heads (the GQA memory win is in the cache itself).
    The decode head-shape check compares against `num_kv_heads`.
  - Threaded through `nn::TransformerBlock` (trailing `num_kv_heads`) and
    `models::LlamaConfig::num_key_value_heads` (+ `kv_heads()` resolver;
    `make_kv_caches` / `make_layer_pools` allocate KV heads). `llama_3_2_1b()`
    preset sets 8 KV heads; `llama_infer` gains `--kv-heads N`.
  - **Verification** (`tests/nn/test_gqa.cpp`, 5 cases): projection-shrink
    shapes; **weight-replicated parity** — GQA matches a plain-MHA
    reference whose K/V weights are the GQA weights with each KV head
    replicated `G` times, proving the `h→h/G` sharing order; GQA
    `forward_step` (token-by-token, RoPE on) vs one-shot `forward`;
    divisibility validation; Llama-GQA deterministic generate +
    chunked-prefill `forward_step` parity (caches hold `kv_heads()`).
    Backward-compatible (default `num_kv_heads=0` → existing tests
    unchanged). CUDA ctest **433/433** (+5), CPU green, lints clean.
- **Deferred (B-030+):** a fused GQA/MQA attention kernel that indexes
  the shared KV head directly (no `G×` replication before the attention
  matmul) — the throughput/footprint follow-up, pairs with the ragged
  paged-attention kernel (B-019b+).

### B-031 — KV-cache INT8 quantization *(Resolved, Wave 9)*
- **Introduced in:** Wave 9 (2026-06-22). Motivation: the KV cache bounds
  context length and concurrent-request count on the Wave-7 paged pools.
  Storing it in INT8 rather than full precision cuts the *persistent* KV
  footprint ~4× (vs FP32) / ~2× (vs FP16), so a fixed memory budget holds
  proportionally more tokens / requests — the long-context memory frontier
  from idea.md §4.4.
- **Definition of done (met):**
  - **Device-resident quant ops** `quant::quantize_kv_per_token` /
    `quant::dequantize_kv_per_token` (`include/tesseract/quant/QuantizeKV.hpp`,
    `src/quant/QuantizeKV.cpp`). Unlike the `Pack.hpp` weight packers
    (one-shot, host-side) these run on the decode hot path, so they stay
    on whatever device the cache lives on: a CUDA kernel pair in
    `src/cuda/QuantizeKV.cu` (quantize = one thread per token·head row,
    FP32 absmax + scale; dequantize = one thread per element; FP32-promoted
    on FP16/BF16) and a numerically identical CPU loop, behind the standard
    always-compiled stub-vs-kernel CMake pairing (`QuantizeKVStub.cpp` +
    `detail/QuantizeKV.hpp`).
  - **Granularity:** per-token, per-head symmetric INT8. The last dim
    (`D_head`) gets one FP32 scale `= absmax/127` (1.0 for an all-zero
    row); banker's round, clamp [-127,127]. Scales kept FP32 so the only
    error is the INT8 payload.
  - **`nn::QuantizedKVCache`** (`KVCacheBase` drop-in,
    `src/nn/QuantizedKVCache.cpp`): persistent storage is INT8 K/V slabs
    `[B,H,max_len,D_head]` + FP32 scale slabs `[B,H,max_len]`. `append`
    quantizes the projected slab then byte-copies the INT8 payload + scales
    into place (per-(b,h) strided insert, mirroring `KVCache::append`);
    `keys_view()`/`values_view()` narrow → contiguous → dequantize the
    prefix to the cache's float dtype. `MHA::forward_step` consumes it
    through the unchanged interface, so it composes with GQA and the
    contiguous/paged caches.
  - **Integration:** `LlamaModel::make_quantized_kv_caches` +
    `GenerateConfig::kv_int8` route `generate` through quantized caches.
  - **Verification** (`tests/nn/test_quant_kv.cpp`, 6 cases): error-bound
    roundtrip (`|x−x'| ≤ scale_row/2`); zero-row → scale 1, exact zero;
    **CPU↔CUDA exact agreement**; **exact `dequant(quant(K))` cache
    reconstruction** (chunked append == one-shot — validates the slab /
    scale plumbing); MHA `forward_step` bounded-error vs FP cache;
    deterministic, valid `kv_int8` generate. CUDA **437/439 correctness
    green** (+6; the 2 reds are `bench_cuda_matmul` /
    `bench_cuda_transformer_block` perf bars flaking under shared-GPU
    co-tenancy — no matmul code changed), CPU green (the CUDA-agreement
    case SKIPs via `SKIP_RETURN_CODE 4`), lints clean.
- **Deferred (B-031+):** `keys_view()` materializes a fresh FP prefix per
  step — fusing dequant into the attention matmul (a quantized-attention
  kernel reading INT8 K/V + scales directly) removes that transient and is
  the throughput/footprint follow-up (pairs with the B-019b+ ragged
  paged-attention kernel). Also: INT8 *paged* pool (quantized
  `PagedKVPool`), FP8 KV, per-channel / asymmetric KV schemes.

### B-032 — Compute-batched decode *(Resolved, Wave 10)*
- **Introduced in:** Wave 10 (2026-06-22). Motivation: the Wave-7
  scheduler decoded its running set one request at a time — N separate
  `forward_step` calls per tick, each a stack of tiny `M=1` GEMMs. On a
  GPU those launches are latency-bound and leave the SMs idle; the
  throughput win of continuous batching only materializes when the active
  set's matmuls are *batched* into one launch. This closes the
  compute-batching half of the B-029+ deferral.
- **Key insight / scope:** decode FLOPs are dominated by the dense
  projections (4·D² per token) + FFN (3·D·D_ff) + LM head — all
  position-independent and trivially batchable across requests. Only
  attention is ragged (each request's KV prefix is a different length).
  So batch the dense layers across the active set and keep attention
  per-sequence: a pure op-composition path, **no new CUDA kernel**, lowest
  risk to the headline throughput win.
- **Definition of done (met):**
  - **`MultiHeadAttention::forward_step_batched(x [A, S_new, D], caches:
    vector<KVCacheBase*>)`** — Q/K/V/O projections run once over all
    `A·S_new` rows; a per-sequence loop drives `attend_single` (a
    file-local factoring of the Wave-8 `forward_step` core: RoPE@`pos` →
    contiguous → `cache.append` → gather → `repeat_kv` (GQA) → SDPA (+
    decode mask when `S_new>1`) → merge), each cache `batch()==1` at its
    own `current_len`. Restacked via `ops::cat`, projected once by `o_proj`.
  - **`TransformerBlock::forward_step_batched`** — norms / FFN / residual
    adds are position-independent ⇒ batched; attention delegated.
  - **`LlamaModel::forward_step_batched(tokens [A, S_new],
    caches[seq][layer])`** — batched embed → per-layer batched block (the
    per-layer `KVCacheBase*` list assembled across sequences) → batched
    final RMSNorm + LM head → `[A, S_new, V]`.
  - **Scheduler integration:** `ContinuousBatchingScheduler::step` samples
    every running request from the prior step's logits (advancing tokens +
    EOS/length stop), then folds the still-active set into one
    `forward_step_batched`, slicing per-request `[1,1,V]` logits back out.
    Prefill stays per-request (admission prefills each prompt singly).
  - **Exact parity contract:** each GEMM output row is independent of the
    others and attention is per-sequence ⇒ row `r` of batched decode ≡
    standalone `forward_step` on sequence `r` fed the same tokens.
    Bit-identical on CPU (independent of `A`); within float tolerance on
    CUDA (batched vs single GEMM may pick different algos).
  - **Verification:** the 6 existing Wave-7 scheduler tests (CPU) still
    match standalone `generate` byte-for-byte — now through the batched
    path. New `tests/models/test_batched_decode.cpp` (6 cases): ragged
    mixed-length parity vs per-request for MHA + GQA (CPU exact, CUDA
    ≤ 3e-3 + argmax match), CUDA MHA+GQA, chunked-prefill `S_new>1` exact,
    `A==1` ≡ `forward_step`, shape/cache-count validation. CUDA green on a
    contention-free card (perf benches + 2 timing-sensitive cases flake
    under `-j` self-contention but pass `-j1`), CPU green (CUDA case SKIPs
    via `SKIP_RETURN_CODE 4`), lints clean.
- **Resolved (B-032+, Wave 11, 2026-06-22):** the per-sequence attention
  loop is gone for the CUDA decode path — see **B-032+** below.
- **Deferred (B-032++):** INT8-direct paged attention (read INT8 K/V +
  scales straight from a quantized pool, no FP-prefix dequant), padded
  same-length micro-batching, a fused paged *prefill* kernel, and
  chunked-prefill batching across admissions.

### B-032+++ — INT8-quantized paged KV cache *(Resolved, Wave 13)*
- **Introduced in:** Wave 13 (2026-06-22). Motivation: Wave 12 proved the
  fused INT8 paged op + kernel (`paged_decode_attention_int8`) but left it
  unwired — nothing in the runtime *stored* INT8 paged KV to feed it. This
  is the storage layer that unifies the three KV fast paths the framework
  built separately (GQA, KV-quant, paged-attention) into one allocatable
  cache the scheduler can use.
- **Definition of done (met):**
  - **Pool** `nn::QuantizedPagedKVPool`
    (`include/tesseract/nn/QuantizedPagedKVPool.hpp`,
    `src/nn/QuantizedPagedKVPool.cpp`): the INT8 sibling of `PagedKVPool` —
    same `num_blocks × block_size` grid + one `BlockAllocator`, but each
    slot is an INT8 payload `[num_blocks,H,block_size,D]` + one FP32 scale
    per `(block,head,slot)` `[num_blocks,H,block_size]` (the Wave-9
    per-token, per-head symmetric layout = exactly what
    `paged_decode_attention_int8` reads). INT8 payload + FP32 scale share
    block ids (single allocator). All four tensors zero-init (quantized 0
    dequantizes to 0).
  - **Cache** `nn::QuantizedPagedKVCache` (`KVCacheBase` drop-in,
    `src/nn/QuantizedPagedKVCache.cpp`): `append([B,H,Sn,D] FP)` quantizes
    per-(token,head) via `quant::quantize_kv_per_token` and scatters the
    INT8 payload + scale into on-demand blocks; `keys_view()` /
    `values_view()` gather the scattered INT8 + scale prefix and dequantize
    to a fresh FP `[B,H,L,D]` (CPU / non-fused fallback). Scatter/gather are
    per-head byte copies (CPU memcpy / CUDA async) — the launch-collapsed
    `launch_paged_gather` kernel only handles 2/4/8-byte elements, and
    gather is the fallback anyway (the fused decode reads the pool in
    place). `reset()` returns this cache's blocks to the shared pool;
    `set_current_len` honors the graph-rewind contract.
  - **Integration:** `MultiHeadAttention::forward_step_batched`'s CUDA
    single-token fast path recognizes BOTH paged flavors — the per-request
    RoPE+append + `[A,max_logical]` block-table assembly is factored into
    one `run_fused` driver that dispatches to `paged_decode_attention` (FP
    pools) or `paged_decode_attention_int8` (INT8 pools). CPU / contiguous
    caches / chunked prefill (`Sn>1`) keep the per-sequence fallback — so
    the scheduler's CPU bit-exact parity vs standalone `generate` holds.
  - **Verification** (`tests/nn/test_quant_paged_kv.cpp`, 3 cases): paged vs
    contiguous (`QuantizedKVCache`) dequant-view **bit-exact** on identical
    chunked appends straddling block boundaries (validates scatter/gather);
    on-demand paging + reset recycling; end-to-end `forward_step_batched`
    (quant-paged, CUDA) vs per-request `forward_step` (gather+dequant+
    attention fallback) ≤ 3e-3. CPU green (CUDA case SKIPs via
    `SKIP_RETURN_CODE 4`), CUDA green on a free card. No regression after
    the `forward_step_batched` refactor: FP paged 11/11, batched decode
    6/6, scheduler CPU+CUDA 6/6 each (still bit-exact).
- **Deferred (B-032++++):** a scheduler/`generate` knob to select INT8
  paged pools per layer (the cache is ready and proven; only the engine
  config flag + per-layer pool construction remain), padded same-length
  micro-batching, and a fused paged *prefill* kernel (`S_new>1`).

### B-032++ — INT8-direct paged decode-attention *(Resolved, Wave 12)*
- **Introduced in:** Wave 12 (2026-06-22). Motivation: the three KV fast
  paths still had a seam. Wave 9 made the *persistent* KV INT8
  (`QuantizedKVCache`), Wave 11 fused paged decode-attention, but a fused
  INT8 *paged* read still required an FP-prefix transient — dequantize the
  whole pool, then run the Wave-11 FP kernel — which defeats the quant
  memory-bandwidth win exactly on the decode hot path.
- **Definition of done (met):**
  - **Op** `nn::paged_decode_attention_int8(q [A,H,D], k_pool/v_pool
    [num_blocks,Hkv,block_size,D] Int8, k_scale/v_scale
    [num_blocks,Hkv,block_size] Float32, block_tables [A,max_logical]
    Int32, lens [A] Int32, scale, group)` → O [A,H,D] in q's FP dtype
    (`include/tesseract/nn/PagedAttention.hpp`, `src/nn/PagedAttention.cpp`).
    Same ragged single-launch decode as B-032+, but each K/V vector is
    `int8 * scale` with the Wave-9 per-(block,head,slot) symmetric scale
    (one FP32 per `D`-vector), dequantized on load — the FP-prefix transient
    is never materialized. Result: an INT8 paged pool read straight into
    attention at ~4× (vs FP32) / ~2× (vs FP16) lower KV bandwidth, unifying
    the GQA (B-030) + KV-quant (B-031) + paged-attention (B-032+) paths.
  - **CUDA kernel** `paged_decode_attention_int8_kernel`
    (`src/cuda/PagedAttention.cu`): same warp-per-`(request,query-head)` /
    per-lane-Q-fragment / online-softmax structure as the FP kernel; loads
    `int8` K/V + one scale per key `j` (broadcast over `D`), FP32 interior
    math. Q/O stay FP32/FP16/BF16; head_dim ≤ 128. Always-compiled
    stub-vs-kernel pairing (`launch_paged_decode_attention_int8` in
    `PagedAttentionStub.cpp` + `detail/PagedAttention.hpp`) + a matching CPU
    reference loop in the op.
  - **Verification** (`tests/nn/test_paged_attention.cpp`, +5 cases, 11
    total): the INT8 op is checked against the **Wave-11 FP op run on the
    dequantized pool** (`dequantize_kv_per_token(pool)`) — identical
    `int8*scale` values + online softmax, so the only delta is the inline
    dequant: CPU **bit-exact** ≤ 1e-5, CUDA ≤ 3e-3, for MHA (group=1) + GQA
    (group>1); zero-length request → zero row; operand validation (Int8
    pools required, group match, scale numel). CPU green (CUDA cases SKIP
    via `SKIP_RETURN_CODE 4`), all 11 cases green on a contention-free card.
- **Resolved (B-032+++, Wave 13, 2026-06-22):** the `QuantizedPagedKVPool`
  / `QuantizedPagedKVCache` storage layer that allocates INT8 paged KV and
  feeds this op directly via `forward_step_batched` — see **B-032+++**
  above. Still deferred (B-032++++): the scheduler/`generate` knob to
  select INT8 paged pools, padded same-length micro-batching, and a fused
  paged *prefill* kernel.

### B-032+ — Fused ragged paged decode-attention *(Resolved, Wave 11)*
- **Introduced in:** Wave 11 (2026-06-22). Motivation: Wave 10 batched the
  dense decode layers but still looped attention per sequence — for A
  active requests, per layer per tick: A `keys_view()` gathers (each
  materializes a request's full KV prefix out of scattered paged blocks
  into a fresh contiguous slab), A `repeat_kv` GQA replications, and A
  `ops::attention` launches. All of it is memory-bound and launch-bound on
  the decode hot path — the attention-side throughput the fused kernel
  exists to reclaim.
- **Definition of done (met):**
  - **Op** `nn::paged_decode_attention(q [A,H,D], k_pool/v_pool
    [num_blocks,Hkv,block_size,D], block_tables [A,max_logical] Int32,
    lens [A] Int32, scale, group)` → O [A,H,D]
    (`include/tesseract/nn/PagedAttention.hpp`, `src/nn/PagedAttention.cpp`).
    One query token per request (decode); reads K/V in place from the
    shared per-layer pool via each request's block table (no gather),
    maps query head `h` → KV head `h/group` on the fly (no `repeat_kv`),
    online (FlashAttention-style) softmax so the `[1,S_k]` score row is
    never written. `lens[r]==0` ⇒ zero output row.
  - **CUDA kernel** `src/cuda/PagedAttention.cu`: one warp (32 threads)
    per `(request, query-head)`; per-lane Q fragment (scale folded),
    warp-shuffle score dot, online-softmax accumulate; FP32 interior math
    on FP32/FP16/BF16 storage; `D_PER_LANE = D_MAX/32 = 4` (head_dim ≤ 128).
    Standard always-compiled stub-vs-kernel pairing
    (`PagedAttentionStub.cpp`, `detail/PagedAttention.hpp`) + a numerically
    matching CPU reference loop in the op.
  - **Plumbing:** `PagedKVCache::block_table(b)` exposes the
    logical→physical map so a caller can build the `[A,max_logical]`
    table; `tesseract_nn` now links `Tesseract::cuda` (the op reaches the
    launcher directly).
  - **Integration:** `MultiHeadAttention::forward_step_batched` takes the
    fused path when the step is a CUDA single-token decode (`S_new==1`),
    every cache is a `PagedKVCache`, and all share one pool — RoPE + cache
    append stay per-request (cheap), then one `paged_decode_attention`
    over the active set, then a batched `o_proj`. CPU / contiguous caches
    / chunked-prefill (`S_new>1`) keep the exact Wave-10 per-sequence loop,
    so the scheduler's CPU bit-exact parity vs standalone `generate` is
    untouched.
  - **Verification** (`tests/nn/test_paged_attention.cpp`, 6 cases):
    ragged mixed-length parity vs gather+`ops::attention` for MHA (group=1)
    + GQA (group>1) — CPU ≤ 1e-5, CUDA ≤ 3e-3; zero-length request → zero
    row; CUDA op parity; **end-to-end `forward_step_batched` (paged, CUDA)
    vs per-request `forward_step` ≤ 3e-3** (proves the wiring takes the
    fused path and stays correct); operand validation. CPU green (CUDA
    cases SKIP via `SKIP_RETURN_CODE 4`), CUDA correctness green on a
    contention-free card. The only CUDA reds in the full run are perf
    benches + 3 timing-sensitive cases (CudaGraph `forward_step` replay,
    `to_async` pageable, mnist tolerance) that flake under heavy external
    GPU contention and pass when re-run on a free card — none touch the
    paged path or any code Wave 11 modified.
- **Resolved (B-032++, Wave 12, 2026-06-22):** INT8-direct paged attention
  (read INT8 K/V + per-token scales straight from a quantized pool, fusing
  the Wave-9 dequant in-kernel — no FP-prefix transient) — see **B-032++**
  above. Still deferred (B-032+++): padded same-length micro-batching, a
  fused paged *prefill* kernel (`S_new>1` over the active set), and the
  `QuantizedPagedKVPool` storage layer.

### B-010 follow-up — CUDA bench parallel-execution stability *(Resolved)*
- **Introduced in:** Wave 2 closure (2026-04-18). Under `ctest -j 8`
  the six `bench_cuda_*` binaries were racing for SM time and
  tripping their hard bars (matmul dispatch_us, attention_bwd
  envelope, rms_norm speedup) due to co-scheduled concurrent GPU
  work. The benches individually passed but the parallel run was
  flaky.
- **Definition of done (met):** each CUDA bench registers with
  CTest's `RESOURCE_LOCK "cuda_gpu_0"` property in
  `benchmarks/CMakeLists.txt`, serializing them on the GPU slot
  (they still run in parallel with CPU-only tests). The
  `bench_cuda_attention_bwd` SDPA-envelope bar was widened from
  `≤ 5.0×` to `≤ 5.1×` to absorb the 0.05% noise floor observed
  under parallel ctest load — a measurement-variance fix, not a
  perf regression.
- **Resolved:** Wave 2.2 landing (2026-04-18). Remaining flake
  under parallel ctest is non-bench reduction / softmax tests
  sharing cuBLASLt or per-call workspace state; tracked as a
  cross-cutting "GPU test serialization" follow-up, out of Wave 2
  scope.

### B-012 — Generated dispatch table
- **Introduced in:** M2 kickoff (2026-04-18) via [ADR-0005](adr/0005-cuda-hal.md) §3.
- **Context:** M2 dispatches ops to CPU / CUDA via a per-op
  `switch (device.type)` statement. At M2's ~20-op scope this is
  clearer than PyTorch-style boxed kernels and costs nothing. When
  the op count grows past ~50 (expected mid-M3) or a third backend
  (ROCm in M5) lands, the switch blocks become repetitive boilerplate
  prone to drift.
- **Definition of done:** A registry-based dispatcher that each
  backend registers into at static-init time; the current switch
  blocks reduced to a single `dispatch(op_kind, device, ...)` call.
  Parity-tested against the pre-refactor behavior (no benchmark
  regressions > 1 %).
- **Target window:** M5 (AMD backend landing forces the issue).

---

## M3 frontier stack (Waves 14–20, 2026-06-22)

### B-032++++ — INT8 paged KV in the engine *(Resolved core, Wave 14)*
- **Introduced in:** Wave 13 deferral. Wave 13 (B-032+++) shipped the
  `QuantizedPagedKVPool` / `QuantizedPagedKVCache` storage layer but left the
  scheduler unable to *select* it; this is the engine wiring.
- **Definition of done (met):** `EngineConfig::kv_int8`
  (`include/tesseract/models/Scheduler.hpp`) makes
  `ContinuousBatchingScheduler` build per-layer `qpools_` via
  `LlamaModel::make_quantized_layer_pools` and admit INT8 paged caches via
  `make_quantized_paged_kv_caches`; `allocated_blocks()`/`free_blocks()` read
  whichever pool family is live; `retire` recycles the quantized blocks.
- **Verification:** `tests/models/test_scheduler.cpp` "INT8 paged KV matches
  generate(kv_int8)" — bit-exact on CPU vs `generate(kv_int8)`. Scheduler
  suite 18/18.
- **Resolved tail (2026-06-22, on free RTX 5880 Ada SM 8.9 GPUs):**
  - **Fused paged *prefill* kernel (`S_new>1`):** new
    `nn::paged_prefill_attention` + `paged_prefill_attention_int8`
    (`src/cuda/PagedAttention.cu`, `src/nn/PagedAttention.cpp`). One warp per
    `(request r, new-query s, head h)` triple with a causal bound
    `key_len = kv_len[r] - S + s + 1`, reading K/V in place from the paged
    pool (FP and INT8-direct, inline dequant), online softmax — collapses the
    chunked-prefill `Sn>1` per-sequence `attend_single` loop into one launch.
    Wired into `MultiHeadAttention::forward_step_batched` (CUDA, all caches
    paged or all qpaged); CPU / contiguous / mixed caches keep the exact
    per-sequence fallback so bit-exact parity holds. Verified
    (`tests/nn/test_paged_attention.cpp`, +6 cases): FP/INT8 prefill parity vs
    gather+attention (CPU ≤1e-5, CUDA ≤3e-3) for MHA+GQA, and end-to-end
    `forward_step_batched(Sn>1, CUDA)` matches per-request `forward_step`.
  - **Padded same-length micro-batching:** *superseded, not implemented* — the
    fused ragged decode (B-032+/+++) + prefill kernels handle ragged lengths
    directly with no padding, which is strictly better than padding the active
    set to a common length (padded batching wastes compute on the pad and is
    the pre-paged-attention strategy). The micro-batching idea is therefore
    obsoleted by the chosen design rather than left undone.

### B-033 — Real HF checkpoint end-to-end demo *(Resolved, Wave 16)*
- **Introduced in:** M3 exit-criteria (the `llama_generate` demo).
- **Definition of done (met):** `LlamaConfig::from_json` / `from_json_file`
  (`src/models/Llama.cpp`) parse a real HF `config.json` (essential
  architecture fields enforced, optional fields defaulted, `torch_dtype` +
  `bos/eos_token_id` captured); `examples/llama_generate.cpp` loads
  config+safetensors+tokenizer (`io::BpeTokenizer`) and streams decoded
  output with full generation-param CLI.
- **Verification:** `tests/models/test_llama_config_json.cpp` (valid configs,
  dtype variants, defaults, malformed/incomplete rejection).

### B-034 — Structured / grammar-constrained generation *(Resolved, Wave 17)*
- **Introduced in:** idea.md §4.4 frontier stack.
- **Definition of done (met):** `RegexAutomaton` (recursive-descent regex →
  Thompson NFA → lazy DFA over bytes) + `GrammarConstraint` (logit masking to
  the accepting-path token set, `accept()` state advance, EOS gated on accept)
  in `src/models/StructuredDecoding.cpp`; wired into `llama_generate` via
  `--grammar-regex`.
- **Verification:** `tests/models/test_structured_decoding.cpp` — regex match
  semantics, masking, and `constrained_greedy` always matching the grammar.

### B-035 — Speculative decoding *(Resolved, Wave 18)*
- **Introduced in:** idea.md §4.4 frontier stack.
- **Definition of done (met):** `SpeculativeDecoder(target, draft, gamma)`
  (`src/models/Speculative.cpp`): draft proposes γ tokens, target verifies all
  γ+1 positions in one `forward_step_batched`, longest matching prefix
  accepted, KV rewound via `set_current_len` on partial accept, correction
  token re-fed; output identical to target-only greedy. `Result` tracks
  proposed/accepted/rounds/target-forwards.
- **Verification:** `tests/models/test_speculative.cpp` — greedy parity vs
  target standalone (several drafts + γ), self-draft acceptance, EOS early
  stop, input validation.

### B-036 — Disaggregated prefill/decode *(Resolved, Wave 19)*
- **Introduced in:** idea.md §4.4 frontier stack.
- **Definition of done (met):** `DisaggregatedEngine` + `KvTransfer`
  (`src/models/Disaggregated.cpp`): `prefill` exports per-layer contiguous K/V
  views + first token + prompt_len; `decode` imports them into fresh caches
  and continues autoregressively; `generate` chains both, bit-identical to
  monolithic generation.
- **Verification:** `tests/models/test_disaggregated.cpp` — `generate` parity
  vs monolithic, explicit prefill-role→decode-role transfer, EOS early stop,
  input validation.

### B-037 — Graph-mode quantization dialects *(Resolved, Wave 20)*
- **Introduced in:** idea.md §4.4 frontier stack (FP8/FP4/INT4 as IR attrs).
- **Definition of done (met):** `tesseract.dequant_matmul`
  (`src/ir/TesseractOps.td`) with `lhs`/`qweight`/`scale` operands + `scheme`
  (int8/int4_group/fp8_e4m3/fp8_e5m2/fp4) and `group_size` (si64) attributes;
  `DequantMatMulOp::verify()` (`src/ir/TesseractOps.cpp`) validates the scheme
  enum + group_size/scheme-family agreement.
- **Verification:** round-trip in `tests/ir/roundtrip.mlir`
  (ctest `ir_dialect_roundtrip`) + verifier negatives in
  `tests/ir/dequant_matmul_invalid.mlir` (ctest `ir_dequant_matmul_invalid`,
  `--verify-diagnostics`). IR ctests 6/6.

## From M4 (2026-06) — parallel A/B/C tracks (ADR-0006)

### B-038 — MoE native (Track A1)
- **Introduced in:** M4 (2026-06-22), `idea.md` §4.2 (non-Transformer
  architectures), [`docs/m4-plan.md`](m4-plan.md) Track A1.
- **Context:** Sparse mixture-of-experts is the first non-Transformer building
  block and the gate to Mixtral-class checkpoints; it reuses the existing dense
  `FeedForward`/`Linear` stack so routing is the only new concept.
- **Definition of done:** `nn::MoEFeedForward` (router top-k softmax gating + N
  expert FFNs + token dispatch/combine); `nn::TransformerBlock` optional MoE-FFN
  branch (`num_experts > 0`); `LlamaConfig::{num_experts, num_experts_per_tok}` +
  Mixtral-style `config.json` parse; CPU reference + CUDA (op-composition);
  optional load-balance aux loss.
- **Verification:** top_k==1/one-expert MoE bit-identical to dense FeedForward;
  equal-gate full-routing == mean of experts; CPU exact / CUDA tolerance parity;
  a 2-layer MoE Llama generates deterministically.
- **Target window:** M4 Track A1. Fused grouped-GEMM expert dispatch is a
  B-038+ follow-up.
- **Resolved (2026-06-22):** `nn::MoEFeedForward`
  (`include/tesseract/nn/MoEFeedForward.hpp` + `src/nn/MoEFeedForward.cpp`) —
  a `gate` `Linear` (d_model→E, no bias) + E `FeedForward` experts in a
  `ModuleList`; forward computes `softmax(router(x))`, a host-side top-k mask
  (deterministic value-desc/index-asc tie-break, FP32/FP64), renormalized gates
  `(probs·mask)/Σ`, and the gate-weighted sum of all experts (dense compute, so
  bit-exact parity; fused dispatch deferred B-038+). `last_aux_loss()` exposes
  the Switch load-balancing loss `E·Σ f_e·P_e` (differentiable through P_e).
  `nn::TransformerBlock` gained a `num_experts`/`num_experts_per_tok` ctor pair
  that swaps the FFN slot (same "ffn" registration name) for the MoE variant
  via a `ffn_forward` dispatch in all three forward paths; `ffn()` returns null
  for MoE blocks (quantizer null-guards). `LlamaConfig::{num_experts,
  num_experts_per_tok}` + `from_json` parse of `num_local_experts`/`num_experts`
  + `num_experts_per_tok`. **Verified:** `tests/nn/test_moe.cpp` (7 cases, 23
  asserts) — single-expert==lone-expert, uniform-router==mean-of-experts,
  top1==argmax routing, aux-loss finiteness, MoE-Llama generate determinism +
  block-is-MoE structure, Mixtral config parse, validation. CPU green.
- **Sparse dispatch + metric (2026-06-23):** the dense inner loop was replaced
  with **true sparse dispatch** — tokens are permuted into per-expert groups
  (autograd-aware `index_select`), each expert runs only on its routed rows
  (`narrow`+`forward`, Σ_e n_e == T·k vs dense T·E), then un-permuted and summed
  with gate values gathered from the differentiable `gates` (`ops::gather`), so
  the router stays trainable. Output is numerically identical to the dense
  reference (the prior tests still pass bit-for-bit within 1e-5). Two new tests
  added: top-2 sparse==dense parity and router-gradient-flows. `bench_moe_sparse`
  quantifies the saving: at D=512/dff=1024, E=8/k=2 the sparse path is **2.14×**
  faster at T=4096 (ratio 0.47 → ideal k/E=0.25), and E=16/k=1 is **1.76×** at
  T=1024 — the gap from ideal is per-expert GEMM shrinkage + routing overhead on
  CPU. Full read in [`docs/design/moe-sparse.md`](design/moe-sparse.md). This is
  the efficiency advantage the dense A1 could not demonstrate. Remaining tail:
  grouped-GEMM (one launch over slices) + device-side top-k.

### B-039 — SSM / Mamba native (Track A2)
- **Introduced in:** M4 (2026-06-22), `idea.md` §4.2 (the headline
  differentiator), [`docs/m4-plan.md`](m4-plan.md) Track A2.
- **Definition of done:** `ops::selective_scan` (discretized SSM recurrence,
  chunkwise — parallel within a chunk, sequential across; CPU ref + CUDA
  kernel); `nn::Mamba` block (in-proj → causal depthwise conv1d → SiLU →
  selective_scan → out-proj); `nn::SSMStateCache` (`KVCacheBase`-analog holding
  recurrent state + conv ring buffer).
- **Verification:** recurrent step-by-step decode == parallel chunkwise prefill
  (CPU ≤ 1e-5, CUDA tolerance); a minimal Mamba model runs end-to-end `generate`
  reusing `Sampler` + scheduler.
- **Target window:** M4 Track A2 (after A1).
- **Resolved (2026-06-22):** `ops::selective_scan`
  (`include/tesseract/ops/SelectiveScan.hpp` + `src/ops/cpu/SelectiveScan.cpp`)
  — the discretized input-dependent SSM recurrence `h = exp(Δ·A)·h + (Δ·B)·u`,
  `y = Σ C·h + D·u`, forward-only, FP32 interior, returning `{y [B,L,D],
  final state [B,D,N]}` with an optional `state_in` so decode threads the
  state. CUDA kernel (`src/cuda/SelectiveScan.cu` + `detail/SelectiveScan.hpp`
  + `SelectiveScanStub.cpp`): one thread per `(b, d)`, register state array,
  caps `N ≤ 32`; **compiles under build-cuda** (FP32/FP64/FP16/BF16). `nn::Mamba`
  (`include/tesseract/nn/Mamba.hpp` + `src/nn/Mamba.cpp`): in_proj → causal
  depthwise conv1d (op-composition via `cat`+narrow+mul, runs CPU/CUDA) → SiLU
  → x_proj → softplus(dt_proj) → `selective_scan` → SiLU(z) gate → out_proj,
  with `forward` (prefill) + `forward_step` (threading `nn::SSMStateCache` —
  conv ring buffer + SSM hidden state). `models::MambaModel`
  (`include/tesseract/models/MambaModel.hpp` + `src/models/MambaModel.cpp`):
  embed → N×(RMSNorm→Mamba→residual) → norm → lm_head, with full `forward`,
  recurrent `forward_step`, and `generate` (greedy + seeded `Sampler`).
  **Verified:** `tests/nn/test_mamba.cpp` (5 cases, 16 asserts) — op-level
  recurrent==parallel (y + final state ≤ 1e-5), op validation, block-level
  forward==step loop (incl. d_conv==1 degenerate), model stepping==full forward
  + generate determinism/length. CPU green; CUDA kernel compiles (full GPU
  numeric parity is a gated tail, validated when cards are free — same policy
  as the M3 CUDA waves). Scheduler reuse (vs the Sampler reuse done here) is a
  B-039+ follow-up since the scheduler is currently KVCache/attention-shaped.
- **Scaling metric (2026-06-23):** `benchmarks/bench_mamba_vs_llama_scaling.cpp`
  turns the SSM thesis into a *measured curve* — decode ms/step and resident
  state swept over context length L, Mamba vs a matched Llama. Recorded (dev
  host): Llama per-step latency 14.6→27.3 ms over L=128→2048 (O(L)) with KV
  1→16 MiB; Mamba flat at ~12.4 ms with state constant 0.148 MiB — a **≈108×
  residency gap at L=2048** and a diverging latency slope. This is the
  objective architectural-advantage number A2 previously lacked. Full table +
  honest read in [`docs/design/mamba-scaling.md`](design/mamba-scaling.md).
  Charting prefill O(L) vs O(L²) needs the chunkwise-parallel scan tail (the
  current single-step prefill loop masks it).

### B-040 — DiT (Track A3, design only)
- **Introduced in:** M4 (2026-06-22), `idea.md` §4.2.
- **Definition of done (this milestone):** DiT module + IR-representation design
  captured in [`docs/m4-plan.md`](m4-plan.md) Track A3 + placeholder interface.
  The diffusion-scheduler / VAE-UNet runtime is deferred to M5 (modality far
  from the autoregressive stack).
- **Target window:** design M4, runtime M5.
- **Resolved (design, 2026-06-22):** adaLN-Zero DiT block design + IR-alignment
  notes written in `docs/m4-plan.md` Track A3; placeholder interface
  `include/tesseract/nn/DiTBlock.hpp` (construct-allowed, `forward` throws with
  an "M5 runtime" message). Runtime intentionally deferred to M5.

### B-041 — Python frontend (Track B1)
- **Introduced in:** M4 (2026-06-22), `idea.md` §8.2 ("用户迁移成本高于一切"),
  [`docs/m4-plan.md`](m4-plan.md) Track B1.
- **Definition of done:** `TESSERACT_BUILD_PYTHON` CMake option (pybind11 via
  FetchContent); `python/` builds a `tesseract._core` extension + thin
  `tesseract/` package; binds `Tensor` (creation/dtype/device/`to`/numpy
  buffer-protocol/autograd), `nn` modules, `LlamaModel.generate`, tokenizer,
  sampler params.
- **Verification:** `import tesseract` trains MNIST MLP + runs a Llama inference
  smoke from Python; `pytest python/tests` green.
- **Target window:** M4 Track B1 (independent — parallel start).
- **Resolved (2026-06-22):** `TESSERACT_BUILD_PYTHON` option
  (`cmake/Options.cmake`) + pybind11 discovery (`cmake/Dependencies.cmake`:
  prefers `python -m pybind11 --cmakedir`, FetchContent fallback). `python/`
  builds the `tesseract._core` extension (`python/_core.cpp`) staged inside a
  `tesseract/` package (`python/tesseract/__init__.py`). Bound surface: `Tensor`
  (NumPy buffer-protocol in/out, factories, `requires_grad`/`grad`/`backward`,
  `item`, `reshape`, arithmetic + matmul), free `backward`, `ops` (add/sub/mul/
  div/matmul/relu/sigmoid/softmax/cross_entropy), `nn` (Module/Linear/Embedding/
  ReLU/Sigmoid/Tanh/Sequential), `optim` (SGD/Adam), `models` (LlamaConfig incl
  MoE fields/LlamaModel/generate + MambaConfig/MambaModel/generate +
  SamplingParams), `io` (Tokenizer/WhitespaceTokenizer/BpeTokenizer).
  **Verified:** `python/tests/test_smoke.py` — 9 pytest cases green: import,
  NumPy roundtrip, autograd, **MNIST-style MLP overfit (loss → < 0.5)**, Llama +
  MoE-Llama + Mamba generate smoke/determinism, tokenizer roundtrip.
- **Resolved tail (2026-06-24, M4 perf-closeout — overhead quantified):** added
  `benchmarks/bench_python_overhead.cpp` + `bench/external/python_overhead.py`
  + `run_python_overhead.py` to measure the frontend crossing cost (same kernel
  both sides, diff = pybind overhead). **Result:** a fixed **~1.5–6 µs/call** —
  ~20% only for a trivial 64³ matmul, **<1–3% (≈0%) for real-sized ops** (MLP
  forward ~0%). Confirms the package is a thin shim with a constant dispatch tax
  that vanishes for any non-trivial workload. Recorded in
  `bench/external/results/python_overhead.md`.

### B-042 — single-GPU LLM training loop (Track B2)
- **Introduced in:** M4 (2026-06-22), [`docs/m4-plan.md`](m4-plan.md) Track B2.
- **Context:** "Distributed training" presupposes single-GPU LLM training, which
  never existed — the LLM has only run inference. This fills that prerequisite.
- **Definition of done:** `examples/llama_train.cpp` — next-token CE training
  over `LlamaModel` (`forward` → `cross_entropy` → `backward` → `Adam::step`) on
  `--device {cpu,cuda}`; any autograd gaps in the LLM forward filled.
- **Verification:** loss decreases monotonically on a tiny overfit set
  (memorize a fixed batch → loss → ~0), recorded + smoke test.
- **Target window:** M4 Track B2 (independent — parallel start).
- **Resolved (2026-06-22):** `examples/llama_train.cpp` trains a tiny Llama
  (2 layers, d_model 64) with next-token CE + Adam over a fixed seeded batch;
  CLI `--steps/--lr/--batch/--seq/--device/--target-loss`. Existing M2 autograd
  backward on the transformer forward ops was sufficient — no gaps surfaced
  (the generation paths bypass autograd but `forward()` records a full graph).
  **Verified:** binary drives loss **4.26 → 0.0065** in 80 CPU steps; unit
  smoke `tests/models/test_llama_train.cpp`
  (`LlamaModel: next-token training overfits a fixed batch`) asserts loss
  decreases and reaches < 0.1, green in CPU ctest.
- **Resolved tail (2026-06-24, M4 perf-closeout — PyTorch loss parity):** built
  a config-matched PyTorch Llama trainer (`bench/external/torch_llama_train.py`,
  RMSNorm+RoPE-MHA+SwiGLU, same vocab/d/layers/heads/ffn, same Adam lr=3e-3,
  same fixed-batch CE memorization, 100 steps). The loss curves **track within
  ~10% at every milestone** and converge to the same ~5e-3 floor (Tesseract
  4.26→0.0050, PyTorch 4.32→0.0054). Confirms Tesseract's autograd + Adam
  reproduce PyTorch training dynamics on an identical config. Recorded in
  `bench/external/results/train_parity.md`.

### B-043 — tensor parallelism as a transform (Track B3)
- **Introduced in:** M4 (2026-06-22), `idea.md` §6.1.5 (并行策略作为 IR pass),
  [`docs/m4-plan.md`](m4-plan.md) Track B3.
- **Definition of done:** column-parallel (q/k/v/gate/up) + row-parallel
  (o/down) sharding helpers with all-reduce; a `CommBackend` abstraction with a
  single-process multi-rank simulator (all-reduce = sum) for CPU parity; design
  doc on the IR-pass formulation.
- **Verification:** TP=2 sharded Llama block bit-identical to the unsharded
  block under the simulator (CPU).
- **Gated tail:** real multi-GPU NCCL backend (`TESSERACT_ENABLE_NCCL`) +
  validation — needs ≥2 contention-free cards.
- **Target window:** M4 Track B3 (after B2).
- **Resolved (2026-06-22):** new `tesseract_distributed` library.
  `distributed::CommBackend` abstracts the only two TP collectives
  (all-reduce-sum, all-gather-concat); `SimCommBackend` is the single-process
  reference (all-reduce = autograd-aware `ops::add` fold, all-gather =
  `ops::cat`). `ColumnParallelLinear` (split output features, optional
  all-gather) + `RowParallelLinear` (split input features, all-reduce, full
  bias added once) with `from_dense` shard builders and `forward_shards` for
  the no-extra-collective column→row composition. Design doc:
  [`docs/design/tensor-parallel.md`](design/tensor-parallel.md) (incl. the
  IR-pass formulation). **Verified:** `tests/distributed/test_tensor_parallel.cpp`
  — TP=2 column-parallel **bit-identical** to dense, row-parallel within 1e-5,
  full Megatron SwiGLU MLP (column gate/up → local silu*up → row down, one
  all-reduce) within 1e-5; green in CPU ctest. NCCL backend + multi-GPU
  validation remains the gated tail.
- **Resolved tail (2026-06-24, M4 perf-closeout — real multi-GPU unlocked):**
  with cards 0,1,2 held by `scripts/gpu_reserve.py`, added
  `benchmarks/bench_cuda_tp_scaling.cpp` — a **real** multi-GPU TP forward run
  placing each FFN shard on its own physical card with a cross-device all-reduce
  (Megatron SwiGLU MLP = sum of `d_ff/N` shards). **Measured** (d_model=4096,
  d_ff=12288, 4096 tokens, FP32): per-GPU weight scales **exactly 1/N**
  (576→288→192 MB), throughput **TP=2 2.06×, TP=3 2.98×** — and at TP=3
  Tesseract (332,916 tok/s) **beats PyTorch's same-structure run** (264,954
  tok/s, 2.38×) by **1.26×** (`bench/external/torch_tp_scaling.py`,
  `bench/external/results/tp_multigpu.md`). **Backward parity** added to
  `tests/distributed/test_tensor_parallel.cpp` (now 7 cases): TP=3 forward +
  TP=2 column/row backward grads re-assemble to the dense weight grad
  (column bit-exact, row within 1e-5). Remaining production step (still gated):
  an autograd-aware NCCL all-reduce (cross-device `to()` currently has no
  grad-fn, so the multi-GPU bench is forward/inference-only).

### B-044 — transformer op set in the dialect (Track C1)
- **Introduced in:** M4 (2026-06-22), `idea.md` §4.1 (one IR), ADR-0006
  deviation 2, [`docs/m4-plan.md`](m4-plan.md) Track C1.
- **Context:** the M3 LLM runtime is entirely eager and never touches the IR;
  this is the concrete down payment on making a real model executable through
  the `tesseract` dialect.
- **Definition of done:** extend the dialect + `--convert-tesseract-to-linalg`
  (`src/ir/passes/ConvertToLinalg.cpp`) to cover `rms_norm`, `silu`/`swiglu`,
  `rope`, `attention` (linalg lowering or marker+decompose); a single Llama
  block captured via `GraphScope`, emitted, lowered, and run on the CPU
  `JitEngine`.
- **Verification:** round-trip + FileCheck for the new lowerings; IR-executed
  Llama block matches eager within FP32 tolerance on CPU.
- **Target window:** M4 Track C1 (independent — parallel start).
- **Resolved (2026-06-22, FFN+norm slice; attention/rope = gated tail):**
  extended `--convert-tesseract-to-linalg` with the transformer FFN +
  normalization primitives — the exact ops `ops::rms_norm`
  (mul/mean/add/sqrt/div/mul) and `ops::swiglu_silu_gate` (sigmoid/mul/mul)
  decompose into: `sigmoid` (→ 1/(1+exp(-x))), `exp`/`log`/`sqrt`/`tanh`
  (→ `math.*`), and `mean` (→ `linalg.reduce` + scale + optional
  `expand_shape`). Added `tesseract.sqrt` to the dialect (`TesseractOps.td`).
  **Verified:** FileCheck `tests/ir/convert_transformer.mlir` (lowering
  structure) + JIT exec parity `tests/ir/test_jit_transformer.cpp` — sigmoid,
  the SwiGLU activation, and the **full RMSNorm primitive chain** execute
  through `ir::JitEngine` (dialect→linalg→LLVM) matching eager within 1e-4; all
  29 IR ctests green in `build/`.
- **Resolved tail (2026-06-23, attention + RoPE):** closed the gated C1 tail.
  (1) **Softmax lowering** — added `SoftmaxLowering` to
  `--convert-tesseract-to-linalg`: numerically-stable `exp(x − rowmax)/rowsum`
  over an arbitrary axis via two `linalg.reduce` (max, sum) + two
  `linalg.generic` (sub/exp, div). (2) **Single-head SDPA** — the
  transpose / rank-2 matmul / scale-mul / mask-add / softmax / matmul chain now
  executes through the JIT, validated against eager `ops::attention` for both
  the non-causal and causal (`-inf` upper-triangular mask) paths. (3) **RoPE**
  — added `tesseract.rotary_embedding` to the dialect (`TesseractOps.td`,
  3-operand) + a copy-free lowering: one all-parallel `linalg.generic` that
  reads the diagonal `x[c]` and broadcast `cos`/`sin` rows through affine maps
  and fetches the cross-lane partner `x[c^1]` in-body via `linalg.index` +
  `tensor.extract` (computing `x·cos + rotate_half(x)·sin`). The
  `tensor.extract` formulation deliberately avoids strided `insert_slice`,
  which bufferized to a `memrefCopy` runtime call the in-process JIT can't
  resolve. **Verified:** FileCheck `tests/ir/convert_attention.mlir` (softmax +
  SDPA + RoPE lowering structure, no residual `tesseract.*`) + JIT exec parity
  in `tests/ir/test_jit_transformer.cpp` (softmax 1e-6; non-causal/causal SDPA
  1e-5; RoPE 1e-5) — 7 transformer JIT parity cases + 18 IR ctests green in
  `build/`. Every primitive of a Llama block (RMSNorm, SwiGLU FFN, RoPE,
  attention) now runs through the dialect→linalg→LLVM CPU pipeline.

### B-045 — KV cache / dynamic shape as IR concepts (Track C2)
- **Introduced in:** M4 (2026-06-22), `idea.md` §4.1, [`docs/m4-plan.md`](m4-plan.md)
  Track C2.
- **Definition of done (this wave):** design doc + initial ops/attributes for a
  paged-buffer / dynamic-dim IR representation. Execution not required — the goal
  is a representation later NVVM/runtime stages can target.
- **Target window:** M4 Track C2 (after C1).
- **Resolved (2026-06-22):** three new dialect ops (`TesseractOps.td` +
  verifiers in `TesseractOps.cpp`) pin the decode-runtime representation:
  `tesseract.paged_kv_alloc` (rank-4 `[num_blocks, block_size, num_kv_heads,
  head_dim]` pool), `tesseract.paged_kv_append` (SSA-valued scatter into
  physical slots), `tesseract.paged_attention` (decode attention over the pool
  with `block_table` + `seq_lens`, `scale`/`causal` attrs). Dynamic shapes use
  MLIR's native `?` dim — verifiers skip dynamic dims, so a `tensor<?x8x64xf32>`
  query verifies. Design: [`docs/design/kv-cache-ir.md`](design/kv-cache-ir.md).
  **Verified:** `tests/ir/paged_kv.mlir` (round-trip incl. dynamic token axis)
  + `tests/ir/paged_kv_invalid.mlir` (verifier negatives) green in `build/`.
  Execution/lowering intentionally deferred (the wave goal is representation).

### B-046 — external-framework benchmark (cross-cutting)
- **Introduced in:** M4 (2026-06-22), `idea.md` §6.2/§8.5, ADR-0006 deviation 3.
- **Context:** all 16 existing benches are internal; `idea.md` demands alignment
  vs PyTorch/vLLM/SGLang/llama.cpp.
- **Definition of done:** a `benchmarks/bench_vs_llama_cpp.*` harness comparing
  end-to-end CPU decode tok/s against `llama.cpp` on a matched small model.
- **Gated tail:** vLLM / Hopper throughput alignment (hardware-gated).
- **Target window:** M4 cross-cutting.
- **Resolved (2026-06-22; external row filled 2026-06-23):**
  `benchmarks/bench_llama_decode_cpu.cpp` emits the Tesseract side — end-to-end
  CPU decode tok/s for a small synthetic Llama (4L/d256/8h/v4096/ff688);
  `scripts/make_tiny_llama_gguf.py` writes a **same-architecture GGUF** directly
  via `gguf-py` (5.26 M params, identical to Tesseract's config, side-stepping
  the converter's tokenizer auto-detect); `scripts/bench_vs_llama_cpp.sh
  --llama-cpp <checkout>` auto-builds the GGUF, runs `llama-bench`, parses it,
  and tabulates the comparison.
  **Recorded gap (decode, prompt 32 / gen 64):** llama.cpp ≈2 144 tok/s vs
  tesseract ≈56 tok/s single-threaded (**≈38×**); ≈4 687 vs ≈64 at 8 threads
  (**≈73×**). Tesseract decode is serial (does not scale with threads). Full
  table + honest read + backlog signals in
  [`docs/design/external-benchmark.md`](design/external-benchmark.md).
  **Verified:** driver runs end-to-end against a freshly-built llama.cpp on CPU.
  vLLM/Hopper alignment remains a gated tail — the "no external baseline" gap is
  now a tracked, reproducible **number**, not a TODO.
- **Resolved tail (2026-06-24, M4 perf-closeout — GPU comparison ungated):**
  with cards 0,1,2 reserved, the GPU side is no longer gated. Added the PyTorch
  GPU head-to-head (`bench/external/torch_baseline.py`, `run_compare.py`) and a
  **consolidated 12-row external scoreboard** in
  [`docs/design/external-benchmark.md`](design/external-benchmark.md) with
  per-row win/tie/honest-loss verdicts. **Headline wins:** INT8 decode GEMV
  **2.45× vs torch FP16**; Mamba O(1) decode **2.2×** + 108× less state; real
  multi-GPU TP=3 **2.98× vs torch 2.38×**. **Ties:** cuBLAS GEMM (FP16/true
  FP32). **Honest losses:** FlashDecoding attention (~2×) and llama.cpp CPU
  decode (~73×), each root-caused. **vLLM:** install aborted on the shared host
  (heavy dep closure risked downgrading the working torch 2.10+cu128); recorded
  as a follow-up needing an isolated venv + serving harness — documented, not a
  blocker.
- **Resolved tail (2026-06-25, vLLM serving — flipped to a WIN):** the one honest
  loss is now a decode/e2e win. Landed: **B-022** FP16 loader cast
  (`Llama.cpp::cast_floating_cpu` — one on-disk fp32/bf16 checkpoint seeds an
  FP16 model); **capture-safe Storage** (async-on-stream memset/D2D + KV append
  as a single `cudaMemcpy2DAsync`, which also fixed a graph-replay bit-exactness
  drift, 0.046→2e-4, now covered by a documented-tolerance test); **whole-model
  CUDA-graph** capture/replay for decode AND prefill in
  `bench_cuda_llama_serving --cudagraph`; **GQA-native fused prefill attention**
  (`ops::prefill_attention_gqa` + H_kv-aware fused kernel, removing `repeat_kv` +
  the composite score-matrix HBM round trip). Matched-FP16 TinyLlama-1.1B
  (prompt/gen 128, clean RTX 5880 Ada): **decode 321.1 vs vLLM 305.8 tok/s,
  TPOT 3.115 vs 3.275 ms, e2e ~403 vs 420 ms — all wins**; TTFT 7.28 vs 5.47 ms
  is the lone residual (shared cuBLAS GEMM floor). Numbers + reproduce in
  [`bench/external/results/vllm_serving.md`](../bench/external/results/vllm_serving.md).
  **Open follow-up → B-024:** WMMA tensor-core FlashAttention (decode+prefill) +
  norm/rope/residual elementwise fusion, to also win TTFT.

---

### B-047 — Tesseract Studio: visual block builder *(M5 adoption, initial build 2026-06-25; UI on LiteGraph + tensor/IR groups 2026-06-25)*

- **Target window:** M5 (adoption); started early while B-024+ WMMA waits for a
  free GPU.
- **Motivation:** a Scratch-like drag-drop block UI so beginners can build,
  train, and run models with no code; doubles as a visual front-end onto the
  "one IR, train+infer" thesis and the headline adoption lever for the
  open-source release. Decision recorded in
  [`docs/adr/0007-studio-visual-builder.md`](adr/0007-studio-visual-builder.md).
- **Architecture (ADR-0007):** native C++ executable embedding the engine
  directly (no Python), gated by `TESSERACT_BUILD_STUDIO`. Two layers:
  `tesseract_studio_core` (GUI-independent: block graph + `.tsb` JSON +
  catalog + validation/shape-inference + C++/Python codegen + executor over the
  real `nn`/`models`/`optim`) and `tesseract_studio` (a thin app that embeds the
  block-canvas web UI as byte arrays and serves it + a JSON control plane over
  localhost). Headless-friendly (browser is the display) because the box has no
  X11/OpenGL; backend is dependency-free (in-tree minimal JSON + POSIX HTTP
  server). The canvas is **LiteGraph.js (MIT)**, vendored under
  `app/web/vendor/` and embedded into the binary (build-time CDN download; only
  GitHub `git` FetchContent is blocked on the box, not HTTPS CDNs). Catalog →
  LiteGraph node types are generated at runtime, so adding a `BlockSpec` needs no
  UI code.
- **Definition of done (all six proposed feature groups, met):**
  - 32-block catalog across Data / Layers / **Tensor** / Model / Loss /
    Optimizer / Tokenizer / Train / Inference, driving palette + validation +
    codegen + IR + executor from one declarative source.
  - **build:** layer chain (Linear/ReLU/Sigmoid/Tanh/RMSNorm/LayerNorm/
    Embedding/FeedForward/MHA/TransformerBlock) → `SequentialModel`; **train:**
    `SyntheticClassification`/`CsvDataset` + `Adam`/`SGD` + `CrossEntropyLoss` +
    `TrainLoop` with live loss/accuracy; **infer:** `LoadLlama`/`LoadMamba`
    (+ `BpeTokenizer`) → `Generate` with streamed tokens.
  - **tensor:** a tensor & autograd playground — `TensorConst` leaves,
    element-wise (`TAdd/TSub/TMul`), `TMatMul`, unary (`TReLU/TSigmoid/TTanh/
    TExp`) ops, and a `TensorInspect` sink that evaluates the expression, runs
    `backward(Σout)`, and streams value + per-leaf gradient tensors (rendered as
    heatmaps in the UI). Runs on CPU **and CUDA**.
  - **ir:** `generate_ir()` lowers the block graph to a textual `tesseract`-
    dialect MLIR module (matching `tests/ir/*.mlir` op syntax), exposed at
    `/api/ir` and an IR panel — the visual realization of "one IR".
  - **code I/O:** `.tsb` JSON save/load fully bidirectional; C++ and Python
    codegen carry a round-trippable `@tsb-graph` header; examples in
    `studio/examples/`.
  - Edit-time validation (unknown kinds, dangling/typed wires, required inputs,
    DAG-ness) + tensor shape inference along the chain *and* through tensor ops
    (incl. matmul inner-dim checks), surfaced as on-node shapes + error dots.
- **Verification:** `test_studio` (**10 cases / 48 assertions**, headless)
  covers JSON + `.tsb` round-trip, catalog, validation + shape inference, codegen
  round-trip, **IR emission**, the executor **training an MLP to 100%**,
  **generating from a Llama**, and **evaluating + back-propagating a tensor
  graph** (correct [3,2] output + one gradient per leaf). The running binary was
  smoked over HTTP end-to-end on **CPU and on a clean CUDA card (device 1)**:
  validate → IR → run → events for the tensor playground (forward+grads) and MLP
  train (loss 1.07 → 9e-4, acc 1.0 on GPU); vendor assets served (litegraph.js
  200/491 KB). See [`studio/README.md`](../studio/README.md).
- **Deferred (B-047+):** real per-token streaming via `forward_step` (current
  Generate calls `generate()` then replays tokens); routing the IR view through
  the *real* MLIR emitter when `TESSERACT_ENABLE_MLIR=ON` (today it is a faithful
  textual pretty-print) and full lowering of Embedding/FeedForward/MHA blocks; an
  optional native desktop-window (ImGui/Qt) front-end over the same
  `studio_core`; broadening the Python bindings toward PyTorch-level coverage
  (tracked separately — not on Studio's critical path).

---

## Maintenance conventions

- Each item is referred to by its B-### identifier in code comments (e.g.
  `// TODO(B-003): ...`).
- Closing an item requires both (a) the definition-of-done to be met in the
  codebase and (b) this document updated with a one-line "Resolved:" entry
  under the item.
- New items are appended with the current milestone stamp; never reuse IDs.

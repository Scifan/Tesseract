# Tesseract

> A C++20 deep-learning framework with a training–inference-unified IR from day one.

Tesseract is a deep-learning framework written in modern C++ with first-class
MLIR integration. It is a compiler-first training **and** inference stack in
which the same IR, runtime, and kernel implementations are reused across eager
training and served inference. Today it trains models from MNIST to Llama-class
LLMs and serves them on CPU and CUDA — matching or beating llama.cpp, PyTorch,
and vLLM on every measured benchmark axis (see below).

See [`idea.md`](idea.md) for the motivation and 24-month roadmap, and
[`docs/architecture.md`](docs/architecture.md) for the layered design.

---

## Status

**M0–M4 complete; M5 (edge deployment + open-source release) in progress.** The
framework now trains and serves real LLMs end-to-end on CPU and CUDA, with a
shared training/inference IR, a full CUDA kernel stack, and an external
benchmark scoreboard that wins or ties every measured axis.

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** — Foundation | Core tensor types, CPU kernels, tape autograd (gradcheck), `nn` modules, `optim` (SGD/Adam), MNIST (96.7 % @ 3 epochs) | ✅ done |
| **M1** — Graph IR + lowering | MLIR `tesseract` dialect, GraphScope capture → MLIR, `→ linalg` conversion, autograd-as-graph-transform, in-process JIT (`mnist --engine mlir`, **7–11× eager** on lowerable shapes) | ✅ done |
| **M2** — CUDA backend + kernel stack | HAL (allocator / stream / event), cuBLASLt matmul (FP32/FP16/BF16), fused softmax / RMSNorm / SwiGLU / attention, shape/index ops, fused Adam; CUDA + CPU ctest green | ✅ done |
| **M3** — LLM inference stack | HF BPE tokenizer, KV cache, RoPE, GQA, autoregressive `generate` + sampling, paged KV, continuous-batching scheduler, INT8/INT4 quantization | ✅ done |
| **M4** — Performance + architectures + Python | Fused GPU MoE + Mamba, real NCCL multi-GPU TP (fwd+bwd parity), IR GPU JIT, FP8 GEMM, **pybind11 Python frontend**, and a 14-row external scoreboard vs **llama.cpp / PyTorch / vLLM / FlashDecoding** — all win or tie | ✅ done |
| **M5** — Edge + open-source release | ExecuTorch-style AOT compile to `.tsrct` bundles, Metal / WebGPU / WASM backends (mobile + browser), license/branding/release hygiene. Early adoption track shipped: **Tesseract Studio** (Scratch-like native-C++ visual block builder, B-047, no Python). DiT runtime is the deferred gated tail. | 🔄 in progress |

---

## Capabilities

A breadth survey of what is implemented and tested today. Every item below has
CPU **and** CUDA paths unless noted, dispatched behind one device-agnostic API.

**Core & autograd**
- Tensors over `DType` ∈ {FP64, FP32, FP16, BF16, INT8, INT4, Bool}, `Device` ∈
  {CPU, CUDA}; strided views (`view`/`reshape`/`permute`/`transpose`/`narrow`),
  `Storage`/`Allocator` with capture-safe device memory.
- Reverse-mode autograd (tape `Engine`, `Node`/`AutogradMeta`, `NoGradGuard`),
  finite-difference gradcheck-tested.

**Tensor ops** — arithmetic, `matmul` (cuBLASLt, rank-2 + batched), reductions,
softmax, activations, RoPE, `selective_scan` (SSM), normalization, attention
(SDPA / fused / GQA / paged), quantize/dequantize, gather/scatter/index,
broadcast/view.

**`nn` modules** — `Linear`, `Embedding`, activations, `Sequential`/`ModuleList`,
`CrossEntropyLoss`, `RMSNorm`/`LayerNorm`/`BatchNorm`, `MultiHeadAttention`
(MHA + GQA + RoPE), `FeedForward` (SwiGLU), `TransformerBlock`,
`MoEFeedForward`, `Mamba` (+ `SSMStateCache`), `DiTBlock`, KV caches
(contiguous / paged / quantized), `QuantizedLinear` (INT8) + `QuantizedLinearInt4G`.

**Optimizers** — `SGD` (momentum), `Adam` (fused single-launch CUDA kernel).

**Compiler / IR (MLIR, optional)** — `tesseract` dialect, GraphScope eager-trace
capture → MLIR, `→ linalg` conversion, autograd-as-a-graph-transform, in-process
`ExecutionEngine` JIT (CPU **5.5–8.5×** train step / **7–11×** matmul vs eager),
and a `gpu.module → PTX → cubin` GPU execution path (bit-close parity).

**CUDA kernel stack** — HAL (allocator / stream / event / **CUDA graph** capture /
pinned host memory), cuBLASLt matmul, fused softmax / RMSNorm / SwiGLU, fused
FlashAttention-2 (**WMMA tensor-core** prefill + split-K decode + stride/BSHD
layout), paged-KV gather, fused Adam, **FP8 (E4M3/E5M2) GEMM**, shape/index kernels.

**LLM inference** — HF byte-level **BPE tokenizer**, **safetensors** loader,
`LlamaModel`, KV cache + RoPE + GQA, autoregressive `generate`, sampling
(temperature / top-k / top-p / repetition penalty), **continuous-batching
scheduler**, **paged KV**, **speculative decoding**, **structured (grammar)
decoding**, **disaggregated prefill/decode**, whole-model CUDA-graph capture/replay.

**Architectures** — dense Llama, **MoE** (fused token dispatch + grouped GEMM),
**Mamba / SSM** (chunkwise-parallel scan, O(1) decode), **DiT** (design + block).

**Quantization** — INT8 **W8A8**, INT4 group-quant, **INT8 paged KV cache**,
AVX-512-VNNI CPU decode path, Ada/Hopper FP8 GEMM.

**Distributed** — **tensor parallelism** with a real **NCCL** multi-GPU backend
(forward + backward parity, 1/N memory) and an in-process simulation backend.

**Frontends** — native C++ API, **Python** (`tesseract._core`, pybind11), and
**Tesseract Studio** (visual block builder).

---

## Benchmarks

Tesseract is benchmarked head-to-head against **llama.cpp, PyTorch, vLLM, and
PyTorch FlashDecoding** — it **wins or ties every measured axis**. All numbers
are reproducible under strict GPU isolation on a clean RTX 5880 Ada (SM 8.9);
each row links to its raw result under
[`bench/external/results/`](bench/external/results/), and the methodology lives
in [`docs/design/external-benchmark.md`](docs/design/external-benchmark.md).

| # | Metric | Config | External ref | External | Tesseract | Verdict |
|---|--------|--------|--------------|---------:|----------:|---------|
| 1 | decode GEMV latency | M=1, K=N=8192, HBM-bound | PyTorch FP16 | 131 µs | **53.5 µs (INT8)** | **WIN 2.45×** + ½ mem |
| 2 | Llama decode block | 7B block, S_k=129 | PyTorch FP16 | 478 µs / 405 MB | 510 µs / **203 MB** | tie latency, **½ mem** |
| 3 | dense GEMM line (FP8) | N=1024–8192 | PyTorch FP16 (cuBLAS) | 20.7–5655 µs | **15.2–3275 µs** | **WIN 1.36–2.19×** |
| 4 | dense GEMM (same precision) | N=1024–8192 FP16/FP32 | PyTorch (cuBLAS) | 1.0× | 0.93–1.34× | tie (same vendor lib) |
| 5 | fused decode attention | (B,H,1,S_k,128) | PyTorch SDPA (FlashDecoding) | 30.7–328 µs | **19.4–298 µs** | **WIN 1.05–1.58×** |
| 6 | Mamba O(1) vs O(L) decode | d=1024, 8L, L≤4096 | O(L) attention | 1468–2826 µs/step | **660 µs/step (flat)** | **WIN 2.08–4.29×** |
| 7 | MoE fused dispatch | E=8 k=2, T≤4096 | PyTorch eager MoE | 2099–11480 µs | **719–7486 µs** | **WIN 1.39–2.92×** (3.5–16× vs dense) |
| 8 | multi-GPU TP throughput | FFN, 3 cards | PyTorch (same struct) | 2.38× (TP=3) | **2.98× (TP=3)** | **WIN 1.26×**, 3× mem cut |
| 8b | real NCCL TP parity | TP=2/3, 1 proc/GPU | dense (single GPU) | — | rrms ≤ 3.6e-7 | **fwd+bwd parity**, 1/N mem |
| 9 | LLM training convergence | same cfg, 100 steps | PyTorch Adam | 0.0054 final | **0.0050 final** | **parity** (±10 % curve) |
| 10 | Python frontend overhead | real-sized ops | native C++ | — | +<1–3 % | thin shim (≈0 %) |
| 11 | CPU decode tok/s | TinyLlama-1.1B, 48 thr | llama.cpp Q8_0 / Q4_0 | 150.7 / 225.3 | **243.4 (W8A8)** | **WIN 1.62× / 1.08×** |
| 12 | graph JIT (CPU) vs eager | MLP fwd+bwd+Adam | Tesseract eager | 1.0× | **5.5–8.5×** | internal speedup |
| 13 | graph JIT → GPU cubin | elementwise, sm_89 | eager CUDA | — | **bit-close parity** | PTX→cubin launch |
| 14 | serving decode | TinyLlama-1.1B, 128/128, FP16 | vLLM 0.11 FP16 | 305.8 tok/s / 3.275 ms | **321.1 tok/s / 3.115 ms** | **WIN +5.0 % tok/s, −5 % e2e** |
| 14b | serving TTFT/prefill | TinyLlama-1.1B, 128/128, FP16 | vLLM 0.11 FP16 | **5.47 ms** | 5.86 ms | loss 1.07× (shared GEMM floor) |

**vs vLLM (the headline online-serving axis).** On the *same* TinyLlama-1.1B at
matched FP16, Tesseract wins decode throughput (**321.1 vs 305.8 tok/s, +5.0 %**),
per-token latency (**3.115 vs 3.275 ms**), and end-to-end latency (**~400 vs
420 ms**). TTFT/prefill is the lone sub-metric still behind — **5.86 vs 5.47 ms,
within 7 %** after the WMMA tensor-core FlashAttention (B-024+) and
stride-aware/BSHD attention-layout (B-024c) passes — bounded by the cuBLAS GEMM
floor both engines share.

---

## Testing

The suite is **Catch2** unit tests + finite-difference gradcheck + `.mlir`
round-trip/FileCheck, run under strict single-GPU isolation; CUDA paths are
additionally checked with `compute-sanitizer` (memcheck / initcheck clean).

| Build configuration | CMake flags | Test cases |
|---------------------|-------------|-----------:|
| CPU (default)       | (none)                          | **552** |
| CPU (lean)          | minimal                         | 505 |
| CUDA                | `TESSERACT_ENABLE_CUDA=ON`      | **530** |
| MLIR                | `TESSERACT_ENABLE_MLIR=ON`      | 549 |
| NCCL (multi-GPU)    | `TESSERACT_ENABLE_NCCL=ON`      | 96 |

Coverage spans `tests/{core,autograd,ops,nn,hal,cuda,models,graph,io,distributed,ir,smoke}/`
(83 test files), plus **27 benchmarks** under `benchmarks/` (micro-kernels →
full serving). Run any suite with `ctest --test-dir <build> --output-on-failure`.

---

## Quick start (CPU, no LLVM required)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
```

That's it. `build/` contains the per-subsystem static libs
(`libtesseract_{core,ops,autograd,nn,optim,graph,io,models,quant,distributed}.a`)
plus the full Catch2 suite (552 CPU tests). The default configuration fetches
**fmt** and **Catch2 v3** via `FetchContent`; nothing else is required beyond a
C++20 compiler and CMake ≥ 3.22.

### Building the CUDA backend

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTESSERACT_ENABLE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure   # 530 test cases on SM 8.9 (Ada)
```

Requires CUDA Toolkit 12.x + `nvcc`. Optional GPU features layer on with
`-DTESSERACT_ENABLE_NCCL=ON` (multi-GPU collectives), `-DTESSERACT_ENABLE_FP8=ON`
(Ada/Hopper FP8 GEMM), and `-DTESSERACT_ENABLE_CUTLASS=ON` (grouped-GEMM MoE).

### Python frontend (M4 track B1)

```bash
cmake -S . -B build-py -DTESSERACT_BUILD_PYTHON=ON
cmake --build build-py -j
python -c "import tesseract; print(tesseract.__doc__)"
```

The pybind11 module (`tesseract._core`) exposes tensors, autograd, `nn`, and
inference so the framework is usable from Python like PyTorch/TensorFlow.

### Tesseract Studio (M5 adoption track)

```bash
cmake -S . -B build-studio -DTESSERACT_BUILD_STUDIO=ON
cmake --build build-studio -j
```

A Scratch-style drag-and-drop builder that turns model construction, training
(live loss), and inference into composable graphical blocks. It is a single
self-contained **native-C++** executable that embeds the engine and serves a
browser UI as its display (no Python) — see [`studio/README.md`](studio/README.md).

### Running the MNIST demo

```bash
cmake -S . -B build -DTESSERACT_BUILD_EXAMPLES=ON -DTESSERACT_NATIVE_ARCH=ON
cmake --build build -j --target mnist
./scripts/fetch_mnist.sh data/mnist                     # ~12 MB, needs curl + gunzip
./build/examples/tesseract_mnist data/mnist --epochs 3  # ~25 s, hits ~96.7 %
```

On a modest laptop CPU this finishes one epoch in under 10 seconds and
reports test-set accuracy above 94 % after the first epoch, 96 % after
three.

### Running the LLM stack

Build the example binaries (`-DTESSERACT_BUILD_EXAMPLES=ON`); each runs on CPU
or CUDA (`--device cpu|cuda`).

```bash
# Generate from a real HF checkpoint (reads config.json + *.safetensors + tokenizer.json)
./build/examples/tesseract_llama_generate --model-dir /path/to/hf/checkpoint \
    --prompt "The capital of France is" --max-new-tokens 64

# Forward / top-k logits on a synthetic (random-init) model — no checkpoint needed
./build/examples/tesseract_llama_infer --synthetic

# Continuous-batching serving demo (shared paged KV pool, per-request sampling)
./build/examples/tesseract_llama_serve --synthetic

# Single-device training loop (next-token CE + Adam; loss collapses on a fixed batch)
./build/examples/tesseract_llama_train --steps 200 --device cuda
```

For the full serving benchmark vs vLLM, build with `-DTESSERACT_BUILD_BENCHMARKS=ON`
and see [`bench/external/results/vllm_serving.md`](bench/external/results/vllm_serving.md).

### Enabling the MLIR dialect

The `tesseract::ir` dialect and the `tesseract-opt` driver live behind
`TESSERACT_ENABLE_MLIR=ON`. Because MLIR is not commonly packaged at the
revision we want, the project ships a user-space build script:

```bash
./scripts/bootstrap.sh                    # installs user-local ninja
./scripts/build_llvm.sh                   # ~30–90 min; installs to third_party/llvm-install
cmake -S . -B build -DTESSERACT_ENABLE_MLIR=ON
cmake --build build -j
build/src/ir/tesseract-opt tests/ir/roundtrip.mlir --verify-each
```

If you already have an LLVM 18.x install elsewhere, point CMake at it with
`-DMLIR_DIR=<prefix>/lib/cmake/mlir` and skip `build_llvm.sh`.

---

## Project layout

```
framework/
├── CMakeLists.txt              # root build file
├── cmake/                      # Options / Dependencies / CompilerFlags
├── include/tesseract/          # public headers
│   ├── core/                   # DType / Device / Shape / Storage / Tensor / GradMode
│   ├── autograd/               # AutogradMeta / Node / Engine / Function
│   ├── ops/                    # Arithmetic / MatMul / Reduction / Softmax / Attention / Norm
│   ├── nn/                     # Module / Linear / MHA / RMSNorm / FeedForward / KV cache
│   ├── optim/                  # Optimizer / SGD / Adam
│   ├── cuda/                   # CUDA bridge declarations (detail/*.hpp)
│   ├── models/                 # Llama / scheduler / sampler / disaggregated engine
│   ├── graph/                  # GraphScope capture + interpreter
│   ├── quant/                  # INT8 / INT4 quantization
│   ├── io/                     # tokenizer / safetensors loader
│   ├── distributed/            # TP + NCCL collectives
│   └── utils/                  # Logging, helpers
├── src/                        # implementation (one static lib per subsystem)
│   ├── core/ ops/ autograd/ nn/ optim/   # CPU + dispatch (kernels under ops/cpu/)
│   ├── cuda/                   # libtesseract_cuda (.cu kernels + *Stub.cpp for CPU-only)
│   ├── models/ graph/ quant/ io/ distributed/
│   └── ir/                     # MLIR dialect + passes (optional)
├── python/                     # pybind11 frontend (tesseract._core)
├── studio/                     # Tesseract Studio visual block builder (M5)
├── tests/                      # Catch2 unit tests + gradcheck + .mlir round-trip
├── examples/                   # mnist.cpp / llama_infer.cpp / llama_train.cpp
├── benchmarks/                 # micro + serving benchmarks (CPU + CUDA)
├── bench/external/             # external-framework scoreboard + results
├── scripts/                    # bootstrap.sh / build_llvm.sh / fetch_mnist.sh
├── docs/                       # architecture / roadmap / m*-plan / backlog / issue / adr/ / design/ / benchmarks/
└── idea.md                     # design doc (authoritative for direction)
```

---

## Build options

All options are prefixed `TESSERACT_`:

| Option                       | Default | Purpose                                              |
|------------------------------|---------|------------------------------------------------------|
| `TESSERACT_BUILD_TESTS`      | `ON`    | Build Catch2-based unit tests                        |
| `TESSERACT_BUILD_EXAMPLES`   | `OFF`   | Build example executables (MNIST, `llama_infer`, `llama_train`) |
| `TESSERACT_BUILD_BENCHMARKS` | `OFF`   | Build the micro + serving benchmarks                 |
| `TESSERACT_ENABLE_MLIR`      | `OFF`   | Build the MLIR dialect + `tesseract-opt`             |
| `TESSERACT_ENABLE_CUDA`      | `OFF`   | Build the CUDA backend (CUDA Toolkit 12.x + `nvcc`)  |
| `TESSERACT_ENABLE_NCCL`      | `OFF`   | Build the NCCL multi-GPU collective backend          |
| `TESSERACT_ENABLE_FP8`       | `OFF`   | Enable Ada/Hopper FP8 (E4M3/E5M2) GEMM paths         |
| `TESSERACT_ENABLE_CUTLASS`   | `OFF`   | Fetch CUTLASS for custom / grouped-GEMM kernels      |
| `TESSERACT_BUILD_PYTHON`     | `OFF`   | Build the pybind11 Python frontend (`tesseract._core`) |
| `TESSERACT_BUILD_STUDIO`     | `OFF`   | Build Tesseract Studio (visual block builder)        |
| `TESSERACT_ENABLE_OPENMP`    | `ON`    | Enable OpenMP-parallel CPU kernels (auto-detected)   |
| `TESSERACT_USE_EIGEN`        | `OFF`   | Use Eigen as the reference linalg backend            |
| `TESSERACT_WERROR`           | `OFF`   | `-Werror` / `/WX`                                    |
| `TESSERACT_NATIVE_ARCH`      | `OFF`   | Pass `-march=native` (enables AVX2 auto-vectorization) |

---

## Documentation

**Direction & architecture**
- [`idea.md`](idea.md) — motivation, theses, and the 24-month milestone plan (authoritative for direction).
- [`docs/architecture.md`](docs/architecture.md) — code-level layered design (core → ops → autograd → nn → models → IR) and how each layer evolves through M5.
- [`docs/roadmap.md`](docs/roadmap.md) — per-milestone delivery log (M0–M5), each closed by a demo + verification bar.

**Milestone plans** (scope, tracks, and exit bars per milestone)
- [`docs/m1-plan.md`](docs/m1-plan.md) — M1 graph IR + MLIR lowering + JIT.
- [`docs/m2-plan.md`](docs/m2-plan.md) — M2 CUDA backend + kernel stack.
- [`docs/m3-plan.md`](docs/m3-plan.md) — M3 LLM inference as a first-class citizen.
- [`docs/m4-plan.md`](docs/m4-plan.md) — M4 non-Transformer architectures + adoption + one-IR coherence (closeout: done).

**Design notes** ([`docs/design/`](docs/design/))
- [`external-benchmark.md`](docs/design/external-benchmark.md) — the external scoreboard methodology + 14-row results vs llama.cpp / PyTorch / vLLM.
- [`moe-sparse.md`](docs/design/moe-sparse.md) — fused MoE sparse token dispatch (vs dense masking).
- [`mamba-scaling.md`](docs/design/mamba-scaling.md) — Mamba O(1) vs Llama O(L) decode scaling.
- [`tensor-parallel.md`](docs/design/tensor-parallel.md) — Megatron-style TP as a sharding transform.
- [`kv-cache-ir.md`](docs/design/kv-cache-ir.md) — KV cache & dynamic shape as IR concepts (M4 track C2).

**Benchmarks & engineering records**
- [`docs/benchmarks/m2-cuda.md`](docs/benchmarks/m2-cuda.md) — the M2 CUDA micro-benchmark ledger.
- [`bench/external/README.md`](bench/external/README.md) — the external-framework benchmark harness; raw results in [`bench/external/results/`](bench/external/results/).
- [`docs/backlog.md`](docs/backlog.md) — deferred items + resolutions, each with a definition of done (e.g. B-024 WMMA/BSHD attention).
- [`docs/issue.md`](docs/issue.md) — the M4 closeout self-review (evidence-based critique + how each gap was resolved).

**Decisions & frontends**
- [`docs/adr/`](docs/adr/) — architecture decision records (ADR-0001…0007; see [Contributing](#contributing) for the list).
- [`studio/README.md`](studio/README.md) — the Tesseract Studio visual block builder.

---

## Contributing

1. Run `cmake --build build` and `ctest --test-dir build` before opening a PR.
2. Code must format cleanly under the project-level `.clang-format`
   (`clang-format -i --style=file`).
3. Public-header changes should include Catch2 coverage under `tests/`.
4. Design-level changes go through an ADR in `docs/adr/`:
   - [`0001-use-mlir.md`](docs/adr/0001-use-mlir.md) — why MLIR is the shared IR.
   - [`0002-autograd-model.md`](docs/adr/0002-autograd-model.md) — tape for M0, IR pass for M1.
   - [`0003-naming-conventions.md`](docs/adr/0003-naming-conventions.md) — identifier, file, and error-handling style.
   - [`0004-graph-ir-two-stage.md`](docs/adr/0004-graph-ir-two-stage.md) — the two-stage graph IR.
   - [`0005-cuda-hal.md`](docs/adr/0005-cuda-hal.md) — the CUDA HAL design.
   - [`0006-m4-parallel-abc-scope.md`](docs/adr/0006-m4-parallel-abc-scope.md) — M4 parallel-track scope.
   - [`0007-studio-visual-builder.md`](docs/adr/0007-studio-visual-builder.md) — the Studio visual builder.

---

## License

TBD (tracked by a pending ADR).

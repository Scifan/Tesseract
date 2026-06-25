<div align="center">

<h1>
  Tesseract
  <img src="https://github.githubassets.com/images/icons/emoji/unicode/1f4a0.png" alt="💠" width="52" height="52" style="vertical-align: middle; margin-left: 10px;">
</h1>

## C++20 训练与推理统一的深度学习框架

*同一 IR、同一运行时 —— 从 MNIST 到 Llama 级 LLM，训练与服务一站搞定。*

<br>

![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-555?style=flat-square)
![C++](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![Backend](https://img.shields.io/badge/Backend-CPU%20%2B%20CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![IR](https://img.shields.io/badge/IR-MLIR-FF6F00?style=flat-square)
![Frontend](https://img.shields.io/badge/Frontend-Python%20%7C%20Studio-4CAF50?style=flat-square)
![Benchmark](https://img.shields.io/badge/Decode%20vs%20vLLM-+5.0%25-43A047?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-552%20CPU%20%7C%20529%20CUDA-2196F3?style=flat-square)
![Status](https://img.shields.io/badge/Status-M5%20进行中-FFC107?style=flat-square)

<br>

[架构设计](docs/architecture.md) · [快速开始](#quick-start-cpu-no-llvm-required) · [里程碑](#status) · [Studio](studio/README.md) · [性能基准](docs/design/external-benchmark.md) · [文档](#documentation)

</div>

---

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


| Milestone                                      | Scope                                                                                                                                                                                                                                                                                              | Status         |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **M0** — Foundation                           | Core tensor types, CPU kernels, tape autograd (gradcheck),`nn` modules, `optim` (SGD/Adam), MNIST (96.7 % @ 3 epochs)                                                                                                                                                                              | ✅ done        |
| **M1** — Graph IR + lowering                  | MLIR`tesseract` dialect, GraphScope capture → MLIR, `→ linalg` conversion, autograd-as-graph-transform, in-process JIT (`mnist --engine mlir`, **7–11× eager** on lowerable shapes)                                                                                                            | ✅ done        |
| **M2** — CUDA backend + kernel stack          | HAL (allocator / stream / event), cuBLASLt matmul (FP32/FP16/BF16), fused softmax / RMSNorm / SwiGLU / attention, shape/index ops, fused Adam; CUDA + CPU ctest green                                                                                                                              | ✅ done        |
| **M3** — LLM inference stack                  | HF BPE tokenizer, KV cache, RoPE, GQA, autoregressive`generate` + sampling, paged KV, continuous-batching scheduler, INT8/INT4 quantization                                                                                                                                                        | ✅ done        |
| **M4** — Performance + architectures + Python | Fused GPU MoE + Mamba, real NCCL multi-GPU TP (fwd+bwd parity), IR GPU JIT, FP8 GEMM,**pybind11 Python frontend**, and a 14-row external scoreboard vs **llama.cpp / PyTorch / vLLM / FlashDecoding** — all win or tie                                                                            | ✅ done        |
| **M5** — Edge + open-source release           | ExecuTorch-style AOT compile to`.tsrct` bundles, Metal / WebGPU / WASM backends (mobile + browser), license/branding/release hygiene. Early adoption track shipped: **Tesseract Studio** (Scratch-like native-C++ visual block builder, B-047, no Python). DiT runtime is the deferred gated tail. | 🔄 in progress |

### Performance highlights (measured, reproducible under strict GPU isolation)

- **vs vLLM 0.11 (FP16, TinyLlama-1.1B, single-stream serving):** decode
  **321.1 vs 305.8 tok/s (+5.0 %)**, TPOT **3.115 vs 3.275 ms**, end-to-end
  **~400 vs 420 ms** — all wins. TTFT/prefill **5.86 vs 5.47 ms** (within 7 %)
  after the WMMA tensor-core FlashAttention (B-024+) and stride-aware/BSHD
  attention-layout (B-024c) passes; bounded by the shared cuBLAS GEMM floor.
- **External scoreboard:** 14 rows vs llama.cpp / PyTorch / vLLM / FlashDecoding /
  FP8 / NCCL — every row a win or tie. Detail in
  [`docs/design/external-benchmark.md`](docs/design/external-benchmark.md) and
  [`bench/external/results/`](bench/external/results/).
- **Test status:** CUDA build 529/529, CPU build 552/552 Catch2 tests green.

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
ctest --test-dir build-cuda --output-on-failure   # 529/529 green on SM 8.9 (Ada)
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
├── docs/                       # architecture.md / roadmap.md / m*-plan.md / adr/ / design/
└── idea.md                     # design doc (authoritative for direction)
```

---

## Build options

All options are prefixed `TESSERACT_`:


| Option                       | Default | Purpose                                                        |
| ------------------------------ | --------- | ---------------------------------------------------------------- |
| `TESSERACT_BUILD_TESTS`      | `ON`    | Build Catch2-based unit tests                                  |
| `TESSERACT_BUILD_EXAMPLES`   | `OFF`   | Build example executables (MNIST,`llama_infer`, `llama_train`) |
| `TESSERACT_BUILD_BENCHMARKS` | `OFF`   | Build the micro + serving benchmarks                           |
| `TESSERACT_ENABLE_MLIR`      | `OFF`   | Build the MLIR dialect +`tesseract-opt`                        |
| `TESSERACT_ENABLE_CUDA`      | `OFF`   | Build the CUDA backend (CUDA Toolkit 12.x +`nvcc`)             |
| `TESSERACT_ENABLE_NCCL`      | `OFF`   | Build the NCCL multi-GPU collective backend                    |
| `TESSERACT_ENABLE_FP8`       | `OFF`   | Enable Ada/Hopper FP8 (E4M3/E5M2) GEMM paths                   |
| `TESSERACT_ENABLE_CUTLASS`   | `OFF`   | Fetch CUTLASS for custom / grouped-GEMM kernels                |
| `TESSERACT_BUILD_PYTHON`     | `OFF`   | Build the pybind11 Python frontend (`tesseract._core`)         |
| `TESSERACT_BUILD_STUDIO`     | `OFF`   | Build Tesseract Studio (visual block builder)                  |
| `TESSERACT_ENABLE_OPENMP`    | `ON`    | Enable OpenMP-parallel CPU kernels (auto-detected)             |
| `TESSERACT_USE_EIGEN`        | `OFF`   | Use Eigen as the reference linalg backend                      |
| `TESSERACT_WERROR`           | `OFF`   | `-Werror` / `/WX`                                              |
| `TESSERACT_NATIVE_ARCH`      | `OFF`   | Pass`-march=native` (enables AVX2 auto-vectorization)          |

---

## Documentation

- [`idea.md`](idea.md) — motivation, theses, and the 24-month milestone plan (authoritative for direction).
- [`docs/architecture.md`](docs/architecture.md) — the layered design (HAL → ops → autograd → nn → models).
- [`docs/roadmap.md`](docs/roadmap.md) — per-milestone delivery log (M0–M5).
- [`docs/m4-plan.md`](docs/m4-plan.md) — the M4 performance-closeout plan + exit bar.
- [`docs/backlog.md`](docs/backlog.md) — the engineering backlog (open + resolved items, e.g. B-024 attention).
- [`docs/design/external-benchmark.md`](docs/design/external-benchmark.md) — the external scoreboard methodology + 14-row results.
- [`docs/adr/`](docs/adr/) — architecture decision records.
- [`studio/README.md`](studio/README.md) — the Tesseract Studio visual builder.

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

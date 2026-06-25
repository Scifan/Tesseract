# Tesseract

> A C++20 deep-learning framework with a training–inference-unified IR from day one.

Tesseract is an experimental deep-learning framework written in modern C++ with
first-class MLIR integration. The immediate goal is to deliver a compiler-first
training and inference stack in which the same IR, runtime, and kernel
implementations are reused across eager training and served inference.

See [`idea.md`](idea.md) for the motivation and 24-month roadmap, and
[`docs/architecture.md`](docs/architecture.md) for the layered design.

---

## Status

This repository is at **M0 (Foundation)**. The current drop contains:

| Area               | Status |
|--------------------|--------|
| Core tensor types (`DType`, `Device`, `Shape`, `Allocator`, `Storage`, `Tensor`) | ✅ implemented + unit-tested |
| Shape ops (`view` / `reshape` / `permute` / `transpose` / `contiguous` / `clone`) | ✅ implemented + unit-tested |
| CPU kernels (add / sub / mul / div / neg / matmul / reductions / activations / softmax / cross-entropy) | ✅ implemented + unit-tested |
| Autograd engine (`AutogradMeta`, `Node`, `Engine::backward`, `NoGradGuard`) | ✅ implemented + gradcheck-tested |
| `nn::Module` / `nn::Linear` / `nn::ReLU` / `nn::Sequential` / `nn::CrossEntropyLoss` | ✅ implemented + regression-tested |
| `optim::SGD` (with momentum) / `optim::Adam` | ✅ implemented + regression-tested |
| MNIST example (`examples/mnist.cpp` + `scripts/fetch_mnist.sh`, `--epochs N`) | ✅ **94.4 % / 96.1 % / 96.7 % at epochs 1/2/3 (CPU)** |
| `bench_matmul` micro-benchmark vs naive baseline | ✅ **~33× faster at 512×512 FP32 (32.7 vs 1.2 GFLOP/s, OpenMP + `-march=native`)** |
| MLIR dialect (`tesseract.constant` / `add` / `matmul`) skeleton | ✅ wired behind `TESSERACT_ENABLE_MLIR=ON` |
| CUDA / ROCm backend | ⏳ M2 |

---

## Quick start (CPU, no LLVM required)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
```

That's it. `build/` contains `libtesseract_{core,ops,autograd,nn,optim}.a`
plus 75 Catch2 tests. The default configuration fetches **fmt** and
**Catch2 v3** via `FetchContent`; nothing else is required beyond a C++20
compiler and CMake ≥ 3.22.

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
│   ├── autograd/               # AutogradMeta / Node / Engine / Function
│   ├── core/                   # DType / Device / Shape / Storage / Tensor / GradMode
│   ├── nn/                     # Module / Linear / Activation / Sequential / Loss
│   ├── ops/                    # Arithmetic / MatMul / Reduction / Activation / Softmax / Loss
│   ├── optim/                  # Optimizer / SGD / Adam
│   └── utils/                  # Logging, helpers
├── src/                        # implementation (one static lib per subsystem)
│   ├── core/                   # libtesseract_core
│   ├── ops/                    # libtesseract_ops (CPU kernels under ops/cpu/)
│   ├── autograd/               # libtesseract_autograd
│   ├── nn/                     # libtesseract_nn
│   ├── optim/                  # libtesseract_optim
│   └── ir/                     # MLIR dialect (optional)
├── tests/                      # Catch2 unit tests + gradcheck + .mlir round-trip
├── examples/                   # mnist.cpp
├── benchmarks/                 # bench_matmul.cpp
├── scripts/                    # bootstrap.sh / build_llvm.sh / fetch_mnist.sh
├── docs/                       # architecture.md / roadmap.md / adr/
└── idea.md                     # design doc (authoritative for direction)
```

---

## Build options

All options are prefixed `TESSERACT_`:

| Option                       | Default | Purpose                                              |
|------------------------------|---------|------------------------------------------------------|
| `TESSERACT_BUILD_TESTS`      | `ON`    | Build Catch2-based unit tests                        |
| `TESSERACT_BUILD_EXAMPLES`   | `OFF`   | Build `examples/mnist.cpp` (run with `scripts/fetch_mnist.sh`) |
| `TESSERACT_BUILD_BENCHMARKS` | `OFF`   | Build `benchmarks/bench_matmul.cpp`                  |
| `TESSERACT_ENABLE_MLIR`      | `OFF`   | Build the MLIR dialect + `tesseract-opt`             |
| `TESSERACT_ENABLE_OPENMP`    | `ON`    | Enable OpenMP-parallel CPU kernels (auto-detected)   |
| `TESSERACT_USE_EIGEN`        | `OFF`   | Use Eigen as the reference linalg backend            |
| `TESSERACT_WERROR`           | `OFF`   | `-Werror` / `/WX`                                    |
| `TESSERACT_NATIVE_ARCH`      | `OFF`   | Pass `-march=native` (enables AVX2 auto-vectorization) |

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

---

## License

TBD (tracked by ADR pending the M3 open-source milestone).

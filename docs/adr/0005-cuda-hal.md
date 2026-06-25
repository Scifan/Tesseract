# ADR-0005: CUDA as an opt-in HAL backend

- **Status:** Accepted (2026-04-18)
- **Supersedes:** none
- **Relates to:** ADR-0001 (MLIR as shared IR), ADR-0004 (two-stage graph IR)
- **Authors:** tesseract core team

## Context

M0 + M1 have delivered a complete CPU training stack: eager + graph IR +
MLIR JIT, 159/159 ctests, software-emulated FP16/BF16 (B-005). M2 adds
a CUDA backend. The question is *how* the CUDA code is layered — this
decision affects every build configuration, every public header, and
every op dispatch for the rest of the project's life.

Three tensions define the design space:

1. **Build friction.** `nvcc` is slow, demands a matching host compiler,
   requires 10–30 GB of CUDA Toolkit artefacts, and is the single most
   common reason contributors can't compile the project locally. Forcing
   CUDA on the CPU developer loop would destroy the fast iteration the
   M0/M1 workflow enjoys today.
2. **Header cleanliness.** Exposing `<cuda_runtime.h>` or any
   `__host__ __device__` declaration in a public header transitively
   infects every downstream consumer — including the `tesseract-opt`
   tool, the tests, and future Python bindings — with a hard CUDA
   Toolkit dependency.
3. **Dispatch clarity.** PyTorch's ATen uses a dynamic vtable to
   dispatch the same `at::add` call to CPU, CUDA, MPS, etc. It's a
   powerful pattern but requires significant infrastructure
   (boxed/unboxed kernels, Dispatcher singleton, fallback rules) that
   we don't need at M2's ~15-op scope.

Prior art:

- **PyTorch ATen.** Dynamic dispatcher + per-backend registration. Heavy
  but general. Justified by their 2000+-op surface.
- **tinygrad.** Pure-Python dispatch on `tensor.device.name`.
  Lightweight but Python-only.
- **Burn (Rust).** Backend is a generic parameter of `Tensor<B, ...>`,
  type-checked at compile time. Elegant, but requires every user call
  site to be monomorphised per backend.
- **Candle (Rust).** Enum-based `Storage::Cpu | Cuda | Metal` with a
  match in each op. Simple, fast, small — and exactly what the current
  `ops::foo` dispatch on `tensor.device().type` already does in
  Tesseract's CPU-only codebase.

## Decision

CUDA is introduced as an **opt-in HAL backend** following four rules.
Together they give us a production-quality GPU path without surrendering
the CPU build's fast-iteration properties.

### 1. Build-time opt-in via `TESSERACT_ENABLE_CUDA=OFF` (default)

A new CMake option gates every CUDA-touching source file and every
`find_package(CUDAToolkit)` call. When OFF:

- No `nvcc`, no CUDA headers, no `cublas*` symbols linked.
- `DeviceType::CUDA` stays in the enum so APIs that take a `Device`
  argument compile, but `default_allocator_for(cuda_device())` throws a
  clean error at the HAL boundary.
- Every op's CUDA dispatch entry is a stub that throws; the public
  `ops::foo(cuda_tensor)` never silently produces a wrong answer.

When ON, the `TESSERACT_HAS_CUDA=1` define is the **single source of
truth** for `#ifdef`-gated code. No `CUDA_VERSION` or
`__CUDACC__` checks above the HAL layer.

This mirrors exactly how `TESSERACT_ENABLE_MLIR` is gated today.

### 2. CUDA code lives in `src/cuda/`; public headers stay plain C++20

`.cu` files and any `<cuda_runtime.h>` / `<cublasLt.h>` include live
under `src/cuda/` and produce a `tesseract_cuda` static library
compiled with `set_source_files_properties(... LANGUAGE CUDA)` and
`CXX_STANDARD 17` (matching cuBLAS's requirements as of CUDA 12.5).

`tesseract_cuda` is linked into the top-level `tesseract` target with
`PRIVATE` visibility. Downstream consumers — tests, benchmarks, the
tool, future Python bindings — include only `include/tesseract/*.hpp`
and see a pure C++20 surface. They do not need `nvcc` to compile.

The bridge between CPU-dispatch code and CUDA kernels goes through
forward-declared free functions in the internal header
`src/cuda/Launch.hpp`:

```cpp
// src/cuda/Launch.hpp (not in include/)
namespace tesseract::cuda {
  void launch_add(const Tensor& a, const Tensor& b, Tensor& out, Stream s);
  // ... one per op ...
}
```

Each CPU-side `ops::foo(Tensor)` entry point switches on
`tensor.device().type` and either runs the existing CPU kernel or
calls `cuda::launch_foo`. This keeps dispatch visible at the call site
and preserves the single-file CPU path for debuggability.

### 3. Dispatch is a switch statement, not a vtable

For M2's ~20-op surface, a `switch (device.type)` block at the entry
point of each public op is both faster and clearer than a generic
dispatcher. Concretely:

```cpp
Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out = add_forward(a, b);       // CPU reference, throws on non-CPU
  // ... autograd wiring unchanged ...
  return out;
}

Tensor add_forward(const Tensor& a, const Tensor& b) {
  switch (a.device().type) {
    case DeviceType::CPU:  return cpu_add_forward(a, b);
#if defined(TESSERACT_HAS_CUDA)
    case DeviceType::CUDA: return cuda::add_forward(a, b);
#endif
    default: TESSERACT_THROW("add: unsupported device {}", a.device().to_string());
  }
}
```

This is the same pattern Candle uses and costs us one extra pair of
braces per op. When we outgrow it (likely during M5 with 4+ backends
live), we graduate to a generated dispatcher under a backlog item
(B-012 "Generated dispatch table"). We do **not** prematurely
introduce PyTorch-style boxed kernels.

### 4. Streams are explicit; per-thread default, no hidden global state

Every CUDA op takes its current stream from a **thread-local**
`StreamGuard` (mirrors `GradMode`). The default stream on CUDA is a
**non-blocking per-thread stream** created by
`cudaStreamCreateWithFlags(cudaStreamNonBlocking)` — *not*
`cudaStreamLegacy` and *not* `cudaStreamPerThread`'s implicit
per-thread mode (which conflates our streams with cuBLAS's). Tests
create their own guard so work never leaks into another test's
timeline.

On CPU, `Stream` is a trivial value type (no-op). This keeps the API
uniform without ifdefs at call sites.

Host-visible operations (`Tensor::item<T>()`, `Tensor::to_string`,
`Tensor::to(cpu_device())`) advertise in their doc comment that they
call `cudaStreamSynchronize` on the source stream. No op silently
synchronizes.

## Consequences

### Positive

- The CPU-only developer loop is never slower than it is today.
- Public headers stay compilable by any C++20 compiler without the
  CUDA Toolkit.
- Onboarding a second GPU backend (ROCm in M5) is a parallel directory
  (`src/hip/`) + a second switch case; no refactor of the HAL.
- The FP16/BF16 storage layout from B-005 matches the CUDA ABI
  exactly, so host `Tensor` → device GEMM needs no re-pack.
- The MLIR JIT (which is CPU-only through M2 per M2.γ scoping) does
  not block on CUDA work; likewise, eager CUDA does not block on the
  MLIR GPU backend (that work is explicitly M2L.2 stretch / M3).

### Negative

- Every new op touches two places (CPU dispatch + CUDA kernel) plus a
  parity test. This is a real friction cost; we accept it as the
  price of keeping dispatch visible and the HAL minimal.
- The switch-based dispatch does not scale to PyTorch's 2000+-op
  surface. We have ~20 ops today and expect ~50 by end of M3; if the
  count grows meaningfully past that, B-012 generates the dispatch
  table from a registry.
- Cross-backend tensors (e.g. `a.device() == CPU`, `b.device() ==
  CUDA`) throw early rather than silently inserting a copy. This
  matches PyTorch's explicit-`.to()` convention and avoids accidental
  host-device round-trips on the hot path.

### Deferred questions

- **Async copies are opt-in for M2.** `Tensor::to(device)` synchronizes
  on the current stream before returning. An async variant
  (`Tensor::to_async(device, stream)`) is deferred to M3 when the
  continuous-batching scheduler actually benefits from overlap.
- **Multi-GPU dispatch.** Even though `Device(DeviceType::CUDA, i)` is
  a valid handle today, M2 ships with a single-device scope
  (tensor.device().index is validated but we don't stress-test
  multi-GPU memory transfers). Distribution comes in M4.
- **Unified Memory (UM) support.** M2 uses only device memory. UM is
  tempting for debugging but has well-documented perf cliffs on
  pre-Hopper hardware; we revisit post-M2 if debuggability wins
  outweigh perf risk.

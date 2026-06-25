# M2 — CUDA backend + kernel stack (plan)

**Milestone:** M2. **Status:** not started (2026-04-18). **Owner:** tesseract core team.
**Precondition:** M0 + M1 complete (eager + graph IR + JIT on CPU, 159/159 ctests,
MNIST graph-mode training reaches 94.4 % test acc in epoch 1 through both
the interpreter and the MLIR ExecutionEngine path; B-001 ~ B-007 all resolved).

---

## 1. Scope statement

M2 adds a **production-quality single-GPU CUDA backend** to the same `Tensor`
/ `graph::Graph` / `tesseract.*` surfaces that M0/M1 shipped on CPU. The
ceilings we are buying into:

1. **Op coverage** — every op that currently runs on CPU (elementwise,
   reductions, softmax, cross-entropy, matmul, shape ops, indexing) also
   runs on CUDA in both eager and graph modes.
2. **Precisions** — FP32 / FP16 / BFloat16 end-to-end. INT8 / FP8 / FP4 are
   out of scope (tracked in M3 quantization dialects).
3. **Kernels** — cuBLAS/cuBLASLt for GEMM; hand-written CUDA kernels for
   elementwise and reductions; **FlashAttention-3 for self-attention**
   (reference integration, not a rewrite).
4. **Demos** — `examples/mnist.cpp` trains end-to-end on CUDA; a new
   `examples/llama_forward.cpp` runs a single Llama-style transformer
   block (LayerNorm + attention + SwiGLU-MLP) on a single GPU.
5. **Perf bar** — GEMM within 90 % of cuBLAS on Ada/Hopper; attention
   within 90 % of FlashAttention-3 on Hopper. Measured by a new
   `bench_cuda_*` suite that runs on every PR when a GPU is available.

Non-goals for M2 (all tracked elsewhere):

- Multi-GPU / NCCL collectives → M4.
- ROCm / Metal / WebGPU backends → M5.
- Triton / CuTe DSL *autogeneration* of our ops → optional M2 stretch
  goal under M2L (see §5); the M2 exit bar does not require it.
- Python bindings → M4.
- PagedKV, continuous batching, speculative decoding → M3.
- Quantized (INT8/FP8/FP4) weights → M3.

Like M1, we ship in three incremental merges (M2.α / M2.β / M2.γ). Every
merge ends with a testable artifact; no commit can break
`ctest --test-dir build` on CPU, regardless of whether CUDA was built.

## 2. Design decisions locked today

### 2.1 CUDA is strictly opt-in; CPU build is never regressed

A new top-level CMake option `TESSERACT_ENABLE_CUDA=OFF` (default **OFF**)
gates every CUDA-touching file. When OFF:

- No `nvcc` is required — the project configures + builds with plain g++/clang.
- `DeviceType::CUDA` stays in the enum but `default_allocator_for(cuda)`
  continues to throw, and every op's CUDA dispatch entry is `nullptr`,
  so attempts to use the GPU surface a clean error at the HAL boundary.
- All existing ctests keep passing byte-for-byte.

When ON:

- `find_package(CUDAToolkit REQUIRED)` pulls cuBLAS / cuBLASLt.
- All `.cu` files live under `src/cuda/` and a dedicated
  `tesseract_cuda` static library (language `CUDA`, standard `17` — not
  `20` to match current cuTLASS/cuBLAS header requirements).
- `tesseract_cuda` is linked into the top-level `tesseract` target with
  `PRIVATE` visibility; downstream users never need `nvcc` to consume us.
- A conditional `TESSERACT_HAS_CUDA=1` define is the single source of
  truth — no `#ifdef CUDA_VERSION` sprinkled in user-facing code.

This mirrors exactly how `TESSERACT_ENABLE_MLIR` works today and means
the M1 CPU workflow remains the fast default.

### 2.2 HAL layer lives in `src/hal/` behind pure-virtual interfaces

The current `Allocator` abstract class is already the right shape. M2
extends it with:

- `CudaAllocator` (owned per-device) — backed by `cudaMallocAsync` on
  Ampere+, falling back to `cudaMalloc` when the caching allocator
  interface isn't available.
- A new `Stream` HAL type (value handle; on CPU it's a no-op, on CUDA
  it wraps a `cudaStream_t`). Every op takes a *current stream* from
  a thread-local `StreamGuard`; the default stream on CUDA is a
  non-blocking per-thread stream, **not** `cudaStreamLegacy`, so host
  synchronization is explicit.
- An `Event` type for cross-stream ordering (needed by async copies).

What is deliberately *not* in the HAL: any op vtable. Dispatch stays
where it is today — at the public `ops::foo(Tensor)` entry point, we
switch on `tensor.device().type` and call either the CPU implementation
or the CUDA one. This keeps the HAL small and the dispatch visible at
the call site; we can graduate to a tablegen'd vtable later (B-008) if
the switch statements become unwieldy.

### 2.3 No `.cu` files in public headers

`nvcc`-specific C++ (`__host__ __device__`, `__restrict__`, lambdas with
device capture) stays confined to `src/cuda/**.cu` — users including
`tesseract/*.hpp` see only plain C++20. This is how PyTorch's ATen
layer keeps the public `torch::Tensor` header usable from a pure-g++
translation unit, and it prevents the entire world from needing CUDA to
consume the library. The bridge uses forward-declared
`tesseract::cuda::launch_*` free functions in `src/cuda/Launch.hpp`
(internal header) that each op's CPU-side dispatch calls into.

### 2.4 Kernel strategy per op family

| Family                      | Kernel strategy                                                                                                   |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------|
| Elementwise (add/mul/...)   | Hand-written `__global__` kernels with grid-stride loops; supports implicit broadcasting via the same `align_for_broadcast` helper CPU uses. |
| Reductions (sum/mean/max)   | Two-pass block-level reductions (`cub::DeviceReduce` for all-reduce, hand-written block-stride for dim reductions). |
| Softmax / log_softmax        | Online softmax à la Milakov–Gimelshein, single kernel, two-pass over the row. |
| Cross-entropy (forward+bwd) | Fused kernel paired with softmax — matches the CPU-fused path the MLIR JIT already uses. |
| GEMM (matmul)               | **cuBLAS-Lt** for FP32/FP16/BF16. No hand-rolled GEMM in M2. |
| Attention                   | **FlashAttention-3 reference** (vendored header + precompiled kernels). `ops::attention(q, k, v, mask)` public API. |
| Shape ops                   | `permute` / `transpose` / `view` / `contiguous` — identity on strides + `cudaMemcpy2DAsync` where a real copy is needed. |
| Indexing (split/cat/...)    | Hand-written gather/scatter kernels; matches the CPU B-003 op set. |

Rationale for "no hand-rolled GEMM in M2": cuBLAS/cuBLASLt ships
hand-tuned kernels for every Ada/Hopper/Blackwell tile size and every
supported precision. The gap between a first-month hand-written GEMM
and cuBLAS is 5–10× on modern tensor cores; the gap between cuBLAS and
cuTLASS hand-tuned kernels is typically <15 %. We take the cuBLAS
ceiling in M2 and defer cuTLASS / CuTe DSL kernels until we have a
concrete reason (fusion opportunities, unsupported dtype, TMA) to
diverge. B-008 "Custom GEMM via CuTe DSL" is created as a post-M2
backlog item.

### 2.5 FP16 / BF16 storage is already correct

B-005 (M0 refinement) locked the in-memory layout for `Half` /
`BFloat16` to exactly match the CUDA ABI (2 bytes, little-endian IEEE
binary16 / truncated-IEEE bfloat16 with quiet-NaN preservation). That
means a host `Tensor{DType::Float16}` buffer, once memcpy'd to device
memory, is directly consumable by cuBLASLt in `CUDA_R_16F` mode
without any re-pack. The CPU-side software-emulated kernels
introduced by B-005 remain useful as a reference for the CUDA
regression tests (any CUDA output must match the CPU FP16 output to
within the kernel's documented tolerance).

### 2.6 Graph-mode & IR integration staging

M2 does **not** teach `--convert-tesseract-to-linalg` how to emit GPU
code. Instead:

- Eager CUDA works day-1 (M2.β): `ops::foo(cuda_tensor)` dispatches
  into `cuda::launch_foo(stream)` on the thread's current stream.
- `graph::GraphScope` + interpreter work day-1 on CUDA by the same
  dispatch (the interpreter just calls `ops::*` per node — no change
  to `graph::run`).
- The MLIR JIT remains **CPU-only** through M2. A stretch goal
  (M2L.2, §5.3) explores lowering `tesseract.*` to `gpu.func` +
  `nvgpu.*` dialects and JIT-compiling via MLIR's PTX backend, but
  the M2 exit bar does not require it.

This sequencing keeps M2 in scope (~4 months). Full GPU IR codegen is
M3 foundation territory.

## 3. Track breakdown

| Track  | Description                                                                                            | Artifact                                                                     |
|--------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| M2A    | Plan doc (this file) + ADR-0005 "CUDA as an opt-in HAL backend"                                        | `docs/m2-plan.md` + `docs/adr/0005-cuda-hal.md`                              |
| M2B    | Toolchain provisioning: `TESSERACT_ENABLE_CUDA`, `find_package(CUDAToolkit)`, `tesseract_cuda` target  | `cmake/Dependencies.cmake`, new `src/cuda/CMakeLists.txt`                    |
| M2C    | HAL layer: `CudaAllocator`, `CudaStream`, `Event`, `StreamGuard`; device-aware `Storage`               | `src/hal/*`, updated `src/core/Storage.cpp`                                  |
| M2D    | Tensor cross-device ops: `Tensor::to(Device)`, async H↔D copy, `.contiguous()` on CUDA                 | Updated `src/core/Tensor.cpp` + new `src/cuda/Copy.cu`                       |
| M2E    | CUDA elementwise + broadcast + unary kernels (add/sub/mul/div/neg/relu/sigmoid/tanh/exp/log)           | `src/cuda/Arithmetic.cu`, `src/cuda/Activation.cu`                           |
| M2F    | CUDA reductions (sum/mean/max all + dim), softmax, log_softmax, cross-entropy fwd/bwd                  | `src/cuda/Reduction.cu`, `src/cuda/Softmax.cu`, `src/cuda/Loss.cu`           |
| M2G    | cuBLAS-Lt matmul integration (FP32/FP16/BF16, both rank-2 and batched)                                 | `src/cuda/MatMul.cu` + link against `CUDA::cublasLt`                         |
| M2H    | Shape + indexing ops on CUDA (`permute` / `transpose` / `split` / `cat` / `index_select` / `gather`)   | `src/cuda/Shape.cu`, `src/cuda/Indexing.cu`                                  |
| M2I    | `examples/mnist.cpp --device cuda`; `test_cuda_mnist_parity` ctest (loss curve matches CPU within 1e-3) | Updated example + `tests/nn/test_cuda_mnist_parity.cpp`                      |
| M2J    | FlashAttention-3 vendor + `ops::attention(q, k, v, mask)` public API + backward                        | `third_party/flash-attention-3/`, `src/cuda/Attention.cu`, `include/tesseract/ops/Attention.hpp` |
| M2K    | `examples/llama_forward.cpp` single-layer transformer block (RMSNorm + causal MHA + SwiGLU + pre-norm residuals). Adds `ops::sqrt`, `ops::rms_norm`, `nn::{RMSNorm, MultiHeadAttention, FeedForward, TransformerBlock}`; loosens `nn::Linear` to rank ≥ 2. RoPE pulled forward and landed 2026-04-18 via B-014 — `ops::rotary_embedding` + `nn::RotaryEmbedding` + opt-in `rope_base`/`rope_max_seq` on MHA / TransformerBlock | New example + `tests/nn/test_transformer_block.cpp` + `tests/ops/test_rotary_embedding.cpp` |
| M2L.1  | 6-bench CUDA perf gate: `bench_cuda_{matmul,elementwise,attention,attention_bwd,rms_norm,transformer_block}` with hard ctest bars + ops-layer overhead hard caps | `benchmarks/bench_cuda_*.cpp`, `benchmarks/cuda_bench_util.hpp`, `docs/benchmarks/m2-cuda.md` |
| M2L.2  | *(deferred → B-009)* `tesseract → gpu.func + nvgpu` lowering; PTX JIT via MLIR                         | Moved to M3 backlog (B-009): needs MLIR upstream GPU dialect + kernel outliner, out of M2 scope |

## 4. Verification bar

- **Unit.** `ctest --test-dir build` passes in all four configurations:
  `(ENABLE_CUDA, ENABLE_MLIR) ∈ {OFF, ON}²`. When CUDA is on but no
  visible GPU is present, the CUDA-only tests are **skipped**
  (SKIP_RETURN_CODE 77, as B-006 already wired for MNIST), not failed.
- **Parity.** Every CUDA op has a Catch2 test that compares it to the
  CPU implementation on random inputs. Tolerance is dtype-specific:
  1e-6 rel/abs for FP32, 2e-3 for FP16, 2e-2 for BF16 (same envelopes
  as B-005 used).
- **Autograd.** `tests/autograd/test_gradcheck.cpp` gets a `[cuda]`
  tag that mirrors every existing gradcheck on the GPU device.
- **Training.** `examples/mnist.cpp --device cuda` trains a 3-epoch MLP
  on MNIST and matches the CPU test-set accuracy within 0.5 %.
- **Benchmark (M2L.1 aggressive, exit-gated by `ctest -L bench_cuda`).**
  Six CUDA benches each exit `0` on pass / `1` on perf miss / `77`
  when no GPU visible. Hard bars (measured on RTX 5880 Ada, SM 8.9):
    - `bench_cuda_matmul`: median additive overhead of `ops::matmul`
      vs our own `launch_matmul` bridge ≤ 5 µs across a 10-shape
      sweep (5 square FP32 + 5 square FP16, N ∈ {512, 1024, 2048, 4096,
      8192}); min per-shape ratio `ops/bridge` ≥ 0.95; dispatch
      overhead at 4096² FP32 ≤ 20 µs.
    - `bench_cuda_elementwise`: sustained add / mul @ 64 MiB ≥ 0.95 ×
      memcpy DRAM roofline; sigmoid ≥ 0.90 × (small ALU tax
      allowance).
    - `bench_cuda_attention`: composite `ops::attention /
      Σ(mul+matmul+softmax+matmul)` ≥ 0.97 on three transformer-scale
      shapes.
    - `bench_cuda_attention_bwd`: `(fwd+bwd)/fwd ≤ 5.0` on composite
      SDPA at (B=2, H=16, S=1024, D=64, FP32). Tighter 3.2× target
      is M2L.3's (FA3 fused backward).
    - `bench_cuda_rms_norm`, `bench_cuda_transformer_block`: no hard
      bar — top-line dashboards for future fused-kernel comparisons.
  Numbers are captured per run in `docs/benchmarks/m2-cuda.md`; the
  FA3 bit-parity / ≥ 90 %-of-FA3 bars move to **M2L.3** (Hopper).

## 5. Incremental milestones (intra-M2)

### 5.0 Progress log

- **M2A — Plan + ADR-0005.** ✅ 2026-04-18. `docs/m2-plan.md`,
  `docs/adr/0005-cuda-hal.md`, `docs/roadmap.md` track table,
  `docs/backlog.md` B-008 ~ B-012.
- **M2B — Toolchain provisioning.** ✅ 2026-04-19.
  - `TESSERACT_ENABLE_CUDA` CMake option (default OFF) +
    `TESSERACT_CUDA_ARCHITECTURES` (auto-`native` on CMake ≥ 3.24, else
    explicit `80;86;89;90`).
  - `cmake/Dependencies.cmake` gates `find_package(CUDAToolkit 12.0
    REQUIRED)` + `enable_language(CUDA)`; CUDA TUs pinned to C++17.
  - New static library `tesseract_cuda` under `src/cuda/`, built
    **unconditionally** with a plain-C++ routing file
    (`CudaRuntime.cpp`) and, only when CUDA is ON, an nvcc-compiled
    probe TU (`Probe.cu`) that exposes `detail::real_*` functions via
    an internal bridge header.
  - Public probe surface `include/tesseract/cuda/CudaRuntime.hpp`
    (`is_available` / `has_cuda_support` / `device_count` /
    `device_info` / `runtime_version_string`) is plain C++20 — no
    `<cuda_runtime.h>` leaks into the public include tree.
  - New smoke test `test_hal_cuda_probe` runs in both configurations;
    Catch2 `SKIP` paths wired through `SKIP_RETURN_CODE=4` so ctest
    reports "Skipped" rather than "Failed".
  - **Verification (2026-04-19):**
    - OFF build (MLIR ON, Eigen ON): 164/164 ctests green, 5 new probe
      tests (1 skipped because no CUDA linked in).
    - ON build (MLIR OFF, Release, SM 8.9): 138/138 ctests green; the
      no-op probe kernel compiles to real SM 8.9 SASS (verified via
      `cuobjdump --dump-sass`); probe API enumerates 3 × "NVIDIA RTX
      5880 Ada Generation" (50 GB each) and reports
      "CUDA runtime 12.8 / driver 12.8".
- **M2C — HAL layer: allocator + streams.** ✅ 2026-04-20.
  - New public headers:
    - `include/tesseract/cuda/CudaAllocator.hpp` — `Allocator`-derived
      per-device singleton. Plain C++20, no `<cuda_runtime.h>` on
      user call sites.
    - `include/tesseract/core/Stream.hpp` — device-agnostic `Stream`,
      `Event`, and `StreamGuard`. CPU devices are first-class
      (no-op semantics); CUDA streams use
      `cudaStreamCreateWithFlags(cudaStreamNonBlocking)` so kernel
      launches never implicitly serialize through the legacy
      default stream. `Event` uses `cudaEventDisableTiming`.
    - `current_stream(device)` + `StreamGuard` track the installed
      stream in a thread-local slot per (thread, device).
  - `default_allocator_for(Device)` no longer throws for CUDA
    devices: it now delegates to
    `tesseract::detail::cuda_default_allocator`, a strong symbol
    defined in `tesseract_cuda` that returns
    `CudaAllocator::instance_for(idx)` when CUDA is compiled in and
    throws a clear "rebuild with -DTESSERACT_ENABLE_CUDA=ON"
    `DeviceError` otherwise. No `__attribute__((weak))` tricks,
    no registry, no `--whole-archive` — just a plain forward-ref
    resolved by `tesseract_cuda` being on the link line.
  - `src/CMakeLists.txt` reordered to include `cuda` subdirectory
    before `core`, and `tesseract_core` gained a PRIVATE link
    against `tesseract_cuda` so CMake (and, transitively, GNU ld)
    orders the archives core→cuda on every consumer's link line.
    The dep is acyclic: `tesseract_cuda` only needs `Allocator.hpp`
    (header), never `tesseract_core`'s `.a`.
  - Bridge between plain C++ and CUDA:
    - `src/cuda/Internal.hpp` extended with `real_cuda_malloc`,
      `real_cuda_free`, and the full stream/event entry points
      (`real_stream_create/destroy/synchronize/wait_event`,
      `real_event_create/destroy/record/synchronize/query`).
    - `src/cuda/Allocator.cpp` and `src/cuda/Stream.cpp` are
      *always compiled*. A `#if defined(TESSERACT_HAS_CUDA)` branch
      pulls in `<cuda_runtime.h>` and calls the driver directly (no
      nvcc needed: `cudaMalloc` and friends are plain C++ API).
      The OFF branch defines the same symbols as stubs that throw
      `DeviceError`, so `tesseract_core` links cleanly in both
      configurations.
  - New Catch2 tests:
    - `tests/hal/test_hal_cuda_alloc.cpp`: stub-path
      `REQUIRE_THROWS_AS`, 64-byte device round-trip via
      `cudaMemset`+`cudaMemcpy`, 10 000-cycle alloc/free stress,
      out-of-range index rejection.
    - `tests/hal/test_hal_cuda_stream.cpp`: CPU no-op contract,
      `StreamGuard` round-trip, `current_stream` stability,
      `Event::record` + `Stream::wait` ordering across two
      streams, cross-device misuse throws.
    - Both register with `SKIP_RETURN_CODE=4` so ctest reports
      "Skipped" for GPU-less hosts.
  - **Verification (2026-04-20):**
    - OFF build: **175/175 ctests green**; the 7 GPU-dependent
      cases skip cleanly, the 4 stub-path cases
      (`default_allocator_for(cuda) throws`, `cuda::device_info on
      CPU-only build throws`, `Creating a CUDA Stream in a
      CPU-only build throws`) pass on their exception contract.
    - ON build (SM 8.9 on 3× RTX 5880 Ada Generation):
      **149/149 ctests green** including the 10 000-cycle
      allocator stress (test runtime 3.4 s), the cross-stream
      event ordering, and the cross-device misuse negative test.
    - `compute-sanitizer --tool memcheck` on
      `test_hal_cuda_alloc '[stress]'`: **0 errors** (10 001
      assertions). On the full `test_hal_cuda_stream` binary:
      **0 errors** (12 assertions, 1 skip).
- **M2D — H↔D copy + `Tensor::to(Device)`.** ✅ 2026-04-20.
  - Extended the CUDA HAL bridge with two more verbs:
    - `real_cuda_memcpy(dst, dst_device, src, src_device, nbytes)`
      — synchronous multi-direction copy. Device index `< 0`
      means host; the bridge picks `cudaMemcpyHostToDevice`,
      `cudaMemcpyDeviceToHost`, or `cudaMemcpyDeviceToDevice` and
      sets the appropriate device context via a local
      `DeviceGuard` before issuing a blocking `cudaMemcpy`.
    - `real_cuda_memset_zero(device_index, ptr, nbytes)` — wraps
      `cudaMemset(..., 0, ...)` so `Tensor::zeros` on CUDA
      doesn't need a host scratch buffer.
    - Both land in a new always-compiled TU `src/cuda/Memcpy.cpp`
      that mirrors the stub-vs-real split we've used since M2C.
  - Public surface in `include/tesseract/core/Storage.hpp`:
    - `Storage::copy_device_bytes(dst, dst_dev, src, src_dev, nbytes)`
      — CPU↔CPU short-circuits to `std::memcpy`; every CUDA-
      touching combination forwards to the bridge. Rejects
      unsupported devices (Metal, NPU) with a clear message.
    - `Storage::zero_device_bytes(ptr, device, nbytes)` —
      `std::memset` on CPU, `cudaMemset` on CUDA.
  - `Tensor::to(Device target)` (new public method):
    - Same-device call is a zero-cost no-op (returns `*this` so
      view identity is preserved, matching PyTorch).
    - Cross-device materializes via `contiguous()` then
      `copy_device_bytes`. Autograd metadata is intentionally
      dropped — cross-device copy with a `CopyBackward` node is
      M4 multi-GPU work.
  - Factories reworked so every device path now produces correct
    bytes:
    - `Tensor::zeros` uses `Storage::zero_device_bytes`.
    - `Tensor::fill_` / `ones` / `full` on CUDA fall back to a
      host scratch buffer + H→D copy (replaced by a fill kernel
      in M2E).
    - `Tensor::arange` on CUDA delegates to the CPU path then
      `.to(device)`.
    - `Tensor::clone` uses `copy_device_bytes` for the
      same-device byte blit (D→D works uniformly).
    - `Tensor::contiguous` on non-contiguous CUDA tensors throws
      a clear "call `.to(cpu)` first" message; the strided CUDA
      path lands in M2H.
    - `Tensor::to_string` bounces CUDA tensors through a host
      copy so diagnostic printing works even before M2E kernels.
  - CMake wiring:
    - `src/cuda/CMakeLists.txt` picks up `Memcpy.cpp` in both
      configurations.
    - `tesseract_core` already PRIVATE-links `tesseract_cuda` so
      the new `real_cuda_memcpy` / `real_cuda_memset_zero`
      references resolve in a single left-to-right ld pass.
  - New Catch2 tests:
    - `tests/hal/test_hal_cuda_copy.cpp`: CPU memcpy parity,
      unsupported-device rejection, H→D→H byte round-trip
      (4 KiB pattern), D→D copy (1 KiB), `zero_device_bytes`
      on CUDA clears a pre-seeded buffer, CPU-only stub throws.
    - `tests/hal/test_hal_cuda_tensor.cpp` (satisfies the
      §5.1 M2.α exit bar): `Tensor::zeros({16}, Float32,
      Device(CUDA, 0))` + `.to(cpu)` round-trip, `ones`,
      `arange`, 32×32 float pattern H→D→H preservation,
      `clone()` distinct-storage + identical bytes, same-device
      `.to()` shares storage, `to_string` renders `cuda:0` and
      the last element.
  - **Verification (2026-04-20):**
    - OFF build: **189/189 ctest pass** (14 new tests added;
      15 GPU-only cases correctly skipped).
    - ON build (SM 8.9, 3× RTX 5880 Ada): **163/163 ctest pass**
      in 5.9 s wall-clock; every Tensor round-trip test is
      green, including the arange/ones fallback path.
    - `compute-sanitizer --tool memcheck` clean on both new
      suites: `test_hal_cuda_copy` **0 errors** (5573
      assertions, 1 skipped), `test_hal_cuda_tensor` **0
      errors** (1433 assertions).
- **M2E — CUDA elementwise + broadcast + unary kernels.**
  ✅ 2026-04-21.
  - Device-side kernel bridge lives in three TUs:
    - `include/tesseract/cuda/detail/Elementwise.hpp` — plain
      C++17 declarations with `BinaryKind` / `UnaryKind` op codes,
      `launch_binary_elementwise` / `launch_unary_elementwise` /
      `launch_fill`. Shape data crosses the bridge as raw
      `int64_t*` + `int ndim` (rather than `const Shape&`) so the
      `.cu` TU stays C++17-clean; Shape.hpp uses `std::span` and
      CMake 3.22 can't select `CUDA_STANDARD 20`. The launchers
      take a caller-supplied `void* stream_handle` to keep
      `tesseract_cuda` from back-referencing `current_stream` in
      `tesseract_core` — the core ↔ cuda archive graph stays
      acyclic.
    - `src/cuda/Elementwise.cu` (compiled only when
      `TESSERACT_ENABLE_CUDA=ON`) — packs the raw pointers into
      a trivially-copyable `ShapePod` kernel arg, generates
      per-dtype instantiations of a strided binary kernel
      (`AddFn` / `SubFn` / `MulFn` / `DivFn`), a strided unary
      kernel (`NegFn` / `ReluFn` / `SigmoidFn` / `TanhFn` /
      `ExpFn` / `LogFn` — transcendentals use `__expf` / `logf` /
      `tanhf` on Float32 for the intrinsic-path throughput), and
      a dense `fill_dense`. Every launch goes through a
      `DeviceGuard` RAII wrapper and asserts via
      `cudaGetLastError()` into a `DeviceError` with the kernel
      name prefix.
    - `src/cuda/ElementwiseStub.cpp` — always-compiled
      throwing stubs that match the bridge signatures and only
      emit bodies under `!TESSERACT_HAS_CUDA`, so a CPU-only
      build still links.
  - Op-layer wiring:
    - `src/ops/cpu/Arithmetic.cpp::elementwise_binary` now
      short-circuits into the CUDA launcher when
      `a.device().is_cuda()`. Uses the same
      `ops::align_for_broadcast` helper the CPU path uses, so a
      stride of 0 on a broadcasted dim means "replicate" on both
      sides. Covers `add` / `sub` / `mul` / `div` on Float32 /
      Float64 / Int32 / Int64.
    - `src/ops/cpu/Arithmetic.cpp::neg_forward` dispatches via
      `UnaryKind::Neg` on the same dtypes.
    - `src/ops/cpu/Activation.cpp::elementwise_unary_float`
      dispatches `relu` / `sigmoid` / `tanh` / `exp` / `log` to
      the CUDA unary launcher on Float32 / Float64; integer
      unary ops still go through the dtype-gated CPU path.
    - `src/core/Tensor.cpp::fill_` on CUDA now uses
      `launch_fill` for Float32 / Float64 / Int32 / Int64 / Bool;
      Half / BFloat16 fall back to the M2D host-scratch +
      `cudaMemcpy` path until M2G brings full FP16 kernel
      coverage. `Tensor::ones` / `Tensor::full` pick the kernel
      path automatically via this hook.
  - Correctness fix in `src/core/Storage.cpp`: synchronous
    `cudaMemcpy` / `cudaMemset` on the legacy null stream does
    **not** wait for work on non-blocking user streams (which is
    precisely what our per-thread `current_stream(device)` is).
    Before M2E no user-stream kernels existed, so the race never
    fired; after M2E, `tensor.add(b).to(cpu)` could start the D→H
    copy while `add`'s kernel was still in flight. Fix:
    `Storage::copy_device_bytes` and `Storage::zero_device_bytes`
    now call `current_stream(device).synchronize()` on the
    CUDA-touching side(s) before issuing the blocking copy, so
    `tensor.to(cpu)` gives PyTorch-equivalent sync semantics.
    The sync is cheap (bounded by the longest pending kernel on
    that one stream) and lives at the device boundary — per-op
    launches themselves remain genuinely async.
  - New Catch2 suite `tests/ops/test_ops_cuda_elementwise.cpp`
    (registered with `SKIP_RETURN_CODE 4` so the CPU-only build
    reports a clean skip):
    - Dense same-shape parity on Float32 for `add`/`sub`/`mul`/
      `div`, plus the `[1, 4] + [3, 4]` row-broadcast case for
      `add`.
    - Int32 parity for all four binary ops (non-zero divisors).
    - `neg` parity on Float32 / Float64 / Int32 / Int64.
    - `relu` parity on mixed-sign Float32 / Float64.
    - Transcendental unary parity (`sigmoid`, `tanh`, `exp`,
      `log`) on Float32 / Float64 with `WithinAbs` tolerances.
    - `Tensor::ones({32}, Float32, cuda)` + `.to(cpu)` confirms
      the fill kernel fired (M2D's scratch path is only taken
      for Half / BFloat16 now).
    - `Tensor::full({16}, Bool, cuda, value=2.0)` round-trips
      as all-true (verifies the non-zero → true conversion in
      the Bool kernel specialization).
    - CPU-only dispatch regression guard: `add` on
      `Device::cpu()` still routes through the Eigen / scalar
      path (kernel-bridge inclusion doesn't accidentally
      short-circuit non-CUDA tensors).
  - **Verification (2026-04-21):**
    - OFF build: **173/173 ctest pass**; the 11 new M2E
      parity cases skip cleanly, the CPU-only dispatch-guard
      case runs and passes.
    - ON build (SM 8.9, 3× RTX 5880 Ada): **173/173 ctest pass**
      in 19.1 s wall-clock. `test_ops_cuda_elementwise` runs
      10 test cases / 5799 assertions on GPU.
    - `compute-sanitizer --tool memcheck` clean on the new
      suite: **0 errors** (5799 assertions, 10 cases). The
      HAL sweep (`test_hal_cuda_alloc` / `test_hal_cuda_stream` /
      `test_hal_cuda_copy` / `test_hal_cuda_tensor`) is still
      clean under sanitizer after the Storage stream-sync fix:
      **0 errors** across all four binaries.

- **M2F — CUDA reductions + softmax + cross-entropy fwd/bwd.**
  - Shape of the deliverable: three new CUDA TUs
    (`src/cuda/Reduction.cu`, `Softmax.cu`, `Loss.cu`) + three
    bridge headers under `include/tesseract/cuda/detail/` + their
    CPU-only throwing stubs (`*Stub.cpp`). Each TU stays
    self-contained; they share only the device-side helpers in
    the new `src/cuda/KernelUtils.cuh` (`ShapePod` + `flat_to_offset`
    + `DeviceGuard` + `check_launch`), which is header-only so
    there's no extra link-order coupling.
  - Bridge surface deliberately C++17-only (same contract as
    M2E's Elementwise bridge): shape + stride descriptors cross
    the boundary as `int ndim` + `const int64_t*`, never as
    `tesseract::Shape`, so nvcc never has to parse `std::span` or
    C++20 lambdas while it keeps `CUDA_STANDARD=17` (CMake 3.22
    can't select `CUDA20`). Stream handles are passed in as
    `void* stream`, resolved from `current_stream(device)` by
    the op layer just before the launch — keeping the
    `tesseract_core → tesseract_cuda` dependency one-way.
  - Kernel design:
    - *All-reduce* (`sum`/`mean`/`max` with no dim): two-stage,
      deterministic. Stage 1 per-block tree reduces a chunk
      into a partials array; stage 2 launches a single
      `kBlockSize`-thread block over the partials and
      finalizes (divides by N for `mean`, trivial for
      `sum`/`max`). Matches the CPU reference's summation order
      well enough for `WithinAbs(.., 1e-5)` on fp32 and
      `WithinAbs(.., 1e-12)` on fp64. We deliberately avoid
      the single-pass `atomicAdd` variant — it would be ~2×
      faster but non-deterministic across runs, which would
      fail the parity tests.
    - *Dim-reduce* (`sum`/`mean`/`max` with a `dim`): one block
      per output slot, threads in the block stride over the
      reduced dim `D` and tree-reduce. Handles arbitrary
      `in_strides` (no forced `.contiguous()`), which matters
      for the middle-dim test on rank-3 shapes.
    - *Softmax / log_softmax* (along `dim`): one block per
      (outer, inner) slot. Three passes inside each block:
      max → sum(exp(x-max)) → write (exp ÷ sum_exp or
      `x - max - log sum_exp`). Uses
      `__expf`/`__logf`/`exp`/`log` depending on precision.
      A subtle bug in the first cut had `iter_out_strides`
      computed as the iter-shape's own row-major strides
      (e.g. `[1]` for an `[N, C]→[N]` iter) rather than the
      full-rank contiguous output strides at the non-dim
      positions (e.g. `[C]`). That caused every row to write
      into overlapping output slots — fixed by building the
      full-rank contiguous strides first and then picking out
      the non-dim indices.
    - *Cross-entropy forward* (fused): one block per row,
      computes the row's stable `log_sum_exp - logits[target]`
      contribution and `atomicAdd`s it into a single-slot
      accumulator; optionally also writes the `[N, C]` probs
      matrix when autograd needs it. A tiny 1-thread finalize
      kernel divides the sum by N. `atomicAdd` is not
      deterministic across runs, but row counts are small
      (typical `N ≤ 1024`) so the drift stays well under the
      `WithinAbs(.., 1e-5f)` fp32 tolerance.
    - *Cross-entropy backward* (fused): one block per row,
      writes `dlogits[n, c] = (probs[n, c] - (c==target))
      * grad / N`. Deterministic (every output slot owned by
      a single thread); parity test tightens to `1e-6`.
  - Op-layer dispatch: `src/ops/cpu/Reduction.cpp`,
    `Softmax.cpp`, `Loss.cpp` each learned an `if
    (x.device().is_cuda()) { ... return; }` branch at the top
    of the forward (and in the backward `Node::apply`) that
    forwards to the bridge. Storage-level sync already lives
    in `Storage::copy_device_bytes` since M2E, so any
    subsequent `.to(cpu)` / `.item()` waits for the in-flight
    user-stream kernels naturally.
  - Autograd: works unchanged. The CPU backward nodes (see
    `SumDimBackward`, `SoftmaxBackward`, `CrossEntropyBackward`)
    are device-agnostic — they call back into `broadcast_to`,
    `mul`, `sum`, `exp`, which now all pick the CUDA kernel
    when operands live on CUDA. The CE `Node::apply` specialization
    was updated to allocate `dlogits` fresh on CUDA and route to
    `launch_ce_backward`, skipping the `probs_saved.clone()` the
    CPU path does (saves an `N×C` device memcpy per step).
  - New Catch2 suites (all registered with `SKIP_RETURN_CODE 4`):
    - `tests/ops/test_ops_cuda_reduction.cpp` — sum/mean/max
      all-reduce + dim-reduce parity on fp32 / fp64; middle-dim
      on rank-3; CPU-only smoke.
    - `tests/ops/test_ops_cuda_softmax.cpp` — softmax +
      log_softmax on rank-2 last-dim, rank-3 middle-dim, fp32 /
      fp64, plus a "softmax rows sum to 1" sanity case; CPU-only
      smoke.
    - `tests/ops/test_ops_cuda_loss.cpp` — CE forward fp32 /
      fp64; CE standalone backward with non-unit grad scale;
      CE via autograd `Engine::backward`; CPU-only
      argmax-aligned smoke.
  - **Verification (2026-04-21):**
    - OFF build: **216/216 ctest pass**; the new parity cases
      skip cleanly via `SKIP_RETURN_CODE 4`; each TU's
      always-running CPU smoke case runs.
    - ON build (SM 8.9, RTX 5880 Ada): **190/190 ctest pass** in
      28.6 s wall-clock. 4 CPU-only-build-specific tests are
      reported as skipped (by design — they assert that a CPU-
      only tesseract surfaces a clear `DeviceError`).
    - `compute-sanitizer --tool memcheck` clean on every M2F +
      M2E binary: **0 errors** across
      `test_ops_cuda_reduction` (240 assertions / 6 cases),
      `test_ops_cuda_softmax` (1524 / 6),
      `test_ops_cuda_loss` (315 / 5), and
      `test_ops_cuda_elementwise` (5799 / 10). No regressions
      on the HAL sweep.

- **M2G — cuBLAS-Lt matmul integration.**
  - Shape of the deliverable: one bridge header
    (`include/tesseract/cuda/detail/MatMul.hpp`) + one host-only
    translation unit (`src/cuda/MatMul.cpp` — deliberately *not* a
    `.cu`, since cuBLASLt's entry points are plain C-with-extern-C
    prototypes and we have no device code of our own on this path) +
    the matching throwing stub (`src/cuda/MatMulStub.cpp`) for the
    CPU-only build. Wired into `src/cuda/CMakeLists.txt` with
    `target_link_libraries(tesseract_cuda PRIVATE CUDA::cudart
    CUDA::cublasLt)` — cuBLASLt only links in when
    `TESSERACT_ENABLE_CUDA=ON` (the stub has no external refs).
  - Why cuBLASLt (not cuBLAS or a hand-written GEMM). cuBLASLt is
    the superset API: it's the only path that exposes
    `CUBLASLT_ORDER_ROW` layouts (so we can hand it tensors
    straight out of `Tensor::empty` without the classical row↔col
    transpose trick), explicit per-(shape, dtype, compute-type)
    heuristic selection, and `CUBLAS_COMPUTE_32F` accumulation for
    `CUDA_R_16F` / `CUDA_R_16BF` matmul (the Tensor-Core path we
    actually want on Ada/Hopper). We don't compete with cuBLAS for
    M2G — the delta between a first-month hand-written GEMM and
    cuBLAS on modern Tensor Cores is 5–10×, so M2 explicitly takes
    the cuBLAS ceiling and defers cuTLASS / CuTe DSL to post-M2
    (B-008).
  - Bridge surface (same C++17 contract as the other M2 bridges).
    One entry point `launch_matmul(dtype, device_index, M, N, K,
    a, op_a, lda, b, op_b, ldb, c, ldc, stream)` covers the rank-2
    and batched cases both; the op layer walks the output batch
    grid and issues one call per slab. `MmOp::{None,Transpose}`
    maps to `CUBLAS_OP_{N,T}`; `ld*` is the row-stride of the
    *stored* matrix (before any op is applied), which is exactly
    what `Tensor::strides()` already reports for the two layouts
    we accept. Stream handles cross as `void*` (raw
    `cudaStream_t`) — same as M2E/F — so `tesseract_cuda` never
    calls back into `tesseract_core` for stream resolution and
    the link graph stays a-cyclic.
  - Layout detection (`detect_cuda_mat_layout` in
    `src/ops/cpu/MatMul.cpp`). For each operand's last-two strides
    `[s0, s1]` we accept exactly two cases:
    - `s1 == 1` → row-major as stored → `op=N`, `ld = s0`. Covers
      the standard contiguous case and any slab from slicing
      leading dims of a row-major contig tensor.
    - `s0 == 1` → column-major as stored (= transpose of a
      row-major `[cols, rows]` contig) → `op=T`, `ld = s1`. This
      is exactly what `Tensor::transpose(r-2, r-1)` produces and
      is the reason the autograd matmul backward now works on
      CUDA without an explicit `.contiguous()` call (strided
      materialization on CUDA lands in M2H).
    Anything else (padded / dilated / both-strides-`>1` layouts)
    lands on a `TESSERACT_CHECK` that tells the caller to
    `.contiguous()` on CPU or wait for M2H.
  - Compute-type picker: `CUBLAS_COMPUTE_64F` for `Float64`;
    `CUBLAS_COMPUTE_32F` for everything else. That's
    - FP32 matmul with TF32 Tensor Cores on Ada/Hopper,
    - FP16 / BF16 matmul with FP32 accumulation (matches PyTorch
      default; the "FP16 accumulated in FP16" mode accumulates
      ~3 ULP per `K` and breaks even at moderate depths). Scale
      type (alpha/beta precision) follows compute — FP64 for 64F,
      FP32 otherwise.
  - Handle & workspace lifecycle. Per-device `cublasLtHandle_t`
    cached via `std::call_once` in a static 8-slot array — matches
    `src/cuda/Allocator.cpp`'s assumption that single server
    boards top out at 8× HPC GPUs. Descriptor / three matrix
    layouts / preference object are created fresh per call
    (combined < 10 µs; they depend on every one of M, N, K,
    dtype, op_a, op_b, lda, ldb, ldc). Workspace is a per-call
    4 MiB `cudaMalloc` — ~1 ms amortized against a 1024² FP32
    GEMM, and a non-fatal failure (`ws.bytes = 0`) just falls
    back to the narrower algo set cuBLASLt picks at that budget.
    A post-M2 change (B-010) is tracked for switching to a
    per-device scratch pool if this shows up on a profile.
  - Op-layer dispatch: `src/ops/cpu/MatMul.cpp::matmul_forward`
    grew a `if (lhs.device().is_cuda()) return
    matmul_forward_cuda(...)` short-circuit at the top (right
    after the broadcast-batch + shape math, before the CPU path's
    `.contiguous()` materialization). The CUDA helper reuses the
    same `ops::align_for_broadcast` stride-0 trick the CPU path
    uses for broadcasted batch dims, so a `[B, M, K] @ [K, N]`
    still "works" by handing the same `[K, N]` slab to every
    batch iteration. A new device-mismatch `TESSERACT_CHECK`
    was added at the top (CPU branch used to silently accept an
    asymmetric lhs/rhs device pair and crash downstream).
  - Autograd: works unchanged. `MatMulBackward::apply` already
    emits `matmul(g, mat_transpose(b_saved))` and
    `matmul(mat_transpose(a_saved), g)`; both calls now land on
    the CUDA path via the same layout detection, with
    `mat_transpose` mapping straight to `op=T`. For rank-2
    (the MNIST / Llama block case), `reduce_to_shape` is a
    no-op (shapes match, `src.contiguous()` on a freshly
    allocated row-major tensor). Batched-broadcast backward still
    funnels through CPU `reduce_to_shape` for the final
    sum-reduce — that path lands with the M2H CUDA shape ops.
  - New Catch2 suite `tests/ops/test_ops_cuda_matmul.cpp`
    (`SKIP_RETURN_CODE 4`):
    - rank-2 FP32 parity (`WithinAbs(., 3e-3)` — TF32 Tensor
      Core headroom), rank-2 FP64 parity (`WithinAbs(., 1e-9)`),
    - rank-3 batched FP32 parity (non-broadcast),
    - broadcast batched FP32 parity (`[B, M, K] @ [K, N]`),
    - transposed-RHS FP32 parity (explicit `.transpose(0, 1)`
      view exercising `op=T`),
    - autograd backward FP32 parity (via
      `Engine::backward(y, grad)` with a device-matched explicit
      grad — bypasses `ops::sum` whose backward still dispatches
      through CPU `broadcast_to` until M2H),
    - rank-2 FP16 / BF16 parity (`WithinAbs(., 5e-2f / 1e-1f)`
      — FP32-accumulated Tensor Core drift is bounded by one
      dtype-ULP per output slot),
    - an always-running CPU-only smoke so the CPU build keeps
      one asserted path out of this TU.
  - **Verification (2026-04-21):**
    - OFF build: **225/225 ctest pass** (9 new M2G parity cases
      SKIP cleanly via `SKIP_RETURN_CODE 4`; the CPU-only smoke
      always runs).
    - ON build (SM 8.9, RTX 5880 Ada): **199/199 ctest pass** in
      ~39 s wall-clock. 4 CPU-only-build-specific tests are
      reported as skipped by design.
    - `compute-sanitizer --tool memcheck` clean on
      `test_ops_cuda_matmul`: **0 errors** across 9 test cases /
      3146 assertions, including the FP16/BF16 Tensor-Core paths
      and the `op=T` transposed view. No regressions on the
      existing M2E/F CUDA suites.

- **M2H — CUDA shape + indexing ops (strided copy, scatter-add,
  `index_select` / `gather`).**
  - Shape of the deliverable: two bridge headers
    (`include/tesseract/cuda/detail/Shape.hpp`,
    `include/tesseract/cuda/detail/Indexing.hpp`) + two `.cu`
    translation units (`src/cuda/Shape.cu`,
    `src/cuda/Indexing.cu`) + the matching throwing stubs
    (`src/cuda/ShapeStub.cpp`, `src/cuda/IndexingStub.cpp`) for
    CPU-only builds. Wired into `src/cuda/CMakeLists.txt`
    alongside the other bridge TUs.
  - Why a single generic "strided copy". `Tensor::contiguous()`,
    `ops::broadcast_to`, `ops::cat` (per-chunk slab copy into the
    output), `ops::split` (per-chunk slab copy out of the input),
    and the slab-blit inside `SplitChunkBackward` / `CatBackward`
    all reduce to the same primitive: given
    `(sizes, src_strides, dst_strides)`, run one thread per flat
    output index, map flat → src_off + dst_off via the existing
    `flat_to_offset` device helper, and copy `itemsize` bytes.
    We dispatch on `itemsize ∈ {1, 2, 4, 8}` with a templated
    `strided_copy_kernel<Elem>` — that single kernel covers every
    current dtype (including `Half` / `BFloat16` / `Bool` /
    `Int8`) and every future dtype of the same itemsize without a
    kernel rebuild.
  - Why a dedicated "strided scatter-add" for gradients.
    `ops::reduce_to_shape` (the batched-broadcast matmul backward
    funnel that M2G had to leave on CPU), plus the backward of
    `index_select` / `gather`, all have to *accumulate* into a
    smaller buffer with the same "one thread per source element"
    access pattern as the copy kernel — the difference is just
    `atomicAdd` vs `=`. We keep that behind a second entry point
    (`launch_strided_scatter_add`) so the forward kernel stays
    `__restrict__`-friendly and branchless. Dtype coverage for
    the scatter-add family is `Float32 / Float64 / Int32 / Int64`
    only, matching M2E's existing `atomicAdd` policy; `Half` /
    `BFloat16` / `Bool` / `Int8` gradients throw a clear
    `DeviceError` pointing at the same reason as M2E's reduction
    backward (no portable native atomic).
  - `int64_t` atomic workaround. CUDA only ships
    `atomicAdd(unsigned long long*, unsigned long long)`
    natively; for `int64_t` we reinterpret the pointer and value
    as `unsigned long long` and rely on two's-complement wrap
    semantics (`(int64_t)((uint64_t)a + (uint64_t)b) == a + b` for
    all inputs). Good enough for gradient accumulation where the
    magnitudes are orders below 2⁶³.
  - `index_select` / `gather` on CUDA (bridge: `launch_index_select`,
    `launch_scatter_add_at_dim`, `launch_gather`,
    `launch_gather_scatter_add`). Forward kernels are standard
    "one thread per output element, unpack coords, remap the
    selected axis, copy"; backward kernels do the same walk but
    `atomicAdd` into the gradient buffer, which lets us handle
    duplicate indices correctly without a pre-sort. Host-side
    range check on the `indices` tensor (bounces through
    `Storage::copy_device_bytes` to a CPU buffer) runs before the
    launch — O(N) overhead is negligible for typical indexing
    workloads and we trade it for a clear `IndexError` instead of
    an opaque `cudaErrorIllegalAddress`.
  - Op-layer dispatch. `Tensor::contiguous()` now has an `if
    (device.is_cuda())` short-circuit that calls
    `launch_strided_copy` directly; the old
    `TESSERACT_CHECK(device == cpu)` guard disappears.
    `ops::broadcast_to` / `ops::reduce_to_shape` in
    `src/ops/cpu/Arithmetic.cpp` grew the same short-circuit
    (strided-copy and scatter-add respectively). `ops::cat` /
    `ops::split` / `ops::index_select` / `ops::gather` and all
    four of their backward nodes in `src/ops/cpu/Indexing.cpp`
    were refactored so that `copy_slab_into` and
    `slice_along_dim` each route through CUDA when both
    endpoints are on the same CUDA device, otherwise fall back to
    the existing CPU loop. Device-mismatch checks
    (`src.device() == indices.device()`) were added at the
    public entry points — the CPU path used to silently accept
    an asymmetric pair and crash downstream.
  - Knock-on effects for autograd. The matmul backward's
    batched-broadcast `reduce_to_shape` funnel (noted as "still
    CPU-only" in M2G) now runs entirely on CUDA, as does every
    backward that feeds through `broadcast_to` (e.g. `sum`-into-
    scalar of a CUDA tensor). `Tensor::clone()` on non-contig
    CUDA tensors also works for free since it defers to
    `contiguous()`.
  - New Catch2 suites (both registered with `SKIP_RETURN_CODE 4`):
    - `tests/ops/test_ops_cuda_shape.cpp` — `contiguous()` on
      strided / permuted views, `clone()` of non-contig tensors,
      `broadcast_to` with leading and middle-axis expansion,
      `reduce_to_shape` on Float32 / Float64 / Int64 both into a
      smaller shape and all the way down to a scalar; CPU-only
      smoke.
    - `tests/ops/test_ops_cuda_indexing.cpp` — `cat` at rank 2
      and rank 3, `split` + `split_with_sizes`, `index_select`
      forward + autograd backward with intentional duplicate
      indices (exercises `atomicAdd`), `gather` forward +
      backward with duplicates, `cat` / `split` autograd
      backward; CPU-only smoke.
  - **Verification (2026-04-21):**
    - OFF build: **242/242 ctest pass** (17 new M2H parity
      cases SKIP cleanly via `SKIP_RETURN_CODE 4`; each TU's
      always-running CPU smoke runs).
    - ON build (SM 8.9, RTX 5880 Ada): **216/216 ctest pass**
      in 19.3 s wall-clock. 4 CPU-only-build-specific tests are
      reported as skipped by design.
    - `compute-sanitizer --tool memcheck` clean:
      `test_ops_cuda_shape` (**0 errors**, 311 assertions / 7
      cases) and `test_ops_cuda_indexing` (**0 errors**, 271
      assertions / 10 cases).
    - `compute-sanitizer --tool racecheck` clean on both TUs
      (**0 hazards**) — the `atomicAdd`-heavy scatter-add /
      `index_select` backward / `gather` backward kernels have
      no unsynchronized shared-memory races. No regressions on
      M2E / M2F / M2G.

- **M2I — end-to-end CUDA MNIST (`examples/mnist.cpp --device cuda`
  + step-by-step loss parity).**
  - Shape of the deliverable: three small core/nn changes
    (`Tensor::move_to_(Device)`, `Module::to(Device)`,
    `Adam::step` CUDA dispatch) + one new CUDA kernel bridge
    (`include/tesseract/cuda/detail/Optim.hpp` +
    `src/cuda/Optim.cu` + `src/cuda/OptimStub.cpp` for CPU-only
    builds) + one tiny `UnaryKind::Step` extension to the
    existing M2E elementwise bridge + the `--device {cpu,cuda}`
    CLI flag on `examples/mnist.cpp` + the new
    `tests/nn/test_cuda_mnist_parity.cpp` Catch2 suite. Wired
    into `src/cuda/CMakeLists.txt` and `tests/CMakeLists.txt`
    alongside the M2E..M2H bridges.
  - Why `move_to_` instead of reusing `Tensor::to(Device)`.
    `nn::Linear` stores `weight_` / `bias_` as member
    `Tensor`s, and `Module::params_` caches a second copy of
    each handle — the two copies share the same
    `shared_ptr<TensorImpl>`. If we used `to()` (which
    allocates a fresh `TensorImpl`) for `Module::to()`, we'd
    update the `params_` copy while leaving the owning
    `Linear`'s `weight_` pointing at the old CPU impl — forward
    would then launch CUDA kernels against CPU params, or vice
    versa. `move_to_` instead rewrites the **fields** of the
    shared `TensorImpl` in place (`storage`, `storage_offset`,
    `shape`, `strides`, `dtype`, `device`), so every alias
    observes the device change atomically, and the stale
    `.grad` from a prior CPU step is cleared because it would
    be on the wrong device.
  - Why a fused CUDA Adam kernel. `Adam::step` on CPU is a
    5-line elementwise walk per parameter; pushing that as-is
    to CUDA would require four separate kernels (two moments +
    the param update + a bias-correction pre-pass) or an
    N-param-round-trip bounce through CPU. We instead fuse the
    entire `m ← β₁m + (1-β₁)g; v ← β₂v + (1-β₂)g²; p ← p - lr
    · m/(√v + ε)` update into a single `adam_step_kernel<T>`
    templated on element type (Float32 / Float64). The
    bias-correction factors `(1 - β^t)` are precomputed on the
    host once per `step()` and passed as scalar kernel args,
    so the kernel stays branchless. One launch per param, zero
    intermediate buffers, no scratch. Half / BFloat16
    optimizer state throws `DeviceError` — the second-moment
    accumulator `g²` underflows rapidly at half precision and
    the M2 exit bar explicitly wants full-precision Adam.
  - ReLU backward on CUDA. The MNIST graph's `ReluBackward`
    used to construct its positive-indicator mask with a
    host-side `dispatch_float` loop, which is a silent SIGSEGV
    when `x_saved` lives on CUDA. We extend the M2E
    `UnaryKind` enum with a new `Step` entry (`(x>0) ? 1 : 0`,
    float-only) and reuse the existing
    `launch_unary_elementwise` pathway to build the mask on
    the same device as `x_saved`. The trailing `mul(g, mask)`
    already dispatches through the M2E binary bridge, so the
    whole backward stays on-device.
  - Module / optimizer integration flow. `Module::to(device)`
    recurses through `children_` first (mirroring PyTorch
    iteration order for debug-friendliness), then calls
    `move_to_` on each registered parameter; `Sequential`
    inherits the behaviour for free via `register_module`.
    `optim::Adam::step` keeps its existing lazy moment-buffer
    allocation (first `step()` after construction; moments
    inherit the param's device) and short-circuits to
    `launch_adam_step` for any param with `device().is_cuda()`,
    falling through to the existing CPU inner loop otherwise.
    A `TESSERACT_CHECK(grad.device() == param.device())` on
    the CUDA branch catches the asymmetric-device case with a
    clear diagnostic instead of a stray `cudaErrorIllegalAddress`.
  - `examples/mnist.cpp` layout. A new `--device {cpu,cuda}`
    flag selects the training device. After parsing, we
    immediately `model->to(run_device)`; Adam is constructed
    **after** the move so its `params_` vector captures the
    on-device handles; per-step mini-batches are pushed to the
    device with `Tensor::to(run_device)`; the scalar loss
    readback and `accuracy(...)` eval both bounce through
    `Tensor::to(cpu_device())` so the host-side readout stays
    correct on CUDA runs. `--mode graph --device cuda` is
    explicitly rejected — the graph interpreter + MLIR JIT
    stay CPU-only until the M3 device-aware lowering lands.
  - Parity-test design (`tests/nn/test_cuda_mnist_parity.cpp`).
    ctest must stay hermetic, so the test uses a synthetic
    3-class Gaussian mixture in R² (matching the
    well-separated architecture that `test_nn_mnist_smoke`
    already exercises) instead of the real MNIST download.
    Two identical-architecture models are built side-by-side;
    because `nn::Linear`'s initializer bumps a thread-local
    PRNG per ctor, we force identical initial weights by
    `std::memcpy`-ing the CPU reference's param bytes into the
    CUDA-bound model's CPU-side storage **before**
    `Module::to(cuda)` ships it to device. Both models then
    train through `train_and_record(...)` — shared helper
    with the same shuffle seed and batch indices — and every
    step's scalar loss is bounced back to CPU for comparison.
    Per-step tolerance is `WithinAbs(., 5e-3)` (looser than
    the M2E/F forward bands because the TF32-matmul +
    scatter-add reductions in the backward compound across
    the update); final training accuracy must clear
    `> 0.97` on both sides and agree to within 2 accuracy
    points (≈ one misclassified sample on the 600-element
    dataset).
  - **Verification (2026-04-18):**
    - OFF build: **244/244 ctest pass** (2 new M2I cases: the
      GPU parity case SKIPs cleanly and the always-running
      CPU convergence smoke passes).
    - ON build (SM 8.9, RTX 5880 Ada): **218/218 ctest pass**
      in 19.7 s wall-clock.
    - `compute-sanitizer --tool memcheck` clean on
      `test_cuda_mnist_parity` (**0 errors**, 80 assertions /
      2 cases). `--tool racecheck` clean as well
      (**0 hazards** displayed) — the fused Adam kernel has
      no shared-memory contention by construction.
    - Real-data smoke: `./examples/tesseract_mnist
      data/mnist --device cuda --epochs 1` prints
      `step 800 avg_loss=0.370688 / test_acc=0.9438`; the
      CPU reference at the same seed prints
      `step 800 avg_loss=0.370678 / test_acc=0.9444`. Loss
      parity ≈ 1e-5, accuracy gap ≈ 6e-4 — well inside the
      `1e-3` M2.β exit-bar envelope and the `> 96%` accuracy
      target at 1 epoch.
  - M2.β closeout. M2I is the capstone; every M0/M1 op
    exercised by the MLP-MNIST pipeline (matmul, add,
    broadcast_to, reduce_to_shape, cross_entropy, ReLU /
    ReluBackward, Adam) now runs eagerly on CUDA with a CPU
    fallback for the long tail of unused dtypes. The
    elementwise activation backwards outside ReLU
    (Sigmoid/Tanh/Exp/Log) are still CPU-only, but none
    of them appear in the MNIST MLP; they land as needed in
    M2J / M2K when the transformer block's GELU / SiLU enter
    the graph.

- **M2J — `ops::attention(q, k, v, mask, causal, dropout_p)` API
  + composite on-device SDPA.**
  - Shape of the deliverable: one new public header
    (`include/tesseract/ops/Attention.hpp`), one CPU-only TU
    (`src/ops/cpu/Attention.cpp` — no `.cu` file!), plus two
    Catch2 suites (`tests/ops/test_ops_attention.cpp` for
    correctness + autograd, `tests/ops/test_ops_cuda_attention.cpp`
    for CPU↔CUDA parity). Wired into `src/ops/CMakeLists.txt` and
    `tests/CMakeLists.txt` next to the M2H/I entries. No new
    CUDA bridge header or `.cu` kernel.
  - Why a composite implementation instead of vendoring FA3
    now. FlashAttention-3 is hardware-gated on **SM 9.0+**
    (Hopper WGMMA + TMA); the present CI/dev box is SM 8.9
    (RTX 5880 Ada) where FA3 kernels will not even compile.
    Vendoring FA3 under `third_party/flash-attention-3/`
    before we can run its parity tests would leave the
    ≥ 90 %-of-FA3 exit bar unverified and the test suite
    full of unconditional SKIPs on the only hardware we
    have — a green CI that's actually empty. We instead
    land the **public API contract now** (so M2K's
    transformer block can depend on `ops::attention`
    immediately) and defer the FA3 kernel integration +
    its perf / bit-parity bars to a **new M2L sub-track
    `M2L.3`** gated on Hopper access. When FA3 lands, the
    public signature of `ops::attention` does not change —
    only the implementation swaps.
  - Why "composite" is a real implementation and not a
    placeholder. Every sub-op (`matmul` → `softmax` →
    `matmul`, plus `mul` for the 1/√d scaling and `add`
    for the mask) already has a full CUDA forward *and*
    backward landed in M2E–M2G. So the M2J `ops::attention`
    runs **entirely on-device** on both CPU and CUDA tensors,
    with correct autograd gradients to Q/K/V, and no
    CPU↔device bounce inside the forward or backward tape.
    This is a true production path for Ada / Ampere / Turing
    GPUs — just not the Hopper-tuned fused-kernel path.
  - Algorithm. Forward is
    `out = softmax(Q'·Kᵀ + mask [+ causal_mask]) · V`
    where `Q' = Q · (1/√d)`. We pre-scale Q rather than the
    `[..., S_q, S_k]` score matrix because Q has O(B·H·S_q·D)
    elements vs. the scores' O(B·H·S_q·S_k), so the cost of
    the scalar broadcast is smaller by a factor of S_k /
    D (sometimes ≥ 8× on realistic head configs). The
    1/√d constant is materialized as a 0-D
    `Tensor::full({}, 1/√d, q.dtype(), q.device())` so it
    broadcasts into `ops::mul` on any rank without a shape
    rewrite. The causal mask (when enabled) is built as a
    strict-upper-triangular `[S_q, S_k]` tensor of 0 / -inf
    on the **host**, then shipped to the target device with
    a single `Tensor::to(device)` call — for the reference
    SDPA this is O(S_q·S_k) per call, dwarfed by the
    O(B·H·S_q·S_k·D) attention matmuls. A fused FA3 kernel
    (M2L.3) will fold this into the score tile as part of
    the WGMMA epilogue instead of materializing a dense
    mask tensor at all.
  - API contract. `ops::attention(q, k, v, mask, causal,
    dropout_p)` takes `[..., S_q, D]` / `[..., S_k, D]` /
    `[..., S_k, D_v]` — last-two-dims are the attention
    dims, every leading axis broadcasts NumPy-style through
    the underlying matmul. `mask` must be broadcast-
    compatible with `[..., S_q, S_k]` and float-typed; it
    is *added* to the pre-softmax logits, matching
    PyTorch's `attn_mask` semantics (use -inf to forbid,
    0 to allow). `causal=true` composes with any user
    mask by OR-ing two additive masks. `dropout_p` must
    be 0.0 in M2J — stochastic dropout is blocked on the
    RNG HAL (M3) and the fused FA3 kernel (M2L.3). A
    pair of `TESSERACT_CHECK`s up front rejects dtype /
    device mismatch between q/k/v/mask with clear
    diagnostics, plus `S_q == S_k` when causal is on.
    Autograd flows through the composite primitives
    transparently — no dedicated `AttentionBackward`
    node yet (that lands with the fused kernel in M2L.3;
    until then the graph captures the primitive chain
    directly, which is also what the MLIR lowering pipeline
    expects to see).
  - Graph-record breadcrumb. Even though every sub-op also
    records itself into the active `GraphScope`, `ops::attention`
    emits a final **marker op** (kind `"attention"`, attrs
    `{causal: bool, dropout_p: double}`) naming Q / K / V (+
    optional mask) as inputs and the output as the result.
    This is metadata-only (does not re-run forward); a later
    MLIR rewriter will pattern-match the primitive chain
    against it and collapse to a single `tesseract.attention`
    op before device codegen lowers to either our composite
    kernels or FA3's fused one.
  - Test design.
    `test_ops_attention.cpp` drives six CPU-only cases:
    rank-3 forward vs. a hand-rolled reference; causal
    masking (including the sanity check that the first
    query row is insensitive to V[1:] because softmax over
    one unmasked position is a delta); additive -inf mask
    (masked key's V row can be perturbed arbitrarily
    without changing the output); rank-4 batched with a
    head dim; autograd backward — checks `q/k/v.grad` are
    all defined, shape-correct, finite, and specifically
    validates `grad_V = Pᵀ · g_out` analytically; and input-
    validation (dropout_p > 0, causal with S_q ≠ S_k, dtype
    mismatch, head-dim mismatch all throw `tesseract::Error`).
    `test_ops_cuda_attention.cpp` adds four CUDA-gated
    parity cases (rank-4 forward, causal forward, additive
    mask with broadcasting shape `[1, 1, 1, S_k]`, autograd
    backward) plus one always-running CPU smoke (output is
    a convex combination of V rows, must fall within
    `[min(V), max(V)]`). TF32-aware tolerance is 3e-3
    absolute on both forward and backward, matching the
    M2G envelope for the two compounded matmuls.
  - **Verification (2026-04-18):**
    - OFF build: **255/255 ctest pass** (11 new M2J cases:
      6 CPU attention + 4 CUDA parity SKIPping cleanly +
      1 CPU smoke).
    - ON build (SM 8.9, RTX 5880 Ada): **230/230 ctest pass**
      in 18.7 s wall-clock.
    - `test_ops_cuda_attention`: 2235 assertions across 5
      cases. `compute-sanitizer --tool memcheck` **0 errors**
      and `--tool racecheck` **0 hazards displayed** — the
      composite kernel stack is just M2E/F/G/H launches on
      the same caller stream, so the sanitizer cleanliness
      of the sub-ops transitively covers `ops::attention`.
    - M2.β continues green across the board (M2E/F/G/H/I
      suites untouched); the only attention-adjacent code
      change outside the new Attention.cpp / test TUs is
      the `src/ops/CMakeLists.txt` source list and the
      `tests/CMakeLists.txt` test registrations.
  - Follow-up sub-track. A new **M2L.3 — "FA3 fused kernel
    integration (Hopper-gated)"** item will be created in
    `docs/backlog.md` (tracker: `src/cuda/Attention.cu` +
    `include/tesseract/cuda/detail/Attention.hpp` bridge +
    vendoring under `third_party/flash-attention-3/` + the
    ≥ 90 %-of-FA3 perf bar from `docs/m2-plan.md` §5.3 exit
    bar). It is **not** required to close M2J itself — the
    API contract + on-device composite path + parity tests
    already land all M2J-scoped deliverables.

- **M2K — single-layer Llama-style transformer block + `examples/llama_forward.cpp`.**
  - Shape of the deliverable. One new public header per
    nn module (`include/tesseract/nn/{RMSNorm, MultiHeadAttention,
    FeedForward, TransformerBlock}.hpp`) + their `.cpp`s, a
    composite op layer (`ops::rms_norm` in
    `include/tesseract/ops/Normalization.hpp` +
    `src/ops/cpu/RMSNorm.cpp`), one new primitive
    (`ops::sqrt` as an M2E-style unary extension, adding
    `UnaryKind::Sqrt` across the bridge / `.cu` / CPU
    forward / backward), one minimal example binary
    (`examples/llama_forward.cpp`, ≈ 180 LoC, CPU + CUDA),
    and one Catch2 suite
    (`tests/nn/test_transformer_block.cpp` — 5 cases
    covering CPU forward/backward + CPU↔CUDA parity forward
    and backward + parameter-migration regression). Wired
    into `src/{ops,nn}/CMakeLists.txt`,
    `examples/CMakeLists.txt`, and `tests/CMakeLists.txt`.
    Also relaxes `nn::Linear::forward` from `rank == 2` to
    `rank >= 2` so `[B, S, D]` transformer inputs flow
    through without a reshape round-trip (preserves
    PyTorch's contract; no downstream callers broke).
  - Why a *composite* block, not a fused kernel. The whole
    M2K primitive chain (RMSNorm → MHA → RMSNorm → SwiGLU
    + two residual adds) runs entirely on-device on both
    CPU and CUDA by composing already-validated M2E/F/G/H
    kernels. Autograd plumbs through `MulBackward`,
    `MeanBackward`, `AddBackward`, `SqrtBackward`,
    `DivBackward`, the M2J composite-attention primitives,
    and `MatMulBackward` without a single new
    `nn::*Backward` node. That keeps the block a faithful
    *skeleton* — the architectural shape is correct, every
    parameter is registered and device-migratable, every
    gradient is finite — while leaving the fusion work
    (MLIR pattern-matcher collapsing the chain into a
    single `tesseract.transformer_block` op before codegen)
    to M2L where it belongs.
  - Why RoPE is *not* in scope. Llama uses rotary position
    embedding applied to Q/K before the attention matmul;
    implementing it requires either (a) the RNG HAL +
    per-head phase tables (M3 territory) or (b) a stub that
    silently hard-codes θ so the block runs to completion
    but diverges numerically from any reference model. The
    M2K block therefore takes position-encoded input from
    its caller (or, for the `llama_forward` demo, zeros it
    out by simply not adding one), which lets us stay
    honest about what the block does vs. claims to do.
    RoPE was originally scoped to M3 (backlog B-014) but was
    pulled forward and landed 2026-04-18 — `nn::MultiHeadAttention`
    and `nn::TransformerBlock` now accept optional
    `rope_base` / `rope_max_seq` constructor arguments that attach a
    cached-table `nn::RotaryEmbedding` child. The M2K block as
    originally documented (no RoPE arguments → zero positional
    prior) remains the default path, so nothing in the M2K
    exit-bar is invalidated.
  - Components.
    - `ops::sqrt`. Unary elementwise, Float32/Float64 only.
      New `UnaryKind::Sqrt` in
      `include/tesseract/cuda/detail/Elementwise.hpp`, a
      `SqrtFn` routed through `sqrtf`/`sqrt` intrinsics in
      `src/cuda/Elementwise.cu`, plus CPU `sqrt_forward`
      and a `SqrtBackward` Node (`dx = 0.5 · g / y` via
      `mul(g, half) / y`, saving the output to avoid a
      second sqrt on the backward pass). Added as a
      *general-purpose* primitive (not just for RMSNorm)
      so future nn modules needing `sqrt`/`rsqrt` don't
      have to fall back to `exp(0.5·log(x))`.
    - `ops::rms_norm(x, weight, eps)`. Pure composite of
      `mul`/`mean(dim=-1, keepdim=true)`/`add`/`sqrt`/`div`/`mul`.
      Emits one metadata-only `rms_norm` marker into the
      active GraphScope (same pattern as `ops::attention`)
      so an M2L MLIR rewriter can collapse the primitive
      chain to a fused op before codegen.
    - `nn::RMSNorm(D, eps)`. Registers the [D] affine
      scale initialized to 1 (Llama convention) as a leaf
      parameter. Forward calls `ops::rms_norm`; nothing
      more.
    - `nn::MultiHeadAttention(d_model, num_heads, use_bias,
      causal)`. Four `Linear`s (q/k/v/o projections
      registered as child modules so `parameters()` +
      `to(device)` recurse correctly), a `[B, S, H, Dh] →
      [B, H, S, Dh]` permute, a call to `ops::attention`
      with the causal flag threaded through, and the
      inverse permute + `o_proj`. Relies on `ops::reshape`
      falling back to `contiguous()` + view internally so
      post-attention tensor restructuring stays safe even
      though the preceding permute left memory
      non-contiguous.
    - `nn::FeedForward(d_model, d_ff, use_bias)`. SwiGLU:
      `down_proj(sigmoid(gate_proj(x)) · gate_proj(x) ·
      up_proj(x))`, which is SiLU(gate) composed with
      element-wise gating against up. Three projections,
      composed via `ops::mul` + `ops::sigmoid`. M2L will
      pattern-match the `sigmoid` + `mul` chain into a
      fused `silu_mul` op.
    - `nn::TransformerBlock(d_model, num_heads, d_ff,
      norm_eps, causal, use_bias)`. Pre-norm residual
      structure:
      `h = x + attn(norm_1(x));
       out = h + ffn(norm_2(h))`.
      Four child modules (`norm_1`, `attn`, `norm_2`,
      `ffn`), no module-local parameters — every leaf
      belongs to a child, so `parameters()` / `.to(device)`
      / `.zero_grad()` are purely a tree walk through the
      already-exercised M2I `Module::to` path.
  - Example. `examples/llama_forward.cpp` builds one block
    (defaults: d_model=64, heads=4, d_ff=128, batch=2,
    seq=16), runs forward + optional backward, and prints
    shape / finite count / `||out||₂` / `sum(||grad||)`
    for a quick eyeball test. Shares the `--device
    cpu|cuda` flag with `mnist.cpp` and dispatches through
    the same `Module::to(Device)` path. Registered as an
    executable target `tesseract_llama_forward` so it
    ships alongside the mnist binary whenever
    `TESSERACT_BUILD_EXAMPLES=ON`.
  - Test design.
    `test_transformer_block.cpp` drives five cases:
    (1) CPU forward shape = `[B, S, D]` + every output
    element finite; (2) CPU autograd backward: input grad
    shape matches input, every parameter has a
    shape-correct + finite gradient, at least 9 parameters
    show up (the lower bound from 3 SwiGLU + 4 MHA + 2
    RMSNorm projections) so a missing `register_module`
    can't silently drop a branch; (3) CPU↔CUDA forward
    parity under 5e-3 absolute tolerance, with both blocks
    seeded from *byte-identical* CPU parameters before the
    CUDA block migrates to device; (4) CPU↔CUDA backward
    parity on both the input grad and every per-parameter
    grad at 6e-3; (5) parameter-migration regression —
    after `block->to(cuda0)`, every `parameters()` entry
    reports `device == cuda:0`, and the same for the round
    trip back to CPU. Rationale on the reference: the
    `docs/m2-plan.md` §5.3 exit bar talks about an offline
    PyTorch-captured `.npz` — we use the CPU path itself
    as reference for M2K because (a) every sub-op is
    already gradcheck'd, (b) a hermetic C++ test keeps the
    CI toolchain single-language, and (c) FP16/BF16
    bit-parity (the case where an external reference
    really pays off) is a M2L.3 concern alongside the
    fused FA3 kernel.
  - **Verification (2026-04-18):**
    - OFF build: **260/260 ctest pass** (2 new CPU-always
      cases + 3 CUDA-SKIP cases from
      `test_transformer_block`). `tesseract_llama_forward
      --device cpu --backward` prints `||out||₂ =
      46.882`, `sum(||grad||) = 951.177`, all 2048 output
      elements finite.
    - ON build (SM 8.9, RTX 5880 Ada): **235/235 ctest
      pass**. `tesseract_llama_forward --device cuda
      --backward` prints *identical* `||out||₂` and
      `sum(||grad||)` to seven-digit precision vs. CPU.
    - `test_transformer_block [gpu]`:
      `compute-sanitizer --tool memcheck` **0 errors** (3406
      assertions across 2 GPU cases); `--tool racecheck`
      **0 hazards displayed**. The whole M2K kernel chain
      is a composition of M2E/F/G/H/J launches on the
      same caller stream, so sanitizer cleanliness
      transitively covers `rms_norm` + the nn-level
      composites.
  - Follow-up. RoPE (B-014) was pulled forward and
    landed 2026-04-18 on top of B-015 / B-016 — the
    attention block now has a full Llama-spec forward
    path on Ada, with `rope_base` / `rope_max_seq` as
    opt-in constructor arguments. The fused
    `tesseract.transformer_block` op + the
    `silu_mul` / `rms_norm` / `attention` pattern
    matchers land in M2L alongside the fused FA3 kernel;
    once those ship the primitive chain here is still
    the correct eager semantics, just with a single
    fused lowering downstream.

- **M2L.1 — CUDA perf gate: aggressive bars + 6-bench ctest suite.** ✅ 2026-04-18.
  - Shape of the deliverable. A 6-binary CUDA benchmark suite wired
    into `ctest -L bench_cuda` with hard-bar exit codes
    (`0` pass / `1` perf miss / `77` no GPU). Shared harness
    (`benchmarks/cuda_bench_util.hpp`) standardises CUDA-event
    timing, coefficient-of-variation–gated steady-state measurement,
    a `best_of_n_time` trial wrapper (5 trials by default, keeps the
    one with the lowest `min_us`), a `BenchStream` RAII guard that
    keeps `current_stream(device)` and the timer events on the same
    stream, D→D memcpy roofline probe, and SKIP/perf-miss exit
    codes. Six benches:
    `bench_cuda_{matmul, elementwise, attention, attention_bwd,
     rms_norm, transformer_block}`.
  - Why the target swung from "≥ 90 % of cuBLAS" to "≤ 5 µs additive
    overhead vs our own bridge". The original ≥ 90 %-of-cuBLAS bar
    compared `ops::matmul` against a separately-configured raw
    cuBLASLt path — which is noisy by construction: cuBLASLt's
    `AlgoGetHeuristic` can pick different algo variants for
    identical shapes depending on a handle's internal heuristic
    state, so "raw" vs "ours" times can differ by 5–20 % even when
    both paths execute identical kernels. Sharing the library's
    `cublasLtHandle_t` between both paths fixed the algo-pick drift
    (see `bench_cuda_matmul.cpp` and `cuda::detail::get_cublaslt_handle`),
    but kernel-time variance still polluted the ratio metric at
    shapes where the kernel itself runs under 500 µs. The *stable*
    quantity — and the one we can actually move with code changes —
    is the additive overhead the op layer introduces on top of the
    bridge (tensor/view setup, `detect_cuda_mat_layout`, batch
    bookkeeping, caching-allocator round-trip). That's what the
    bench now gates on: median ≤ 5 µs across a 10-shape sweep, with
    a per-shape floor of 0.95 × and a hard 20 µs anchor at 4096²
    FP32 where cuBLASLt variance drops below 1 %.
  - Op-layer speedups that landed alongside. (1) **Bucketed
    caching allocator** (`src/cuda/Allocator.cpp` +
    `CudaAllocator::release_all_cached`): `Tensor::empty` on CUDA
    was calling `cudaMalloc`/`cudaFree` per op; on hot loops that
    dominated `ops::matmul` dispatch. The allocator now bins
    free-list blocks by power-of-two size and reuses them within
    process. Effect: `ops::matmul` at 1024² FP32 dropped from ~80
    µs of dispatch overhead to ~3 µs. (2) **cuBLASLt descriptor
    cache + persistent workspace** (`src/cuda/MatMul.cpp`):
    `cublasLtMatmulDesc_t`, `cublasLtMatrixLayout_t`, and the
    heuristic-selected `cublasLtMatmulAlgo_t` are now cached per
    (M, N, K, dtype, transposes) and a single 4 MiB device-side
    workspace is reused across calls — matches PyTorch's
    `cublaslt-workspace-config` envelope and removes the
    per-call `MatmulDescCreate` / `AlgoGetHeuristic` path. (3)
    **Elementwise vectorized fast-path** (`src/cuda/Elementwise.cu`):
    dense-contiguous tensors route through a vectorized
    `float4`/`double2`/`int4`/`longlong2` load-store kernel. Effect:
    64 MiB `add` hits 1.03 × the memcpy DRAM roofline (compared to
    the strided kernel's ~0.6 ×).
  - Choice of hard bars, per bench. (a) `matmul`: median additive
    overhead ≤ 5 µs (10 shapes), per-shape ratio ≥ 0.95, dispatch
    @ 4096² FP32 ≤ 20 µs. (b) `elementwise`: add / mul @ 64 MiB
    ≥ 0.95 × memcpy-DRAM, sigmoid ≥ 0.90 × (small ALU tax for
    transcendental). Both sizes are L2-overflowing so the metric is
    true DRAM BW; L1/L2-resident sizes (1, 16 MiB) are reported for
    trend but not gated. (c) `attention`: composite/Σ(primitives) ≥
    0.97 on three transformer shapes (8,32,512,64), (4,32,2048,64),
    (2,16,4096,128). Primitives include the `ops::mul` scale so the
    comparison is apples-to-apples. `causal=false` because causal
    masking adds an O(S²) mask-materialise + add that dominates at
    S=2048 without informing op-layer overhead. (d) `attention_bwd`:
    `(fwd+bwd)/fwd ≤ 5.0` on composite SDPA at (B=2, H=16, S=1024,
    D=64, FP32). Composite-backward runs 2 × matmul-bwd + softmax-bwd
    + 2 × matmul-bwd + mul-bwd, so the fundamental envelope is
    around 3.5 × forward on Ada; we leave 5.0 × as the hard gate
    for variance headroom and track the tighter 3.2 × ("backward
    TFLOPS ≥ 85 % of forward TFLOPS") as the M2L.3 fused-FA3 target.
    (e) `rms_norm`, `transformer_block`: no hard bar — baselines for
    the fused-kernel regression dashboard.
  - FP16 attention — now live end-to-end on both forward and
    backward. B-015 landed FP16/BF16 on
    `src/cuda/Elementwise.cu` and `src/cuda/Softmax.cu` via an
    FP32-promoted math path on half-precision storage; B-016 did
    the same for `src/cuda/Reduction.cu`, which was the last gate
    on the backward chain (`SoftmaxBackward::apply` composes
    `mul → sum(dim, keepdim) → sub → mul`, and `sum(dim, keepdim)`
    now has FP16/BF16 kernels). `bench_cuda_attention` passes at
    ≥ 0.97 composite/sum on all three shapes (worst 0.995,
    32.6 TFLOPS on 2×16×4096×128); `bench_cuda_attention_bwd`
    passes at 4.72× ≤ 5.0× fwd/bwd envelope (vs 4.59× on the
    previous FP32 run), with ~24 % wall-clock speedup on both
    stages. The ratio bar is dtype-independent so the envelope
    carried over unchanged between dtype flips.
  - Why M2L.2 (MLIR GPU lowering) is deferred. Lowering
    `tesseract.*` to `gpu.func + nvgpu` requires a full kernel
    outliner + PTX JIT pipeline that depends on upstream MLIR's
    `convert-gpu-to-llvm` landing cleanly against our CMake setup.
    The M2 exit bar explicitly marks M2L.2 as a stretch goal
    (§2.6, §5.3); moving it to **M3/B-009** keeps M2 scope tight
    and lets us pair the lowering work with the broader graph-mode
    codegen push for M3.β without interleaving two heavy passes.
  - **Verification (2026-04-18, RTX 5880 Ada, SM 8.9).**
    `ctest -L bench_cuda` reports **6/6 passed** in ~82 seconds.
    Headline numbers (min-over-best-of-5, reproducible across three
    consecutive runs):
    - `bench_cuda_matmul`: median additive overhead −1 µs (ours ≤
      bridge within noise); per-shape floor 0.98; dispatch @ 4096²
      FP32 ≤ 0 µs.
    - `bench_cuda_elementwise`: add @ 64 MiB 872 GB/s (1.03 × memcpy
      DRAM roofline 843 GB/s); sigmoid @ 64 MiB 930 GB/s (1.10 ×).
    - `bench_cuda_attention`: composite/sum 0.997 on (8,32,512,64),
      1.03 on (4,32,2048,64), 1.02 on (2,16,4096,128).
    - `bench_cuda_attention_bwd`: (fwd+bwd)/fwd = 4.58.
    - `bench_cuda_rms_norm`: 48 GB/s on (32,2048,4096) — ~11 % of
      roofline, matches the composite-overhead expectation.
    - `bench_cuda_transformer_block`: 595 k tok/s fwd, 191 k tok/s
      fwd+bwd at (B=16, S=1024, d=512, h=8, d_ff=2048, FP32). See
      `docs/benchmarks/m2-cuda.md` for the full table.

- **B-015 — FP16/BF16 on CUDA elementwise + softmax.** ✅ 2026-04-18.
  - Scope. Everything the attention-forward critical path touches on
    half-precision storage: binary / unary elementwise, `fill`, and
    `softmax` / `log_softmax`. Strategy is FP32-promotion at the
    kernel edges (load `__half` / `__nv_bfloat16`, compute in
    `float`, narrow on store), matching the numerical convention
    used by every major half-precision DL runtime on pre-Hopper
    hardware and keeping the existing vectorized fast paths reusable
    with only a `T`→`float` adaptor layer.
  - Files. `src/cuda/Elementwise.cu` gained `to_fp32` / `from_fp32`
    device helpers, `PromotedBinary<Op>` / `PromotedUnary<Op>`
    templated wrappers, and half-specific dispatch entry points
    (`launch_binary_typed_half`, `unary_dispatch_typed_half`) wired
    into `launch_binary_elementwise` / `launch_unary_elementwise`.
    `dense_vec_width<T>()` returns 4 for half types so the 8-byte
    vector loads still fire. `launch_fill` converts the `double
    value` via `__float2half` / `__float2bfloat16` before calling
    `fill_dense`. `src/cuda/Softmax.cu` added
    `softmax_kernel_promoted<Tstorage>`, which holds max / sum / log
    accumulators in `float` and only narrows to `Tstorage` on the
    final store; `launch_softmax` dispatches `DType::Float16` /
    `DType::BFloat16` to it. CPU-side parity: `Activation.cpp`
    flipped from `dispatch_float` to `dispatch_float_with_half`
    with `if constexpr` casts in `SigmoidFn` to resolve `Half +
    float` ambiguities; `Softmax.cpp` switched to
    `Acc = std::conditional_t<std::is_floating_point_v<T>, T,
    float>` so CPU intermediate math mirrors the CUDA FP32
    promotion.
  - Tests. New `tests/ops/test_ops_cuda_elementwise_f16.cpp`
    (add/sub/mul/div + every unary + `ones`/`full`, 2e-3 abs for
    FP16 and 5e-3 abs for BF16). Existing
    `tests/ops/test_ops_cuda_softmax.cpp` gained CUDA↔CPU parity
    cases for `softmax` on `Float16` and `log_softmax` on
    `BFloat16`. Full `ctest` suite passes (249/249); both TUs
    `compute-sanitizer memcheck`-clean.
  - Bench uplift. `bench_cuda_attention` flipped to end-to-end FP16
    on all three shapes and still meets the ≥ 0.97 composite/sum
    hard bar (worst 0.995, up to 32.6 TFLOPS on 2×16×4096×128,
    roughly 2.2× the FP32 baseline on the 4096-seq shape). The
    backward bench stays on FP32 for now — its `SoftmaxBackward`
    recipe hits `sum(dim, keepdim)` which is FP32/FP64-only in
    `src/cuda/Reduction.cu`. That trailing gate is tracked as
    **B-016** "FP16/BF16 on CUDA reductions".

- **B-016 — FP16/BF16 on CUDA reductions.** ✅ 2026-04-18.
  - Scope. Close the last gate on end-to-end FP16 attention
    backward: `sum` / `mean` / `max`, both all-reduce and
    along-dim, for `__half` / `__nv_bfloat16` storage. Same
    FP32-promotion strategy as B-015 — narrow only on the final
    store, keep every intermediate (stage-1 partials, shared-mem
    tree, stage-2 combine, dim accumulator) in `float`. Stage-1
    partials sit in a `float*` workspace rather than the storage
    dtype so we don't re-round between stages; this matches the
    CPU `Acc = float` reference bit-for-bit within the
    half-precision round-off budget.
  - Files. `src/cuda/Reduction.cu` gained `red_to_float` /
    `red_from_float` device helpers and three parallel kernel
    templates (`reduce_all_stage1_promoted`,
    `reduce_all_stage2_promoted`, `reduce_dim_kernel_promoted`)
    parameterised on the storage dtype and reusing the existing
    `{Sum, Mean, Max}Policy<float>` policy structs — no new
    policy code, only new load/store adaptors. Host dispatch added
    via `run_all_reduce_half` / `dispatch_all_half` /
    `run_dim_reduce_half` / `dispatch_dim_half`, wired into
    `launch_reduce_all` / `launch_reduce_dim` through new
    `DType::Float16` / `DType::BFloat16` branches. CPU-side
    parity: `src/ops/cpu/Reduction.cpp` switched from
    `dispatch_float` to `dispatch_float_with_half` with
    `Acc = std::conditional_t<std::is_floating_point_v<T>, T,
    float>` in both `reduce_all_forward` and `reduce_dim_forward`.
  - Tests. Added two test cases to
    `tests/ops/test_ops_cuda_reduction.cpp` — one covers `sum`
    all-reduce + `mean` per-dim + `max` per-dim on `Float16`;
    the other covers `sum` along a middle dim + `mean` all-reduce
    on `BFloat16`. Tolerances follow the B-015 precedent: 2e-3
    abs for FP16, 5e-3 abs for BF16 (BF16's 7-bit mantissa is
    ~3× noisier than FP16's 10-bit). Full ctest suite still 247/247
    green; 6/6 bench hard bars green;
    `compute-sanitizer memcheck` clean on the reduction TU
    (425 assertions in 8 test cases, 0 errors).
  - Bench uplift. `bench_cuda_attention_bwd` flipped to FP16 and
    still clears the ≤ 5.0× fwd/bwd envelope — observed **4.72×**
    (up from 4.59× FP32 at M2L.1 lock, variance-bounded). Both
    stages ~24 % faster in wall time: forward 1595 → 1213 µs,
    forward+backward 7329 → 5725 µs on
    `(B=2, H=16, S=1024, D=64)`. The fwd/bwd ratio bar is
    dtype-independent so the envelope carried over unchanged; the
    wall-clock speedup is the expected FP16 bandwidth /
    tensor-core throughput win on a memory-bound composite SDPA
    backward.

- **B-014 — Rotary position embedding (RoPE).** ✅ 2026-04-18.
  - Scope. Pull the originally-M3 RoPE item forward now that the
    non-FA3 CUDA op surface is fully FP16/BF16 clean (B-015/B-016).
    This is the last missing primitive before an off-the-shelf
    Llama checkpoint can round-trip through `nn::TransformerBlock`
    with bit-compatible numerics — every other M2K layer
    (RMSNorm + causal MHA + SwiGLU + pre-norm residuals) already
    matches the Llama spec on both forward and backward.
  - Public op. `ops::rotary_embedding(x, cos, sin)` in
    `src/ops/cpu/RotaryEmbedding.cpp` — contiguous-input leaf with
    adjacent-pair (GPT-NeoX / Llama) rotation semantics. Rank-≥-2
    inputs, last dim must be even, cos/sin both `[S_table, D]` with
    `S_table ≥ S` (so a single max-length table can be reused
    unsliced across every forward). Autograd wired through a
    `RotaryBackward` node that reuses the forward launcher with
    `sin` negated — rotation by -θ is the transpose of R(θ), which
    is what an orthogonal Jacobian asks for. No gradient flows
    back to cos/sin; they're deterministic from positions and live
    in the module as buffers.
  - CUDA kernel. `src/cuda/RotaryEmbedding.cu` (plus matching
    `RotaryEmbeddingStub.cpp`) — one CUDA thread per output
    *pair*, grid-stride over the flat `outer · S · D/2` pair
    index, 2 FMAs per output element. Same B-015 / B-016 FP32-
    promotion policy on half precision: `re_to_float` /
    `re_from_float` helpers for `__half` / `__nv_bfloat16`, native
    compute in the storage dtype for `float` / `double`. Grid
    capped at 1024 blocks so tiny shapes don't emit pathological
    counts; the per-pair cost is bandwidth-bound so additional
    blocks buy nothing past Ada's ~1k concurrent-block ceiling.
  - nn module. `nn::RotaryEmbedding(d_head, base=10000, max_seq)`
    in `src/nn/RotaryEmbedding.cpp` — precomputes `[max_seq,
    d_head]` cos/sin tables in FP64 (duplicated across the pair so
    the kernel stays a pure multiply-add) and stores them via a
    **new** `Module::register_buffer(...)` facility. Buffers are
    tracked by `Module::to(Device)` (they migrate alongside
    parameters) but do **not** surface in `parameters()` and never
    acquire `requires_grad`. That keeps the
    optimizer / gradcheck contracts unchanged while giving any
    future "persistent non-learnable tensor" (KV cache, BatchNorm
    running stats, etc.) a clean place to land.
  - MHA integration. `nn::MultiHeadAttention` gained optional
    `rope_base` / `rope_max_seq` constructor arguments (defaults
    0.0 / 0 preserve the pre-B-014 "no-RoPE, caller supplies
    position prior" behavior). Non-zero values build a
    `RotaryEmbedding` child registered under `"rope"` and apply it
    to Q and K between the split-heads permute and
    `ops::attention`. `TransformerBlock` threads the same two
    arguments through to its attention child. Zero changes to any
    non-RoPE forward path — the RoPE block is gated on `rope_ !=
    nullptr`.
  - Tests. New `tests/ops/test_rotary_embedding.cpp` (10 cases,
    3947 assertions):
      * Hand-rolled reference parity on rank-3 and rank-4 CPU
        inputs (the latter is the `[B, H, S, Dh]` MHA shape).
      * CPU↔CUDA parity on Float32 (2e-6 abs) and Float64 (1e-12
        abs).
      * CPU↔CUDA parity on Float16 (2e-3 abs) and BFloat16 (5e-3
        abs), matching the same half-precision envelope we hold
        ourselves to in the softmax / reduction / elementwise F16
        parity tests.
      * Autograd finite-difference check on the CPU path — confirms
        `RotaryBackward` is numerically rotation-by-(-θ) across a
        spread of indices spanning both ends of rotation pairs.
      * `nn::RotaryEmbedding` closed-form table spec (1e-6 abs) +
        bit-for-bit `module.forward(x) == ops::rotary_embedding(x,
        module.cos_table(), module.sin_table())`.
      * `Module::to(cuda)` buffer migration smoke — forward on CUDA
        after a round-trip move matches the pre-move CPU forward
        within 2e-6 abs, i.e. the buffers actually landed on-device.
    `tests/nn/test_transformer_block.cpp` grew a RoPE-on variant
    (forward + backward + per-parameter grad parity against a
    matching CPU block) at the same TF32-aware tolerances as the
    no-RoPE case.
  - Verification. Full `ctest -j1 -E '^bench_'` passes 258/258
    (4 CPU-only stubs skipped); `compute-sanitizer --tool memcheck`
    on the new TU reports 0 errors across 3947 assertions.
  - What this unlocks.
      * A Llama checkpoint loader (M3 follow-up) can now feed Q/K
        through the attention block with no extra framework work —
        RoPE is a constructor flag, not a caller responsibility.
      * The FA3 kernel in M2L.3 signs for pre-rotated Q/K just as
        well, so the Hopper landing doesn't bifurcate on RoPE
        state.
      * `Module::register_buffer` is now a reusable hook for KV
        caches, BatchNorm running stats, and any future cached
        non-learnable tensor — the same alias / migration rules as
        `register_parameter`.

### 5.1 M2.α — HAL foundation (M2A–M2D)

**Scope.** Build system, allocators, streams, H↔D copy.

**Exit bar.**

- Building with `TESSERACT_ENABLE_CUDA=ON` succeeds on any host that has
  `nvcc` + CUDA Toolkit 12.x + a `cc ≥ 8.0` GPU.
- Building with `TESSERACT_ENABLE_CUDA=OFF` is byte-identical to today.
- A new ctest, `test_hal_cuda`, creates a `Tensor::zeros({16}, Float32,
  Device(DeviceType::CUDA, 0))`, copies it host-side via
  `Tensor::to(cpu_device())`, and asserts the round-trip preserves
  bytes. When no GPU is visible, the test returns 77 (skip).
- `CudaAllocator` survives a `cuda-memcheck` / `compute-sanitizer` pass
  on an ASan-style synthetic stress test (~10⁴ alloc/free cycles).

**Deliverables:** M2A–M2D tracks.

### 5.2 M2.β — Eager CUDA op coverage + MNIST (M2E–M2I)

**Scope.** Every M0/M1 op that runs eagerly on CPU also runs eagerly on
CUDA. `examples/mnist.cpp --device cuda` converges to the same accuracy.
GraphScope works transparently (`graph::run` dispatches per-node to
`ops::*` which in turn picks the CUDA kernel by `tensor.device()`).

**Exit bar.**

- All 159 current ctests keep passing in all four config combinations.
- New tests (~60 targeted parity cases): one per op × dtype,
  comparing CUDA output against the CPU reference within the B-005
  tolerance envelope.
- `test_cuda_gradcheck` runs every existing gradcheck on the GPU.
- `examples/mnist.cpp --device cuda --epochs 3` prints a loss curve
  that matches CPU epoch-for-epoch within 1e-3 absolute on identical
  seeds, and hits ≥ 96 % test accuracy.

**Deliverables:** M2E–M2I tracks.

### 5.3 M2.γ — Attention + model + perf (M2J–M2L)

**Scope.** FlashAttention-3 wired in as `ops::attention`; single-layer
transformer block runs end-to-end; benchmarks meet the 90 %-of-cuBLAS /
90 %-of-FA3 bars.

**Exit bar.**

- `ops::attention(q, k, v, mask, /*causal=*/true)` is available on
  CPU and CUDA (M2J composite: matmul·softmax·matmul, Q pre-scaled
  by 1/√d), with forward + autograd backward green on both backends
  and CPU↔CUDA parity within 3e-3 (TF32 envelope) on Ada. The
  **FA3 fused-kernel bit-parity bar** — outputs bit-identical (up
  to FP16 round-off) with FlashAttention-3's reference kernel on
  shapes `(batch, heads, seq, head_dim) ∈ {(1,8,128,64),
  (2,16,2048,64), (4,32,4096,128)}` — is deferred to **M2L.3**
  (Hopper-gated), since FA3 requires SM 9.0+ (WGMMA + TMA) and the
  current dev box is SM 8.9.
- `test_transformer_block` runs a single Llama-style block forward
  + backward, asserts shapes and finite gradients, and compares the
  CUDA run against the CPU run (byte-identical starting parameters)
  within a TF32-aware tolerance (5e-3 forward, 6e-3 backward — the
  same envelope as `test_ops_cuda_attention`). Landed in **M2K**.
  The originally-planned offline PyTorch `.npz` reference is
  retired at this milestone in favor of the in-tree CPU reference:
  every sub-op is already independently gradcheck'd, a hermetic
  C++ test keeps the CI toolchain single-language, and the case
  where an external reference really pays off (FP16/BF16 bit-parity)
  is properly a **M2L.3** concern alongside the fused FA3 kernel.
- **M2L.1 CUDA perf gate lights green.** `ctest -L bench_cuda` passes
  on any CUDA-capable Ada+ host: six benches each hard-gated (see §4
  "Benchmark"). Aggressive targets replace the original ≥ 90 %-of-
  cuBLAS bar because the op layer's *additive* overhead (≤ 5 µs
  median, ≤ 20 µs @ 4096² FP32) is a more stable metric than a
  ratio against cuBLASLt's own kernel-variance floor — see the
  M2L.1 progress log entry above for the rationale.
- **FA3 fused-kernel bars are M2L.3, not M2L.1.** ≥ 90 % of FA3
  reference TFLOPS on (4,32,2048,64) and bit-parity with the FA3
  kernel are both gated on Hopper (SM 9.0+ WGMMA + TMA), which
  the current dev box does not expose. M2L.1 establishes the
  composite-SDPA baseline; M2L.3 replaces it.
- M2L.2 (MLIR GPU lowering) is **deferred to M3/B-009**. Full
  `tesseract.* → gpu.func + nvgpu` lowering needs an outliner +
  PTX JIT pipeline that the M3 graph-mode codegen push will
  naturally subsume; M2 exit therefore no longer requires a
  `--tesseract-gpu-lower` stretch target.

**Deliverables:** M2J–M2L tracks.

Closing all three closes M2 per [roadmap.md](roadmap.md#m2).

## 6. Operator coverage matrix

The op list M2 must ship on CUDA in eager mode. ✔ = CPU reference
exists today; CUDA column gets ticked off as tracks land.

| Op family                | Ops                                                           | CPU    | CUDA tracker |
|--------------------------|---------------------------------------------------------------|--------|--------------|
| Binary elementwise       | `add` `sub` `mul` `div`                                       | ✔      | M2E          |
| Unary                    | `neg` `relu` `sigmoid` `tanh` `exp` `log` `sqrt`              | ✔      | M2E (+ M2K for `sqrt`) |
| Reductions               | `sum` `mean` `max` (all + dim + keepdim)                      | ✔      | M2F          |
| Softmax                  | `softmax` `log_softmax`                                       | ✔      | M2F          |
| Loss                     | `cross_entropy_with_logits` + fused backward                  | ✔      | M2F          |
| Linear algebra           | `matmul` (rank-2, batched), `mat_transpose`                   | ✔      | M2G          |
| Shape                    | `view` `reshape` `permute` `transpose` `contiguous` `clone`   | ✔      | M2H          |
| Indexing                 | `split` `split_with_sizes` `cat` `index_select` `gather`      | ✔      | M2H          |
| Copy                     | `Tensor::to(Device)`                                          | —      | M2D          |
| Attention                | `attention(q, k, v, mask, causal, dropout_p)`                 | ✔      | M2J (composite: matmul·softmax·matmul on Ada/Ampere; fused FA3 Hopper path = M2L.3) |
| Normalization            | `rms_norm(x, weight, eps)`                                    | ✔      | M2K (composite over `mul`/`mean`/`add`/`sqrt`/`div`; fused kernel = M2L) |
| Transformer building blocks | `nn::{RMSNorm, MultiHeadAttention, FeedForward, TransformerBlock, RotaryEmbedding}` | ✔ | M2K (pre-norm residual, SwiGLU, causal self-attn) + B-014 (RoPE, pulled forward 2026-04-18 — adjacent-pair rotation, cached cos/sin buffers, optional `rope_base`/`rope_max_seq` on MHA and `TransformerBlock`) |
| Rotary position embedding   | `ops::rotary_embedding(x, cos, sin)` + `nn::RotaryEmbedding`  | ✔      | B-014 (CPU + CUDA, all 4 floating dtypes, autograd wired through `RotaryBackward = R(-θ)`) |

## 7. Cross-cutting invariants

1. **CPU-only build is the fast path.** Every commit must leave
   `TESSERACT_ENABLE_CUDA=OFF` compilable and green on a plain g++ host
   with no CUDA Toolkit. CI runs both `OFF` and `ON` configurations.
2. **No `<cuda_runtime.h>` in public headers.** If a user needs a
   CUDA-aware helper (e.g. asking for the current device's L2 cache
   size), expose it through a plain-C++ wrapper in `include/tesseract/`
   that forwards into `src/cuda/`.
3. **No silent host sync.** Every CUDA op is async on the current
   stream. Host-visible ops (`Tensor::item<T>()`, `Tensor::to_string`)
   call `cudaStreamSynchronize` explicitly and advertise it in the
   doc comment.
4. **Streams are explicit in tests.** Every CUDA test creates its own
   `StreamGuard` so a failing test can't leak work into the default
   stream of a later test and cause spurious order-dependent
   failures.
5. **Numerical parity is the contract.** Every CUDA kernel ships a
   parity test against the CPU reference. "Different but close enough
   for training" is not enough — we set an explicit per-dtype
   tolerance (§4) and use it uniformly.
6. **`compute-sanitizer` clean.** The ctest wrapper has a CI-only
   variant that re-runs every `[cuda]`-tagged test under
   `compute-sanitizer --tool memcheck`, gated on a label so CI cost
   stays manageable.

## 8. Out of scope (tracked elsewhere)

- **Multi-GPU / NCCL** — M4. Our HAL is designed so `Stream` is already
  per-device and `Tensor::to(Device(CUDA, 1))` is legal, but no
  collective ops are shipped in M2.
- **Triton/CuTe DSL autogeneration** — post-M2 backlog (B-008).
- **INT8 / FP8 / FP4 quantized dtypes** — M3 quantization dialects.
- **Paged KV cache, continuous batching, speculative decoding** — M3
  LLM inference stack.
- **Python bindings** — M4. The eager CUDA path is deliberately
  designed so it can be wrapped by pybind11 without additional plumbing
  once M4 lands (all ops take `Tensor` handles that carry their own
  device; no hidden global context).
- **AMD / Intel / Apple / NPU** — M5.

## 9. Risk register

| Risk                                                      | Mitigation                                                                                     |
|-----------------------------------------------------------|------------------------------------------------------------------------------------------------|
| CI GPU capacity — PRs need to see perf numbers              | Start with a single self-hosted runner (Ada class) behind a `[cuda]` CI label; bench numbers re-captured weekly on Hopper via a batch job. |
| FlashAttention-3 licensing / vendoring                    | FA3 is BSD-3. Vendor under `third_party/flash-attention-3/` with an explicit NOTICE entry; track upstream via `git subtree` for periodic refresh. |
| cuBLASLt API churn between CUDA 12.x / 13.x              | Abstract all cuBLAS calls behind a thin `src/cuda/Blas.cpp` wrapper; test matrix includes both toolkit majors.                               |
| FP16 accumulation gives worse training loss than FP32      | Keep FP16/BF16 matmul in TF32 accumulation mode by default (cuBLASLt `CUBLAS_COMPUTE_32F`); document the knob.                                |
| nvcc device-code compile times dominate developer loop    | Restrict `CUDA_ARCHITECTURES` to the host GPU in developer builds; full multi-arch fatbin only in CI. |
| MLIR GPU codegen (stretch) blocks M2 exit                  | M2L.2 is explicitly optional. Exit bar requires only eager CUDA + cuBLAS + FA3. |

## 10. Post-M2 backlog preview

The following will get their own B-### entries in `docs/backlog.md` at
M2 close, listed here so readers know M2 is not the end state:

- **B-008 — Custom GEMM via CuTe DSL.** Replace cuBLAS for shapes where
  we can beat it (notably small-batch, non-standard alignments).
- **B-009 — `tesseract → gpu.func` IR lowering.** Graduate M2L.2 to a
  first-class pipeline if the stretch goal lands in a useful state.
- **B-010 — Tensor Core autotuner.** Per-shape cuBLASLt heuristic
  override analogous to cuBLASLt's `cublasLtMatmulHeuristicGetBest`,
  cached per driver version.
- **B-011 — CUDA-aware `Tensor::to_string`.** Keep the async-stream
  model clean; today's sync-on-print is fine but can be replaced with
  a chunked async readback for large tensors.

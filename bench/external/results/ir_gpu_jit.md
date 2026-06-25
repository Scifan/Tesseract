# Phase 8 (B-009 tail) — GPU JIT: gpu.module → NVVM → PTX → cubin → launch

Strict isolation (clean RTX 5880 Ada, sm_89, CUDA 12.8, local LLVM/MLIR 18.1
built **with the NVPTX target**). This closes the long-standing B-009 tail:
the graph compiler now lowers a data-parallel graph all the way to a real
**cubin** and runs it on the GPU, matching eager bit-for-bit-close.

## What was built

* **Device compilation pipeline** (`GpuJitEngine` + the extended
  `--convert-tesseract-to-gpu`): the existing tesseract → linalg → bufferize →
  parallel-loops → gpu + outlining chain, now followed in-process by
    `nvvm-attach-target{chip=sm_89}` → `convert-gpu-to-nvvm` →
    `convert-nvvm-to-llvm` → `reconcile-unrealized-casts` →
    `gpu-module-to-binary{format=bin}`
  which invokes the in-tree NVPTX backend + `ptxas` to serialize the outlined
  kernel to a real cubin ELF (`gpu.binary` with a `#gpu.object`). Pinned by
  `tests/ir/gpu_to_cubin.mlir` FileCheck (`ir_gpu_to_cubin`) — runs offline,
  no GPU needed.
* **Elementwise fusion in the GPU pipeline**: named linalg ops are generalized
  and elementwise chains fused **before** bufferization, so e.g. `mul → relu`
  outlines into **one** kernel/launch instead of two — fewer launches and the
  single-kernel contract the launcher relies on.
* **`GpuJitEngine`** (`src/ir/GpuJitEngine.cpp`) — the `build_for_gpu` half of
  the JIT (the CPU half is `JitEngine` via `mlir::ExecutionEngine`). It runs
  the pipeline, extracts {kernel name, constant grid/block, operand→buffer
  map, cubin bytes} from the lowered IR, then drives the **CUDA driver API**
  directly: `cuModuleLoadData(cubin)` → `cuModuleGetFunction` →
  H2D `cuMemcpyHtoD` inputs → `cuLaunchKernel` (memref C-ABI: alloc/aligned
  ptr, offset, sizes, strides per operand; scalar constants like relu's 0.0
  threshold passed inline) → `cuCtxSynchronize` → D2H outputs. No MLIR CUDA
  runtime dependency.

## Eager-CUDA parity (`tests/ir/test_gpu_jit_parity.cpp`)

Compiled-cubin output vs the eager kernels, ≤1e-5:

| graph                 | shape    | result |
|-----------------------|----------|--------|
| `add(a,b)`            | 64×64    | PASS (4096 elems exact) |
| `relu(mul(a,b))` fused | 128×32  | PASS (4096 elems exact) |

**8196 assertions, all pass.** Self-skips cleanly when no CUDA device is
visible, so CPU-only CI stays green.

## Notes

* The LLVM build lacked `libmlir_cuda_runtime.so` (the MLIR host `gpu-to-llvm`
  + `mgpu*` runner path), and that host conversion also aborts without the
  ConvertToLLVM dialect extensions. Rather than depend on it, `GpuJitEngine`
  owns the launch via the CUDA driver — more controllable and matches the
  framework's own device-memory model.
* Scope: single-kernel elementwise graphs (the op set that lowers to
  `gpu.launch_func`), Float32, static shapes. Multi-kernel graphs throw a
  clear "unsupported" — the device-compile path itself is general.

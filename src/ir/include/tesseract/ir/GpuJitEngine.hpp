#pragma once

// M4 Phase 8 (B-009 tail) — GPU JIT execution path.
//
// `GpuJitEngine` is the device analogue of `JitEngine`. It takes a
// `graph::Graph`, emits it as `func.func` IR, and drives the full device
// lowering chain
//     --convert-tesseract-to-gpu          (tesseract → linalg → bufferize →
//                                           parallel-loops → gpu + outlining)
//     --nvvm-attach-target{chip=sm_89}
//     --convert-gpu-to-nvvm
//     --convert-nvvm-to-llvm
//     --reconcile-unrealized-casts
//     --gpu-module-to-binary              (NVVM → PTX → cubin/fatbin via ptxas)
// to a serialized `gpu.binary`. The cubin is then loaded with the CUDA driver
// API (`cuModuleLoadData`) and the outlined kernel is launched directly
// (`cuLaunchKernel`) with device buffers marshaled from the caller's Tensors.
//
// This is the "build_for_gpu" half of the JIT: where `JitEngine` JIT-compiles
// for the host CPU via `mlir::ExecutionEngine`, `GpuJitEngine` AOT-compiles
// the outlined kernel to a real cubin and runs it on the GPU. The
// `bench/ir` + `tests/ir` parity gates check its output against eager CUDA.
//
// Scope / invariants:
//   * Supports graphs that lower to a single `gpu.launch_func` (the
//     data-parallel elementwise op set: add/sub/mul/div/relu/... and their
//     fusions). Multi-kernel graphs throw a clear "unsupported" error.
//   * Every input / param / output tensor must be contiguous, row-major,
//     and Float32 (the lowering pins f32). Inputs/params live on host; they
//     are copied H2D, the kernel runs, and outputs are copied D2H.
//   * Not thread-safe: owns one MLIR context + one CUDA module + the
//     retained primary context for `device_index`.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/Value.hpp"

namespace tesseract::ir {

struct GpuJitEngineOptions {
  std::string function_name = "tesseract_gpu_entry";
  // SM target. Defaults to Ada (sm_89); override for other architectures.
  std::string chip = "sm_89";
  int opt_level = 3;
  int device_index = 0;
  bool dump_ir = false;
};

class GpuJitEngine {
 public:
  using Options = GpuJitEngineOptions;

  // Lowers + serializes + loads the graph's kernel eagerly so `invoke` is a
  // pure marshal + `cuLaunchKernel`. Throws if the graph does not lower to a
  // single kernel, if any op lacks a GPU lowering rule, or if the device
  // compilation / module load fails.
  explicit GpuJitEngine(const graph::Graph& g, Options opts = {});
  ~GpuJitEngine();

  GpuJitEngine(const GpuJitEngine&) = delete;
  GpuJitEngine& operator=(const GpuJitEngine&) = delete;
  GpuJitEngine(GpuJitEngine&&) noexcept;
  GpuJitEngine& operator=(GpuJitEngine&&) noexcept;

  // Launch the cubin kernel. `bindings` must cover every graph input + param
  // (same contract as `graph::run`). Returns fresh host Tensors, one per
  // `g.outputs()`, filled by a D2H copy after the launch completes.
  std::vector<Tensor> invoke(
      const std::unordered_map<graph::ValueId, Tensor>& bindings) const;

  // True iff this build has MLIR + CUDA support and at least one usable
  // CUDA device is visible. Tests/benches gate on this.
  static bool available();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tesseract::ir

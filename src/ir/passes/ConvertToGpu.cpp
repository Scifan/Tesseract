// Wave 15 (B-009 / M2L.2) — tesseract → GPU dialect lowering.
//
// `--convert-tesseract-to-gpu` lowers the data-parallel tesseract ops all the
// way to the MLIR GPU dialect: `gpu.module` + `gpu.func` kernels invoked by
// `gpu.launch_func`. It is a pass *pipeline* that reuses the upstream, battle-
// tested lowering chain rather than hand-rolling kernel outlining:
//
//   tesseract.*                         (elementwise / reductions)
//     │  --convert-tesseract-to-linalg
//     ▼
//   linalg.* on tensors
//     │  one-shot-bufferize
//     ▼
//   linalg.* on memrefs
//     │  --convert-linalg-to-parallel-loops
//     ▼
//   scf.parallel
//     │  --gpu-map-parallel-loops  (annotate with gpu mapping)
//     │  --convert-parallel-loops-to-gpu
//     ▼
//   gpu.launch { ... }
//     │  --gpu-kernel-outlining
//     ▼
//   gpu.module @… { gpu.func @…_kernel } + gpu.launch_func
//
// This is the device-IR half of B-009. The remaining half — lowering the
// `gpu.module` to NVVM and JIT-compiling it to PTX/cubin for execution +
// eager-CUDA parity — is gated on a local LLVM built WITH the NVPTX target
// (`scripts/build_llvm.sh` currently builds `host` only) and on a free GPU
// for the parity/benchmark gates; both are out of reach here, so this pass
// stops at verified `gpu.*` IR (FileCheck-pinned) and the NVVM/PTX stage is
// tracked as the B-009 tail.

#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "tesseract/ir/Passes.hpp"

namespace tesseract::ir {

void buildConvertTesseractToGpuPipeline(mlir::OpPassManager& pm) {
  // 1. tesseract → linalg (on tensors).
  pm.addPass(createConvertTesseractToLinalgPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Generalize named linalg ops (linalg.add/mul/...) to linalg.generic so the
  // elementwise fusion pass can see them, then fuse chains (e.g. mul → relu)
  // into a single op *before* bufferization so they outline into ONE gpu
  // kernel instead of one launch per op. This cuts kernel-launch overhead and
  // keeps the single-kernel contract `GpuJitEngine` relies on for simple
  // elementwise graphs.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgGeneralizationPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // 2. Bufferize: tensors → memrefs (parallel-loop lowering is buffer-based).
  mlir::bufferization::OneShotBufferizationOptions bufferize_opts;
  bufferize_opts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferize_opts));

  // 3. linalg (memrefs) → scf.parallel.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertLinalgToParallelLoopsPass());

  // 4. Annotate scf.parallel with a GPU block/thread mapping, then convert
  //    the mapped loops into a gpu.launch region.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
  pm.addPass(mlir::createParallelLoopToGpuPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // 5. Outline each gpu.launch body into a gpu.module + gpu.func, leaving a
  //    gpu.launch_func at the original site.
  pm.addPass(mlir::createGpuKernelOutliningPass());
}

void registerConvertTesseractToGpuPipeline() {
  mlir::PassPipelineRegistration<>(
      "convert-tesseract-to-gpu",
      "Lower data-parallel tesseract ops to the GPU dialect "
      "(gpu.module/gpu.func/gpu.launch_func) via linalg → parallel-loops → "
      "gpu + kernel outlining.",
      [](mlir::OpPassManager& pm) { buildConvertTesseractToGpuPipeline(pm); });
}

}  // namespace tesseract::ir

#pragma once

// Public entry points for Tesseract-owned MLIR passes. Only compiled when
// TESSERACT_ENABLE_MLIR=ON. Each pass follows the upstream convention:
//
//   * `createFooPass()` returns a unique_ptr<Pass> that can be added to a
//     user-built PassManager programmatically.
//   * `registerFooPass()` wires the pass into MLIR's global pass registry so
//     `tesseract-opt --foo` picks it up.
//
// The single bulk entry `registerTesseractPasses()` calls every individual
// registration; `tesseract-opt.cpp` invokes it at startup.

#include <memory>

#include "mlir/Pass/Pass.h"

namespace tesseract::ir {

// tesseract.* → linalg/tensor/arith lowering. See
// src/ir/passes/ConvertToLinalg.cpp for the rewrite patterns.
std::unique_ptr<mlir::Pass> createConvertTesseractToLinalgPass();

void registerConvertTesseractToLinalgPass();

// Reverse-mode AD applied in-place to every tesseract.function in the
// module. Extends the function from
//     (inputs..., params...) -> (outputs...)
// to
//     (inputs..., params..., grad_outputs...) -> (outputs..., dparams...)
// where `dparams` are the gradients of each tesseract.param in declaration
// order. See src/ir/passes/Backward.cpp.
std::unique_ptr<mlir::Pass> createBackwardPass();

void registerBackwardPass();

// Permutes `linalg.matmul`'s iteration space from the default
// `(m, n, k)` (K-innermost, stride-N access on RHS) to `(m, k, n)`
// (N-innermost, contiguous access on RHS + output). This unblocks
// LLVM LoopVectorize / SLP on the default `--convert-linalg-to-loops`
// lowering; see `src/ir/passes/InterchangeMatmul.cpp` for the full
// rationale and B-007 in `docs/backlog.md`.
std::unique_ptr<mlir::Pass> createInterchangeMatmulPass();

void registerInterchangeMatmulPass();

// Wave 15 (B-009): tesseract → GPU dialect lowering pipeline. Builds the
// pass chain (tesseract → linalg → bufferize → parallel-loops → gpu →
// kernel-outlining) producing gpu.module/gpu.func/gpu.launch_func. Exposed
// both as a builder (to embed in a larger PassManager) and a registration
// (so `tesseract-opt --convert-tesseract-to-gpu` picks it up).
void buildConvertTesseractToGpuPipeline(mlir::OpPassManager& pm);

void registerConvertTesseractToGpuPipeline();

// Bulk registration. Call once from `main` (e.g. tesseract-opt).
void registerTesseractPasses();

}  // namespace tesseract::ir

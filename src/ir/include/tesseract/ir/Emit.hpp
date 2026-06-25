#pragma once

// C++-Graph → MLIR (tesseract dialect) emitter.
//
// Converts a Stage-1 `tesseract::graph::Graph` into an MLIR `ModuleOp` whose
// body holds a single `tesseract.graph` containing one `tesseract.function`
// whose body mirrors the recorded op stream. Only compiled when
// TESSERACT_ENABLE_MLIR=ON (the entire src/ir/ directory is gated on that
// flag in CMake).
//
// The emitter is intentionally 1:1 — no canonicalization, no fusion, no
// folding. Transformations happen as MLIR passes afterwards. See ADR-0004.

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "tesseract/graph/Graph.hpp"

namespace tesseract::ir {

struct EmitOptions {
  // Symbol name for the emitted `tesseract.function`.
  std::string function_name = "main";
};

// Emit the given graph into a fresh `builtin.module`. The returned module is
// verified before returning; on verification failure an exception is thrown.
// The caller must ensure `ctx` has the `tesseract` and `builtin` / `func`
// dialects loaded (or at least allow them to be loaded — the emitter will
// `getOrLoadDialect<TesseractDialect>()` itself).
mlir::OwningOpRef<mlir::ModuleOp> emit_mlir(mlir::MLIRContext& ctx,
                                            const graph::Graph& graph,
                                            const EmitOptions& opts = {});

// Alternate emission path that targets `func.func` directly, bypassing the
// `tesseract.graph` / `tesseract.function` / `tesseract.param` structural
// wrappers. The resulting module is:
//
//     builtin.module {
//       func.func @<opts.function_name>(
//           inputs..., params...) -> (outputs...)
//           attributes { llvm.emit_c_interface }
//       { body..., func.return vals }
//     }
//
// This is the shape the standard MLIR CPU lowering pipeline
// (`--convert-tesseract-to-linalg` → `--one-shot-bufferize` →
// `--convert-linalg-to-loops` → ... → `--convert-func-to-llvm`) expects,
// and is consumed by `tesseract::ir::JitEngine`. The body still uses the
// tesseract dialect on ranked tensors; lowering happens as pass passes.
// The module is `mlir::verify()`-checked before return.
mlir::OwningOpRef<mlir::ModuleOp> emit_func_mlir(mlir::MLIRContext& ctx,
                                                 const graph::Graph& graph,
                                                 const EmitOptions& opts = {});

}  // namespace tesseract::ir

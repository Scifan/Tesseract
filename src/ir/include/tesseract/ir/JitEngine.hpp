#pragma once

// M1I.2.b — MLIR ExecutionEngine wrapper (Stage-2 execution path).
//
// Takes a `graph::Graph`, emits it as `func.func` IR, runs the full CPU
// lowering pipeline
//     --convert-tesseract-to-linalg
//     --one-shot-bufferize{bufferize-function-boundaries=true}
//     --buffer-results-to-out-params
//     --convert-linalg-to-loops
//     --lower-affine
//     --convert-scf-to-cf
//     --expand-strided-metadata
//     --finalize-memref-to-llvm
//     --convert-func-to-llvm
//     --convert-arith-to-llvm
//     --convert-cf-to-llvm
//     --reconcile-unrealized-casts
// and hands the result to `mlir::ExecutionEngine`. `invoke(bindings)`
// marshals the caller's `Tensor`s through the StridedMemRef C ABI and
// returns freshly-allocated output Tensors.
//
// Status: Phase-1 — only graphs whose every op is covered by
// `--convert-tesseract-to-linalg` can be JITed. At M1I.2.a that set is
// {add, sub, mul, div, matmul, sum}. Backward rules and activation /
// broadcast / transpose lowerings land in follow-up phases.
//
// Invariants:
//   * All input / param / output tensors must be contiguous and row-major.
//     The memref signature the JIT sees assumes identity layout maps.
//   * The graph's op kinds must all have a lowering rule; unknown kinds
//     surface as a loud `TESSERACT_THROW` during construction.
//   * Not thread-safe: a `JitEngine` wraps one MLIR context + one
//     ExecutionEngine and is intended to be owned by a single training
//     loop at a time.

#include <memory>
#include <unordered_map>
#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/Value.hpp"

namespace tesseract::ir {

struct JitEngineOptions {
  std::string function_name = "tesseract_entry";
  // JIT optimization level; defaults to -O0 to keep capture-to-run time
  // bounded. Downstream callers that need speed can crank it to 2/3.
  int opt_level = 0;
  // If true, dumps the pre- and post-lowering IR to stderr — useful for
  // pipeline debugging.
  bool dump_ir = false;
};

class JitEngine {
 public:
  using Options = JitEngineOptions;

  // Builds + lowers + JITs the graph eagerly so that `invoke` is just a
  // pointer marshal + native call. Throws if any op in `g` has no
  // lowering rule or if the lowering / JIT pipeline fails.
  explicit JitEngine(const graph::Graph& g, Options opts = {});
  ~JitEngine();

  JitEngine(const JitEngine&) = delete;
  JitEngine& operator=(const JitEngine&) = delete;
  JitEngine(JitEngine&&) noexcept;
  JitEngine& operator=(JitEngine&&) noexcept;

  // Execute the JITed function. `bindings` must cover every graph input,
  // param, and constant (same contract as `graph::run`). Returns fresh
  // owning Tensors, one per `g.outputs()`, allocated by the caller here
  // and filled by the JIT via `buffer-results-to-out-params` plumbing.
  std::vector<Tensor> invoke(
      const std::unordered_map<graph::ValueId, Tensor>& bindings) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tesseract::ir

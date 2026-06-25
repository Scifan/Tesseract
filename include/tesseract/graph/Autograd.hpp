#pragma once

// Reverse-mode autograd as a C++-level graph transform.
//
// `build_backward` mirrors the MLIR `--tesseract-backward` pass (see
// src/ir/passes/Backward.cpp) but operates directly on `graph::Graph`
// instead of `mlir::ModuleOp`. It is used by the graph interpreter
// (see `tesseract/graph/Interpreter.hpp`) to execute the backward pass
// without requiring the MLIR JIT to be online. The two implementations
// share the same *rule table*; keeping them in lockstep is a hard
// invariant — tests/graph/test_graph_autograd.cpp enforces it.
//
// Exit semantics: on return,
//   * `g` has `g.outputs().size()` new inputs, one per original output,
//     appended to `g.inputs()`. Each cotangent input's shape/dtype/device
//     matches the corresponding original output. Cotangents for the
//     original outputs are exposed through `BackwardResult::cotangents`
//     in the same order as `g.outputs()` before the call.
//   * `g` has `g.params().size()` new outputs appended to `g.outputs()`,
//     one per param, in `g.params()` declaration order. Their ValueIds
//     are exposed through `BackwardResult::dparams`. If a param does
//     not participate in the forward graph, its gradient is a zero
//     tensor of matching shape/dtype (emitted via `tesseract.zeros_like`
//     semantics — concretely a `sub(p, p)` placeholder today; the
//     interpreter materializes it as zeros).
//
// Supported ops (exactly the set covered by the M1H MLIR pass):
//   add, sub, mul, neg, matmul, sum, mean, transpose, broadcast_to,
//   relu, cross_entropy_with_logits.

#include <vector>

#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/Value.hpp"

namespace tesseract::graph {

struct BackwardResult {
  std::vector<ValueId> cotangents;
  std::vector<ValueId> dparams;
};

BackwardResult build_backward(Graph& g);

}  // namespace tesseract::graph

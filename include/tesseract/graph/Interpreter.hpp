#pragma once

// Reference interpreter for `graph::Graph` (M1I.2).
//
// `run(g, bindings)` walks the ops of `g` in declaration order, dispatches
// each to the matching `tesseract::ops::*` eager kernel, and returns the
// tensor values for `g.outputs()` in order. The interpreter is the first
// end-to-end path that runs the backward-extended graph from
// `graph::build_backward` without going through MLIR; graph-mode training
// (see examples/mnist.cpp) is built directly on top of it.
//
// Invariants:
//   * Must NOT be called from inside a live `GraphScope`. The interpreter
//     itself uses `NoGradGuard` so dispatched eager ops do not record into
//     the autograd tape either.
//   * Every `ValueId` in `g.inputs()`, `g.params()`, and `g.constants()`
//     must have a `Tensor` binding in `bindings`. Shapes / dtypes must
//     match the declared Value metadata exactly.
//   * Every op kind used in `g` must be in the dispatch table
//     (see src/graph/Interpreter.cpp). Adding a new op usually means one
//     new entry there plus a matching backward rule in
//     src/graph/Autograd.cpp.
//
// Returns a vector of tensors sized `g.outputs().size()`, one per output
// in declaration order.

#include <unordered_map>
#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/Value.hpp"

namespace tesseract::graph {

std::vector<Tensor> run(const Graph& g,
                        const std::unordered_map<ValueId, Tensor>& bindings);

}  // namespace tesseract::graph

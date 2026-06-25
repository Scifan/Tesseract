#pragma once

// Stage-1 C++-level graph IR for Tesseract. See docs/adr/0004 for the design
// rationale. The Graph is an SSA list of Ops operating over Values; it does
// not own storage and does not run computation. It is produced by a
// `graph::GraphScope` during recording and consumed downstream by:
//
//   * `graph::run(g, inputs)` — reference re-execution through the eager ops
//     (used today for parity testing; will be replaced by MLIR-lowered code
//     paths in M1.γ).
//   * `graph::emit_mlir(g)` — straight 1:1 translation to the `tesseract`
//     dialect. Only available when `TESSERACT_ENABLE_MLIR=ON`.
//   * `graph::compute_backward(g, loss_id)` — reverse-mode AD as a graph
//     transform (M1H).

#include <cstdint>
#include <string>
#include <vector>

#include "tesseract/graph/Value.hpp"

namespace tesseract::graph {

struct Op {
  // Mnemonic as used in the `tesseract` dialect ("add", "matmul",
  // "reshape", ...). Mirrors `TesseractOps.td`.
  std::string kind;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
  AttrMap attrs;

  // For M1.α we only need the mnemonic; richer debug info (source location,
  // stack trace) follows in M1.β together with the MLIR emitter.
};

class Graph {
 public:
  Graph() = default;

  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) = default;
  Graph& operator=(Graph&&) = default;

  // -------------------- Value construction -------------------- //

  // Allocate a new Value with the given kind + metadata. Ops are expected to
  // call this to materialize their results.
  ValueId new_value(Shape shape, DType dtype, Device device,
                    ValueKind kind = ValueKind::kIntermediate,
                    std::string name = {});

  // -------------------- Inputs / params / outputs -------------------- //

  // Convenience wrappers that also record the value in the corresponding
  // role list. The returned id participates in the op list the moment the
  // first op references it.
  ValueId add_input(Shape shape, DType dtype, Device device, std::string name = {});
  ValueId add_param(Shape shape, DType dtype, Device device, std::string name = {});
  ValueId add_constant(Shape shape, DType dtype, Device device, std::string name = {});

  void mark_output(ValueId id);

  // -------------------- Op construction -------------------- //

  // Append a new op and return the index into `ops_`. Used by `ops::` entry
  // points through the `Recorder` helpers.
  std::size_t add_op(std::string kind,
                     std::vector<ValueId> inputs,
                     std::vector<ValueId> outputs,
                     AttrMap attrs = {});

  // -------------------- Accessors -------------------- //

  const std::vector<Value>& values() const noexcept { return values_; }
  const std::vector<Op>& ops() const noexcept { return ops_; }
  const std::vector<ValueId>& inputs() const noexcept { return inputs_; }
  const std::vector<ValueId>& outputs() const noexcept { return outputs_; }
  const std::vector<ValueId>& params() const noexcept { return params_; }
  const std::vector<ValueId>& constants() const noexcept { return constants_; }

  const Value& value(ValueId id) const;

  std::size_t num_values() const noexcept { return values_.size(); }
  std::size_t num_ops() const noexcept { return ops_.size(); }

  // Pretty-printer for humans (and for CI diffs). Produces a stable textual
  // form similar to MLIR; it is NOT the MLIR textual form.
  std::string to_string() const;

 private:
  std::vector<Value> values_;
  std::vector<Op> ops_;
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;
  std::vector<ValueId> params_;
  std::vector<ValueId> constants_;
};

}  // namespace tesseract::graph

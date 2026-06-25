#pragma once

// Thread-local RAII scope that installs an active `Graph` so that the
// `tesseract::ops::*` entry points record SSA ops alongside (or instead of)
// eager execution.
//
// Model:
//   * Scope installs itself on construction and restores the previous scope
//     on destruction. Nested scopes are rejected at construction time: at
//     M1, a single active graph per thread is enough (nested graphs would
//     require op semantics we haven't defined yet).
//   * The default behavior is *eager + trace*: ops continue to execute
//     against the CPU backend and also append themselves to the active
//     graph. This preserves M0 numerics and simplifies testing.
//
// Hook usage from an op entry point looks like:
//
//     Tensor add(const Tensor& a, const Tensor& b) {
//       Tensor out = add_forward(a, b);
//       autograd::maybe_attach(...);
//       graph::maybe_record("add", {a, b}, {out});
//       return out;
//     }
//
// The `maybe_record` helper is a no-op when no scope is active.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/Value.hpp"

namespace tesseract {

class Tensor;
struct TensorImpl;

namespace graph {

class GraphScope {
 public:
  GraphScope();
  ~GraphScope();

  GraphScope(const GraphScope&) = delete;
  GraphScope& operator=(const GraphScope&) = delete;
  GraphScope(GraphScope&&) = delete;
  GraphScope& operator=(GraphScope&&) = delete;

  Graph& graph() noexcept { return graph_; }
  const Graph& graph() const noexcept { return graph_; }

  // These three fields are intentionally public: they form the internal
  // state shared with the namespace-level recorder helpers (below) and the
  // graph::emit_mlir emitter in M1.β. Wrapping them in accessors adds noise
  // without adding safety — the module is the trust boundary.
  Graph graph_;
  // TensorImpl* -> ValueId map. We hold a shared_ptr to the underlying impl
  // so that the tensor cannot be destroyed while the scope is live (and
  // hence while we still reference it by raw pointer).
  std::unordered_map<TensorImpl*, ValueId> value_of_;
  std::vector<std::shared_ptr<TensorImpl>> pinned_;
};

// Thread-local active scope pointer; returns nullptr outside a scope.
GraphScope* active_scope() noexcept;

// Convenience predicate — usually simpler than comparing against nullptr at
// the op level.
bool is_recording() noexcept;

// ---------------- Binding helpers used by ops / module code ----------------

// Register `t` as a graph input / param / constant and return the ValueId.
// Safe to call multiple times for the same tensor: subsequent calls just
// return the existing id (kind is NOT upgraded).
ValueId bind_input(const Tensor& t, std::string name = {});
ValueId bind_param(const Tensor& t, std::string name = {});
ValueId bind_constant(const Tensor& t, std::string name = {});

// Return the ValueId currently bound to `t`, or `kInvalidValueId` if the
// tensor is not yet in the active graph.
ValueId value_id_of(const Tensor& t);

// Ensure that `t` has a ValueId in the active graph. If it does not, create
// an intermediate Value (treating `t` as an already-materialized input).
// Returns `kInvalidValueId` when there is no active scope — callers must
// guard on `is_recording()`.
ValueId ensure_value(const Tensor& t, ValueKind kind = ValueKind::kInput);

// Mark `t` as a graph output. No-op outside a scope.
void mark_output(const Tensor& t);

// Append a new op referencing the ValueIds bound to `inputs` (creating them
// lazily as kInput values) and allocate fresh ValueIds for each output.
// Returns the op index or SIZE_MAX outside a scope.
std::size_t maybe_record(std::string kind,
                         std::vector<const Tensor*> inputs,
                         std::vector<const Tensor*> outputs,
                         AttrMap attrs = {});

}  // namespace graph
}  // namespace tesseract

#include "tesseract/graph/GraphScope.hpp"

#include <limits>
#include <utility>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::graph {

namespace {

// Thread-local active scope. M1 only supports a single scope per thread;
// nested scopes will come in M2 once sub-graph semantics are defined.
thread_local GraphScope* g_active_scope = nullptr;

}  // namespace

GraphScope::GraphScope() {
  TESSERACT_CHECK(g_active_scope == nullptr,
                  "graph::GraphScope: nested scopes are not supported at M1. "
                  "Close the outer scope before opening a new one.");
  g_active_scope = this;
}

GraphScope::~GraphScope() {
  if (g_active_scope == this) {
    g_active_scope = nullptr;
  }
}

GraphScope* active_scope() noexcept { return g_active_scope; }

bool is_recording() noexcept { return g_active_scope != nullptr; }

namespace {

ValueId bind_with_kind(const Tensor& t, ValueKind kind, std::string name) {
  GraphScope* scope = active_scope();
  TESSERACT_CHECK(scope != nullptr, "graph::bind: no active GraphScope");
  TESSERACT_CHECK(t.defined(), "graph::bind: tensor is undefined");

  TensorImpl* key = t.impl().get();
  auto it = scope->value_of_.find(key);
  if (it != scope->value_of_.end()) {
    return it->second;
  }

  // New binding: create a Value and pin the TensorImpl to keep the pointer
  // valid for the scope's lifetime.
  ValueId id = kInvalidValueId;
  switch (kind) {
    case ValueKind::kInput:
      id = scope->graph_.add_input(t.shape(), t.dtype(), t.device(), std::move(name));
      break;
    case ValueKind::kParam:
      id = scope->graph_.add_param(t.shape(), t.dtype(), t.device(), std::move(name));
      break;
    case ValueKind::kConstant:
      id = scope->graph_.add_constant(t.shape(), t.dtype(), t.device(), std::move(name));
      break;
    case ValueKind::kIntermediate:
      id = scope->graph_.new_value(t.shape(), t.dtype(), t.device(),
                                   ValueKind::kIntermediate, std::move(name));
      break;
  }
  scope->value_of_.emplace(key, id);
  scope->pinned_.push_back(t.impl());
  return id;
}

}  // namespace

ValueId bind_input(const Tensor& t, std::string name) {
  return bind_with_kind(t, ValueKind::kInput, std::move(name));
}

ValueId bind_param(const Tensor& t, std::string name) {
  return bind_with_kind(t, ValueKind::kParam, std::move(name));
}

ValueId bind_constant(const Tensor& t, std::string name) {
  return bind_with_kind(t, ValueKind::kConstant, std::move(name));
}

ValueId value_id_of(const Tensor& t) {
  GraphScope* scope = active_scope();
  if (scope == nullptr || !t.defined()) return kInvalidValueId;
  auto it = scope->value_of_.find(t.impl().get());
  return it == scope->value_of_.end() ? kInvalidValueId : it->second;
}

ValueId ensure_value(const Tensor& t, ValueKind kind) {
  GraphScope* scope = active_scope();
  if (scope == nullptr) return kInvalidValueId;
  TESSERACT_CHECK(t.defined(), "graph::ensure_value: tensor is undefined");

  TensorImpl* key = t.impl().get();
  auto it = scope->value_of_.find(key);
  if (it != scope->value_of_.end()) return it->second;
  return bind_with_kind(t, kind, {});
}

void mark_output(const Tensor& t) {
  GraphScope* scope = active_scope();
  if (scope == nullptr) return;
  ValueId id = ensure_value(t, ValueKind::kIntermediate);
  scope->graph_.mark_output(id);
}

std::size_t maybe_record(std::string kind,
                         std::vector<const Tensor*> inputs,
                         std::vector<const Tensor*> outputs,
                         AttrMap attrs) {
  GraphScope* scope = active_scope();
  if (scope == nullptr) return std::numeric_limits<std::size_t>::max();

  std::vector<ValueId> in_ids;
  in_ids.reserve(inputs.size());
  for (const Tensor* t : inputs) {
    TESSERACT_CHECK(t != nullptr, "graph::maybe_record: null input tensor pointer");
    in_ids.push_back(ensure_value(*t, ValueKind::kInput));
  }

  std::vector<ValueId> out_ids;
  out_ids.reserve(outputs.size());
  for (const Tensor* t : outputs) {
    TESSERACT_CHECK(t != nullptr, "graph::maybe_record: null output tensor pointer");
    TESSERACT_CHECK(t->defined(), "graph::maybe_record: output tensor is undefined");
    // Outputs are always new tensors produced by the op, so we create a
    // fresh intermediate Value and bind it.
    TensorImpl* key = t->impl().get();
    // It is technically legal for an op to return one of its inputs
    // unchanged (e.g. contiguous() of a contiguous tensor). In that case we
    // reuse the existing binding rather than create a duplicate Value.
    auto it = scope->value_of_.find(key);
    if (it != scope->value_of_.end()) {
      out_ids.push_back(it->second);
    } else {
      ValueId id = scope->graph_.new_value(t->shape(), t->dtype(), t->device(),
                                           ValueKind::kIntermediate);
      scope->value_of_.emplace(key, id);
      scope->pinned_.push_back(t->impl());
      out_ids.push_back(id);
    }
  }

  return scope->graph_.add_op(std::move(kind), std::move(in_ids), std::move(out_ids),
                              std::move(attrs));
}

}  // namespace tesseract::graph

#include "tesseract/graph/Interpreter.hpp"

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/ops/Softmax.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

// Graph interpreter. Each op kind maps to a single-line call into the
// existing M0 eager kernels. Keeping this file free of per-op tensor
// algorithms means any numerics that disagree with eager mode are
// localized to the eager kernels themselves, not the dispatch layer.
//
// Important: run(...) must execute outside any GraphScope and wraps the
// body in NoGradGuard, because the eager kernels call into
// `graph::maybe_record` and `autograd::attach_grad_fn` themselves. The
// interpreter does not want either side-effect — the graph is already
// the authoritative representation.

namespace tesseract::graph {

namespace {

// ----- Attr accessors --------------------------------------------------- //

int64_t require_int(const AttrMap& attrs, const std::string& key,
                    const std::string& op_kind) {
  auto it = attrs.find(key);
  TESSERACT_CHECK(it != attrs.end(),
                  "graph::run('{}'): missing int attribute '{}'", op_kind, key);
  const int64_t* p = std::get_if<int64_t>(&it->second);
  TESSERACT_CHECK(p != nullptr,
                  "graph::run('{}'): attribute '{}' is not an int", op_kind,
                  key);
  return *p;
}

bool optional_bool(const AttrMap& attrs, const std::string& key,
                   bool dflt = false) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return dflt;
  const bool* p = std::get_if<bool>(&it->second);
  return p ? *p : dflt;
}

std::vector<int64_t> require_i64_vec(const AttrMap& attrs,
                                     const std::string& key,
                                     const std::string& op_kind) {
  auto it = attrs.find(key);
  TESSERACT_CHECK(it != attrs.end(),
                  "graph::run('{}'): missing i64-vector attribute '{}'",
                  op_kind, key);
  const std::vector<int64_t>* p =
      std::get_if<std::vector<int64_t>>(&it->second);
  TESSERACT_CHECK(p != nullptr,
                  "graph::run('{}'): attribute '{}' is not an i64 vector",
                  op_kind, key);
  return *p;
}

// ----- Tensor environment ---------------------------------------------- //

class Env {
 public:
  Env(const Graph& g,
      const std::unordered_map<ValueId, Tensor>& bindings)
      : g_(g) {
    env_.reserve(bindings.size() + g.ops().size() * 2);
    for (const auto& [id, t] : bindings) env_.emplace(id, t);
  }

  const Tensor& get(ValueId id) const {
    auto it = env_.find(id);
    TESSERACT_CHECK(it != env_.end(),
                    "graph::run: no tensor bound for %{} ('{}')", id,
                    g_.value(id).name);
    return it->second;
  }

  void put(ValueId id, Tensor t) { env_.emplace(id, std::move(t)); }

 private:
  const Graph& g_;
  std::unordered_map<ValueId, Tensor> env_;
};

// ----- Per-op dispatchers ---------------------------------------------- //

Tensor dispatch(const Op& op, Env& env) {
  const auto& ins = op.inputs;
  const auto& attrs = op.attrs;

  // Element-wise arithmetic.
  if (op.kind == "add") return ops::add(env.get(ins[0]), env.get(ins[1]));
  if (op.kind == "sub") return ops::sub(env.get(ins[0]), env.get(ins[1]));
  if (op.kind == "mul") return ops::mul(env.get(ins[0]), env.get(ins[1]));
  if (op.kind == "div") return ops::div(env.get(ins[0]), env.get(ins[1]));
  if (op.kind == "neg") return ops::neg(env.get(ins[0]));

  // Linear algebra.
  if (op.kind == "matmul")
    return ops::matmul(env.get(ins[0]), env.get(ins[1]));

  // Reductions.
  if (op.kind == "sum") {
    if (attrs.find("dim") == attrs.end()) return ops::sum(env.get(ins[0]));
    return ops::sum(env.get(ins[0]), require_int(attrs, "dim", op.kind),
                    optional_bool(attrs, "keepdim"));
  }
  if (op.kind == "mean") {
    if (attrs.find("dim") == attrs.end()) return ops::mean(env.get(ins[0]));
    return ops::mean(env.get(ins[0]), require_int(attrs, "dim", op.kind),
                     optional_bool(attrs, "keepdim"));
  }
  if (op.kind == "max") {
    if (attrs.find("dim") == attrs.end()) return ops::max(env.get(ins[0]));
    return ops::max(env.get(ins[0]), require_int(attrs, "dim", op.kind),
                    optional_bool(attrs, "keepdim"));
  }

  // Activations.
  if (op.kind == "relu") return ops::relu(env.get(ins[0]));
  if (op.kind == "sigmoid") return ops::sigmoid(env.get(ins[0]));
  if (op.kind == "tanh") return ops::tanh(env.get(ins[0]));
  if (op.kind == "exp") return ops::exp(env.get(ins[0]));
  if (op.kind == "log") return ops::log(env.get(ins[0]));
  if (op.kind == "softmax")
    return ops::softmax(env.get(ins[0]), require_int(attrs, "dim", op.kind));

  // Shape ops.
  if (op.kind == "view")
    return ops::view(env.get(ins[0]),
                     Shape(require_i64_vec(attrs, "shape", op.kind)));
  if (op.kind == "reshape")
    return ops::reshape(env.get(ins[0]),
                        Shape(require_i64_vec(attrs, "shape", op.kind)));
  if (op.kind == "permute") {
    const auto axes = require_i64_vec(attrs, "axes", op.kind);
    return ops::permute(env.get(ins[0]), std::span<const int64_t>(axes));
  }
  if (op.kind == "transpose")
    return ops::transpose(env.get(ins[0]), require_int(attrs, "dim_a", op.kind),
                          require_int(attrs, "dim_b", op.kind));
  if (op.kind == "contiguous") return ops::contiguous(env.get(ins[0]));
  if (op.kind == "clone") return ops::clone(env.get(ins[0]));
  if (op.kind == "broadcast_to")
    return ops::broadcast_to(env.get(ins[0]),
                             Shape(require_i64_vec(attrs, "shape", op.kind)));

  // Loss.
  if (op.kind == "cross_entropy_with_logits")
    return ops::cross_entropy_with_logits(env.get(ins[0]), env.get(ins[1]));

  // Fused backward kernels (emitted by graph::build_backward).
  if (op.kind == "relu_backward")
    return ops::relu_backward(env.get(ins[0]), env.get(ins[1]));
  if (op.kind == "cross_entropy_with_logits_backward")
    return ops::cross_entropy_with_logits_backward(
        env.get(ins[0]), env.get(ins[1]), env.get(ins[2]));

  TESSERACT_CHECK(false,
                  "graph::run: no interpreter rule registered for op '{}'. "
                  "Add it to src/graph/Interpreter.cpp.",
                  op.kind);
  return {};
}

}  // namespace

std::vector<Tensor> run(const Graph& g,
                        const std::unordered_map<ValueId, Tensor>& bindings) {
  TESSERACT_CHECK(active_scope() == nullptr,
                  "graph::run: cannot run a graph while a GraphScope is "
                  "active (that would re-record ops into the scope). Close "
                  "the scope before invoking the interpreter.");
  NoGradGuard nogg;

  // Verify all inputs / params / constants are bound, with matching shape
  // and dtype (catches misaligned training-loop state early).
  auto check_bound = [&](ValueId id, const char* role) {
    auto it = bindings.find(id);
    TESSERACT_CHECK(it != bindings.end(),
                    "graph::run: {} %{} ('{}') has no tensor binding", role,
                    id, g.value(id).name);
    const Tensor& t = it->second;
    const Value& v = g.value(id);
    TESSERACT_CHECK(t.defined(), "graph::run: {} %{} binding is undefined",
                    role, id);
    TESSERACT_CHECK(t.shape() == v.shape,
                    "graph::run: {} %{} shape mismatch (bound {} vs "
                    "expected {})",
                    role, id, t.shape().to_string(), v.shape.to_string());
    TESSERACT_CHECK(t.dtype() == v.dtype,
                    "graph::run: {} %{} dtype mismatch", role, id);
  };
  for (ValueId id : g.inputs()) check_bound(id, "input");
  for (ValueId id : g.params()) check_bound(id, "param");
  for (ValueId id : g.constants()) check_bound(id, "constant");

  Env env(g, bindings);

  for (const Op& op : g.ops()) {
    TESSERACT_CHECK(op.outputs.size() == 1,
                    "graph::run: op '{}' has {} outputs; only single-output "
                    "ops are supported today",
                    op.kind, op.outputs.size());
    Tensor out = dispatch(op, env);
    env.put(op.outputs[0], std::move(out));
  }

  std::vector<Tensor> results;
  results.reserve(g.outputs().size());
  for (ValueId id : g.outputs()) results.push_back(env.get(id));
  return results;
}

}  // namespace tesseract::graph

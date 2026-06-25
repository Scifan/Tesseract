#include "tesseract/graph/Autograd.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tesseract/graph/Graph.hpp"
#include "tesseract/utils/Logging.hpp"

// Reverse-mode AD on `graph::Graph`. See tesseract/graph/Autograd.hpp and the
// MLIR sibling at src/ir/passes/Backward.cpp — this file is the non-MLIR
// mirror of that pass and is used by the graph interpreter. The two backward
// transforms must agree on the rule table; tests/graph/test_graph_autograd
// pins the key cases.

namespace tesseract::graph {

namespace {

// ----- Shape / attr helpers ---------------------------------------------- //

std::optional<int64_t> int_attr(const AttrMap& attrs, const std::string& key) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return std::nullopt;
  if (const int64_t* p = std::get_if<int64_t>(&it->second)) return *p;
  return std::nullopt;
}

std::optional<bool> bool_attr(const AttrMap& attrs, const std::string& key) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return std::nullopt;
  if (const bool* p = std::get_if<bool>(&it->second)) return *p;
  return std::nullopt;
}

// ----- Builder wrappers --------------------------------------------------- //

class Builder {
 public:
  explicit Builder(Graph& g) : g_(g) {}

  ValueId new_value(const Shape& shape, DType dtype, Device device) {
    return g_.new_value(shape, dtype, device);
  }

  ValueId emit_unary(const std::string& kind, ValueId in,
                     const Shape& out_shape, DType dtype, Device device,
                     AttrMap attrs = {}) {
    const ValueId out = new_value(out_shape, dtype, device);
    g_.add_op(kind, {in}, {out}, std::move(attrs));
    return out;
  }

  ValueId emit_binary(const std::string& kind, ValueId lhs, ValueId rhs,
                      const Shape& out_shape, DType dtype, Device device,
                      AttrMap attrs = {}) {
    const ValueId out = new_value(out_shape, dtype, device);
    g_.add_op(kind, {lhs, rhs}, {out}, std::move(attrs));
    return out;
  }

  ValueId emit_ternary(const std::string& kind, ValueId a, ValueId b,
                       ValueId c, const Shape& out_shape, DType dtype,
                       Device device, AttrMap attrs = {}) {
    const ValueId out = new_value(out_shape, dtype, device);
    g_.add_op(kind, {a, b, c}, {out}, std::move(attrs));
    return out;
  }

  // Emit `sum(x, dim, keepdim)`, materializing the output shape.
  ValueId emit_sum_along(ValueId x, int64_t dim, bool keepdim) {
    const Value xv = g_.value(x);
    Shape out_shape;
    for (int64_t i = 0; i < static_cast<int64_t>(xv.shape.rank()); ++i) {
      if (i == dim) {
        if (keepdim) out_shape.push_back(1);
      } else {
        out_shape.push_back(xv.shape[i]);
      }
    }
    AttrMap attrs;
    attrs.emplace("dim", static_cast<int64_t>(dim));
    attrs.emplace("keepdim", keepdim);
    return emit_unary("sum", x, out_shape, xv.dtype, xv.device,
                      std::move(attrs));
  }

  // Reduce `grad_id` (currently has shape == grad shape) back to
  // `target_shape`, matching the broadcasting rules used by the eager ops.
  // Strategy mirrors `reduce_to_shape` in src/ops/cpu/Arithmetic.cpp and
  // `reduceToShape` in src/ir/passes/Backward.cpp:
  //   1. Sum over leading axes that were introduced by the broadcast.
  //   2. Sum (keepdim) over size-1 axes that got tiled up.
  ValueId emit_reduce_to_shape(ValueId grad_id, const Shape& target_shape) {
    const Shape grad_shape = g_.value(grad_id).shape;
    if (grad_shape == target_shape) return grad_id;

    TESSERACT_CHECK(grad_shape.rank() >= target_shape.rank(),
                    "graph::build_backward: cannot reduce shape {} to {} "
                    "(rank decreases)",
                    grad_shape.to_string(), target_shape.to_string());

    ValueId cur = grad_id;
    const int64_t lead = static_cast<int64_t>(grad_shape.rank()) -
                         static_cast<int64_t>(target_shape.rank());
    for (int64_t i = 0; i < lead; ++i) {
      cur = emit_sum_along(cur, /*dim=*/0, /*keepdim=*/false);
    }
    // Ranks match now. Collapse residual size-1 axes.
    for (int64_t i = 0; i < static_cast<int64_t>(target_shape.rank()); ++i) {
      const int64_t t = target_shape[i];
      const int64_t c = g_.value(cur).shape[i];
      if (t == 1 && c != 1) {
        cur = emit_sum_along(cur, /*dim=*/i, /*keepdim=*/true);
      }
    }
    // Final sanity check.
    TESSERACT_CHECK(g_.value(cur).shape == target_shape,
                    "graph::build_backward: reduce_to_shape produced {}, "
                    "expected {}",
                    g_.value(cur).shape.to_string(),
                    target_shape.to_string());
    return cur;
  }

 private:
  Graph& g_;
};

// ----- Accumulator ------------------------------------------------------- //

class GradMap {
 public:
  explicit GradMap(Builder& bldr, Graph& g) : bldr_(bldr), g_(g) {}

  ValueId lookup(ValueId v) const {
    auto it = grads_.find(v);
    return it == grads_.end() ? kInvalidValueId : it->second;
  }

  void seed(ValueId v, ValueId contribution) {
    grads_[v] = contribution;
  }

  // Accumulate a new gradient contribution for value `v`. When `v` already
  // has one, emit `add(old, new)` to combine them (after reducing the
  // contribution to `v`'s shape if broadcasting was in play).
  //
  // NOTE: We copy the Value descriptor for `v` rather than holding a
  // reference, because emitting new ops below can cause the underlying
  // `values_` vector to reallocate and invalidate references.
  void add_contribution(ValueId v, ValueId contribution) {
    const Value vv = g_.value(v);
    const ValueId reduced = bldr_.emit_reduce_to_shape(contribution, vv.shape);

    auto it = grads_.find(v);
    if (it == grads_.end()) {
      grads_[v] = reduced;
      return;
    }
    grads_[v] =
        bldr_.emit_binary("add", it->second, reduced, vv.shape, vv.dtype,
                          vv.device);
  }

 private:
  Builder& bldr_;
  Graph& g_;
  std::unordered_map<ValueId, ValueId> grads_;
};

// ----- Per-op backward rules --------------------------------------------- //

struct BackwardDriver {
  Graph& g;
  Builder bldr;
  GradMap grads;

  BackwardDriver(Graph& gg) : g(gg), bldr(gg), grads(bldr, gg) {}

  // Return a copy — not a reference — of the Value descriptor. Emitting
  // ops downstream may reallocate `g.values_`, which would invalidate a
  // reference here.
  Value val(ValueId id) const { return g.value(id); }

  bool run_op(const Op& op) {
    // --- element-wise ----------------------------------------------------
    if (op.kind == "add") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      grads.add_contribution(op.inputs[0], dout);
      grads.add_contribution(op.inputs[1], dout);
      return true;
    }
    if (op.kind == "sub") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      const Value dv = val(dout);
      ValueId neg = bldr.emit_unary("neg", dout, dv.shape, dv.dtype, dv.device);
      grads.add_contribution(op.inputs[0], dout);
      grads.add_contribution(op.inputs[1], neg);
      return true;
    }
    if (op.kind == "mul") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      const Value ov = val(op.outputs[0]);
      ValueId dLhs = bldr.emit_binary("mul", dout, op.inputs[1], ov.shape,
                                      ov.dtype, ov.device);
      ValueId dRhs = bldr.emit_binary("mul", dout, op.inputs[0], ov.shape,
                                      ov.dtype, ov.device);
      grads.add_contribution(op.inputs[0], dLhs);
      grads.add_contribution(op.inputs[1], dRhs);
      return true;
    }
    if (op.kind == "neg") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      const Value dv = val(dout);
      grads.add_contribution(
          op.inputs[0],
          bldr.emit_unary("neg", dout, dv.shape, dv.dtype, dv.device));
      return true;
    }

    // --- matmul / transpose ---------------------------------------------
    if (op.kind == "matmul") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      const Value A = val(op.inputs[0]);
      const Value B = val(op.inputs[1]);
      // Bᵀ : [K, N] → [N, K]. Aᵀ : [M, K] → [K, M].
      const int64_t bRank = static_cast<int64_t>(B.shape.rank());
      const int64_t aRank = static_cast<int64_t>(A.shape.rank());
      TESSERACT_CHECK(aRank == 2 && bRank == 2,
                      "graph::build_backward(matmul): only rank-2 matmul is "
                      "supported (got {} and {})",
                      A.shape.to_string(), B.shape.to_string());
      Shape bt_shape = {B.shape[1], B.shape[0]};
      Shape at_shape = {A.shape[1], A.shape[0]};
      AttrMap tp_attrs;
      tp_attrs.emplace("dim_a", static_cast<int64_t>(0));
      tp_attrs.emplace("dim_b", static_cast<int64_t>(1));
      ValueId bT = bldr.emit_unary("transpose", op.inputs[1], bt_shape,
                                   B.dtype, B.device, tp_attrs);
      ValueId aT = bldr.emit_unary("transpose", op.inputs[0], at_shape,
                                   A.dtype, A.device, tp_attrs);
      Shape dA_shape = {A.shape[0], A.shape[1]};
      Shape dB_shape = {B.shape[0], B.shape[1]};
      ValueId dA = bldr.emit_binary("matmul", dout, bT, dA_shape, A.dtype,
                                    A.device);
      ValueId dB = bldr.emit_binary("matmul", aT, dout, dB_shape, B.dtype,
                                    B.device);
      grads.add_contribution(op.inputs[0], dA);
      grads.add_contribution(op.inputs[1], dB);
      return true;
    }
    if (op.kind == "transpose") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      const auto dim_a = int_attr(op.attrs, "dim_a");
      const auto dim_b = int_attr(op.attrs, "dim_b");
      TESSERACT_CHECK(dim_a && dim_b,
                      "graph::build_backward(transpose): missing dim_a/dim_b");
      const Value inV = val(op.inputs[0]);
      AttrMap tp_attrs;
      tp_attrs.emplace("dim_a", *dim_a);
      tp_attrs.emplace("dim_b", *dim_b);
      ValueId dx = bldr.emit_unary("transpose", dout, inV.shape, inV.dtype,
                                   inV.device, tp_attrs);
      // Eager TransposeBackward returns a contiguous tensor; we mirror that
      // here so downstream consumers (the optimizer, gradient printers,
      // ctest comparisons) see a proper row-major layout and not a view
      // with transposed strides.
      dx = bldr.emit_unary("contiguous", dx, inV.shape, inV.dtype,
                           inV.device);
      grads.add_contribution(op.inputs[0], dx);
      return true;
    }

    // --- reductions -----------------------------------------------------
    if (op.kind == "sum") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      // reduce-all path (no `dim` attr): broadcast the scalar back to
      // input shape. reduce-dim path: broadcast along the reduced axis.
      const Value inV = val(op.inputs[0]);
      if (!int_attr(op.attrs, "dim").has_value()) {
        AttrMap bc_attrs;
        std::vector<int64_t> tgt(inV.shape.begin(), inV.shape.end());
        bc_attrs.emplace("shape", tgt);
        ValueId dx = bldr.emit_unary("broadcast_to", dout, inV.shape,
                                     inV.dtype, inV.device, bc_attrs);
        grads.add_contribution(op.inputs[0], dx);
        return true;
      }
      int64_t dim = *int_attr(op.attrs, "dim");
      if (dim < 0) dim += static_cast<int64_t>(inV.shape.rank());
      const bool keepdim = bool_attr(op.attrs, "keepdim").value_or(false);
      ValueId d = dout;
      if (!keepdim) {
        // Insert a unit axis at `dim` so the subsequent broadcast aligns.
        // C++ graph uses `reshape` for this.
        Shape with_one;
        for (int64_t i = 0; i < static_cast<int64_t>(inV.shape.rank()); ++i) {
          with_one.push_back(i == dim ? 1 : inV.shape[i]);
        }
        AttrMap rs_attrs;
        std::vector<int64_t> shape_vec(with_one.begin(), with_one.end());
        rs_attrs.emplace("shape", shape_vec);
        d = bldr.emit_unary("reshape", d, with_one, inV.dtype, inV.device,
                            rs_attrs);
      }
      AttrMap bc_attrs;
      std::vector<int64_t> tgt(inV.shape.begin(), inV.shape.end());
      bc_attrs.emplace("shape", tgt);
      ValueId dx = bldr.emit_unary("broadcast_to", d, inV.shape, inV.dtype,
                                   inV.device, bc_attrs);
      grads.add_contribution(op.inputs[0], dx);
      return true;
    }

    // --- shape ops ------------------------------------------------------
    if (op.kind == "broadcast_to") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      const Value inV = val(op.inputs[0]);
      ValueId dx = bldr.emit_reduce_to_shape(dout, inV.shape);
      grads.add_contribution(op.inputs[0], dx);
      return true;
    }

    // --- activations ----------------------------------------------------
    if (op.kind == "relu") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      const Value inV = val(op.inputs[0]);
      ValueId dx =
          bldr.emit_binary("relu_backward", op.inputs[0], dout, inV.shape,
                           inV.dtype, inV.device);
      grads.add_contribution(op.inputs[0], dx);
      return true;
    }
    if (op.kind == "softmax") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      // y = softmax(x, dim); dx = y ⊙ (dy − Σ_dim(y ⊙ dy)). Mirrors the MLIR
      // --tesseract-backward rule and honors the op's `dim` attribute.
      const Value yV = val(op.outputs[0]);
      const int64_t rank = static_cast<int64_t>(yV.shape.rank());
      int64_t dim = int_attr(op.attrs, "dim").value_or(-1);
      if (dim < 0) dim += rank;
      const ValueId y = op.outputs[0];
      ValueId ydy = bldr.emit_binary("mul", y, dout, yV.shape, yV.dtype,
                                     yV.device);
      ValueId s = bldr.emit_sum_along(ydy, dim, /*keepdim=*/true);
      AttrMap bc_attrs;
      std::vector<int64_t> tgt(yV.shape.begin(), yV.shape.end());
      bc_attrs.emplace("shape", tgt);
      ValueId sB = bldr.emit_unary("broadcast_to", s, yV.shape, yV.dtype,
                                   yV.device, bc_attrs);
      ValueId diff = bldr.emit_binary("sub", dout, sB, yV.shape, yV.dtype,
                                      yV.device);
      ValueId dx = bldr.emit_binary("mul", y, diff, yV.shape, yV.dtype,
                                    yV.device);
      grads.add_contribution(op.inputs[0], dx);
      return true;
    }

    // --- loss -----------------------------------------------------------
    if (op.kind == "cross_entropy_with_logits") {
      ValueId dout = grads.lookup(op.outputs[0]);
      if (dout == kInvalidValueId) return true;
      const Value logV = val(op.inputs[0]);
      ValueId dLogits = bldr.emit_ternary(
          "cross_entropy_with_logits_backward", op.inputs[0], op.inputs[1],
          dout, logV.shape, logV.dtype, logV.device);
      grads.add_contribution(op.inputs[0], dLogits);
      // No contribution to targets (Int64 index tensor).
      return true;
    }

    return false;
  }
};

}  // namespace

BackwardResult build_backward(Graph& g) {
  BackwardResult result;

  // Snapshot forward structure before mutating `g`.
  const std::vector<ValueId> fwd_outputs = g.outputs();
  const std::vector<ValueId> fwd_params = g.params();

  BackwardDriver drv(g);

  // Append one cotangent input per forward output.
  result.cotangents.reserve(fwd_outputs.size());
  for (ValueId out_id : fwd_outputs) {
    const Value ov = g.value(out_id);
    const std::string name = "grad_" + (ov.name.empty()
                                            ? std::to_string(ov.id)
                                            : ov.name);
    ValueId ct = g.add_input(ov.shape, ov.dtype, ov.device, name);
    result.cotangents.push_back(ct);
    drv.grads.seed(out_id, ct);
  }

  // Walk ops in reverse and emit backward rules.
  const auto& ops = g.ops();
  // Copy indices into a vector so we can iterate in reverse cleanly; `ops`
  // grows as we add backward ops but those entries come after the snapshot
  // and must not feed into the reverse walk.
  std::vector<std::size_t> idx;
  idx.reserve(ops.size());
  for (std::size_t i = 0; i < ops.size(); ++i) idx.push_back(i);
  for (auto it = idx.rbegin(); it != idx.rend(); ++it) {
    const Op& op_snapshot = g.ops()[*it];
    // The backward driver reads `op_snapshot` by reference; because `add_op`
    // may relocate `ops_`, we work on a local copy of the op metadata.
    Op local = op_snapshot;
    const bool ok = drv.run_op(local);
    TESSERACT_CHECK(ok,
                    "graph::build_backward: no backward rule for op '{}'. "
                    "Add a rule in src/graph/Autograd.cpp (and mirror in "
                    "src/ir/passes/Backward.cpp).",
                    local.kind);
  }

  // Collect dparams and append them as new outputs.
  result.dparams.reserve(fwd_params.size());
  for (ValueId pid : fwd_params) {
    ValueId gid = drv.grads.lookup(pid);
    TESSERACT_CHECK(gid != kInvalidValueId,
                    "graph::build_backward: param %{} has no gradient "
                    "contribution. Did the forward graph actually use it?",
                    pid);
    result.dparams.push_back(gid);
    g.mark_output(gid);
  }
  return result;
}

}  // namespace tesseract::graph

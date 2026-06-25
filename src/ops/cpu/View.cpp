#include "tesseract/ops/View.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "tesseract/autograd/Function.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/graph/GraphScope.hpp"

namespace tesseract::ops {

namespace {

// ViewBackward / ReshapeBackward are identical up to name: the gradient
// flowing in has the same numel as the original tensor, so we simply reshape
// it back. We always materialize a contiguous result because consumers (the
// engine's `ops::add` accumulator) require it.
struct ReshapeBackward : Node {
  Shape input_shape;
  bool is_view_op = false;  // purely informational (node name)
  std::string_view name() const override {
    return is_view_op ? "ViewBackward" : "ReshapeBackward";
  }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    Tensor gc = g.is_contiguous() ? g : g.contiguous();
    std::vector<Tensor> outs(1);
    if (next_edges[0].requires_grad) outs[0] = gc.view(input_shape);
    return outs;
  }
};

struct PermuteBackward : Node {
  // Inverse permutation: inv[axes[i]] = i. We store it as a raw vector rather
  // than a Shape because Shape is an int64_t container of dims and kMaxRank
  // already covers our needs, but a plain vector avoids confusing semantics.
  std::vector<int64_t> inv_axes;
  std::string_view name() const override { return "PermuteBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    Tensor gp = g.permute(std::span<const int64_t>(inv_axes.data(), inv_axes.size()));
    outs[0] = gp.contiguous();
    return outs;
  }
};

struct TransposeBackward : Node {
  int64_t dim_a;
  int64_t dim_b;
  std::string_view name() const override { return "TransposeBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    outs[0] = g.transpose(dim_a, dim_b).contiguous();
    return outs;
  }
};

// Contiguous and Clone both have identity gradient (shape preserved, values
// copied element-wise). Even when the forward was a no-op (input already
// contiguous), producing a contiguous clone in backward keeps the accumulator
// contract simple.
struct IdentityBackward : Node {
  std::string_view node_name;
  std::string_view name() const override { return node_name; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (next_edges[0].requires_grad) {
      outs[0] = g.is_contiguous() ? g : g.contiguous();
    }
    return outs;
  }
};

// Attach a 1-input backward Node if `self` is part of the graph. Returns
// `out` unchanged for convenience.
template <typename NodeT, typename Configure>
Tensor maybe_record(const Tensor& self, Tensor out, Configure&& configure) {
  if (!is_grad_enabled() || !autograd::any_requires_grad(self)) return out;
  auto node = std::make_shared<NodeT>();
  configure(*node);
  node->next_edges = { autograd::edge_for(self) };
  autograd::attach_grad_fn(out, node);
  return out;
}

}  // namespace

namespace {

std::vector<int64_t> shape_to_attr(const Shape& s) {
  std::vector<int64_t> out;
  out.reserve(s.rank());
  for (std::size_t i = 0; i < s.rank(); ++i) out.push_back(s[i]);
  return out;
}

}  // namespace

Tensor view(const Tensor& self, Shape new_shape) {
  Shape input_shape = self.shape();
  Shape ns_attr = new_shape;
  Tensor out = self.view(std::move(new_shape));
  Tensor res = maybe_record<ReshapeBackward>(self, std::move(out),
                                             [&](ReshapeBackward& n) {
                                               n.input_shape = input_shape;
                                               n.is_view_op = true;
                                             });
  graph::maybe_record("view", {&self}, {&res},
                      {{"shape", shape_to_attr(ns_attr)}});
  return res;
}

Tensor reshape(const Tensor& self, Shape new_shape) {
  Shape input_shape = self.shape();
  Shape ns_attr = new_shape;
  Tensor out = self.reshape(std::move(new_shape));
  Tensor res = maybe_record<ReshapeBackward>(self, std::move(out),
                                             [&](ReshapeBackward& n) {
                                               n.input_shape = input_shape;
                                               n.is_view_op = false;
                                             });
  graph::maybe_record("reshape", {&self}, {&res},
                      {{"shape", shape_to_attr(ns_attr)}});
  return res;
}

Tensor permute(const Tensor& self, std::span<const int64_t> axes) {
  Tensor out = self.permute(axes);
  std::vector<int64_t> axes_vec(axes.begin(), axes.end());
  if (is_grad_enabled() && autograd::any_requires_grad(self)) {
    std::vector<int64_t> inv(axes.size());
    for (std::size_t i = 0; i < axes.size(); ++i) {
      inv[static_cast<std::size_t>(axes[i])] = static_cast<int64_t>(i);
    }
    auto node = std::make_shared<PermuteBackward>();
    node->inv_axes = std::move(inv);
    node->next_edges = { autograd::edge_for(self) };
    autograd::attach_grad_fn(out, node);
  }
  graph::maybe_record("permute", {&self}, {&out},
                      {{"axes", std::move(axes_vec)}});
  return out;
}

Tensor permute(const Tensor& self, std::initializer_list<int64_t> axes) {
  return permute(self, std::span<const int64_t>(axes.begin(), axes.size()));
}

Tensor transpose(const Tensor& self, int64_t dim_a, int64_t dim_b) {
  Tensor out = self.transpose(dim_a, dim_b);
  Tensor res = maybe_record<TransposeBackward>(self, std::move(out),
                                               [&](TransposeBackward& n) {
                                                 n.dim_a = dim_a;
                                                 n.dim_b = dim_b;
                                               });
  graph::maybe_record("transpose", {&self}, {&res},
                      {{"dim_a", static_cast<int64_t>(dim_a)},
                       {"dim_b", static_cast<int64_t>(dim_b)}});
  return res;
}

Tensor contiguous(const Tensor& self) {
  Tensor out = self.contiguous();
  Tensor res = maybe_record<IdentityBackward>(self, std::move(out),
                                              [&](IdentityBackward& n) {
                                                n.node_name = "ContiguousBackward";
                                              });
  graph::maybe_record("contiguous", {&self}, {&res});
  return res;
}

Tensor clone(const Tensor& self) {
  Tensor out = self.clone();
  Tensor res = maybe_record<IdentityBackward>(self, std::move(out),
                                              [&](IdentityBackward& n) {
                                                n.node_name = "CloneBackward";
                                              });
  graph::maybe_record("clone", {&self}, {&res});
  return res;
}

}  // namespace tesseract::ops

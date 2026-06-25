#pragma once

#include <memory>
#include <vector>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/Tensor.hpp"

namespace tesseract::autograd {

// Helper: make an Edge from a forward-input tensor. Captures its grad_fn /
// leaf-impl / requires_grad flag atomically so the engine has all it needs at
// backward time.
inline Node::Edge edge_for(const Tensor& t) {
  Node::Edge e;
  if (!t.defined()) return e;
  const AutogradMeta* am = t.autograd_meta();
  if (!am) return e;
  e.requires_grad = am->requires_grad || am->grad_fn;
  e.grad_fn = am->grad_fn;
  if (!e.grad_fn && am->requires_grad) {
    // Leaf tensor that accumulates gradient directly.
    e.leaf_impl = t.impl();
  }
  return e;
}

// Attach `node` as `out`'s producer and mark `out` as needing grad downstream.
inline void attach_grad_fn(Tensor& out, std::shared_ptr<Node> node) {
  auto* am = out.mutable_autograd_meta();
  am->grad_fn = std::move(node);
  // `requires_grad` remains false on non-leaf tensors; the presence of
  // `grad_fn` suffices to mark them as participating in the graph.
}

// True if any of the arguments requires grad. Used to guard backward wiring.
inline bool any_requires_grad(const Tensor& a) { return a.requires_grad(); }
inline bool any_requires_grad(const Tensor& a, const Tensor& b) {
  return a.requires_grad() || b.requires_grad();
}

}  // namespace tesseract::autograd

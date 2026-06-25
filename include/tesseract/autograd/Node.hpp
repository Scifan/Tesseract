#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "tesseract/core/Tensor.hpp"

namespace tesseract {

// A node in the autograd graph. Every backward op derives from this base and
// overrides `apply`. For M0 each Node has exactly one output (most DL ops).
//
// `next_edges` wires the Node back into the graph: for each forward input,
//   - `grad_fn` is the producer Node (or nullptr if the input is a leaf)
//   - `leaf_impl` is the underlying TensorImpl if the input is a leaf that
//     requires grad (the engine accumulates `grad` into `leaf_impl->grad`)
//
// `input_requires_grad[i]` records whether forward input `i` requires grad at
// the moment the op was recorded. The engine uses this to skip work for inputs
// whose gradients will be thrown away.
class Node : public std::enable_shared_from_this<Node> {
 public:
  struct Edge {
    std::shared_ptr<Node> grad_fn;      // producer node, or nullptr for leaves
    std::weak_ptr<TensorImpl> leaf_impl;
    bool requires_grad{false};
  };

  virtual ~Node() = default;

  // Compute gradients for each input given `grad_output` w.r.t. this Node's
  // sole output. Returns one gradient per forward input (same order). Return
  // an undefined Tensor for inputs that require no gradient.
  virtual std::vector<Tensor> apply(const Tensor& grad_output) = 0;

  // Identifier for debug / introspection.
  virtual std::string_view name() const = 0;

  std::vector<Edge> next_edges;
};

}  // namespace tesseract

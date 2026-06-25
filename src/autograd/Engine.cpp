#include "tesseract/autograd/Engine.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract {

void Engine::backward(const Tensor& root, Tensor grad) {
  if (!root.defined()) return;

  const AutogradMeta* root_am = root.autograd_meta();
  if (!root_am || !root_am->grad_fn) {
    // Either a leaf with no gradient to propagate, or an intermediate without
    // recorded autograd. Either way, nothing to do.
    return;
  }

  if (!grad.defined()) {
    TESSERACT_CHECK(root.numel() == 1,
                    "Engine::backward: implicit grad only valid for scalar outputs "
                    "(got shape {})", root.shape().to_string());
    grad = Tensor::ones(root.shape(), root.dtype(), root.device());
  } else {
    TESSERACT_CHECK(grad.shape() == root.shape(),
                    "Engine::backward: grad shape {} != root shape {}",
                    grad.shape().to_string(), root.shape().to_string());
    TESSERACT_CHECK(grad.dtype() == root.dtype(),
                    "Engine::backward: grad dtype {} != root dtype {}",
                    dtype_name(grad.dtype()), dtype_name(root.dtype()));
  }

  // Topological sort (iterative DFS, post-order).
  std::vector<Node*> topo;
  {
    std::unordered_set<Node*> visited;
    std::vector<std::pair<Node*, std::size_t>> stack;
    Node* root_node = root_am->grad_fn.get();
    stack.emplace_back(root_node, 0);
    visited.insert(root_node);
    while (!stack.empty()) {
      auto& [n, child_idx] = stack.back();
      if (child_idx < n->next_edges.size()) {
        const auto& e = n->next_edges[child_idx++];
        if (e.grad_fn && visited.insert(e.grad_fn.get()).second) {
          stack.emplace_back(e.grad_fn.get(), 0);
        }
      } else {
        topo.push_back(n);
        stack.pop_back();
      }
    }
  }

  // Accumulate grads per Node. Process from consumer (root) toward producers.
  std::unordered_map<Node*, Tensor> acc;
  acc[root_am->grad_fn.get()] = grad;

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    Node* node = *it;
    auto found = acc.find(node);
    if (found == acc.end()) continue;
    Tensor grad_out = std::move(found->second);
    acc.erase(found);

    std::vector<Tensor> grads_in = node->apply(grad_out);
    TESSERACT_CHECK(grads_in.size() == node->next_edges.size(),
                    "Node '{}' returned {} grads, expected {}", node->name(),
                    grads_in.size(), node->next_edges.size());

    for (std::size_t i = 0; i < grads_in.size(); ++i) {
      Tensor& gi = grads_in[i];
      const auto& edge = node->next_edges[i];
      if (!edge.requires_grad || !gi.defined()) continue;

      if (edge.grad_fn) {
        auto& dst = acc[edge.grad_fn.get()];
        if (dst.defined()) {
          dst = ops::add(dst, gi);
        } else {
          dst = std::move(gi);
        }
      } else {
        // Leaf accumulation.
        auto leaf = edge.leaf_impl.lock();
        if (!leaf || !leaf->autograd_meta || !leaf->autograd_meta->requires_grad) continue;
        Tensor& leaf_grad = leaf->autograd_meta->grad;
        if (leaf_grad.defined()) {
          leaf_grad = ops::add(leaf_grad, gi);
        } else {
          leaf_grad = std::move(gi);
        }
      }
    }
  }
}

}  // namespace tesseract

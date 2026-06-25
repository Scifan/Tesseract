#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tesseract/core/Tensor.hpp"

namespace tesseract::nn {

// Base class for composable neural-network modules. Modules own parameters
// (leaf tensors with `requires_grad=true`) and child sub-modules. `forward`
// is defined by subclasses; `parameters()` returns a flat list of parameters
// used by optimizers.
//
// Not copyable; always wrapped in `std::shared_ptr<Module>` when used
// polymorphically through `Sequential`.
class Module : public std::enable_shared_from_this<Module> {
 public:
  Module() = default;
  virtual ~Module() = default;
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  // Every subclass implements this. For modules taking multiple inputs, add
  // overloads as needed; the base class does not constrain the signature.
  virtual Tensor forward(const Tensor& x) {
    (void)x;
    TESSERACT_THROW("Module::forward(Tensor) not implemented by subclass");
  }

  // Flat list of parameters (recursive through children) in insertion order.
  std::vector<Tensor> parameters() const;

  // Flat list of (qualified-name, parameter) pairs. Names follow the
  // PyTorch convention: children's names are concatenated with `.` so a
  // transformer's Q projection surfaces as e.g.
  // `layers.0.attn.q_proj.weight`. Order is insertion order (parent
  // params first, then each child recursively). Used primarily by
  // checkpoint loaders that need to match HF-style tensor names.
  std::vector<std::pair<std::string, Tensor>> named_parameters() const;

  // Flat list of (qualified-name, buffer) pairs. Same naming rules as
  // `named_parameters`. Used when a loader needs to restore e.g.
  // BatchNorm running stats or KV caches alongside learnable weights.
  std::vector<std::pair<std::string, Tensor>> named_buffers() const;

  // Zero out `.grad` for every parameter (recursive).
  void zero_grad();

  // Move every registered parameter (recursively through children) to
  // `target_device`. Implemented via `Tensor::move_to_`, so the change is
  // observed by every Tensor handle currently aliasing a given
  // parameter's impl — in particular, the `weight_` / `bias_` members
  // of `nn::Linear` stay in sync with the copies stored in `params_`.
  // Returns `this` for call chaining.
  Module* to(Device target_device);

  // Train / eval mode toggle. Recurses through every registered child so
  // `model->train(false)` flips the whole tree to eval — exactly the
  // semantics `nn.Module.train()` gives you in PyTorch. Without the
  // recursion a `nn::Sequential` would happily keep a leaf
  // `BatchNorm2d` in `training=true` while its parent thinks it's in
  // eval, which is the classic "running stats keep drifting at
  // inference" bug. Modules that care about the flag (BatchNorm,
  // Dropout, …) read it off their local `is_training()` in `forward()`.
  //
  // `eval()` is the customary alias for `train(false)`.
  void train(bool on = true);
  void eval() { train(false); }
  bool is_training() const noexcept { return training_; }

 protected:
  // Register a named parameter. The same name must be unique within a module.
  // Stores a copy of the Tensor handle (shared ptr); mutations to the
  // underlying storage (e.g. optimizer updates) are reflected everywhere.
  void register_parameter(std::string name, Tensor tensor);

  // Register a named buffer. Buffers are per-module persistent tensors
  // that are moved along with parameters by `Module::to()` but are
  // **not** returned by `parameters()` and do **not** participate in
  // autograd — e.g. cached RoPE cos/sin tables, running mean/var for
  // a future BatchNorm, K/V caches for inference. Same alias rules as
  // `register_parameter`: the Tensor handle stored here shares its
  // impl with whatever member the subclass holds, so `.to(Device)`
  // rewrites both in lockstep.
  void register_buffer(std::string name, Tensor tensor);

  // Register a child module.
  void register_module(std::string name, std::shared_ptr<Module> child);

  // Replace an already-registered child module with a new one under the
  // same `name`, preserving its position in `children_`. Used by
  // in-place transformations that need to swap a sub-module without
  // rebuilding the parent — e.g. `LlamaModel::quantize_` which flips
  // every registered `nn::Linear` projection to its INT8 / INT4-group
  // quantized replacement while keeping `named_parameters()` /
  // `named_buffers()` walk order deterministic.
  //
  // Throws when `name` is not an existing child (creating a new child
  // should go through `register_module` instead — the distinction is
  // intentional so typos don't silently reshape the module tree).
  void replace_module(const std::string& name, std::shared_ptr<Module> child);

  // Non-const accessor for derived classes that want to iterate /
  // introspect their own child map (same visibility contract as
  // `register_module`). Currently used by `quantize_` walkers which
  // need to find every `Linear` child and swap it out.
  std::vector<std::pair<std::string, std::shared_ptr<Module>>>& children() {
    return children_;
  }
  const std::vector<std::pair<std::string, std::shared_ptr<Module>>>&
      children() const { return children_; }

 private:
  void collect_parameters(std::vector<Tensor>& out) const;
  void collect_named_parameters(
      const std::string& prefix,
      std::vector<std::pair<std::string, Tensor>>& out) const;
  void collect_named_buffers(
      const std::string& prefix,
      std::vector<std::pair<std::string, Tensor>>& out) const;

  bool training_{true};

  // Ordered name->Tensor so iteration is deterministic.
  std::vector<std::pair<std::string, Tensor>> params_;
  std::vector<std::pair<std::string, Tensor>> buffers_;
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
};

}  // namespace tesseract::nn

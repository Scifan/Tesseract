#include "tesseract/nn/Module.hpp"

#include <utility>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

void Module::register_parameter(std::string name, Tensor tensor) {
  TESSERACT_CHECK(tensor.defined(), "Module::register_parameter('{}'): tensor undefined", name);
  // Parameters must be leaves so the optimizer can update them in-place and
  // the autograd engine can accumulate into `.grad`.
  tensor.set_requires_grad(true);
  params_.emplace_back(std::move(name), std::move(tensor));
}

void Module::register_buffer(std::string name, Tensor tensor) {
  TESSERACT_CHECK(tensor.defined(), "Module::register_buffer('{}'): tensor undefined", name);
  // Buffers stay detached from autograd — the whole point is "moves
  // with the module, but has no gradient". Explicitly clear the flag
  // in case the caller handed us a grad-carrying handle by accident.
  tensor.set_requires_grad(false);
  buffers_.emplace_back(std::move(name), std::move(tensor));
}

void Module::register_module(std::string name, std::shared_ptr<Module> child) {
  TESSERACT_CHECK(child != nullptr, "Module::register_module('{}'): null child", name);
  children_.emplace_back(std::move(name), std::move(child));
}

void Module::replace_module(const std::string& name,
                            std::shared_ptr<Module> child) {
  TESSERACT_CHECK(child != nullptr,
                  "Module::replace_module('{}'): null child", name);
  for (auto& entry : children_) {
    if (entry.first == name) {
      entry.second = std::move(child);
      return;
    }
  }
  TESSERACT_THROW("Module::replace_module: no existing child named '{}'", name);
}

void Module::collect_parameters(std::vector<Tensor>& out) const {
  for (const auto& p : params_) out.push_back(p.second);
  for (const auto& c : children_) c.second->collect_parameters(out);
}

std::vector<Tensor> Module::parameters() const {
  std::vector<Tensor> out;
  collect_parameters(out);
  return out;
}

void Module::collect_named_parameters(
    const std::string& prefix,
    std::vector<std::pair<std::string, Tensor>>& out) const {
  for (const auto& p : params_) {
    out.emplace_back(prefix + p.first, p.second);
  }
  for (const auto& c : children_) {
    const std::string child_prefix = prefix + c.first + ".";
    c.second->collect_named_parameters(child_prefix, out);
  }
}

void Module::collect_named_buffers(
    const std::string& prefix,
    std::vector<std::pair<std::string, Tensor>>& out) const {
  for (const auto& b : buffers_) {
    out.emplace_back(prefix + b.first, b.second);
  }
  for (const auto& c : children_) {
    const std::string child_prefix = prefix + c.first + ".";
    c.second->collect_named_buffers(child_prefix, out);
  }
}

std::vector<std::pair<std::string, Tensor>> Module::named_parameters() const {
  std::vector<std::pair<std::string, Tensor>> out;
  collect_named_parameters("", out);
  return out;
}

std::vector<std::pair<std::string, Tensor>> Module::named_buffers() const {
  std::vector<std::pair<std::string, Tensor>> out;
  collect_named_buffers("", out);
  return out;
}

void Module::zero_grad() {
  for (auto& p : parameters()) {
    auto* am = p.mutable_autograd_meta();
    am->grad = Tensor{};
  }
}

void Module::train(bool on) {
  training_ = on;
  // Recurse into every registered child so calling `train(false)` on the
  // root flips the entire sub-tree in one call. Matches `nn.Module.train`
  // in PyTorch: otherwise a composite like `Sequential(BatchNorm2d, …)`
  // would leak a `training=true` leaf into what the outer caller thinks
  // is pure inference. Leaves without children are a no-op on the loop.
  for (auto& c : children_) {
    c.second->train(on);
  }
}

Module* Module::to(Device target_device) {
  // Recurse through children first — order is immaterial for correctness
  // but keeping parents last mirrors the PyTorch iteration order (helpful
  // when someone is stepping through with a debugger).
  for (auto& c : children_) {
    c.second->to(target_device);
  }
  // Move the leaves. `Tensor::move_to_` rewrites the shared TensorImpl's
  // storage/device/shape/strides fields in place, which is why the
  // Module's owner (Linear, etc.) still sees the update through its own
  // `weight_` / `bias_` members — they alias the same impl.
  for (auto& p : params_) {
    p.second.move_to_(target_device);
  }
  // Buffers follow parameters onto the target device so the whole
  // module graph stays consistent (e.g. an RMSNorm weight on CUDA
  // next to a RoPE cos/sin table on CPU would be a correctness bug).
  // Aliasing rules are identical to parameters — `move_to_` rewrites
  // the shared impl, so the subclass member handle sees the update
  // without us having to re-store it anywhere.
  for (auto& b : buffers_) {
    b.second.move_to_(target_device);
  }
  return this;
}

}  // namespace tesseract::nn

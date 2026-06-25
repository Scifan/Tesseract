#pragma once

#include <memory>
#include <string>
#include <utility>

#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// Ordered, index-addressable container of child modules.
//
// Use this when you have a homogeneous stack of sub-modules (transformer
// blocks, residual-ffn towers, etc.) and you want them to surface in
// `named_parameters()` as `<holder>.0.…`, `<holder>.1.…`, matching the
// PyTorch / HF convention for `nn.ModuleList`.
//
// Unlike `Sequential`, `ModuleList` does not define a composed forward —
// the containing module iterates the children however it likes (e.g. the
// Llama stack threads residual connections across blocks manually). A
// default-thrown `forward(Tensor)` is inherited from Module; calling it
// is a programmer error.
class ModuleList : public Module {
 public:
  ModuleList() = default;

  // Append `child` under the next integer-string name ("0", "1", …). The
  // name is both what `named_parameters()` uses as a prefix and what
  // tests/loaders will see in qualified paths.
  void append(std::shared_ptr<Module> child) {
    register_module(std::to_string(size_), std::move(child));
    ++size_;
  }

  // Append under an explicit string name (e.g. for mixed-topology
  // containers that still want named rather than indexed access).
  void append(std::string name, std::shared_ptr<Module> child) {
    register_module(std::move(name), std::move(child));
    ++size_;
  }

  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

 private:
  std::size_t size_{0};
};

}  // namespace tesseract::nn

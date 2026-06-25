#pragma once

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// Compose modules in order: forward runs each sub-module sequentially,
// passing its output to the next. The caller supplies modules via a
// braced-init list of `std::shared_ptr<Module>`.
class Sequential : public Module {
 public:
  Sequential() = default;
  Sequential(std::initializer_list<std::shared_ptr<Module>> modules);

  Sequential& add(std::shared_ptr<Module> m);

  Tensor forward(const Tensor& x) override;

  std::size_t size() const noexcept { return items_.size(); }

 private:
  std::vector<std::shared_ptr<Module>> items_;
};

}  // namespace tesseract::nn

#include "tesseract/nn/Sequential.hpp"

#include <string>

namespace tesseract::nn {

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
  for (auto m : modules) add(std::move(m));
}

Sequential& Sequential::add(std::shared_ptr<Module> m) {
  const std::size_t idx = items_.size();
  items_.push_back(m);
  register_module(std::to_string(idx), std::move(m));
  return *this;
}

Tensor Sequential::forward(const Tensor& x) {
  Tensor y = x;
  for (auto& m : items_) y = m->forward(y);
  return y;
}

}  // namespace tesseract::nn

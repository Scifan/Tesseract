#pragma once

#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// Stateless activations. These exist as Modules so they can be plugged into
// `Sequential`; prefer the functional `tesseract::ops::relu` etc. when
// composing by hand.

class ReLU : public Module {
 public:
  Tensor forward(const Tensor& x) override;
};

class Sigmoid : public Module {
 public:
  Tensor forward(const Tensor& x) override;
};

class Tanh : public Module {
 public:
  Tensor forward(const Tensor& x) override;
};

}  // namespace tesseract::nn

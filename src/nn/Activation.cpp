#include "tesseract/nn/Activation.hpp"

#include "tesseract/ops/Activation.hpp"

namespace tesseract::nn {

Tensor ReLU::forward(const Tensor& x) { return ops::relu(x); }
Tensor Sigmoid::forward(const Tensor& x) { return ops::sigmoid(x); }
Tensor Tanh::forward(const Tensor& x) { return ops::tanh(x); }

}  // namespace tesseract::nn

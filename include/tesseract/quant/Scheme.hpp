#pragma once

// Wave 3.3 (B-021): quantization scheme descriptors + the `nn::Linear`
// ↦ quantized-module factory.
//
// A `Scheme` is a small tagged value that travels alongside the module
// walker (`LlamaModel::quantize_`, `MultiHeadAttention::quantize_`,
// `FeedForward::quantize_`) and tells it *which* quantized layer type
// to swap an `nn::Linear` with. Keeping the choice in a value rather
// than templating every walker on a static `Scheme` tag means:
//   * callers can build a scheme at runtime from a config file / CLI,
//   * we can extend with new methods (GPTQ, per-token activation
//     quant, FP8, …) without touching every walker signature,
//   * tests can parametrize a single harness over both INT8 and INT4.
//
// The helper `quantize_linear` is the canonical "fp `nn::Linear` →
// quantized drop-in module" conversion. Every walker delegates to it
// so the dispatch table lives in exactly one place.

#include <cstdint>
#include <memory>

namespace tesseract::nn {
class Module;
class Linear;
}  // namespace tesseract::nn

namespace tesseract::quant {

// Which quantization method to apply. Extend here as new schemes
// land; the switch inside `quantize_linear` is the only other site
// that has to grow.
enum class Method : std::uint8_t {
  Int8Symmetric,        // Wave 3.1: per-output-channel symmetric INT8
  Int4GroupSymmetric,   // Wave 3.2: per-group symmetric INT4 (group size in `group_size`)
};

// Concrete scheme descriptor. `group_size` is only meaningful for
// `Int4GroupSymmetric` — ignored otherwise. Default group-size of 128
// matches the GPTQ / AWQ / llama.cpp Q4_0 convention.
struct Scheme {
  Method method{Method::Int8Symmetric};
  std::int64_t group_size{128};

  static Scheme int8_symmetric() {
    Scheme s;
    s.method = Method::Int8Symmetric;
    s.group_size = 0;  // unused
    return s;
  }

  static Scheme int4_group_symmetric(std::int64_t group_size = 128) {
    Scheme s;
    s.method = Method::Int4GroupSymmetric;
    s.group_size = group_size;
    return s;
  }
};

// Convert a fully-populated `nn::Linear` into a quantized drop-in
// module (currently `nn::QuantizedLinear` for INT8 and
// `nn::QuantizedLinearInt4G` for INT4). The returned module:
//   * retains `src.bias()` as a trainable parameter (cloned, same
//     dtype as `src.weight()`),
//   * carries the quantized weight + scale(s) as **buffers** so a
//     subsequent `Module::to(device)` migrates them together,
//   * has a `forward(x)` contract equivalent to `src.forward(x)` up
//     to the quantization error documented on the respective
//     packer / op.
//
// `src.weight()` must already be in a floating-point dtype
// (Float32 / Float16 / BFloat16). FP64 is rejected — quantizing
// from FP64 is never what an inference stack wants.
//
// Throws if `scheme.method` is not a known variant or if the
// per-scheme input constraints (e.g. `in_features % group_size == 0`
// for INT4) are violated.
std::shared_ptr<nn::Module> quantize_linear(const nn::Linear& src,
                                            const Scheme& scheme);

}  // namespace tesseract::quant

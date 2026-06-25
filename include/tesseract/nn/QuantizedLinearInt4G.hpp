#pragma once

// Wave 3.2 (B-021): INT4 per-group symmetric weight-only quantized
// Linear layer.
//
// Drop-in inference replacement for `nn::Linear`, analogous to
// `nn::QuantizedLinear` (Wave 3.1) but with the group-symmetric
// INT4 storage layout produced by `quant::pack_int4_group`:
//
//   q_weight   : Int8    [out_features, in_features / 2]
//                (two signed 4-bit nibbles per byte; low = even-k,
//                 high = odd-k; values in [-7, 7])
//   weight_scale : Float32 [out_features, in_features / group_size]
//
// Forward:
//   y = ops::dequantize_matmul_int4_group(x, q_weight, weight_scale,
//                                         group_size) (+ bias)
//
// Buffer / parameter / autograd conventions match QuantizedLinear:
//   * `q_weight` and `weight_scale` are buffers (move with
//     `Module::to(device)` and serialize, but are not parameters and
//     are never differentiated through). Autograd on the input `x`
//     is routed through a composite `matmul(dequantize(...))` inside
//     the op so trainable upstream layers keep working.
//   * `bias` (optional) stays a parameter so a downstream optimizer
//     can still tune it — same convention PyTorch's QLinear uses.
//
// The `group_size` is stored per-instance rather than being a class
// template so the same class can hold layers packed at different
// group sizes (e.g. GPTQ-style 32 vs. the 128 llama.cpp default).

#include <memory>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

class QuantizedLinearInt4G : public Module {
 public:
  // Construct directly from a pre-packed weight + scale (produced by
  // `quant::pack_int4_group`). `bias` may be undefined. `group_size`
  // must match the packer's group size; it's validated against the
  // scale tensor's inner dim.
  QuantizedLinearInt4G(Tensor q_weight,
                       Tensor weight_scale,
                       Tensor bias,
                       DType compute_dtype,
                       int64_t group_size);

  // Convenience factory: quantize an existing `nn::Linear`'s weight
  // via `quant::pack_int4_group(group_size)` and clone its bias.
  // The source Linear is untouched.
  static std::shared_ptr<QuantizedLinearInt4G> from_linear(
      const Linear& src, int64_t group_size = 128);

  Tensor forward(const Tensor& x) override;

  const Tensor& q_weight()     const { return q_weight_; }
  const Tensor& weight_scale() const { return weight_scale_; }
  const Tensor& bias()         const { return bias_; }
  bool  has_bias()      const noexcept { return use_bias_; }
  DType compute_dtype() const noexcept { return compute_dtype_; }
  int64_t group_size()  const noexcept { return group_size_; }

  int64_t in_features()  const noexcept { return in_features_; }
  int64_t out_features() const noexcept { return out_features_; }

 private:
  int64_t in_features_{};
  int64_t out_features_{};
  int64_t group_size_{0};
  DType   compute_dtype_{DType::Float32};
  bool    use_bias_{false};

  Tensor q_weight_;      // [out, in/2],  Int8   (two nibbles / byte)
  Tensor weight_scale_;  // [out, in/G],  Float32
  Tensor bias_;          // [out],        compute_dtype (if use_bias_)
};

}  // namespace tesseract::nn

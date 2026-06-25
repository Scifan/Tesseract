#pragma once

// Wave 3 (B-021): INT8 weight-only quantized Linear layer.
//
// Drop-in inference replacement for `nn::Linear`: takes a packed INT8
// weight + FP32 per-output-channel scale (produced by
// `tesseract::quant::pack_int8_symmetric`) and an optional FP bias,
// and implements `y = x @ dequantize(q_w, scale)^T + bias` via
// `ops::dequantize_matmul_int8`. No materialized FP weight is ever
// kept in memory on the inference path.
//
// Interface differences vs. `nn::Linear`:
//   * `weight()` / `bias()` are not present; the frozen quantized
//     representation is exposed as `q_weight()` (Int8 [out, in]) and
//     `weight_scale()` (Float32 [out]).
//   * The module does NOT register `q_weight` / `weight_scale` as
//     `parameters()` — they are inference-only buffers. They ARE
//     registered via `register_buffer` so `Module::to(device)` and
//     checkpoint saving see them. Bias, when present, stays a real
//     parameter so the layer remains partially trainable (same
//     convention PyTorch's `torch.ao.quantization.QLinear` uses).
//   * `compute_dtype` controls the dtype of the bias and the
//     activation accumulator narrowing. Must match `x.dtype()` at
//     call time.
//
// Factory: `QuantizedLinear::from_linear(fp_linear)` quantizes a
// trained `nn::Linear`'s weight in-place via
// `quant::pack_int8_symmetric` and copies the bias across. The source
// `nn::Linear` is left untouched — callers that want to free the
// original FP weight should drop it explicitly.

#include <memory>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

class QuantizedLinear : public Module {
 public:
  // Construct directly from a pre-packed weight + scale (produced by
  // `quant::pack_int8_symmetric`). `bias` may be undefined — in that
  // case no bias term is added. `compute_dtype` is the expected
  // activation dtype; it's stored here so callers can pre-check
  // their runtime dtype matches the layer's expectation.
  QuantizedLinear(Tensor q_weight,
                  Tensor weight_scale,
                  Tensor bias,
                  DType compute_dtype);

  // Convenience factory: take an already-initialized `nn::Linear`
  // (FP32 / FP16 / BF16), quantize its weight via
  // `quant::pack_int8_symmetric`, copy its bias (if any), and
  // return a fresh QuantizedLinear. The source `Linear` is not
  // modified. `compute_dtype` defaults to the source Linear's
  // weight dtype.
  static std::shared_ptr<QuantizedLinear> from_linear(const Linear& src);

  Tensor forward(const Tensor& x) override;

  const Tensor& q_weight() const    { return q_weight_; }
  const Tensor& weight_scale() const { return weight_scale_; }
  const Tensor& bias() const         { return bias_; }
  bool  has_bias() const noexcept    { return use_bias_; }
  DType compute_dtype() const noexcept { return compute_dtype_; }

  int64_t in_features()  const noexcept { return in_features_; }
  int64_t out_features() const noexcept { return out_features_; }

 private:
  int64_t in_features_{};
  int64_t out_features_{};
  DType   compute_dtype_{DType::Float32};
  bool    use_bias_{false};

  Tensor q_weight_;      // [out, in], Int8
  Tensor weight_scale_;  // [out],     Float32
  Tensor bias_;          // [out],     compute_dtype (if use_bias_)
};

}  // namespace tesseract::nn

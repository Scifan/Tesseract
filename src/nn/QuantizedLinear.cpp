#include "tesseract/nn/QuantizedLinear.hpp"

#include <memory>
#include <utility>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Quant.hpp"
#include "tesseract/quant/Pack.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

QuantizedLinear::QuantizedLinear(Tensor q_weight,
                                 Tensor weight_scale,
                                 Tensor bias,
                                 DType compute_dtype)
    : compute_dtype_(compute_dtype) {
  TESSERACT_CHECK(q_weight.defined(),
                  "QuantizedLinear: q_weight must be defined");
  TESSERACT_CHECK(q_weight.rank() == 2,
                  "QuantizedLinear: q_weight must be rank-2 [out, in], got {}",
                  q_weight.shape().to_string());
  TESSERACT_CHECK(q_weight.dtype() == DType::Int8,
                  "QuantizedLinear: q_weight must be Int8, got {}",
                  dtype_name(q_weight.dtype()));

  TESSERACT_CHECK(weight_scale.defined(),
                  "QuantizedLinear: weight_scale must be defined");
  TESSERACT_CHECK(weight_scale.rank() == 1,
                  "QuantizedLinear: weight_scale must be rank-1 [out], got {}",
                  weight_scale.shape().to_string());
  TESSERACT_CHECK(weight_scale.dtype() == DType::Float32,
                  "QuantizedLinear: weight_scale must be Float32, got {}",
                  dtype_name(weight_scale.dtype()));
  TESSERACT_CHECK(weight_scale.shape()[0] == q_weight.shape()[0],
                  "QuantizedLinear: weight_scale[out]={} does not match "
                  "q_weight[out]={}", weight_scale.shape()[0],
                  q_weight.shape()[0]);

  TESSERACT_CHECK(q_weight.device() == weight_scale.device(),
                  "QuantizedLinear: q_weight / weight_scale device mismatch "
                  "({} vs {})", q_weight.device().to_string(),
                  weight_scale.device().to_string());

  out_features_ = q_weight.shape()[0];
  in_features_  = q_weight.shape()[1];
  use_bias_     = bias.defined();

  q_weight_     = std::move(q_weight);
  weight_scale_ = std::move(weight_scale);

  // Register the frozen tensors as buffers — they move with the module
  // via `Module::to()` and appear in `named_buffers()`, but are not
  // part of `parameters()` (they're integer / inference-only).
  register_buffer("q_weight",     q_weight_);
  register_buffer("weight_scale", weight_scale_);

  if (use_bias_) {
    TESSERACT_CHECK(bias.rank() == 1,
                    "QuantizedLinear: bias must be rank-1 [out], got {}",
                    bias.shape().to_string());
    TESSERACT_CHECK(bias.shape()[0] == out_features_,
                    "QuantizedLinear: bias[out]={} does not match "
                    "q_weight[out]={}", bias.shape()[0], out_features_);
    TESSERACT_CHECK(bias.dtype() == compute_dtype_,
                    "QuantizedLinear: bias dtype {} does not match "
                    "compute_dtype {}", dtype_name(bias.dtype()),
                    dtype_name(compute_dtype_));
    TESSERACT_CHECK(bias.device() == q_weight_.device(),
                    "QuantizedLinear: bias / q_weight device mismatch "
                    "({} vs {})", bias.device().to_string(),
                    q_weight_.device().to_string());
    bias_ = std::move(bias);
    // Bias stays a real trainable parameter — the quantization is
    // weight-only. A frozen bias would need `register_buffer` instead
    // and no optimizer would touch it; using `register_parameter`
    // keeps the layer composable inside an otherwise-trainable stack.
    register_parameter("bias", bias_);
  }
}

std::shared_ptr<QuantizedLinear> QuantizedLinear::from_linear(const Linear& src) {
  const DType compute_dtype = src.weight().dtype();
  auto [q_w, scale] = quant::pack_int8_symmetric(src.weight());

  // Clone the bias so the source Linear's bias storage is not shared
  // with the quantized layer's (optimizer state stays disjoint).
  Tensor bias_copy;
  if (src.has_bias()) {
    bias_copy = src.bias().clone();
  }

  return std::make_shared<QuantizedLinear>(
      std::move(q_w), std::move(scale), std::move(bias_copy), compute_dtype);
}

Tensor QuantizedLinear::forward(const Tensor& x) {
  TESSERACT_CHECK(x.defined(),
                  "QuantizedLinear::forward: input tensor is undefined");
  TESSERACT_CHECK(x.rank() >= 2,
                  "QuantizedLinear::forward: expected rank >= 2 input "
                  "[..., in_features], got {}", x.shape().to_string());
  const int64_t last = x.shape()[x.rank() - 1];
  TESSERACT_CHECK(last == in_features_,
                  "QuantizedLinear::forward: input feature dim {} != "
                  "in_features {}", last, in_features_);
  TESSERACT_CHECK(x.dtype() == compute_dtype_,
                  "QuantizedLinear::forward: input dtype {} does not match "
                  "compute_dtype {}", dtype_name(x.dtype()),
                  dtype_name(compute_dtype_));

  // Wave 4.4 (B-026) — eval-mode fast path.
  //
  // The underlying `ops::dequantize_matmul_int8` op routes through an
  // autograd fallback whenever `is_grad_enabled() && x.requires_grad()`
  // that materializes the full FP weight `[N, K]` on-device and runs a
  // regular `matmul` — correct, but 4× memory blowup for INT8 and 8×
  // for INT4G, and the dequant alloc is a fresh tensor every forward.
  //
  // When `this` is in eval mode (`is_training() == false`) we know the
  // caller is running inference and does NOT need a differentiable
  // path, so we install a `NoGradGuard` for the whole forward. The
  // guard is thread-local and nests correctly with any outer
  // `NoGradGuard` the caller already has, and the op-layer's
  // autograd-fallback branch short-circuits on `!is_grad_enabled()`
  // before touching the dequant-weight materializer. Net effect:
  // `model.eval()` alone is sufficient to pin the fused inference
  // kernel regardless of `x.requires_grad()`.
  //
  // Train mode (`is_training() == true`) stays exactly as before —
  // if autograd is enabled and `x.requires_grad()` then the op-layer
  // fallback still fires, so LoRA / fine-tune stacks that wrap a
  // frozen quantized backbone in a trainable adapter keep working.
  if (!is_training()) {
    NoGradGuard nogg;
    Tensor y = ops::dequantize_matmul_int8(x, q_weight_, weight_scale_);
    if (use_bias_) y = ops::add(y, bias_);
    return y;
  }

  Tensor y = ops::dequantize_matmul_int8(x, q_weight_, weight_scale_);
  if (use_bias_) {
    y = ops::add(y, bias_);
  }
  return y;
}

}  // namespace tesseract::nn

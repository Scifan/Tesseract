#include "tesseract/nn/QuantizedLinearInt4G.hpp"

#include <memory>
#include <utility>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Quant.hpp"
#include "tesseract/quant/Pack.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

QuantizedLinearInt4G::QuantizedLinearInt4G(Tensor q_weight,
                                           Tensor weight_scale,
                                           Tensor bias,
                                           DType compute_dtype,
                                           int64_t group_size)
    : group_size_(group_size),
      compute_dtype_(compute_dtype) {
  TESSERACT_CHECK(q_weight.defined(),
                  "QuantizedLinearInt4G: q_weight must be defined");
  TESSERACT_CHECK(q_weight.rank() == 2,
                  "QuantizedLinearInt4G: q_weight must be rank-2 "
                  "[out, in/2], got {}", q_weight.shape().to_string());
  TESSERACT_CHECK(q_weight.dtype() == DType::Int8,
                  "QuantizedLinearInt4G: q_weight must be Int8 "
                  "(packed nibbles), got {}", dtype_name(q_weight.dtype()));

  TESSERACT_CHECK(weight_scale.defined(),
                  "QuantizedLinearInt4G: weight_scale must be defined");
  TESSERACT_CHECK(weight_scale.rank() == 2,
                  "QuantizedLinearInt4G: weight_scale must be rank-2 "
                  "[out, in/group_size], got {}",
                  weight_scale.shape().to_string());
  TESSERACT_CHECK(weight_scale.dtype() == DType::Float32,
                  "QuantizedLinearInt4G: weight_scale must be Float32, got {}",
                  dtype_name(weight_scale.dtype()));

  TESSERACT_CHECK(group_size >= 2 && (group_size % 2) == 0,
                  "QuantizedLinearInt4G: group_size must be even and >=2, "
                  "got {}", group_size);

  out_features_ = q_weight.shape()[0];
  in_features_  = q_weight.shape()[1] * 2;  // two nibbles per packed byte

  TESSERACT_CHECK(weight_scale.shape()[0] == out_features_,
                  "QuantizedLinearInt4G: weight_scale[out]={} does not match "
                  "q_weight[out]={}", weight_scale.shape()[0], out_features_);
  TESSERACT_CHECK(in_features_ % group_size == 0,
                  "QuantizedLinearInt4G: in_features ({}) must be a multiple "
                  "of group_size ({})", in_features_, group_size);
  TESSERACT_CHECK(weight_scale.shape()[1] == in_features_ / group_size,
                  "QuantizedLinearInt4G: weight_scale[groups]={} must equal "
                  "in_features/group_size={}",
                  weight_scale.shape()[1], in_features_ / group_size);

  TESSERACT_CHECK(q_weight.device() == weight_scale.device(),
                  "QuantizedLinearInt4G: q_weight / weight_scale device "
                  "mismatch ({} vs {})", q_weight.device().to_string(),
                  weight_scale.device().to_string());

  use_bias_     = bias.defined();
  q_weight_     = std::move(q_weight);
  weight_scale_ = std::move(weight_scale);

  register_buffer("q_weight",     q_weight_);
  register_buffer("weight_scale", weight_scale_);

  if (use_bias_) {
    TESSERACT_CHECK(bias.rank() == 1,
                    "QuantizedLinearInt4G: bias must be rank-1 [out], got {}",
                    bias.shape().to_string());
    TESSERACT_CHECK(bias.shape()[0] == out_features_,
                    "QuantizedLinearInt4G: bias[out]={} does not match "
                    "q_weight[out]={}", bias.shape()[0], out_features_);
    TESSERACT_CHECK(bias.dtype() == compute_dtype_,
                    "QuantizedLinearInt4G: bias dtype {} does not match "
                    "compute_dtype {}", dtype_name(bias.dtype()),
                    dtype_name(compute_dtype_));
    TESSERACT_CHECK(bias.device() == q_weight_.device(),
                    "QuantizedLinearInt4G: bias / q_weight device mismatch "
                    "({} vs {})", bias.device().to_string(),
                    q_weight_.device().to_string());
    bias_ = std::move(bias);
    register_parameter("bias", bias_);
  }
}

std::shared_ptr<QuantizedLinearInt4G> QuantizedLinearInt4G::from_linear(
    const Linear& src, int64_t group_size) {
  const DType compute_dtype = src.weight().dtype();
  auto [q_w, scale] = quant::pack_int4_group(src.weight(), group_size);

  Tensor bias_copy;
  if (src.has_bias()) {
    bias_copy = src.bias().clone();
  }

  return std::make_shared<QuantizedLinearInt4G>(
      std::move(q_w), std::move(scale), std::move(bias_copy),
      compute_dtype, group_size);
}

Tensor QuantizedLinearInt4G::forward(const Tensor& x) {
  TESSERACT_CHECK(x.defined(),
                  "QuantizedLinearInt4G::forward: input tensor is undefined");
  TESSERACT_CHECK(x.rank() >= 2,
                  "QuantizedLinearInt4G::forward: expected rank >= 2 input "
                  "[..., in_features], got {}", x.shape().to_string());
  const int64_t last = x.shape()[x.rank() - 1];
  TESSERACT_CHECK(last == in_features_,
                  "QuantizedLinearInt4G::forward: input feature dim {} != "
                  "in_features {}", last, in_features_);
  TESSERACT_CHECK(x.dtype() == compute_dtype_,
                  "QuantizedLinearInt4G::forward: input dtype {} does not "
                  "match compute_dtype {}", dtype_name(x.dtype()),
                  dtype_name(compute_dtype_));

  // Wave 4.4 (B-026) — eval-mode fast path, same contract as
  // `QuantizedLinear::forward`: when `is_training() == false` we
  // install a `NoGradGuard` so the op-layer never reaches its
  // autograd-fallback branch and we skip the full FP weight
  // materialization (which for INT4G is an 8× memory blow-up vs the
  // packed nibble storage). Train mode keeps the existing behavior:
  // autograd on `x` flows through the fallback's `matmul` path so
  // LoRA-style fine-tuning with a frozen INT4G backbone still works.
  if (!is_training()) {
    NoGradGuard nogg;
    Tensor y = ops::dequantize_matmul_int4_group(x, q_weight_, weight_scale_,
                                                 group_size_);
    if (use_bias_) y = ops::add(y, bias_);
    return y;
  }

  Tensor y = ops::dequantize_matmul_int4_group(x, q_weight_, weight_scale_,
                                               group_size_);
  if (use_bias_) {
    y = ops::add(y, bias_);
  }
  return y;
}

}  // namespace tesseract::nn

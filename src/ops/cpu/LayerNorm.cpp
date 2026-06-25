#include "tesseract/ops/Normalization.hpp"

#include "tesseract/autograd/Function.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/detail/RMSNorm.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::ops {

Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  double eps) {
  TESSERACT_CHECK(x.defined() && weight.defined(),
                  "layer_norm: x and weight must be defined tensors");
  TESSERACT_CHECK(dtype_is_floating(x.dtype()),
                  "layer_norm: floating-point dtype required, got {}",
                  dtype_name(x.dtype()));
  TESSERACT_CHECK(x.dtype() == weight.dtype(),
                  "layer_norm: x/weight dtype mismatch ({} vs {})",
                  dtype_name(x.dtype()), dtype_name(weight.dtype()));
  TESSERACT_CHECK(x.device() == weight.device(),
                  "layer_norm: x/weight device mismatch ({} vs {})",
                  x.device().to_string(), weight.device().to_string());
  TESSERACT_CHECK(x.rank() >= 1,
                  "layer_norm: x must be rank >= 1, got {}", x.shape().to_string());
  TESSERACT_CHECK(weight.rank() == 1,
                  "layer_norm: weight must be rank-1 [D], got {}",
                  weight.shape().to_string());
  const int64_t d = x.shape()[x.rank() - 1];
  TESSERACT_CHECK(weight.shape()[0] == d,
                  "layer_norm: weight size {} != x last dim {}",
                  weight.shape()[0], d);
  const bool use_bias = bias.defined();
  if (use_bias) {
    TESSERACT_CHECK(bias.dtype() == x.dtype(),
                    "layer_norm: bias dtype {} != x dtype {}",
                    dtype_name(bias.dtype()), dtype_name(x.dtype()));
    TESSERACT_CHECK(bias.device() == x.device(),
                    "layer_norm: bias device {} != x device {}",
                    bias.device().to_string(), x.device().to_string());
    TESSERACT_CHECK(bias.rank() == 1 && bias.shape()[0] == d,
                    "layer_norm: bias shape {} != [{}]",
                    bias.shape().to_string(), d);
  }
  TESSERACT_CHECK(eps > 0.0, "layer_norm: eps must be > 0, got {}", eps);

  // Wave 2 (B-022) inference fast path: on CUDA, when no autograd is
  // active, dispatch to the one-block-per-row fused kernel. See the
  // matching comment in `src/ops/cpu/RMSNorm.cpp` for the memory-
  // bandwidth rationale — LayerNorm is even more lopsided than
  // RMSNorm (the composite does a dedicated mean-center pass on top
  // of the variance pass), so the fused kernel's one-pass-two-stats
  // path is a strict win.
  const bool want_fused_cuda =
      x.device().is_cuda() &&
      x.is_contiguous() && weight.is_contiguous() &&
      (!use_bias || bias.is_contiguous()) &&
      !(is_grad_enabled() &&
        (autograd::any_requires_grad(x, weight) ||
         (use_bias && bias.requires_grad())));
  if (want_fused_cuda) {
    int64_t outer = 1;
    for (int64_t i = 0; i + 1 < static_cast<int64_t>(x.rank()); ++i) {
      outer *= x.shape()[i];
    }
    const int64_t D = x.shape()[x.rank() - 1];
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    Stream s = current_stream(x.device());
    cuda::detail::launch_layer_norm(
        x.dtype(), x.device().index, outer, D,
        x.raw_data(), weight.raw_data(),
        use_bias ? bias.raw_data() : nullptr,
        eps, out.raw_data(), s.native_handle());
    if (use_bias) {
      graph::maybe_record("layer_norm", {&x, &weight, &bias}, {&out},
                          {{"eps", eps}});
    } else {
      graph::maybe_record("layer_norm", {&x, &weight}, {&out},
                          {{"eps", eps}});
    }
    return out;
  }

  // Composite forward over primitives — every op below has CUDA + CPU
  // backends and a backward node, so autograd flows through without
  // custom glue. Matches PyTorch's `F.layer_norm(x, [D], weight, bias, eps)`
  // (biased variance, not Bessel-corrected — same as the fused ATen op).
  //
  //   mu      = mean(x, dim=-1, keepdim=True)          [..., 1]
  //   xc      = x - mu                                 [..., D]
  //   var     = mean(xc * xc, dim=-1, keepdim=True)    [..., 1]
  //   denom   = sqrt(var + eps)                        [..., 1]
  //   yhat    = xc / denom                             [..., D]
  //   y       = yhat * weight                          [D] broadcasts
  //   y      += bias                                   (only when bias defined)
  const int64_t last = static_cast<int64_t>(x.rank()) - 1;
  Tensor mu    = mean(x, last, /*keepdim=*/true);
  Tensor xc    = sub(x, mu);
  Tensor sq    = mul(xc, xc);
  Tensor var   = mean(sq, last, /*keepdim=*/true);
  Tensor eps_t = Tensor::full({}, eps, x.dtype(), x.device());
  Tensor denom = sqrt(add(var, eps_t));
  Tensor yhat  = div(xc, denom);
  Tensor y     = mul(yhat, weight);
  if (use_bias) {
    y = add(y, bias);
  }

  // Graph-mode marker for future pattern-matching fusion.
  if (use_bias) {
    graph::maybe_record("layer_norm", {&x, &weight, &bias}, {&y},
                        {{"eps", eps}});
  } else {
    graph::maybe_record("layer_norm", {&x, &weight}, {&y},
                        {{"eps", eps}});
  }
  return y;
}

}  // namespace tesseract::ops

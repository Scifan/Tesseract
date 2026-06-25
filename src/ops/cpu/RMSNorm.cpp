#include "tesseract/ops/Normalization.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "tesseract/autograd/Function.hpp"
#include "tesseract/autograd/Node.hpp"
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

Tensor rms_norm(const Tensor& x, const Tensor& weight, double eps) {
  TESSERACT_CHECK(x.defined() && weight.defined(),
                  "rms_norm: x and weight must be defined tensors");
  TESSERACT_CHECK(dtype_is_floating(x.dtype()),
                  "rms_norm: floating-point dtype required, got {}",
                  dtype_name(x.dtype()));
  TESSERACT_CHECK(x.dtype() == weight.dtype(),
                  "rms_norm: x/weight dtype mismatch ({} vs {})",
                  dtype_name(x.dtype()), dtype_name(weight.dtype()));
  TESSERACT_CHECK(x.device() == weight.device(),
                  "rms_norm: x/weight device mismatch ({} vs {})",
                  x.device().to_string(), weight.device().to_string());
  TESSERACT_CHECK(x.rank() >= 1,
                  "rms_norm: x must be rank >= 1, got {}", x.shape().to_string());
  TESSERACT_CHECK(weight.rank() == 1,
                  "rms_norm: weight must be rank-1 [D], got {}",
                  weight.shape().to_string());
  const int64_t d = x.shape()[x.rank() - 1];
  TESSERACT_CHECK(weight.shape()[0] == d,
                  "rms_norm: weight size {} != x last dim {}",
                  weight.shape()[0], d);
  TESSERACT_CHECK(eps > 0.0, "rms_norm: eps must be > 0, got {}", eps);

  // Wave 2 (B-022) inference fast path: on CUDA, when no autograd is
  // active, dispatch to the one-block-per-row fused kernel. This
  // collapses the composite's 5-6 per-element passes (mul → mean →
  // add → sqrt → div → mul) into a single 2-pass kernel, which is
  // the full memory-bandwidth win on what is otherwise memory-bound.
  // Training / autograd-active paths continue through the composite
  // below — the primitive nodes already wire mean/add/sqrt/div/mul
  // backward, and we deliberately don't duplicate that plumbing in
  // a custom RMSNormBackward until a concrete training benchmark
  // demands it.
  const bool want_fused_cuda =
      x.device().is_cuda() &&
      x.is_contiguous() && weight.is_contiguous() &&
      !(is_grad_enabled() && autograd::any_requires_grad(x, weight));
  if (want_fused_cuda) {
    int64_t outer = 1;
    for (int64_t i = 0; i + 1 < static_cast<int64_t>(x.rank()); ++i) {
      outer *= x.shape()[i];
    }
    const int64_t D = x.shape()[x.rank() - 1];
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    Stream s = current_stream(x.device());
    cuda::detail::launch_rms_norm(
        x.dtype(), x.device().index, outer, D,
        x.raw_data(), weight.raw_data(), eps,
        out.raw_data(), s.native_handle());
    graph::maybe_record("rms_norm", {&x, &weight}, {&out}, {{"eps", eps}});
    return out;
  }

  // Composite forward — every kernel here is already CUDA-resident, so
  // the whole path runs on-device for both CPU and CUDA tensors and
  // autograd flows through the primitive backward nodes (`MulBackward`,
  // `MeanBackward`, `AddBackward`, `SqrtBackward`, `DivBackward`,
  // broadcast-reduce through `reduce_to_shape`) without custom glue.
  //
  //   sq    = x * x                             [..., D]
  //   ms    = mean(sq, dim=-1, keepdim=True)    [..., 1]
  //   denom = sqrt(ms + eps)                    [..., 1]
  //   yhat  = x / denom                         broadcasts [..., 1] against [..., D]
  //   y     = yhat * weight                     [D] broadcasts against [..., D]
  //
  // The `eps` is staged as a 0-D scalar tensor on the source device so
  // `add` never pulls it back to host memory. Memory note: the whole
  // chain keeps at most one `[..., D]`-sized intermediate alive
  // concurrently (the eager `sq`) — the reduction output is small,
  // `sqrt` is in-place-sized, and `div`/`mul` allocate fresh outputs
  // that immediately replace their inputs as the next live value.
  Tensor sq    = mul(x, x);
  Tensor ms    = mean(sq, static_cast<int64_t>(x.rank()) - 1, /*keepdim=*/true);
  Tensor eps_t = Tensor::full({}, eps, x.dtype(), x.device());
  Tensor denom = sqrt(add(ms, eps_t));
  Tensor yhat  = div(x, denom);
  Tensor y     = mul(yhat, weight);

  // Metadata-only marker so a future MLIR pattern-matcher (M2L+) can
  // collapse the primitive chain into a fused `tesseract.rms_norm`
  // before codegen. Autograd is already plumbed through the inner
  // primitives; the marker doesn't re-run forward.
  graph::maybe_record("rms_norm", {&x, &weight}, {&y},
                      {{"eps", eps}});
  return y;
}

}  // namespace tesseract::ops

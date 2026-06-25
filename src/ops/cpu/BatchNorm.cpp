// Wave 2b (B-020) — composite BatchNorm for rank-2/3 (BN1d) and rank-4 (BN2d)
// inputs. Lives in the `cpu/` tree for filesystem parity with the other
// composite normalizations, but is fully device-agnostic: every primitive
// it reaches for (mean, sub, mul, add, div, sqrt, reshape) already has
// CPU + CUDA backends and a matching autograd node, so both forward and
// backward cross the CUDA boundary for free.
//
// Running-stats writeback goes through `Storage::copy_device_bytes` under
// a `NoGradGuard`, which keeps the buffers outside the autograd graph and
// gives us in-place semantics via shared-impl aliasing with
// `nn::BatchNorm{1d,2d}::register_buffer`.
//
// Math (PyTorch-verbatim):
//   training:
//     mu   = mean(x,          dims=all-but-C)              [1, C, 1, ...]
//     xc   = x - mu
//     var  = mean(xc * xc,    dims=all-but-C)              [1, C, 1, ...]    (biased, 1/N_red)
//     denom = sqrt(var + eps)
//     y    = (xc / denom) * weight + bias
//     running_mean ← (1-momentum) · running_mean + momentum · mu_1D
//     running_var  ← (1-momentum) · running_var  + momentum · var_1D · (N_red / (N_red - 1))
//   eval:
//     y    = (x - running_mean) / sqrt(running_var + eps) * weight + bias

#include "tesseract/ops/Normalization.hpp"

#include <cstdint>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Storage.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::ops {

namespace {

// Reduce `x` along every dimension except `channel_dim` using repeated
// `mean(..., keepdim=true)`. Returns a tensor of the same rank as x
// with size 1 in every reduced dimension and size C on the channel
// axis — so the result broadcasts back onto x without any further
// reshape. Iterating from the last dim downwards keeps dim indices
// stable across successive calls (each keepdim=true mean preserves
// rank).
Tensor mean_over_non_channel(const Tensor& x, int64_t channel_dim) {
  Tensor acc = x;
  for (int64_t d = static_cast<int64_t>(x.rank()) - 1; d >= 0; --d) {
    if (d == channel_dim) continue;
    acc = mean(acc, d, /*keepdim=*/true);
  }
  return acc;
}

// Reshape a broadcast-ready stats tensor `[1, C, 1, ...]` down to the
// flat `[C]` buffer layout. Handles the common case where the input is
// already contiguous (just a view), and falls through to the generic
// `reshape` path otherwise.
Tensor flatten_to_channel(const Tensor& broadcast_stats, int64_t C) {
  return broadcast_stats.reshape({C});
}

// In-place byte copy of `src`'s data into `dst`'s storage. `dst` is
// passed by value on purpose — it's a cheap handle copy whose TensorImpl
// is shared with the caller's buffer, so writing through `raw_data()`
// reaches the buffer every other Tensor handle sees.
void write_running_stat(Tensor dst, const Tensor& src) {
  TESSERACT_CHECK(dst.shape() == src.shape(),
                  "batch_norm: running stat shape {} != computed stat shape {}",
                  dst.shape().to_string(), src.shape().to_string());
  TESSERACT_CHECK(dst.dtype() == src.dtype(),
                  "batch_norm: running stat dtype {} != computed stat dtype {}",
                  dtype_name(dst.dtype()), dtype_name(src.dtype()));
  TESSERACT_CHECK(dst.device() == src.device(),
                  "batch_norm: running stat device {} != computed stat device {}",
                  dst.device().to_string(), src.device().to_string());
  TESSERACT_CHECK(dst.is_contiguous(),
                  "batch_norm: running stat buffer must be contiguous");
  const Tensor src_c = src.is_contiguous() ? src : src.contiguous();
  Storage::copy_device_bytes(dst.raw_data(), dst.device(),
                             src_c.raw_data(), src_c.device(),
                             dst.nbytes());
}

}  // namespace

Tensor batch_norm(const Tensor& x,
                  const Tensor& weight,
                  const Tensor& bias,
                  const Tensor& running_mean,
                  const Tensor& running_var,
                  bool training,
                  double momentum,
                  double eps) {
  // --- Shape + dtype + device guards ---------------------------------------
  TESSERACT_CHECK(x.defined(), "batch_norm: x must be defined");
  TESSERACT_CHECK(dtype_is_floating(x.dtype()),
                  "batch_norm: floating-point dtype required, got {}",
                  dtype_name(x.dtype()));
  TESSERACT_CHECK(x.rank() == 2 || x.rank() == 3 || x.rank() == 4,
                  "batch_norm: supported ranks are 2 (BN1d-2D), 3 (BN1d-3D), "
                  "and 4 (BN2d); got shape {}", x.shape().to_string());
  TESSERACT_CHECK(eps > 0.0, "batch_norm: eps must be > 0, got {}", eps);
  TESSERACT_CHECK(momentum >= 0.0 && momentum <= 1.0,
                  "batch_norm: momentum must be in [0, 1], got {}", momentum);

  const int64_t C = x.shape()[1];

  auto check_param_like_c = [&](const Tensor& t, const char* name) {
    if (!t.defined()) return;
    TESSERACT_CHECK(t.rank() == 1 && t.shape()[0] == C,
                    "batch_norm: {} shape {} != [{}]",
                    name, t.shape().to_string(), C);
    TESSERACT_CHECK(t.dtype() == x.dtype(),
                    "batch_norm: {} dtype {} != x dtype {}",
                    name, dtype_name(t.dtype()), dtype_name(x.dtype()));
    TESSERACT_CHECK(t.device() == x.device(),
                    "batch_norm: {} device {} != x device {}",
                    name, t.device().to_string(), x.device().to_string());
  };
  check_param_like_c(weight, "weight");
  check_param_like_c(bias, "bias");

  TESSERACT_CHECK(running_mean.defined() && running_var.defined(),
                  "batch_norm: running_mean / running_var buffers must be "
                  "defined (pass nn::BatchNorm1d/2d or allocate them "
                  "explicitly; PyTorch's track_running_stats=False is NOT "
                  "supported in Wave 2b)");
  check_param_like_c(running_mean, "running_mean");
  check_param_like_c(running_var,  "running_var");

  // --- Count of elements reduced per channel --------------------------------
  // For a rank-r input `[N, C, d₂, ..., d_{r-1}]`, every non-channel
  // dim is reduced away, giving `N_red = N · d₂ · … · d_{r-1}`. Used
  // for both the biased/unbiased running-var correction below and the
  // single-row smoke guard (can't normalize an N_red == 1 batch — the
  // biased variance collapses to 0 and `sqrt(0 + eps)` is fine, but
  // the unbiased correction `N/(N-1)` divides by zero).
  int64_t n_red = 1;
  for (int64_t d = 0; d < x.rank(); ++d) {
    if (d == 1) continue;
    n_red *= x.shape()[d];
  }
  TESSERACT_CHECK(n_red > 0,
                  "batch_norm: reduction over non-channel dims is empty "
                  "(x.shape()={}) — nothing to normalize",
                  x.shape().to_string());

  // Broadcast shape for the `[C]` buffers (and the computed stats): 1 on
  // every axis except channel, C on channel. E.g. x = [N,C,H,W] →
  // bshape = [1, C, 1, 1]. Lets us use the exact same forward arithmetic
  // for both the eval path (broadcast running stats) and the training
  // path (broadcast freshly-computed batch stats).
  Shape bshape(std::vector<int64_t>(static_cast<size_t>(x.rank()), 1));
  bshape[1] = C;

  Tensor eps_t = Tensor::full({}, eps, x.dtype(), x.device());

  Tensor mu_b;   // [1, C, 1, ...] — mean used in forward
  Tensor var_b;  // [1, C, 1, ...] — variance (biased) used in forward

  if (training) {
    // --- Training path: compute batch stats, normalize, update running ----
    // mean over every non-channel dim, keepdim=true so the broadcast back
    // onto `x` is a one-liner.
    mu_b = mean_over_non_channel(x, /*channel_dim=*/1);
    Tensor xc = sub(x, mu_b);
    Tensor sq = mul(xc, xc);
    var_b = mean_over_non_channel(sq, /*channel_dim=*/1);

    // Running-stats writeback. Wrapped in a NoGradGuard so none of the
    // intermediate ops (`mul`, `add`, etc.) attach grad_fns to the
    // running buffers — they're supposed to live entirely outside the
    // autograd graph. The batch stats themselves (`mu_b`, `var_b`)
    // still carry grad_fns at this point because `is_grad_enabled()`
    // was true during their construction; NoGradGuard only suppresses
    // grad-fn creation for ops invoked from *inside* its scope.
    //
    // Sharing-semantics reminder: `running_mean` / `running_var`
    // are handles into the Module's buffers. The byte-copy at the
    // bottom writes into their storage, which is the same storage
    // every other Tensor handle aliasing those buffers (including
    // the Module's `register_buffer` entry) observes.
    {
      NoGradGuard nogg;
      Tensor mu_1d  = flatten_to_channel(mu_b, C);
      Tensor var_1d = flatten_to_channel(var_b, C);

      // PyTorch semantics: running_var tracks the UNBIASED estimator
      // (Bessel correction n/(n-1)), while the forward normalization
      // uses the biased variance. When N_red == 1 the correction is
      // undefined — match PyTorch's behavior of leaving running_var
      // untouched for that edge case rather than blowing up the train
      // loop on the first single-sample batch.
      const double one_minus_m = 1.0 - momentum;
      Tensor keep_m = Tensor::full({}, one_minus_m, x.dtype(), x.device());
      Tensor new_m  = Tensor::full({}, momentum,   x.dtype(), x.device());

      Tensor running_mean_new =
          add(mul(running_mean, keep_m), mul(mu_1d, new_m));
      write_running_stat(running_mean, running_mean_new);

      if (n_red > 1) {
        const double bessel = static_cast<double>(n_red) /
                              static_cast<double>(n_red - 1);
        Tensor bessel_t = Tensor::full({}, bessel, x.dtype(), x.device());
        Tensor var_unbiased = mul(var_1d, bessel_t);
        Tensor running_var_new =
            add(mul(running_var, keep_m), mul(var_unbiased, new_m));
        write_running_stat(running_var, running_var_new);
      }
    }
  } else {
    // --- Eval path: broadcast the running buffers back out ----------------
    mu_b  = running_mean.reshape(bshape);
    var_b = running_var.reshape(bshape);
  }

  // --- Shared forward: (x - mu) / sqrt(var + eps) * weight + bias -----------
  Tensor xc    = sub(x, mu_b);
  Tensor denom = sqrt(add(var_b, eps_t));
  Tensor yhat  = div(xc, denom);

  Tensor y = yhat;
  if (weight.defined()) {
    // Go through `ops::reshape` rather than `Tensor::reshape` so the view
    // carries a `ReshapeBackward` — without it the weight grad just drops
    // off the graph when autograd walks backwards from `mul`.
    Tensor w_b = reshape(weight, bshape);
    y = mul(y, w_b);
  }
  if (bias.defined()) {
    Tensor b_b = reshape(bias, bshape);
    y = add(y, b_b);
  }

  // Graph-mode marker for the pattern-matcher. We intentionally do NOT
  // list `running_mean` / `running_var` as inputs here — they are
  // stateful buffers mutated as a side effect, not differentiable
  // dependencies of `y`, and folding them in would mislead any
  // downstream fusion pass into thinking the running stats gate the
  // forward output.
  std::vector<const Tensor*> gs_inputs{&x};
  if (weight.defined()) gs_inputs.push_back(&weight);
  if (bias.defined())   gs_inputs.push_back(&bias);
  graph::maybe_record("batch_norm", gs_inputs, {&y},
                      {{"eps", eps},
                       {"momentum", momentum},
                       {"training", training ? 1.0 : 0.0}});
  return y;
}

}  // namespace tesseract::ops

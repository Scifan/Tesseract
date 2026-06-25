#include "tesseract/nn/Mamba.hpp"

#include <cmath>
#include <vector>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/CausalConv1d.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Indexing.hpp"
#include "tesseract/ops/SelectiveScan.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

namespace {

Tensor silu(const Tensor& x) { return ops::mul(x, ops::sigmoid(x)); }

// softplus(x) = log(1 + exp(x)). Adequate for the moderate dt-projection
// magnitudes of the minimal models we run; a numerically hardened
// max(x,0)+log1p(exp(-|x|)) form is a follow-up if large-dt overflow appears.
Tensor softplus(const Tensor& x) {
  Tensor ones = Tensor::ones(x.shape(), x.dtype(), x.device());
  return ops::log(ops::add(ops::exp(x), ones));
}

}  // namespace

Mamba::Mamba(int64_t d_model, int64_t d_state, int64_t d_conv, int64_t expand,
             int64_t dt_rank, DType dtype)
    : d_model_(d_model),
      d_inner_(expand * d_model),
      d_state_(d_state),
      d_conv_(d_conv),
      dt_rank_(dt_rank > 0 ? dt_rank : (d_model + 15) / 16),
      dtype_(dtype) {
  TESSERACT_CHECK(d_model > 0 && d_state > 0 && d_conv >= 1 && expand >= 1,
                  "Mamba: d_model/d_state/expand must be > 0 and d_conv >= 1");
  TESSERACT_CHECK(dtype == DType::Float32 || dtype == DType::Float64,
                  "Mamba: only Float32/Float64 supported at A2 (got {})",
                  dtype_name(dtype));

  in_proj_  = std::make_shared<Linear>(d_model_, 2 * d_inner_, false, dtype);
  x_proj_   = std::make_shared<Linear>(d_inner_, dt_rank_ + 2 * d_state_, false,
                                       dtype);
  dt_proj_  = std::make_shared<Linear>(dt_rank_, d_inner_, true, dtype);
  out_proj_ = std::make_shared<Linear>(d_inner_, d_model_, false, dtype);
  register_module("in_proj",  in_proj_);
  register_module("x_proj",   x_proj_);
  register_module("dt_proj",  dt_proj_);
  register_module("out_proj", out_proj_);

  // Conv as an averaging filter (deterministic, stable); bias zero.
  conv_weight_ = Tensor::full(Shape({d_inner_, d_conv_}),
                              1.0 / static_cast<double>(d_conv_), dtype);
  conv_bias_   = Tensor::zeros(Shape({d_inner_}), dtype);
  // A = -exp(A_log); init A_log[d,n] = log(n+1) so A is the canonical
  // descending-negative spectrum (matches the Mamba S4D-real init).
  a_log_ = Tensor::empty(Shape({d_inner_, d_state_}), dtype);
  dispatch_float(dtype, [&]<typename T>() {
    T* p = a_log_.data_ptr<T>();
    for (int64_t d = 0; d < d_inner_; ++d)
      for (int64_t n = 0; n < d_state_; ++n)
        p[d * d_state_ + n] = static_cast<T>(std::log(static_cast<double>(n + 1)));
  });
  d_skip_ = Tensor::ones(Shape({d_inner_}), dtype);

  for (Tensor* t : {&conv_weight_, &conv_bias_, &a_log_, &d_skip_})
    t->set_requires_grad(true);
  register_parameter("conv_weight", conv_weight_);
  register_parameter("conv_bias",   conv_bias_);
  register_parameter("A_log",       a_log_);
  register_parameter("D",           d_skip_);
}

Tensor Mamba::conv1d_forward(const Tensor& x) {
  // x: [B, L, d_inner] → causal depthwise conv1d → [B, L, d_inner].
  const int64_t B = x.shape()[0];
  const int64_t L = x.shape()[1];
  const int64_t K = d_conv_;

  // Fused CUDA path: one kernel does the K-tap causal dot product per output
  // element, replacing the host-orchestrated pad-cat + K elementwise passes.
  // FP32 only; Mamba forward is inference (NoGradGuard) so no autograd edge is
  // needed here.
  if (x.device().is_cuda() && x.dtype() == DType::Float32 &&
      conv_weight_.dtype() == DType::Float32) {
    Tensor xc = x.contiguous();
    Tensor wc = conv_weight_.contiguous();
    Tensor bc = conv_bias_.contiguous();
    Tensor out = Tensor::empty(Shape({B, L, d_inner_}), DType::Float32,
                               x.device());
    Stream s = current_stream(x.device());
    cuda::detail::launch_causal_conv1d(
        x.device().index, B, L, d_inner_, K, xc.data_ptr<float>(),
        wc.data_ptr<float>(), bc.data_ptr<float>(), out.data_ptr<float>(),
        s.native_handle());
    return out;
  }

  Tensor xpad = x;
  if (K > 1) {
    Tensor pad = Tensor::zeros(Shape({B, K - 1, d_inner_}), x.dtype(),
                               x.device());
    xpad = ops::cat({pad, x}, 1);  // [B, L+K-1, d_inner]
  }
  Tensor out;
  for (int64_t k = 0; k < K; ++k) {
    Tensor seg = xpad.narrow(1, k, L);  // [B, L, d_inner]
    Tensor wk = conv_weight_.narrow(1, k, 1).contiguous().reshape(
        Shape({1, 1, d_inner_}));
    Tensor term = ops::mul(seg, wk);  // broadcast over B, L
    out = out.defined() ? ops::add(out, term) : term;
  }
  Tensor bias_r = conv_bias_.reshape(Shape({1, 1, d_inner_}));
  return ops::add(out, bias_r);
}

Tensor Mamba::forward(const Tensor& x) {
  TESSERACT_CHECK(x.rank() == 3 && x.shape()[2] == d_model_,
                  "Mamba::forward: expected [B, L, {}], got {}", d_model_,
                  x.shape().to_string());
  NoGradGuard nogg;

  Tensor xz = in_proj_->forward(x);  // [B, L, 2*d_inner]
  auto parts = ops::split_with_sizes(xz, {d_inner_, d_inner_}, 2);
  Tensor x_in = parts[0];
  Tensor z    = parts[1];

  Tensor x_conv = silu(conv1d_forward(x_in));  // [B, L, d_inner]
  Tensor dbl = x_proj_->forward(x_conv);       // [B, L, dt_rank+2N]
  auto dparts = ops::split_with_sizes(dbl, {dt_rank_, d_state_, d_state_}, 2);
  Tensor delta = softplus(dt_proj_->forward(dparts[0]));  // [B, L, d_inner]
  Tensor A = ops::neg(ops::exp(a_log_));                  // [d_inner, N]

  auto res = ops::selective_scan(x_conv, delta, A, dparts[1], dparts[2],
                                 d_skip_);
  Tensor y = ops::mul(res.y, silu(z));   // [B, L, d_inner]
  return out_proj_->forward(y);          // [B, L, d_model]
}

Tensor Mamba::forward_step(const Tensor& x, SSMStateCache& cache) {
  TESSERACT_CHECK(x.rank() == 3 && x.shape()[1] == 1 && x.shape()[2] == d_model_,
                  "Mamba::forward_step: expected [B, 1, {}], got {}", d_model_,
                  x.shape().to_string());
  NoGradGuard nogg;
  const int64_t K = d_conv_;

  Tensor xz = in_proj_->forward(x);  // [B, 1, 2*d_inner]
  auto parts = ops::split_with_sizes(xz, {d_inner_, d_inner_}, 2);
  Tensor x_in = parts[0];  // [B, 1, d_inner]
  Tensor z    = parts[1];

  Tensor x_conv;
  if (K > 1) {
    // window = [conv_state (K-1) | x_in (1)] → [B, K, d_inner]; the conv output
    // of its last position is the causal conv for the new token.
    Tensor window = ops::cat({cache.conv_state(), x_in}, 1);
    Tensor conv_full = conv1d_forward(window);          // [B, K, d_inner]
    x_conv = conv_full.narrow(1, K - 1, 1).contiguous();  // [B, 1, d_inner]
    cache.set_conv_state(window.narrow(1, 1, K - 1).contiguous());
  } else {
    x_conv = conv1d_forward(x_in);
  }
  x_conv = silu(x_conv);

  Tensor dbl = x_proj_->forward(x_conv);  // [B, 1, dt_rank+2N]
  auto dparts = ops::split_with_sizes(dbl, {dt_rank_, d_state_, d_state_}, 2);
  Tensor delta = softplus(dt_proj_->forward(dparts[0]));  // [B, 1, d_inner]
  Tensor A = ops::neg(ops::exp(a_log_));

  auto res = ops::selective_scan(x_conv, delta, A, dparts[1], dparts[2],
                                 d_skip_, cache.ssm_state());
  cache.set_ssm_state(res.state);
  Tensor y = ops::mul(res.y, silu(z));
  return out_proj_->forward(y);
}

SSMStateCache Mamba::make_state_cache(int64_t batch) const {
  return SSMStateCache(batch, d_inner_, d_state_, d_conv_, dtype_,
                       conv_weight_.device());
}

}  // namespace tesseract::nn

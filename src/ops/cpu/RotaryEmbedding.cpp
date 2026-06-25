#include "tesseract/ops/RotaryEmbedding.hpp"

#include <memory>
#include <type_traits>

#include "tesseract/autograd/Function.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/detail/RotaryEmbedding.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::ops {

namespace {

// Shared forward launcher — takes a *prepared* contiguous `x` plus
// `cos` / `sin` of shape `[S, D]` and writes a contiguous output of
// the same shape as `x`. The autograd backward also lands here with
// `sin` negated (rotation by -θ is the transpose of R(θ), which is
// what a rotation's Jacobian asks for). Sharing the launcher keeps
// the fp16/bf16 promotion policy in one place.
Tensor rotary_forward_contig(const Tensor& x,
                             const Tensor& cos,
                             const Tensor& sin) {
  const int64_t r = x.rank();
  const int64_t S = x.shape()[r - 2];
  const int64_t D = x.shape()[r - 1];
  int64_t outer = 1;
  for (int64_t i = 0; i + 2 < r; ++i) outer *= x.shape()[i];

  Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());

  if (x.device().is_cuda()) {
    Stream s = current_stream(x.device());
    cuda::detail::launch_rotary_embedding(
        x.dtype(), x.device().index,
        outer, S, D,
        x.raw_data(), cos.raw_data(), sin.raw_data(),
        out.raw_data(),
        s.native_handle());
    return out;
  }

  // CPU reference. Inputs are already contiguous (the public op
  // makes sure of that before we land here), so we compute flat
  // offsets inline — RoPE is per-pair with zero cross-slot state,
  // so the tight loop is the cleanest expression and avoids the
  // IndexIter overhead.
  //
  // FP32 promotion matches the CUDA kernel: fp16 / bf16 cast to
  // `float` at load, compute in float, narrow on store. For
  // float/double the `Acc` alias collapses to the storage type and
  // we get bit-for-bit native math.
  dispatch_float_with_half(x.dtype(), [&]<typename T>() {
    using Acc = std::conditional_t<std::is_floating_point_v<T>, T, float>;
    const T* px = x.data_ptr<T>();
    const T* pc = cos.data_ptr<T>();
    const T* ps = sin.data_ptr<T>();
    T* po = out.data_ptr<T>();
    const int64_t half_D = D / 2;
    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t p = 0; p < S; ++p) {
        const int64_t x_base = (o * S + p) * D;
        const int64_t t_base = p * D;
        for (int64_t j = 0; j < half_D; ++j) {
          const Acc a  = static_cast<Acc>(px[x_base + 2 * j]);
          const Acc b  = static_cast<Acc>(px[x_base + 2 * j + 1]);
          const Acc c0 = static_cast<Acc>(pc[t_base + 2 * j]);
          const Acc s0 = static_cast<Acc>(ps[t_base + 2 * j]);
          const Acc c1 = static_cast<Acc>(pc[t_base + 2 * j + 1]);
          const Acc s1 = static_cast<Acc>(ps[t_base + 2 * j + 1]);
          po[x_base + 2 * j]     = static_cast<T>(a * c0 - b * s0);
          po[x_base + 2 * j + 1] = static_cast<T>(a * s1 + b * c1);
        }
      }
    }
  });
  return out;
}

// Autograd backward. The forward's Jacobian is a 2×2 rotation per
// pair, which is orthogonal: `J^T == R(-θ)`. So `dL/dx = R(-θ) ·
// dL/dout` — the same kernel invoked with `sin → -sin`. No grad is
// routed back to `cos` / `sin` because those are a deterministic
// function of positions (cached in the `nn::RotaryEmbedding` module
// as non-parameter buffers). `next_edges[0]` is the only wired edge.
struct RotaryBackward : Node {
  Tensor cos_saved;
  Tensor sin_saved;
  std::string_view name() const override { return "RotaryBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    // neg_sin = sin * (-1) — a single elementwise launch (CUDA or
    // CPU), no allocation beyond the output tensor. We intentionally
    // don't cache neg_sin on the node because a given
    // `nn::RotaryEmbedding` instance is shared across many forwards
    // and we don't want to hold a tensor alive in every backward
    // node it produced.
    Tensor minus_one = Tensor::full({}, -1.0, sin_saved.dtype(),
                                    sin_saved.device());
    Tensor neg_sin = mul(sin_saved, minus_one);
    Tensor gc = g.is_contiguous() ? g : g.contiguous();
    outs[0] = rotary_forward_contig(gc, cos_saved, neg_sin);
    return outs;
  }
};

}  // namespace

Tensor rotary_embedding(const Tensor& x, const Tensor& cos, const Tensor& sin) {
  TESSERACT_CHECK(x.defined() && cos.defined() && sin.defined(),
                  "rotary_embedding: x, cos, sin must all be defined");
  TESSERACT_CHECK(x.rank() >= 2,
                  "rotary_embedding: x must be rank >= 2 [..., S, D], got {}",
                  x.shape().to_string());
  const int64_t r = x.rank();
  const int64_t S = x.shape()[r - 2];
  const int64_t D = x.shape()[r - 1];
  TESSERACT_CHECK(D % 2 == 0,
                  "rotary_embedding: last dim D must be even, got {}", D);
  TESSERACT_CHECK(cos.rank() == 2 && sin.rank() == 2,
                  "rotary_embedding: cos/sin must be rank-2 [S_table, D], got "
                  "cos={}, sin={}",
                  cos.shape().to_string(), sin.shape().to_string());
  TESSERACT_CHECK(cos.shape()[0] >= S && cos.shape()[1] == D,
                  "rotary_embedding: cos shape {} incompatible with "
                  "[S>={}, D={}]",
                  cos.shape().to_string(), S, D);
  TESSERACT_CHECK(sin.shape()[0] >= S && sin.shape()[1] == D,
                  "rotary_embedding: sin shape {} incompatible with "
                  "[S>={}, D={}]",
                  sin.shape().to_string(), S, D);
  TESSERACT_CHECK(x.dtype() == cos.dtype() && x.dtype() == sin.dtype(),
                  "rotary_embedding: dtype mismatch (x={}, cos={}, sin={})",
                  dtype_name(x.dtype()), dtype_name(cos.dtype()),
                  dtype_name(sin.dtype()));
  TESSERACT_CHECK(dtype_is_floating(x.dtype()),
                  "rotary_embedding: floating-point dtype required, got {}",
                  dtype_name(x.dtype()));
  TESSERACT_CHECK(x.device() == cos.device() && x.device() == sin.device(),
                  "rotary_embedding: device mismatch (x={}, cos={}, sin={})",
                  x.device().to_string(), cos.device().to_string(),
                  sin.device().to_string());

  // The kernel expects contiguous inputs. Callers (e.g.
  // `nn::MultiHeadAttention` after split-heads + permute) often hand
  // us non-contiguous tensors; the `.contiguous()` guards here
  // normalize the layout up-front so the kernel path stays single.
  // Autograd sees these as no-ops on already-contig inputs and as a
  // standard `ContiguousBackward` on strided ones.
  Tensor xc = x.is_contiguous() ? x : x.contiguous();
  Tensor cc = cos.is_contiguous() ? cos : cos.contiguous();
  Tensor sc = sin.is_contiguous() ? sin : sin.contiguous();

  Tensor out = rotary_forward_contig(xc, cc, sc);

  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<RotaryBackward>();
    n->cos_saved = cc;
    n->sin_saved = sc;
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("rotary_embedding", {&x, &cos, &sin}, {&out}, {});
  return out;
}

}  // namespace tesseract::ops

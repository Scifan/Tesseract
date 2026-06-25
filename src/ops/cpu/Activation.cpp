#include "tesseract/ops/Activation.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/autograd/Function.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/Elementwise.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/utils/Logging.hpp"

#include "IndexIter.hpp"

namespace tesseract::ops {

namespace {

template <typename Fn>
Tensor elementwise_unary_float(const Tensor& x, Fn&& fn, const char* name,
                               cuda::detail::UnaryKind cuda_kind) {
  TESSERACT_CHECK(x.defined(), "{}: operand is undefined", name);
  TESSERACT_CHECK(dtype_is_floating(x.dtype()),
                  "{}: floating-point dtype required, got {}", name, dtype_name(x.dtype()));

  Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());

  // CUDA dispatch for the M2E / B-015 unary suite. Float16 / BFloat16
  // reach both the CUDA launcher (since B-015 widened its dtype
  // switch) and the CPU loop below (via `dispatch_float_with_half`).
  // The CPU math uses `std::exp<float>` etc. through Half's implicit
  // `operator float()`, widening transparently, matching the
  // FP32-promoted CUDA semantics.
  if (x.device().is_cuda()) {
    Stream s = current_stream(x.device());
    cuda::detail::launch_unary_elementwise(
        cuda_kind, x.dtype(), x.device().index,
        static_cast<int>(x.shape().rank()),
        x.shape().data(), x.strides().data(),
        out.raw_data(), x.raw_data(),
        s.native_handle());
    return out;
  }

  const Shape& xs = x.strides();
  dispatch_float_with_half(x.dtype(), [&]<typename T>() {
    const T* px = x.data_ptr<T>();
    T* po = out.data_ptr<T>();
    detail::for_each_index(x.shape(), [&](int64_t flat, const detail::IndexArray& idx) {
      po[flat] = fn.template operator()<T>(px[detail::offset_of(idx, xs)]);
    });
  });
  return out;
}

struct ReluFn    { template <typename T> T operator()(T v) const { return v > T{0} ? v : T{0}; } };
// The transcendental functors widen to `float` on half types before
// arithmetic. `T{1} + std::exp(-v)` would otherwise be ambiguous for
// `T = Half` / `BFloat16`: both `operator+(Half,Half)` (via the `float
// → Half` converting ctor) and `operator+(float,float)` (via the
// `Half → float` conversion) are viable. The if-constexpr branch keeps
// the float/double code path byte-identical to the pre-B-015 version.
struct SigmoidFn {
  template <typename T> T operator()(T v) const {
    if constexpr (std::is_floating_point_v<T>) {
      return static_cast<T>(T{1} / (T{1} + std::exp(-v)));
    } else {
      const float fv = static_cast<float>(v);
      return static_cast<T>(1.0f / (1.0f + std::exp(-fv)));
    }
  }
};
struct TanhFn    { template <typename T> T operator()(T v) const { return static_cast<T>(std::tanh(v)); } };
struct ExpFn     { template <typename T> T operator()(T v) const { return static_cast<T>(std::exp(v)); } };
struct LogFn     { template <typename T> T operator()(T v) const { return static_cast<T>(std::log(v)); } };
struct SqrtFn    { template <typename T> T operator()(T v) const { return static_cast<T>(std::sqrt(v)); } };

Tensor relu_forward(const Tensor& x)    { return elementwise_unary_float(x, ReluFn{},    "relu",    cuda::detail::UnaryKind::Relu); }
Tensor sigmoid_forward(const Tensor& x) { return elementwise_unary_float(x, SigmoidFn{}, "sigmoid", cuda::detail::UnaryKind::Sigmoid); }
Tensor tanh_forward(const Tensor& x)    { return elementwise_unary_float(x, TanhFn{},    "tanh",    cuda::detail::UnaryKind::Tanh); }
Tensor exp_forward(const Tensor& x)     { return elementwise_unary_float(x, ExpFn{},     "exp",     cuda::detail::UnaryKind::Exp); }
Tensor log_forward(const Tensor& x)     { return elementwise_unary_float(x, LogFn{},     "log",     cuda::detail::UnaryKind::Log); }
Tensor sqrt_forward(const Tensor& x)    { return elementwise_unary_float(x, SqrtFn{},    "sqrt",    cuda::detail::UnaryKind::Sqrt); }

// ---------- Backward nodes ----------

// grad_in = grad_out * (x > 0). Save input.
//
// The positive-indicator mask is materialized on the same device as
// the saved input so the trailing `mul(g, mask)` stays on-device —
// CUDA parameters can't tolerate the CPU fallback below dereferencing
// their storage as a host pointer. The CUDA path dispatches into
// `launch_unary_elementwise` with `UnaryKind::Step`; the CPU path
// keeps the original row-by-row loop.
struct ReluBackward : Node {
  Tensor x_saved;
  std::string_view name() const override { return "ReluBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    Tensor mask = Tensor::empty(x_saved.shape(), x_saved.dtype(), x_saved.device());
    if (x_saved.device().is_cuda()) {
      Stream s = current_stream(x_saved.device());
      cuda::detail::launch_unary_elementwise(
          cuda::detail::UnaryKind::Step, x_saved.dtype(),
          x_saved.device().index,
          static_cast<int>(x_saved.shape().rank()),
          x_saved.shape().data(), x_saved.strides().data(),
          mask.raw_data(), x_saved.raw_data(),
          s.native_handle());
    } else {
      const Shape& xs = x_saved.strides();
      dispatch_float_with_half(x_saved.dtype(), [&]<typename T>() {
        const T* px = x_saved.data_ptr<T>();
        T* pm = mask.data_ptr<T>();
        detail::for_each_index(x_saved.shape(), [&](int64_t flat, const detail::IndexArray& idx) {
          pm[flat] = px[detail::offset_of(idx, xs)] > T{0} ? T{1} : T{0};
        });
      });
    }
    outs[0] = mul(g, mask);
    return outs;
  }
};

// grad_in = grad_out * y * (1 - y). Save output y.
struct SigmoidBackward : Node {
  Tensor y_saved;
  std::string_view name() const override { return "SigmoidBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    Tensor one = Tensor::ones(y_saved.shape(), y_saved.dtype(), y_saved.device());
    Tensor one_minus_y = sub(one, y_saved);
    Tensor gy = mul(g, y_saved);
    outs[0] = mul(gy, one_minus_y);
    return outs;
  }
};

// grad_in = grad_out * (1 - y^2). Save output y.
struct TanhBackward : Node {
  Tensor y_saved;
  std::string_view name() const override { return "TanhBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    Tensor y2 = mul(y_saved, y_saved);
    Tensor one = Tensor::ones(y_saved.shape(), y_saved.dtype(), y_saved.device());
    Tensor d = sub(one, y2);
    outs[0] = mul(g, d);
    return outs;
  }
};

// grad_in = grad_out * y. Save output y = exp(x).
struct ExpBackward : Node {
  Tensor y_saved;
  std::string_view name() const override { return "ExpBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    outs[0] = mul(g, y_saved);
    return outs;
  }
};

// grad_in = grad_out / x. Save input x.
struct LogBackward : Node {
  Tensor x_saved;
  std::string_view name() const override { return "LogBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    outs[0] = div(g, x_saved);
    return outs;
  }
};

// grad_in = grad_out * 0.5 / sqrt(x) = 0.5 * grad_out / y. Saving the
// output `y = sqrt(x)` rather than the input costs the same memory
// (both are the same shape/dtype) but avoids an extra sqrt on the
// backward pass. The `0.5` multiplier is staged as a 0-D scalar
// tensor on the saved device so `mul` / `div` broadcast uniformly
// across CPU and CUDA without the node reaching into any platform
// API directly.
struct SqrtBackward : Node {
  Tensor y_saved;
  std::string_view name() const override { return "SqrtBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    Tensor half = Tensor::full({}, 0.5, y_saved.dtype(), y_saved.device());
    outs[0] = div(mul(g, half), y_saved);
    return outs;
  }
};

}  // namespace

Tensor relu(const Tensor& x) {
  Tensor out = relu_forward(x);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<ReluBackward>();
    n->x_saved = x;
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("relu", {&x}, {&out});
  return out;
}

Tensor sigmoid(const Tensor& x) {
  Tensor out = sigmoid_forward(x);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<SigmoidBackward>();
    n->y_saved = out;
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("sigmoid", {&x}, {&out});
  return out;
}

Tensor tanh(const Tensor& x) {
  Tensor out = tanh_forward(x);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<TanhBackward>();
    n->y_saved = out;
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("tanh", {&x}, {&out});
  return out;
}

Tensor exp(const Tensor& x) {
  Tensor out = exp_forward(x);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<ExpBackward>();
    n->y_saved = out;
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("exp", {&x}, {&out});
  return out;
}

Tensor log(const Tensor& x) {
  Tensor out = log_forward(x);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<LogBackward>();
    n->x_saved = x;
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("log", {&x}, {&out});
  return out;
}

Tensor sqrt(const Tensor& x) {
  Tensor out = sqrt_forward(x);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<SqrtBackward>();
    n->y_saved = out;
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("sqrt", {&x}, {&out});
  return out;
}

// Fused relu backward (dx = dout * (x > 0)). Used both by the eager
// tape's `ReluBackward::apply` and by direct callers constructing a
// graph. No autograd plumbing — the forward `relu` op owns that. When a
// GraphScope is active we record a single `relu_backward` node so JIT
// passes see the op atomically instead of a mul + mask pair.
Tensor relu_backward(const Tensor& x, const Tensor& grad_out) {
  TESSERACT_CHECK(x.defined() && grad_out.defined(),
                  "relu_backward: operand is undefined");
  TESSERACT_CHECK(dtype_is_floating(x.dtype()),
                  "relu_backward: floating-point dtype required, got {}",
                  dtype_name(x.dtype()));
  TESSERACT_CHECK(x.dtype() == grad_out.dtype(),
                  "relu_backward: dtype mismatch ({} vs {})",
                  dtype_name(x.dtype()), dtype_name(grad_out.dtype()));
  TESSERACT_CHECK(x.shape() == grad_out.shape(),
                  "relu_backward: shape mismatch ({} vs {})",
                  x.shape().to_string(), grad_out.shape().to_string());
  NoGradGuard nogg;
  Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
  // Fused single-pass kernel (dx = dout * (x > 0)). Previously this was
  // implemented as `mul(grad_out, mask)`, which leaked an intermediate
  // `mask` tensor into graph-mode capture; fusing keeps the op atomic
  // for the recorder and is cheaper in eager mode too.
  const Shape& xs = x.strides();
  const Shape& gs = grad_out.strides();
  dispatch_float_with_half(x.dtype(), [&]<typename T>() {
    const T* px = x.data_ptr<T>();
    const T* pg = grad_out.data_ptr<T>();
    T* po = out.data_ptr<T>();
    detail::for_each_index(x.shape(), [&](int64_t flat, const detail::IndexArray& idx) {
      const T v = px[detail::offset_of(idx, xs)];
      po[flat] = v > T{0} ? pg[detail::offset_of(idx, gs)] : T{0};
    });
  });
  graph::maybe_record("relu_backward", {&x, &grad_out}, {&out});
  return out;
}

}  // namespace tesseract::ops

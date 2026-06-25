#include "tesseract/ops/Softmax.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/autograd/Function.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/Softmax.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/utils/Logging.hpp"

#include "IndexIter.hpp"

namespace tesseract::ops {

namespace {

int64_t normalize_dim(int64_t dim, int64_t rank) {
  if (dim < 0) dim += rank;
  TESSERACT_CHECK(dim >= 0 && dim < rank, "softmax: dim {} out of range for rank {}", dim, rank);
  return dim;
}

Tensor softmax_impl_forward(const Tensor& x, int64_t dim, bool take_log) {
  TESSERACT_CHECK(x.defined(), "softmax: operand undefined");
  TESSERACT_CHECK(dtype_is_floating(x.dtype()),
                  "softmax: floating-point dtype required, got {}", dtype_name(x.dtype()));

  const int64_t r = x.rank();
  TESSERACT_CHECK(r > 0, "softmax: input must have rank >= 1");
  dim = normalize_dim(dim, r);
  const int64_t D = x.shape()[dim];

  Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());

  // CUDA path (M2F). The kernel consumes the tensor's raw sizes /
  // strides, handles the max + sum + normalize passes, and writes
  // `out` as row-major contiguous (which matches what `Tensor::empty`
  // just allocated). `take_log` picks softmax vs log_softmax.
  if (x.device().is_cuda()) {
    Stream s = current_stream(x.device());
    cuda::detail::launch_softmax(
        take_log, x.dtype(), x.device().index,
        static_cast<int>(x.shape().rank()),
        static_cast<int>(dim),
        x.shape().data(), x.strides().data(),
        x.raw_data(), out.raw_data(),
        s.native_handle());
    return out;
  }

  const Shape& xs = x.strides();
  const Shape out_strides_contig = x.shape().contiguous_strides();
  const int64_t xs_dim = xs[dim];
  const int64_t os_dim = out_strides_contig[dim];

  Shape iter_shape;
  for (int64_t i = 0; i < r; ++i) {
    if (i != dim) iter_shape.push_back(x.shape()[i]);
  }

  // B-015: accumulate softmax in `Acc` — equal to `T` for float/double
  // (preserves the prior bit-for-bit semantics on FP32/FP64) and
  // upgraded to `float` for `Half` / `BFloat16` so that
  // `std::numeric_limits<Acc>::lowest()` is meaningful and the
  // intermediate `exp`/`log` math matches the CUDA FP32-promoted
  // kernel in `src/cuda/Softmax.cu`.
  dispatch_float_with_half(x.dtype(), [&]<typename T>() {
    using Acc = std::conditional_t<std::is_floating_point_v<T>, T, float>;
    const T* px = x.data_ptr<T>();
    T* po = out.data_ptr<T>();
    detail::for_each_index(iter_shape, [&](int64_t, const detail::IndexArray& iter_idx) {
      detail::IndexArray full{};
      std::size_t k = 0;
      for (int64_t i = 0; i < r; ++i) {
        if (i == dim) full[i] = 0;
        else full[i] = iter_idx[k++];
      }
      const int64_t x_base = detail::offset_of(full, xs);
      const int64_t o_base = detail::offset_of(full, out_strides_contig);

      Acc m = std::numeric_limits<Acc>::lowest();
      for (int64_t d = 0; d < D; ++d) {
        const Acc v = static_cast<Acc>(px[x_base + d * xs_dim]);
        if (v > m) m = v;
      }
      Acc s = Acc{0};
      for (int64_t d = 0; d < D; ++d) {
        const Acc v = static_cast<Acc>(px[x_base + d * xs_dim]);
        s = s + std::exp(v - m);
      }
      if (take_log) {
        const Acc log_z = m + std::log(s);
        for (int64_t d = 0; d < D; ++d) {
          const Acc v = static_cast<Acc>(px[x_base + d * xs_dim]);
          po[o_base + d * os_dim] = static_cast<T>(v - log_z);
        }
      } else {
        const Acc inv_s = Acc{1} / s;
        for (int64_t d = 0; d < D; ++d) {
          const Acc v = static_cast<Acc>(px[x_base + d * xs_dim]);
          po[o_base + d * os_dim] = static_cast<T>(std::exp(v - m) * inv_s);
        }
      }
    });
  });
  return out;
}

struct SoftmaxBackward : Node {
  Tensor y_saved;
  int64_t dim{0};
  std::string_view name() const override { return "SoftmaxBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    // dx = (g - sum(g * y, dim, keepdim=true)) * y
    Tensor gy = mul(g, y_saved);
    Tensor s = sum(gy, dim, /*keepdim=*/true);
    Tensor diff = sub(g, s);
    outs[0] = mul(diff, y_saved);
    return outs;
  }
};

struct LogSoftmaxBackward : Node {
  Tensor y_saved;  // log_softmax output; softmax = exp(y_saved)
  int64_t dim{0};
  std::string_view name() const override { return "LogSoftmaxBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    // dx = g - softmax * sum(g, dim, keepdim=true)
    Tensor sm = exp(y_saved);
    Tensor gs = sum(g, dim, /*keepdim=*/true);
    Tensor scaled = mul(sm, gs);
    outs[0] = sub(g, scaled);
    return outs;
  }
};

}  // namespace

Tensor softmax(const Tensor& x, int64_t dim) {
  Tensor out = softmax_impl_forward(x, dim, /*take_log=*/false);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<SoftmaxBackward>();
    n->y_saved = out;
    n->dim = normalize_dim(dim, x.rank());
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("softmax", {&x}, {&out}, {{"dim", static_cast<int64_t>(dim)}});
  return out;
}

Tensor log_softmax(const Tensor& x, int64_t dim) {
  Tensor out = softmax_impl_forward(x, dim, /*take_log=*/true);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<LogSoftmaxBackward>();
    n->y_saved = out;
    n->dim = normalize_dim(dim, x.rank());
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("log_softmax", {&x}, {&out}, {{"dim", static_cast<int64_t>(dim)}});
  return out;
}

}  // namespace tesseract::ops

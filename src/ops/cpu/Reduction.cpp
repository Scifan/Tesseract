#include "tesseract/ops/Reduction.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/autograd/Function.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/Reduction.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/utils/Logging.hpp"

#include "IndexIter.hpp"

namespace tesseract::ops {

namespace {

int64_t normalize_dim(int64_t dim, int64_t rank) {
  if (dim < 0) dim += rank;
  TESSERACT_CHECK(dim >= 0 && dim < rank, "dim {} out of range for rank {}", dim, rank);
  return dim;
}

Shape reduced_shape(const Shape& in, int64_t dim, bool keepdim) {
  Shape out;
  const int64_t r = static_cast<int64_t>(in.rank());
  for (int64_t i = 0; i < r; ++i) {
    if (i == dim) {
      if (keepdim) out.push_back(1);
    } else {
      out.push_back(in[i]);
    }
  }
  return out;
}

struct SumStrategy {
  template <typename T> struct State { T value{T{0}}; };
  template <typename T> static State<T> init() { return {}; }
  template <typename T> static void combine(State<T>& s, T v) { s.value = static_cast<T>(s.value + v); }
  template <typename T> static T finalize(State<T>& s, int64_t) { return s.value; }
};

struct MeanStrategy {
  template <typename T> struct State { T value{T{0}}; };
  template <typename T> static State<T> init() { return {}; }
  template <typename T> static void combine(State<T>& s, T v) { s.value = static_cast<T>(s.value + v); }
  template <typename T> static T finalize(State<T>& s, int64_t n) {
    return static_cast<T>(s.value / static_cast<T>(n));
  }
};

struct MaxStrategy {
  template <typename T> struct State { T value{std::numeric_limits<T>::lowest()}; };
  template <typename T> static State<T> init() { return {}; }
  template <typename T> static void combine(State<T>& s, T v) { if (v > s.value) s.value = v; }
  template <typename T> static T finalize(State<T>& s, int64_t) { return s.value; }
};

// Map the CPU policy tag → the CUDA bridge's `ReduceKind`. The CUDA
// TU uses its own template-parametric policies (see
// `src/cuda/Reduction.cu`), but at the bridge boundary a plain enum
// keeps the header free of template machinery.
template <typename S>
cuda::detail::ReduceKind reduce_kind() {
  if constexpr (std::is_same_v<S, SumStrategy>)  return cuda::detail::ReduceKind::Sum;
  if constexpr (std::is_same_v<S, MeanStrategy>) return cuda::detail::ReduceKind::Mean;
  if constexpr (std::is_same_v<S, MaxStrategy>)  return cuda::detail::ReduceKind::Max;
}

template <typename S>
Tensor reduce_all_forward(const Tensor& x) {
  TESSERACT_CHECK(x.numel() > 0, "reduce: cannot reduce an empty tensor");
  Tensor out = Tensor::empty({}, x.dtype(), x.device());

  // CUDA path. Float32 / Float64 native; Float16 / BFloat16 via the
  // FP32-promoted accumulator path (B-016). Integer dtypes still
  // throw a clear `DeviceError` — they land on the CPU reference
  // because reductions over int tensors are rare enough not to
  // justify a dedicated kernel. We send the whole input as a flat
  // numeric array; the kernel does not need the shape, only `numel`.
  //
  // Contiguity: the CUDA two-stage all-reduce is a contiguous-input
  // kernel (it treats `x` as a flat `T*`). For strided / broadcasted
  // inputs we materialize via `.contiguous()` first — M2D made
  // `Tensor::contiguous` a strict "must already be contig" check on
  // CUDA, so we fall back to the host-side elementwise copy there
  // by bouncing `.to(cpu)` → `.contiguous()` → `.to(cuda)`. This
  // rarely fires in practice because the compute graph produces
  // contiguous intermediates by construction.
  if (x.device().is_cuda()) {
    Tensor xc = x;
    if (!x.is_contiguous()) {
      xc = x.to(cpu_device()).contiguous().to(x.device());
    }
    Stream s = current_stream(x.device());
    cuda::detail::launch_reduce_all(
        reduce_kind<S>(), x.dtype(), x.device().index,
        xc.numel(),
        xc.raw_data(), out.raw_data(),
        s.native_handle());
    return out;
  }

  // B-016: accumulate in `Acc` — equal to `T` for `float`/`double`,
  // widened to `float` for `Half`/`BFloat16`. Mirrors the CUDA
  // `reduce_all_*_promoted` kernels in `src/cuda/Reduction.cu` so
  // the CPU and CUDA paths agree bit-for-bit inside the half-
  // precision round-off budget (2e-3 abs for FP16, 5e-3 for BF16).
  const Shape& xs = x.strides();
  dispatch_float_with_half(x.dtype(), [&]<typename T>() {
    using Acc = std::conditional_t<std::is_floating_point_v<T>, T, float>;
    const T* px = x.data_ptr<T>();
    auto s = S::template init<Acc>();
    detail::for_each_index(x.shape(), [&](int64_t, const detail::IndexArray& idx) {
      S::template combine<Acc>(s, static_cast<Acc>(px[detail::offset_of(idx, xs)]));
    });
    *out.data_ptr<T>() = static_cast<T>(
        S::template finalize<Acc>(s, x.numel()));
  });
  return out;
}

template <typename S>
Tensor reduce_dim_forward(const Tensor& x, int64_t dim, bool keepdim) {
  const int64_t r = x.rank();
  dim = normalize_dim(dim, r);
  const int64_t D = x.shape()[dim];
  TESSERACT_CHECK(D > 0, "reduce_along_dim: size along dim {} is zero", dim);

  const Shape out_shape = reduced_shape(x.shape(), dim, keepdim);
  Tensor out = Tensor::empty(out_shape, x.dtype(), x.device());

  Shape iter_shape;
  for (int64_t i = 0; i < r; ++i) {
    if (i != dim) iter_shape.push_back(x.shape()[i]);
  }

  // CUDA path (M2F). The kernel honors arbitrary `in_strides` so we
  // pass the tensor's own sizes / strides directly; no forced
  // contiguify. The output is always written as row-major contiguous
  // with `dim` removed — which is exactly what
  // `Tensor::empty(reduced_shape(..., keepdim=false), ...)` just
  // allocated above. For `keepdim=true` we still allocate with the
  // size-1 dim retained; the CUDA kernel writes through the output's
  // raw buffer in the same `outer_inner` linear order, so the byte
  // layout matches a rank-`ndim` contiguous view whose `dim` slot
  // has size 1.
  if (x.device().is_cuda()) {
    Stream s = current_stream(x.device());
    cuda::detail::launch_reduce_dim(
        reduce_kind<S>(), x.dtype(), x.device().index,
        static_cast<int>(x.shape().rank()),
        static_cast<int>(dim),
        x.shape().data(), x.strides().data(),
        x.raw_data(), out.raw_data(),
        s.native_handle());
    return out;
  }

  const Shape& xs = x.strides();
  const int64_t dim_stride = xs[dim];

  // B-016: same `Acc = conditional float` widening as reduce_all
  // above; matches `reduce_dim_kernel_promoted` in the CUDA TU.
  dispatch_float_with_half(x.dtype(), [&]<typename T>() {
    using Acc = std::conditional_t<std::is_floating_point_v<T>, T, float>;
    const T* px = x.data_ptr<T>();
    T* po = out.data_ptr<T>();
    detail::for_each_index(iter_shape, [&](int64_t flat_out, const detail::IndexArray& iter_idx) {
      detail::IndexArray full{};
      std::size_t k = 0;
      for (int64_t i = 0; i < r; ++i) {
        if (i == dim) full[i] = 0;
        else full[i] = iter_idx[k++];
      }
      const int64_t base_off = detail::offset_of(full, xs);
      auto s = S::template init<Acc>();
      for (int64_t m = 0; m < D; ++m) {
        S::template combine<Acc>(s, static_cast<Acc>(px[base_off + m * dim_stride]));
      }
      po[flat_out] = static_cast<T>(S::template finalize<Acc>(s, D));
    });
  });
  return out;
}

// Build the "expanded" version of a reduced-grad tensor so that broadcast_to
// produces the shape of the original input. For keepdim=false with a dim,
// reshape to insert a 1 at `dim`. For all-reduce, grad is already a scalar.
Tensor unsqueeze_if_needed(const Tensor& grad, int64_t dim_plus_one_if_applicable,
                           int64_t rank) {
  if (!grad.defined() || grad.rank() == rank) return grad;
  // Here we know grad.rank() == rank - 1 and we need to insert a size-1 at
  // position dim. Build the new shape.
  Shape new_shape;
  for (int64_t i = 0; i < rank; ++i) {
    if (i == dim_plus_one_if_applicable) new_shape.push_back(1);
    else new_shape.push_back(grad.shape()[i > dim_plus_one_if_applicable ? i - 1 : i]);
  }
  return grad.reshape(new_shape);
}

struct SumAllBackward : Node {
  Shape in_shape;
  DType dtype;
  Device device;
  std::string_view name() const override { return "SumAllBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    outs[0] = broadcast_to(g, in_shape);
    return outs;
  }
};

struct SumDimBackward : Node {
  Shape in_shape;
  int64_t dim{0};
  bool keepdim{false};
  std::string_view name() const override { return "SumDimBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    Tensor g_expanded = keepdim ? g : unsqueeze_if_needed(g, dim, static_cast<int64_t>(in_shape.rank()));
    outs[0] = broadcast_to(g_expanded, in_shape);
    return outs;
  }
};

struct MeanAllBackward : Node {
  Shape in_shape;
  int64_t n{1};
  std::string_view name() const override { return "MeanAllBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    Tensor scale = Tensor::full({}, 1.0 / static_cast<double>(n), g.dtype(), g.device());
    Tensor g_scaled = mul(g, scale);
    outs[0] = broadcast_to(g_scaled, in_shape);
    return outs;
  }
};

struct MeanDimBackward : Node {
  Shape in_shape;
  int64_t dim{0};
  bool keepdim{false};
  int64_t n{1};
  std::string_view name() const override { return "MeanDimBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    Tensor scale = Tensor::full({}, 1.0 / static_cast<double>(n), g.dtype(), g.device());
    Tensor g_scaled = mul(g, scale);
    Tensor g_expanded = keepdim ? g_scaled
                                : unsqueeze_if_needed(g_scaled, dim, static_cast<int64_t>(in_shape.rank()));
    outs[0] = broadcast_to(g_expanded, in_shape);
    return outs;
  }
};

}  // namespace

Tensor sum(const Tensor& x) {
  Tensor out = reduce_all_forward<SumStrategy>(x);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<SumAllBackward>();
    n->in_shape = x.shape();
    n->dtype = x.dtype();
    n->device = x.device();
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("sum", {&x}, {&out});
  return out;
}

Tensor sum(const Tensor& x, int64_t dim, bool keepdim) {
  Tensor out = reduce_dim_forward<SumStrategy>(x, dim, keepdim);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<SumDimBackward>();
    n->in_shape = x.shape();
    n->dim = normalize_dim(dim, x.rank());
    n->keepdim = keepdim;
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("sum", {&x}, {&out},
                      {{"dim", static_cast<int64_t>(dim)}, {"keepdim", keepdim}});
  return out;
}

Tensor mean(const Tensor& x) {
  Tensor out = reduce_all_forward<MeanStrategy>(x);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<MeanAllBackward>();
    n->in_shape = x.shape();
    n->n = x.numel();
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("mean", {&x}, {&out});
  return out;
}

Tensor mean(const Tensor& x, int64_t dim, bool keepdim) {
  Tensor out = reduce_dim_forward<MeanStrategy>(x, dim, keepdim);
  if (is_grad_enabled() && autograd::any_requires_grad(x)) {
    auto n = std::make_shared<MeanDimBackward>();
    n->in_shape = x.shape();
    n->dim = normalize_dim(dim, x.rank());
    n->keepdim = keepdim;
    n->n = x.shape()[n->dim];
    n->next_edges = { autograd::edge_for(x) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("mean", {&x}, {&out},
                      {{"dim", static_cast<int64_t>(dim)}, {"keepdim", keepdim}});
  return out;
}

// `max` backward is not implemented at M0. Using `max` in a graph that will
// backprop will raise at call time if `requires_grad` is set.
Tensor max(const Tensor& x) {
  TESSERACT_CHECK(!(is_grad_enabled() && autograd::any_requires_grad(x)),
                  "max: backward not implemented at M0. Wrap in NoGradGuard to use forward only.");
  Tensor out = reduce_all_forward<MaxStrategy>(x);
  graph::maybe_record("max", {&x}, {&out});
  return out;
}

Tensor max(const Tensor& x, int64_t dim, bool keepdim) {
  TESSERACT_CHECK(!(is_grad_enabled() && autograd::any_requires_grad(x)),
                  "max: backward not implemented at M0. Wrap in NoGradGuard to use forward only.");
  Tensor out = reduce_dim_forward<MaxStrategy>(x, dim, keepdim);
  graph::maybe_record("max", {&x}, {&out},
                      {{"dim", static_cast<int64_t>(dim)}, {"keepdim", keepdim}});
  return out;
}

}  // namespace tesseract::ops

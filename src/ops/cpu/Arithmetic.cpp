#include "tesseract/ops/Arithmetic.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/autograd/Function.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/Elementwise.hpp"
#include "tesseract/cuda/detail/Shape.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Broadcast.hpp"
#include "tesseract/utils/Logging.hpp"

#include "IndexIter.hpp"

#if defined(TESSERACT_HAS_EIGEN)
#define EIGEN_DONT_ALIGN_STATICALLY 1
#include <Eigen/Core>
#endif

namespace tesseract::ops {

namespace {

// ---------------- Forward kernels (pure) ----------------

struct AddOp { template <typename T> static T apply(T x, T y) { return static_cast<T>(x + y); } };
struct SubOp { template <typename T> static T apply(T x, T y) { return static_cast<T>(x - y); } };
struct MulOp { template <typename T> static T apply(T x, T y) { return static_cast<T>(x * y); } };
struct DivOp {
  template <typename T>
  static T apply(T x, T y) {
    if constexpr (is_tesseract_floating_v<T>) {
      // Matches IEEE semantics: 1/0 -> inf, 0/0 -> nan. No throw.
      return static_cast<T>(x / y);
    } else {
      TESSERACT_CHECK(y != 0, "integer div: division by zero");
      return static_cast<T>(x / y);
    }
  }
};

#if defined(TESSERACT_HAS_EIGEN)
// Vectorized fast path for the aligned case (both operands same shape and
// contiguous — i.e. no broadcasting, no stride tricks). Eigen's Array
// coefficient-wise ops vectorize with the full SIMD width available on
// the host. The output `out` is already the same shape/layout so we can
// Map straight onto its buffer. Returns `false` when the slow
// broadcast-aware path must still be used.
//
// `Half` / `BFloat16` are our own user-defined types and Eigen has no
// `NumTraits` for them, so we short-circuit those instantiations to the
// scalar loop where the widening-to-float arithmetic lives.
template <typename Op, typename T>
bool try_eigen_elementwise(const Tensor& a, const Tensor& b, Tensor& out) {
  if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
    (void)a; (void)b; (void)out;
    return false;
  } else {
    if (!(a.shape() == b.shape() && a.shape() == out.shape())) return false;
    if (!a.is_contiguous() || !b.is_contiguous() || !out.is_contiguous()) return false;
    using ArrT = Eigen::Array<T, Eigen::Dynamic, 1>;
    using MapC = Eigen::Map<const ArrT, Eigen::Unaligned>;
    using MapO = Eigen::Map<ArrT, Eigen::Unaligned>;
    const int64_t n = a.numel();
    MapC A(a.data_ptr<T>(), n);
    MapC B(b.data_ptr<T>(), n);
    MapO O(out.data_ptr<T>(), n);
    if constexpr (std::is_same_v<Op, AddOp>) O = A + B;
    else if constexpr (std::is_same_v<Op, SubOp>) O = A - B;
    else if constexpr (std::is_same_v<Op, MulOp>) O = A * B;
    else if constexpr (std::is_same_v<Op, DivOp>) {
      if constexpr (std::is_floating_point_v<T>) O = A / B;
      else return false;  // leave integer div to the scalar path so the
                          // zero-divisor check fires with a clean message
    }
    else return false;
    return true;
  }
}
#endif

template <typename Op>
Tensor elementwise_binary(const Tensor& a, const Tensor& b, const char* name) {
  TESSERACT_CHECK(a.defined() && b.defined(), "{}: operand is undefined", name);
  TESSERACT_CHECK(a.dtype() == b.dtype(),
                  "{}: dtype mismatch ({} vs {})", name,
                  dtype_name(a.dtype()), dtype_name(b.dtype()));
  TESSERACT_CHECK(a.device() == b.device(),
                  "{}: device mismatch ({} vs {})", name,
                  a.device().to_string(), b.device().to_string());

  const Shape out_shape = broadcast_shape(a.shape(), b.shape());
  Tensor out = Tensor::empty(out_shape, a.dtype(), a.device());

  // CUDA backend — dispatch to the M2E elementwise launchers before any
  // of the CPU scaffolding below runs. Shape/stride padding goes through
  // the same `align_for_broadcast` helper the CPU loop uses, so a stride
  // of 0 on a broadcasted dim means "replicate", exactly like on CPU.
  // We pass the native stream handle down so the CUDA TU doesn't need
  // to call back into `tesseract_core` (`current_stream` lives there);
  // that keeps the core ↔ cuda archive link graph acyclic.
  if (a.device().is_cuda()) {
    Shape a_bs, b_bs;
    align_for_broadcast(a.shape(), a.strides(), out_shape, a_bs);
    align_for_broadcast(b.shape(), b.strides(), out_shape, b_bs);
    cuda::detail::BinaryKind kind;
    if constexpr (std::is_same_v<Op, AddOp>) kind = cuda::detail::BinaryKind::Add;
    else if constexpr (std::is_same_v<Op, SubOp>) kind = cuda::detail::BinaryKind::Sub;
    else if constexpr (std::is_same_v<Op, MulOp>) kind = cuda::detail::BinaryKind::Mul;
    else if constexpr (std::is_same_v<Op, DivOp>) kind = cuda::detail::BinaryKind::Div;
    else {
      static_assert(sizeof(Op) == 0, "unknown binary op in CUDA dispatch");
    }
    Stream s = current_stream(a.device());
    // Bridge takes raw `int64_t*` + `int ndim` rather than `const Shape&`
    // — the nvcc-compiled `Elementwise.cu` TU is C++17-only (CMake 3.22
    // can't select `CUDA20`), and `Shape.hpp` drags in `std::span` +
    // explicit-template lambdas (both C++20). `align_for_broadcast`
    // already padded `a_bs` / `b_bs` to `out_shape.rank()`, so the
    // three stride arrays all share ndim.
    cuda::detail::launch_binary_elementwise(
        kind, a.dtype(), a.device().index,
        static_cast<int>(out_shape.rank()),
        out_shape.data(), a_bs.data(), b_bs.data(),
        out.raw_data(), a.raw_data(), b.raw_data(),
        s.native_handle());
    return out;
  }

#if defined(TESSERACT_HAS_EIGEN)
  // Same-shape + contiguous fast path (no broadcast, no stride trickery).
  // This covers the overwhelming majority of training-time call sites
  // (parameter + gradient, logits - target, elementwise masks). Falls
  // through to the index-iterator path for broadcasted/strided inputs.
  //
  // Note: the half-precision types short-circuit `try_eigen_elementwise`
  // to `false` because Eigen has no `NumTraits` for them — they always
  // land on the scalar loop below, which widens to FP32 via the
  // `Half`/`BFloat16` operator overloads.
  bool used_eigen = false;
  dispatch_numeric_with_half(a.dtype(), [&]<typename T>() {
    used_eigen = try_eigen_elementwise<Op, T>(a, b, out);
  });
  if (used_eigen) return out;
#endif

  Shape a_strides, b_strides;
  align_for_broadcast(a.shape(), a.strides(), out_shape, a_strides);
  align_for_broadcast(b.shape(), b.strides(), out_shape, b_strides);

  dispatch_numeric_with_half(a.dtype(), [&]<typename T>() {
    const T* pa = a.data_ptr<T>();
    const T* pb = b.data_ptr<T>();
    T* po = out.data_ptr<T>();
    detail::for_each_index(out_shape, [&](int64_t flat, const detail::IndexArray& idx) {
      const int64_t ao = detail::offset_of(idx, a_strides);
      const int64_t bo = detail::offset_of(idx, b_strides);
      po[flat] = Op::template apply<T>(pa[ao], pb[bo]);
    });
  });
  return out;
}

Tensor add_forward(const Tensor& a, const Tensor& b) { return elementwise_binary<AddOp>(a, b, "add"); }
Tensor sub_forward(const Tensor& a, const Tensor& b) { return elementwise_binary<SubOp>(a, b, "sub"); }
Tensor mul_forward(const Tensor& a, const Tensor& b) { return elementwise_binary<MulOp>(a, b, "mul"); }
Tensor div_forward(const Tensor& a, const Tensor& b) { return elementwise_binary<DivOp>(a, b, "div"); }

Tensor neg_forward(const Tensor& a) {
  TESSERACT_CHECK(a.defined(), "neg: operand is undefined");
  Tensor out = Tensor::empty(a.shape(), a.dtype(), a.device());

  if (a.device().is_cuda()) {
    Stream s = current_stream(a.device());
    cuda::detail::launch_unary_elementwise(
        cuda::detail::UnaryKind::Neg, a.dtype(), a.device().index,
        static_cast<int>(a.shape().rank()),
        a.shape().data(), a.strides().data(),
        out.raw_data(), a.raw_data(),
        s.native_handle());
    return out;
  }

  const Shape& s = a.strides();
  dispatch_numeric_with_half(a.dtype(), [&]<typename T>() {
    const T* pa = a.data_ptr<T>();
    T* po = out.data_ptr<T>();
    detail::for_each_index(a.shape(), [&](int64_t flat, const detail::IndexArray& idx) {
      const int64_t ao = detail::offset_of(idx, s);
      po[flat] = static_cast<T>(-pa[ao]);
    });
  });
  return out;
}

// ---------------- Backward nodes ----------------

struct AddBackward : Node {
  Shape a_shape, b_shape;
  std::string_view name() const override { return "AddBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(2);
    if (next_edges[0].requires_grad) outs[0] = reduce_to_shape(g, a_shape);
    if (next_edges[1].requires_grad) outs[1] = reduce_to_shape(g, b_shape);
    return outs;
  }
};

struct SubBackward : Node {
  Shape a_shape, b_shape;
  std::string_view name() const override { return "SubBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(2);
    if (next_edges[0].requires_grad) outs[0] = reduce_to_shape(g, a_shape);
    if (next_edges[1].requires_grad) outs[1] = reduce_to_shape(neg_forward(g), b_shape);
    return outs;
  }
};

struct MulBackward : Node {
  Tensor a_saved, b_saved;
  std::string_view name() const override { return "MulBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(2);
    if (next_edges[0].requires_grad) outs[0] = reduce_to_shape(mul_forward(g, b_saved), a_saved.shape());
    if (next_edges[1].requires_grad) outs[1] = reduce_to_shape(mul_forward(g, a_saved), b_saved.shape());
    return outs;
  }
};

struct DivBackward : Node {
  Tensor a_saved, b_saved;
  std::string_view name() const override { return "DivBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(2);
    if (next_edges[0].requires_grad) outs[0] = reduce_to_shape(div_forward(g, b_saved), a_saved.shape());
    if (next_edges[1].requires_grad) {
      Tensor b2 = mul_forward(b_saved, b_saved);
      Tensor num = mul_forward(g, a_saved);
      Tensor grad_b = neg_forward(div_forward(num, b2));
      outs[1] = reduce_to_shape(grad_b, b_saved.shape());
    }
    return outs;
  }
};

struct NegBackward : Node {
  std::string_view name() const override { return "NegBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (next_edges[0].requires_grad) outs[0] = neg_forward(g);
    return outs;
  }
};

// Backward of a cross-device copy `out = src.to(dst_device)`: route the
// gradient back to the source device (`grad_src = grad_out.to(src_device)`).
// This makes tensor/data-parallel graphs differentiable across GPUs —
// `RowParallelLinear` all-reduce, sharded `.to()` moves, etc. The copy is
// numerically the identity, so the backward is just the inverse copy.
struct CopyBackward : Node {
  Device src_device;
  std::string_view name() const override { return "CopyBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (next_edges[0].requires_grad) outs[0] = g.to(src_device);
    return outs;
  }
};

// The hook `Tensor::to` calls (registered below at static init). Re-checks
// grad-mode / requires_grad so it is a true no-op on inference tensors.
void to_autograd_hook(Tensor& out, const Tensor& src) {
  if (!is_grad_enabled() || !autograd::any_requires_grad(src)) return;
  auto n = std::make_shared<CopyBackward>();
  n->src_device = src.device();
  n->next_edges = {autograd::edge_for(src)};
  autograd::attach_grad_fn(out, n);
}

// Register at static init. This TU (the elementwise ops) is referenced by
// essentially every consumer, so it is always linked and the initializer
// always runs; the hook is only consulted at runtime, well after all
// static initialization completes, so init order is irrelevant.
const bool g_to_hook_registered = [] {
  ::tesseract::detail::set_to_autograd_hook(&to_autograd_hook);
  return true;
}();

}  // namespace

// ---------------- Public API ----------------

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out = add_forward(a, b);
  if (is_grad_enabled() && autograd::any_requires_grad(a, b)) {
    auto n = std::make_shared<AddBackward>();
    n->a_shape = a.shape();
    n->b_shape = b.shape();
    n->next_edges = { autograd::edge_for(a), autograd::edge_for(b) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("add", {&a, &b}, {&out});
  return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
  Tensor out = sub_forward(a, b);
  if (is_grad_enabled() && autograd::any_requires_grad(a, b)) {
    auto n = std::make_shared<SubBackward>();
    n->a_shape = a.shape();
    n->b_shape = b.shape();
    n->next_edges = { autograd::edge_for(a), autograd::edge_for(b) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("sub", {&a, &b}, {&out});
  return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  Tensor out = mul_forward(a, b);
  if (is_grad_enabled() && autograd::any_requires_grad(a, b)) {
    auto n = std::make_shared<MulBackward>();
    n->a_saved = a;
    n->b_saved = b;
    n->next_edges = { autograd::edge_for(a), autograd::edge_for(b) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("mul", {&a, &b}, {&out});
  return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
  Tensor out = div_forward(a, b);
  if (is_grad_enabled() && autograd::any_requires_grad(a, b)) {
    auto n = std::make_shared<DivBackward>();
    n->a_saved = a;
    n->b_saved = b;
    n->next_edges = { autograd::edge_for(a), autograd::edge_for(b) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("div", {&a, &b}, {&out});
  return out;
}

Tensor neg(const Tensor& a) {
  Tensor out = neg_forward(a);
  if (is_grad_enabled() && autograd::any_requires_grad(a)) {
    auto n = std::make_shared<NegBackward>();
    n->next_edges = { autograd::edge_for(a) };
    autograd::attach_grad_fn(out, n);
  }
  graph::maybe_record("neg", {&a}, {&out});
  return out;
}

Tensor broadcast_to(const Tensor& src, const Shape& target_shape) {
  TESSERACT_CHECK(src.defined(), "broadcast_to: src is undefined");
  // Even if the tensor's shape already matches, we still need to record
  // the op so graph-mode capture sees a single broadcast_to node instead
  // of leaking an unbound `contiguous()` result into the graph.
  Tensor out;
  if (src.shape() == target_shape) {
    out = src.contiguous();
  } else {
    TESSERACT_CHECK(is_broadcastable_to(src.shape(), target_shape),
                    "broadcast_to: {} cannot broadcast to {}",
                    src.shape().to_string(), target_shape.to_string());

    out = Tensor::empty(target_shape, src.dtype(), src.device());
    Shape aligned_strides;
    align_for_broadcast(src.shape(), src.strides(), target_shape, aligned_strides);

    // CUDA path (M2H): broadcast_to is exactly a strided read with
    // stride-0 entries on the broadcast axes, writing into a freshly
    // allocated dense-contig output. `launch_strided_copy` handles
    // this uniformly — we pass `aligned_strides` as source strides
    // (zeros for expansions) and the contiguous strides as dst.
    if (src.device().is_cuda()) {
      const Shape out_strides = target_shape.contiguous_strides();
      Stream s = current_stream(src.device());
      cuda::detail::launch_strided_copy(
          src.dtype(), src.device().index,
          static_cast<int>(target_shape.rank()),
          target_shape.data(),
          aligned_strides.data(),
          out_strides.data(),
          src.raw_data(), out.raw_data(),
          s.native_handle());
    } else {
      dispatch_numeric_with_half(src.dtype(), [&]<typename T>() {
        const T* ps = src.data_ptr<T>();
        T* po = out.data_ptr<T>();
        detail::for_each_index(target_shape, [&](int64_t flat, const detail::IndexArray& idx) {
          po[flat] = ps[detail::offset_of(idx, aligned_strides)];
        });
      });
    }
  }

  graph::maybe_record(
      "broadcast_to", {&src}, {&out},
      {{"shape", std::vector<int64_t>(target_shape.begin(), target_shape.end())}});
  return out;
}

Tensor reduce_to_shape(const Tensor& src, const Shape& target_shape) {
  TESSERACT_CHECK(src.defined(), "reduce_to_shape: src is undefined");

  if (src.shape() == target_shape) return src.contiguous();

  TESSERACT_CHECK(is_broadcastable_to(target_shape, src.shape()),
                  "reduce_to_shape: {} is not broadcast-reducible to {}",
                  src.shape().to_string(), target_shape.to_string());

  const std::size_t rs = src.shape().rank();
  const std::size_t rt = target_shape.rank();
  const std::size_t offset = rs - rt;
  Shape aligned_target;
  aligned_target.resize(rs);
  for (std::size_t i = 0; i < rs; ++i) {
    aligned_target[i] = (i < offset) ? 1 : target_shape[i - offset];
  }

  Tensor out = Tensor::zeros(target_shape, src.dtype(), src.device());
  const Shape& ss = src.strides();
  const Shape out_strides_contig = target_shape.contiguous_strides();

  // CUDA path (M2H): reduce-to-shape is a strided scatter-add where
  // "reduced" dims on the destination have stride 0 (so every src
  // element on that dim lands in the same dst slot). We build an
  // `aligned_dst_strides` array of rank `rs` by mapping each src axis
  // to its counterpart in the contiguous target layout; reduced axes
  // get 0. That feeds straight into `launch_strided_scatter_add`.
  if (src.device().is_cuda()) {
    Shape aligned_dst_strides;
    aligned_dst_strides.resize(rs);
    for (std::size_t d = 0; d < rs; ++d) {
      if (aligned_target[d] == 1) {
        aligned_dst_strides[d] = 0;
      } else {
        const int64_t td = d - offset;
        aligned_dst_strides[d] = out_strides_contig[td];
      }
    }
    Stream s = current_stream(src.device());
    cuda::detail::launch_strided_scatter_add(
        src.dtype(), src.device().index,
        static_cast<int>(src.shape().rank()),
        src.shape().data(),
        ss.data(),
        aligned_dst_strides.data(),
        src.raw_data(), out.raw_data(),
        s.native_handle());
    return out;
  }

  dispatch_numeric_with_half(src.dtype(), [&]<typename T>() {
    const T* pa = src.data_ptr<T>();
    T* po = out.data_ptr<T>();
    detail::for_each_index(src.shape(), [&](int64_t, const detail::IndexArray& idx) {
      const int64_t src_off = detail::offset_of(idx, ss);
      int64_t dst_off = 0;
      for (std::size_t d = 0; d < rs; ++d) {
        if (aligned_target[d] == 1) continue;
        const int64_t td = d - offset;
        dst_off += idx[d] * out_strides_contig[td];
      }
      po[dst_off] = static_cast<T>(po[dst_off] + pa[src_off]);
    });
  });
  return out;
}

}  // namespace tesseract::ops

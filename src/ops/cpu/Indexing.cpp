// B-003 — Autograd-aware `cat` / `split` / `index_select` / `gather`.
//
// These four ops share a common "mix-a-view-with-scatter" skeleton:
//
//   * cat        : N inputs → 1 output (one contiguous tensor laid out as
//                  [input_0 | input_1 | ... | input_{N-1}] along `dim`).
//   * split      : 1 input  → N outputs (disjoint slabs along `dim`).
//   * index_select: `out[..., i, ...] = src[..., idx[i], ...]` along `dim`.
//   * gather     : element-wise scatter/gather with `out.shape == idx.shape`.
//
// The backwards all boil down to "put grad slabs back where the forward
// pulled from, summing on collision". The autograd engine already
// accumulates gradients into leaves via `ops::add` (see `autograd/Engine.cpp`
// around `acc[edge.grad_fn.get()] = ops::add(...)`), so we exploit that:
//   * `split` installs a *separate* one-input `SplitChunkBackward` node per
//     output chunk. Each chunk's backward emits a zero-padded grad that
//     covers the parent's full shape but is non-zero only within its own
//     slab. The engine then sums all N contributions at the shared parent
//     edge — equivalent to a single `cat`.
//   * `cat` is modeled as one N-input node whose `apply` returns N slices
//     of the incoming gradient.
//   * `index_select` / `gather` are single-input nodes whose `apply`
//     materializes a zero tensor of the input's shape and scatter-adds the
//     gradient element-wise using the saved Int64 index tensor.

#include "tesseract/ops/Indexing.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "tesseract/autograd/Function.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Storage.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/Indexing.hpp"
#include "tesseract/cuda/detail/Shape.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::cuda::detail {
// Defined in src/cuda/Stream.cpp. True iff `stream` is mid graph-capture.
bool real_stream_is_capturing(int device_index, void* stream) noexcept;
}  // namespace tesseract::cuda::detail

#include "IndexIter.hpp"

namespace tesseract::ops {

namespace {

// ----------------------------------------------------------------------------
// Small helpers
// ----------------------------------------------------------------------------

int64_t normalize_dim(int64_t dim, std::size_t rank, const char* where) {
  const int64_t r = static_cast<int64_t>(rank);
  const int64_t d = dim < 0 ? dim + r : dim;
  TESSERACT_CHECK(d >= 0 && d < r,
                  "{}: dim {} out of range for rank {}", where, dim, rank);
  return d;
}

// Write `slab` (contiguous, shape == `dst.shape()` except along `dim` where
// it is `size`) into `dst[..., start:start+size, ...]` along `dim`.
// Arbitrary (possibly non-contiguous) `dst` strides are honored; `slab` is
// assumed contiguous.
template <typename T>
void copy_slab_into_cpu(Tensor& dst, int64_t dim, int64_t start,
                        const Tensor& slab) {
  const std::size_t r = dst.shape().rank();
  const Shape& ds = dst.strides();
  T* pd = dst.data_ptr<T>();
  const T* ps = slab.data_ptr<T>();
  detail::for_each_index(slab.shape(),
                         [&](int64_t flat, const detail::IndexArray& idx) {
    int64_t off = 0;
    for (std::size_t d = 0; d < r; ++d) {
      const int64_t i = (d == static_cast<std::size_t>(dim)) ? idx[d] + start
                                                             : idx[d];
      off += i * ds[d];
    }
    pd[off] = ps[flat];
  });
}

// Dispatching wrapper: CUDA takes the strided-copy kernel; CPU stays
// on the templated scalar loop above. On CUDA we express the "offset
// into `dst` on `dim`" as a byte offset added to the dst base pointer
// before the launch, so the kernel itself only sees dense strides +
// a single base pointer — matches the M2G/M2H bridge convention of
// pushing offset arithmetic into the op layer.
void copy_slab_into(Tensor& dst, int64_t dim, int64_t start, const Tensor& slab) {
  if (dst.device().is_cuda()) {
    // `slab` is assumed contig (the CPU path also requires this).
    // Materialize on CUDA if it isn't, so we can hand the kernel a
    // dense source stride array.
    Tensor slab_c = slab.is_contiguous() ? slab : slab.contiguous();
    const Shape& slab_shape = slab_c.shape();
    const Shape slab_strides = slab_shape.contiguous_strides();
    const std::size_t elem = dtype_size(dst.dtype());
    auto* dst_bytes = static_cast<std::byte*>(dst.raw_data());
    const int64_t byte_off = start * dst.strides()[static_cast<std::size_t>(dim)]
                             * static_cast<int64_t>(elem);
    Stream s = current_stream(dst.device());
    cuda::detail::launch_strided_copy(
        dst.dtype(), dst.device().index,
        static_cast<int>(slab_shape.rank()),
        slab_shape.data(),
        slab_strides.data(),
        dst.strides().data(),
        slab_c.raw_data(),
        dst_bytes + byte_off,
        s.native_handle());
    return;
  }
  dispatch_numeric(dst.dtype(),
                   [&]<typename T>() { copy_slab_into_cpu<T>(dst, dim, start, slab); });
}

// Pull out `src[..., start:start+size, ...]` along `dim` as a freshly
// allocated contiguous tensor. Used by both the `cat` backward
// (split an incoming grad) and the public `split` forward.
Tensor slice_along_dim(const Tensor& src, int64_t dim, int64_t start,
                       int64_t size) {
  const std::size_t r = src.shape().rank();
  Shape out_shape = src.shape();
  out_shape[static_cast<std::size_t>(dim)] = size;
  Tensor out = Tensor::empty(out_shape, src.dtype(), src.device());

  const Shape& ss = src.strides();
  if (src.device().is_cuda()) {
    // Express the slice as a strided copy from `src + start*s[dim]`
    // into `out` with dense contig strides. The kernel sees dense
    // source strides except for the dim-offset which we bake into
    // the base pointer below.
    const Shape out_strides_contig = out_shape.contiguous_strides();
    const std::size_t elem = dtype_size(src.dtype());
    const auto* src_bytes = static_cast<const std::byte*>(src.raw_data());
    const int64_t byte_off = start * ss[static_cast<std::size_t>(dim)]
                             * static_cast<int64_t>(elem);
    Stream s = current_stream(src.device());
    cuda::detail::launch_strided_copy(
        src.dtype(), src.device().index,
        static_cast<int>(r),
        out_shape.data(),
        ss.data(),
        out_strides_contig.data(),
        src_bytes + byte_off,
        out.raw_data(),
        s.native_handle());
    return out;
  }

  dispatch_numeric(src.dtype(), [&]<typename T>() {
    const T* pa = src.data_ptr<T>();
    T* po = out.data_ptr<T>();
    detail::for_each_index(out_shape,
                           [&](int64_t flat, const detail::IndexArray& idx) {
      int64_t off = 0;
      for (std::size_t d = 0; d < r; ++d) {
        const int64_t i = (d == static_cast<std::size_t>(dim)) ? idx[d] + start
                                                               : idx[d];
        off += i * ss[d];
      }
      po[flat] = pa[off];
    });
  });
  return out;
}

// ----------------------------------------------------------------------------
// Backward nodes
// ----------------------------------------------------------------------------

// `cat` collapses N inputs into one output. The gradient is sliced back.
struct CatBackward : Node {
  int64_t dim{0};
  std::vector<int64_t> sizes;  // one per input, summing to out.shape()[dim]
  std::vector<Shape> input_shapes;
  std::string_view name() const override { return "CatBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    const Tensor gc = g.is_contiguous() ? g : g.contiguous();
    std::vector<Tensor> outs(next_edges.size());
    int64_t start = 0;
    for (std::size_t i = 0; i < next_edges.size(); ++i) {
      const int64_t sz = sizes[i];
      if (next_edges[i].requires_grad) {
        outs[i] = slice_along_dim(gc, dim, start, sz);
      }
      start += sz;
    }
    return outs;
  }
};

// One backward node per `split` output chunk. Scatters the chunk's grad
// into a zero-filled parent-shape tensor. The engine sums contributions
// from different chunks at the shared parent edge.
struct SplitChunkBackward : Node {
  Shape parent_shape;
  int64_t dim{0};
  int64_t start{0};
  std::string_view name() const override { return "SplitChunkBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(1);
    if (!next_edges[0].requires_grad) return outs;
    const Tensor gc = g.is_contiguous() ? g : g.contiguous();
    Tensor full = Tensor::zeros(parent_shape, gc.dtype(), gc.device());
    // `copy_slab_into` now dispatches CPU / CUDA internally, so the
    // backward reads the same on both devices.
    copy_slab_into(full, dim, start, gc);
    outs[0] = std::move(full);
    return outs;
  }
};

// Scatter-add for `index_select` backward:
//   grad_src[..., indices[k], ...] += grad_out[..., k, ...]
// using a Kahan-free straight sum (matches the engine's leaf accumulator).
struct IndexSelectBackward : Node {
  Shape parent_shape;
  Tensor indices_saved;  // rank-1 Int64
  int64_t dim{0};
  std::string_view name() const override { return "IndexSelectBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(2);
    if (next_edges[0].requires_grad) {
      const Tensor gc = g.is_contiguous() ? g : g.contiguous();
      Tensor full = Tensor::zeros(parent_shape, gc.dtype(), gc.device());

      if (gc.device().is_cuda()) {
        // CUDA path: atomic scatter-add into `full`. The indices
        // tensor is already contig Int64 (enforced at forward time).
        Stream s = current_stream(gc.device());
        cuda::detail::launch_scatter_add_at_dim(
            gc.dtype(), gc.device().index,
            static_cast<int>(gc.shape().rank()),
            static_cast<int>(dim),
            gc.shape().data(),
            gc.strides().data(),
            full.strides().data(),
            gc.raw_data(),
            indices_saved.data_ptr<int64_t>(),
            full.raw_data(),
            s.native_handle());
        outs[0] = std::move(full);
        return outs;
      }

      const std::size_t r = parent_shape.rank();
      const int64_t K = indices_saved.shape()[0];
      const int64_t* pidx = indices_saved.data_ptr<int64_t>();

      dispatch_numeric(gc.dtype(), [&]<typename T>() {
        const Shape& fs = full.strides();
        T* pf = full.data_ptr<T>();
        const T* pg = gc.data_ptr<T>();
        detail::for_each_index(gc.shape(),
                               [&](int64_t flat, const detail::IndexArray& idx) {
          const int64_t k = idx[static_cast<std::size_t>(dim)];
          (void)K;
          const int64_t row = pidx[k];
          int64_t off = 0;
          for (std::size_t d = 0; d < r; ++d) {
            const int64_t i = (d == static_cast<std::size_t>(dim)) ? row : idx[d];
            off += i * fs[d];
          }
          pf[off] = static_cast<T>(pf[off] + pg[flat]);
        });
      });
      outs[0] = std::move(full);
    }
    // outs[1] = undefined — indices never needs grad.
    return outs;
  }
};

// Scatter-add for `gather` backward (PyTorch GatherElements convention):
//   grad_src[i_0, ..., indices[i_0, ..., i_d, ...], ..., i_{r-1}]
//     += grad_out[i_0, ..., i_d, ..., i_{r-1}]
struct GatherBackward : Node {
  Shape parent_shape;
  Tensor indices_saved;  // same rank as src, int64
  int64_t dim{0};
  std::string_view name() const override { return "GatherBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(2);
    if (next_edges[0].requires_grad) {
      const Tensor gc = g.is_contiguous() ? g : g.contiguous();
      Tensor full = Tensor::zeros(parent_shape, gc.dtype(), gc.device());

      if (gc.device().is_cuda()) {
        Stream s = current_stream(gc.device());
        cuda::detail::launch_gather_scatter_add(
            gc.dtype(), gc.device().index,
            static_cast<int>(gc.shape().rank()),
            static_cast<int>(dim),
            gc.shape().data(),
            gc.strides().data(),
            indices_saved.strides().data(),
            full.strides().data(),
            gc.raw_data(),
            indices_saved.data_ptr<int64_t>(),
            full.raw_data(),
            s.native_handle());
        outs[0] = std::move(full);
        return outs;
      }

      const std::size_t r = parent_shape.rank();
      const Shape& istr = indices_saved.strides();
      const int64_t* pidx = indices_saved.data_ptr<int64_t>();

      dispatch_numeric(gc.dtype(), [&]<typename T>() {
        const Shape& fs = full.strides();
        T* pf = full.data_ptr<T>();
        const T* pg = gc.data_ptr<T>();
        detail::for_each_index(gc.shape(),
                               [&](int64_t flat, const detail::IndexArray& idx) {
          int64_t idx_off = 0;
          for (std::size_t d = 0; d < r; ++d) idx_off += idx[d] * istr[d];
          const int64_t sel = pidx[idx_off];
          int64_t off = 0;
          for (std::size_t d = 0; d < r; ++d) {
            const int64_t i = (d == static_cast<std::size_t>(dim)) ? sel : idx[d];
            off += i * fs[d];
          }
          pf[off] = static_cast<T>(pf[off] + pg[flat]);
        });
      });
      outs[0] = std::move(full);
    }
    return outs;
  }
};

}  // namespace

// ============================================================================
// Public API
// ============================================================================

Tensor cat(const std::vector<Tensor>& tensors, int64_t dim) {
  TESSERACT_CHECK(!tensors.empty(), "cat: need at least one tensor");
  const Tensor& first = tensors.front();
  TESSERACT_CHECK(first.defined(), "cat: tensor[0] is undefined");
  const std::size_t r = first.shape().rank();
  TESSERACT_CHECK(r > 0, "cat: cannot concatenate rank-0 tensors");
  const int64_t d = normalize_dim(dim, r, "cat");

  // Shape / dtype / device checks, plus accumulate the size along `dim`.
  int64_t cat_dim_total = 0;
  std::vector<int64_t> sizes;
  sizes.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const Tensor& t = tensors[i];
    TESSERACT_CHECK(t.defined(), "cat: tensor[{}] is undefined", i);
    TESSERACT_CHECK(t.dtype() == first.dtype(),
                    "cat: dtype mismatch at index {} ({} vs {})", i,
                    dtype_name(t.dtype()), dtype_name(first.dtype()));
    TESSERACT_CHECK(t.device() == first.device(),
                    "cat: device mismatch at index {} ({} vs {})", i,
                    t.device().to_string(), first.device().to_string());
    TESSERACT_CHECK(t.shape().rank() == r,
                    "cat: rank mismatch at index {} ({} vs {})", i,
                    t.shape().to_string(), first.shape().to_string());
    for (std::size_t k = 0; k < r; ++k) {
      if (k == static_cast<std::size_t>(d)) continue;
      TESSERACT_CHECK(t.shape()[k] == first.shape()[k],
                      "cat: shape mismatch at index {} dim {} ({} vs {})",
                      i, k, t.shape()[k], first.shape()[k]);
    }
    sizes.push_back(t.shape()[static_cast<std::size_t>(d)]);
    cat_dim_total += sizes.back();
  }

  Shape out_shape = first.shape();
  out_shape[static_cast<std::size_t>(d)] = cat_dim_total;
  Tensor out = Tensor::empty(out_shape, first.dtype(), first.device());

  int64_t start = 0;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const Tensor src = tensors[i].is_contiguous() ? tensors[i]
                                                  : tensors[i].contiguous();
    // `copy_slab_into` dispatches CPU / CUDA internally, so the
    // concat loop is device-agnostic.
    copy_slab_into(out, d, start, src);
    start += sizes[i];
  }

  // Autograd wiring.
  bool any_rg = false;
  for (const auto& t : tensors) if (t.requires_grad()) { any_rg = true; break; }
  if (is_grad_enabled() && any_rg) {
    auto node = std::make_shared<CatBackward>();
    node->dim = d;
    node->sizes = sizes;
    node->input_shapes.reserve(tensors.size());
    node->next_edges.reserve(tensors.size());
    for (const auto& t : tensors) {
      node->input_shapes.push_back(t.shape());
      node->next_edges.push_back(autograd::edge_for(t));
    }
    autograd::attach_grad_fn(out, node);
  }

  // Graph capture.
  std::vector<const Tensor*> in_ptrs;
  in_ptrs.reserve(tensors.size());
  for (const auto& t : tensors) in_ptrs.push_back(&t);
  graph::maybe_record("cat", std::move(in_ptrs), {&out},
                      {{"dim", static_cast<int64_t>(d)}});
  return out;
}

std::vector<Tensor> split_with_sizes(const Tensor& src,
                                     const std::vector<int64_t>& sizes,
                                     int64_t dim) {
  TESSERACT_CHECK(src.defined(), "split_with_sizes: src is undefined");
  const std::size_t r = src.shape().rank();
  const int64_t d = normalize_dim(dim, r, "split_with_sizes");
  int64_t total = 0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    TESSERACT_CHECK(sizes[i] >= 0,
                    "split_with_sizes: sizes[{}] = {} must be non-negative",
                    i, sizes[i]);
    total += sizes[i];
  }
  TESSERACT_CHECK(total == src.shape()[static_cast<std::size_t>(d)],
                  "split_with_sizes: sum(sizes) = {} != src.shape[{}] = {}",
                  total, d, src.shape()[static_cast<std::size_t>(d)]);

  std::vector<Tensor> outs;
  outs.reserve(sizes.size());
  int64_t start = 0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    Tensor chunk = slice_along_dim(src, d, start, sizes[i]);
    outs.push_back(std::move(chunk));
    start += sizes[i];
  }

  // Autograd: one chunk-specific node per output. They all back-propagate
  // to the same parent edge, and the engine sums the zero-padded slabs.
  if (is_grad_enabled() && src.requires_grad()) {
    int64_t s = 0;
    for (std::size_t i = 0; i < outs.size(); ++i) {
      auto node = std::make_shared<SplitChunkBackward>();
      node->parent_shape = src.shape();
      node->dim = d;
      node->start = s;
      node->next_edges = { autograd::edge_for(src) };
      autograd::attach_grad_fn(outs[i], node);
      s += sizes[i];
    }
  }

  std::vector<const Tensor*> out_ptrs;
  out_ptrs.reserve(outs.size());
  for (const auto& t : outs) out_ptrs.push_back(&t);
  graph::maybe_record("split", {&src}, std::move(out_ptrs),
                      {{"dim", static_cast<int64_t>(d)},
                       {"sizes", std::vector<int64_t>(sizes)}});
  return outs;
}

std::vector<Tensor> split(const Tensor& src, int64_t size, int64_t dim) {
  TESSERACT_CHECK(src.defined(), "split: src is undefined");
  TESSERACT_CHECK(size > 0, "split: size must be positive, got {}", size);
  const std::size_t r = src.shape().rank();
  const int64_t d = normalize_dim(dim, r, "split");
  const int64_t total = src.shape()[static_cast<std::size_t>(d)];
  std::vector<int64_t> sizes;
  for (int64_t s = 0; s < total; s += size) {
    sizes.push_back(std::min<int64_t>(size, total - s));
  }
  if (sizes.empty()) sizes.push_back(0);  // degenerate: total == 0.
  return split_with_sizes(src, sizes, d);
}

Tensor index_select(const Tensor& src, int64_t dim, const Tensor& indices) {
  TESSERACT_CHECK(src.defined(), "index_select: src is undefined");
  TESSERACT_CHECK(indices.defined(), "index_select: indices is undefined");
  TESSERACT_CHECK(indices.dtype() == DType::Int64,
                  "index_select: indices must be Int64, got {}",
                  dtype_name(indices.dtype()));
  TESSERACT_CHECK(indices.rank() == 1,
                  "index_select: indices must be rank-1, got shape {}",
                  indices.shape().to_string());
  TESSERACT_CHECK(src.device() == indices.device(),
                  "index_select: device mismatch ({} vs {})",
                  src.device().to_string(), indices.device().to_string());
  const std::size_t r = src.shape().rank();
  TESSERACT_CHECK(r > 0, "index_select: src must have rank >= 1");
  const int64_t d = normalize_dim(dim, r, "index_select");
  const int64_t dim_size = src.shape()[static_cast<std::size_t>(d)];
  const int64_t K = indices.shape()[0];

  Shape out_shape = src.shape();
  out_shape[static_cast<std::size_t>(d)] = K;
  Tensor out = Tensor::empty(out_shape, src.dtype(), src.device());

  const Tensor idx = indices.is_contiguous() ? indices : indices.contiguous();

  if (src.device().is_cuda()) {
    const Shape out_strides_contig = out_shape.contiguous_strides();
    Stream s = current_stream(src.device());

    // Range-check CUDA indices on host. Matching the CPU path's
    // per-element check means a bad index throws a `TESSERACT_CHECK`
    // before the kernel launch instead of producing out-of-bounds
    // reads in flight. For the batch-size K values in typical
    // training workloads (label lookups, attention masks, ...) the
    // O(K) host copy is negligible next to the forward kernel cost.
    //
    // The host copy is a device→host sync, which is illegal while the
    // stream is capturing a CUDA graph (and would also serialize the
    // decode loop). Skip it during capture: the closure is driven twice
    // outside capture first (CudaGraph::capture warmup), so the very same
    // indices have already been validated by the time we record the graph.
    if (!cuda::detail::real_stream_is_capturing(src.device().index,
                                                s.native_handle())) {
      std::vector<std::byte> idx_host(static_cast<std::size_t>(K) * sizeof(int64_t));
      Storage::copy_device_bytes(idx_host.data(), cpu_device(),
                                 idx.raw_data(), idx.device(),
                                 idx_host.size());
      const auto* pidx_h = reinterpret_cast<const int64_t*>(idx_host.data());
      for (int64_t k = 0; k < K; ++k) {
        TESSERACT_CHECK(pidx_h[k] >= 0 && pidx_h[k] < dim_size,
                        "index_select: indices[{}] = {} out of range [0, {})",
                        k, pidx_h[k], dim_size);
      }
    }

    cuda::detail::launch_index_select(
        src.dtype(), src.device().index,
        static_cast<int>(r),
        static_cast<int>(d),
        out_shape.data(),
        src.strides().data(),
        out_strides_contig.data(),
        src.raw_data(),
        idx.data_ptr<int64_t>(),
        out.raw_data(),
        s.native_handle());
  } else {
    const int64_t* pidx = idx.data_ptr<int64_t>();
    const Shape& ss = src.strides();

    dispatch_numeric(src.dtype(), [&]<typename T>() {
      const T* ps = src.data_ptr<T>();
      T* po = out.data_ptr<T>();
      detail::for_each_index(out_shape,
                             [&](int64_t flat, const detail::IndexArray& idx_nd) {
        const int64_t k = idx_nd[static_cast<std::size_t>(d)];
        const int64_t row = pidx[k];
        TESSERACT_CHECK(row >= 0 && row < dim_size,
                        "index_select: indices[{}] = {} out of range [0, {})",
                        k, row, dim_size);
        int64_t off = 0;
        for (std::size_t dd = 0; dd < r; ++dd) {
          const int64_t ii = (dd == static_cast<std::size_t>(d)) ? row : idx_nd[dd];
          off += ii * ss[dd];
        }
        po[flat] = ps[off];
      });
    });
  }

  if (is_grad_enabled() && src.requires_grad()) {
    auto node = std::make_shared<IndexSelectBackward>();
    node->parent_shape = src.shape();
    node->indices_saved = idx;  // contiguous, shared
    node->dim = d;
    node->next_edges = { autograd::edge_for(src), autograd::edge_for(indices) };
    autograd::attach_grad_fn(out, node);
  }

  graph::maybe_record("index_select", {&src, &indices}, {&out},
                      {{"dim", static_cast<int64_t>(d)}});
  return out;
}

Tensor gather(const Tensor& src, int64_t dim, const Tensor& indices) {
  TESSERACT_CHECK(src.defined(), "gather: src is undefined");
  TESSERACT_CHECK(indices.defined(), "gather: indices is undefined");
  TESSERACT_CHECK(indices.dtype() == DType::Int64,
                  "gather: indices must be Int64, got {}",
                  dtype_name(indices.dtype()));
  TESSERACT_CHECK(indices.rank() == src.rank(),
                  "gather: indices.rank ({}) must equal src.rank ({})",
                  indices.rank(), src.rank());
  TESSERACT_CHECK(src.device() == indices.device(),
                  "gather: device mismatch ({} vs {})",
                  src.device().to_string(), indices.device().to_string());
  const std::size_t r = src.shape().rank();
  TESSERACT_CHECK(r > 0, "gather: src must have rank >= 1");
  const int64_t d = normalize_dim(dim, r, "gather");

  // Every non-`dim` axis of `indices` must fit inside the corresponding
  // axis of `src`; along `dim` the output shape follows `indices`.
  for (std::size_t k = 0; k < r; ++k) {
    if (k == static_cast<std::size_t>(d)) continue;
    TESSERACT_CHECK(indices.shape()[k] <= src.shape()[k],
                    "gather: indices.shape[{}] = {} exceeds src.shape[{}] = {}",
                    k, indices.shape()[k], k, src.shape()[k]);
  }

  const Shape out_shape = indices.shape();
  Tensor out = Tensor::empty(out_shape, src.dtype(), src.device());

  const Tensor idx = indices.is_contiguous() ? indices : indices.contiguous();
  const Shape& istr = idx.strides();
  const Shape& ss = src.strides();
  const int64_t src_dim = src.shape()[static_cast<std::size_t>(d)];

  if (src.device().is_cuda()) {
    // Host-side range check on the full index tensor. Matches the
    // CPU loop's per-element TESSERACT_CHECK. For gather this is the
    // entire `indices` numel, which is typically bounded (token
    // tables, attention lookups) but we pay the host-bounce rather
    // than rely on an in-kernel assert. If this ever becomes a hot
    // path we can move it to a small device-side bounds kernel.
    const int64_t idx_numel = idx.numel();
    std::vector<std::byte> idx_host(static_cast<std::size_t>(idx_numel) * sizeof(int64_t));
    Storage::copy_device_bytes(idx_host.data(), cpu_device(),
                               idx.raw_data(), idx.device(),
                               idx_host.size());
    const auto* pidx_h = reinterpret_cast<const int64_t*>(idx_host.data());
    for (int64_t k = 0; k < idx_numel; ++k) {
      TESSERACT_CHECK(pidx_h[k] >= 0 && pidx_h[k] < src_dim,
                      "gather: indices[{}] = {} out of range [0, {})",
                      k, pidx_h[k], src_dim);
    }

    const Shape out_strides_contig = out_shape.contiguous_strides();
    Stream s = current_stream(src.device());
    cuda::detail::launch_gather(
        src.dtype(), src.device().index,
        static_cast<int>(r),
        static_cast<int>(d),
        out_shape.data(),
        ss.data(),
        istr.data(),
        out_strides_contig.data(),
        src.raw_data(),
        idx.data_ptr<int64_t>(),
        out.raw_data(),
        s.native_handle());
  } else {
    const int64_t* pidx = idx.data_ptr<int64_t>();
    dispatch_numeric(src.dtype(), [&]<typename T>() {
      const T* ps = src.data_ptr<T>();
      T* po = out.data_ptr<T>();
      detail::for_each_index(out_shape,
                             [&](int64_t flat, const detail::IndexArray& idx_nd) {
        int64_t idx_off = 0;
        for (std::size_t dd = 0; dd < r; ++dd) idx_off += idx_nd[dd] * istr[dd];
        const int64_t sel = pidx[idx_off];
        TESSERACT_CHECK(sel >= 0 && sel < src_dim,
                        "gather: indices[{}] = {} out of range [0, {})",
                        idx_off, sel, src_dim);
        int64_t off = 0;
        for (std::size_t dd = 0; dd < r; ++dd) {
          const int64_t ii = (dd == static_cast<std::size_t>(d)) ? sel : idx_nd[dd];
          off += ii * ss[dd];
        }
        po[flat] = ps[off];
      });
    });
  }

  if (is_grad_enabled() && src.requires_grad()) {
    auto node = std::make_shared<GatherBackward>();
    node->parent_shape = src.shape();
    node->indices_saved = idx;
    node->dim = d;
    node->next_edges = { autograd::edge_for(src), autograd::edge_for(indices) };
    autograd::attach_grad_fn(out, node);
  }

  graph::maybe_record("gather", {&src, &indices}, {&out},
                      {{"dim", static_cast<int64_t>(d)}});
  return out;
}

}  // namespace tesseract::ops

#include "tesseract/core/Tensor.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <ostream>
#include <sstream>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/core/Allocator.hpp"
#include "tesseract/core/Storage.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/PinnedHostAllocator.hpp"
#include "tesseract/cuda/detail/Elementwise.hpp"
#include "tesseract/cuda/detail/Shape.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract {

namespace detail {
// Registered by the autograd/ops layer (see ops::register_to_autograd_hook).
// A plain function pointer keeps `core` free of any autograd-graph types.
namespace {
ToAutogradHook g_to_autograd_hook = nullptr;
}  // namespace
void set_to_autograd_hook(ToAutogradHook hook) noexcept {
  g_to_autograd_hook = hook;
}
ToAutogradHook get_to_autograd_hook() noexcept { return g_to_autograd_hook; }
}  // namespace detail

// ============================ TensorImpl =================================== //

TensorImpl::TensorImpl() = default;

TensorImpl::~TensorImpl() = default;

TensorImpl::TensorImpl(std::shared_ptr<Storage> s, int64_t off, Shape sh, Shape st, DType dt,
                       Device dv)
    : storage(std::move(s)),
      storage_offset(off),
      shape(std::move(sh)),
      strides(std::move(st)),
      dtype(dt),
      device(dv) {
  TESSERACT_CHECK(shape.rank() == strides.rank(),
                  "shape rank {} != strides rank {}", shape.rank(), strides.rank());
}

std::size_t TensorImpl::byte_offset() const noexcept {
  return static_cast<std::size_t>(storage_offset) * dtype_size(dtype);
}

bool TensorImpl::is_contiguous() const noexcept {
  const auto r = shape.rank();
  if (r == 0) return true;
  int64_t expected = 1;
  for (std::size_t i = r; i-- > 0;) {
    // Size-1 dims have unconstrained stride (PyTorch convention).
    if (shape[i] != 1 && strides[i] != expected) return false;
    expected *= shape[i];
  }
  return true;
}

// ============================ Helpers ====================================== //

namespace {

// Strided tensor iteration helper. Invokes `fn(strided_offset_in_elements,
// flat_index)` for every logical position in `shape`.
template <typename F>
void for_each_position(const Shape& shape, const Shape& strides, F&& fn) {
  const std::size_t rank = shape.rank();
  if (rank == 0) {
    fn(int64_t{0}, int64_t{0});
    return;
  }
  std::array<int64_t, Shape::kMaxRank> idx{};
  const int64_t n = shape.numel();
  for (int64_t flat = 0; flat < n; ++flat) {
    int64_t offset = 0;
    for (std::size_t d = 0; d < rank; ++d) offset += idx[d] * strides[d];
    fn(offset, flat);
    for (std::size_t d = rank; d-- > 0;) {
      if (++idx[d] < shape[d]) break;
      idx[d] = 0;
      if (d == 0) return;  // iteration complete
    }
  }
}

std::shared_ptr<TensorImpl> make_empty_impl(Shape shape, DType dtype, Device device) {
  TESSERACT_CHECK(dtype_is_implemented(dtype), "dtype {} is not yet implemented",
                  dtype_name(dtype));
  for (std::size_t i = 0; i < shape.rank(); ++i) {
    TESSERACT_CHECK(shape[i] >= 0, "shape dim {} is negative ({})", i, shape[i]);
  }
  Allocator* alloc = default_allocator_for(device);
  const int64_t numel = shape.numel();
  const std::size_t nbytes = static_cast<std::size_t>(numel) * dtype_size(dtype);
  auto storage = Storage::make_owning(nbytes, alloc);
  Shape strides = shape.contiguous_strides();
  return std::make_shared<TensorImpl>(std::move(storage), 0, shape, strides, dtype, device);
}

// Dispatch a templated lambda over the currently implemented dtypes.
template <typename F>
void dispatch_dtype(DType dt, F&& f) {
  switch (dt) {
    case DType::Float32:  f.template operator()<float>(); return;
    case DType::Float64:  f.template operator()<double>(); return;
    case DType::Float16:  f.template operator()<Half>(); return;
    case DType::BFloat16: f.template operator()<BFloat16>(); return;
    case DType::Int32:    f.template operator()<int32_t>(); return;
    case DType::Int64:    f.template operator()<int64_t>(); return;
    case DType::Int8:     f.template operator()<int8_t>(); return;
    case DType::Bool:     f.template operator()<bool>(); return;
    default:
      TESSERACT_THROW("dtype {} is not handled in dispatch_dtype", dtype_name(dt));
  }
}

}  // namespace

// ============================ Factories ==================================== //

Tensor Tensor::empty(Shape shape, DType dtype, Device device) {
  return Tensor(make_empty_impl(std::move(shape), dtype, device));
}

Tensor Tensor::empty_pinned(Shape shape, DType dtype) {
  TESSERACT_CHECK(dtype_is_implemented(dtype),
                  "dtype {} is not yet implemented", dtype_name(dtype));
  for (std::size_t i = 0; i < shape.rank(); ++i) {
    TESSERACT_CHECK(shape[i] >= 0,
                    "shape dim {} is negative ({})", i, shape[i]);
  }
  // Pinned storage reports CPU device identity — that's the whole
  // point (every downstream code path keeps treating this as a CPU
  // tensor). The only difference from `Tensor::empty(..., cpu)` is
  // the allocator we route the storage through; cudaHostAlloc is
  // called inside `PinnedHostAllocator::allocate`.
  Allocator* alloc = cuda::pinned_host_allocator();
  const int64_t numel = shape.numel();
  const std::size_t nbytes =
      static_cast<std::size_t>(numel) * dtype_size(dtype);
  auto storage = Storage::make_owning(nbytes, alloc);
  Shape strides = shape.contiguous_strides();
  return Tensor(std::make_shared<TensorImpl>(
      std::move(storage), 0, shape, strides, dtype, cpu_device()));
}

Tensor Tensor::zeros(Shape shape, DType dtype, Device device) {
  Tensor t = empty(std::move(shape), dtype, device);
  if (t.numel() > 0) {
    // Storage::zero_device_bytes picks the right backend (std::memset
    // for CPU, cudaMemset for CUDA) so the factory works uniformly
    // across devices. For every numeric dtype the bit-pattern 0 maps
    // to the value 0, which is the only fill this helper supports;
    // non-zero `full` goes through the slower `fill_` path below.
    Storage::zero_device_bytes(t.raw_data(), t.device(), t.nbytes());
  }
  return t;
}

Tensor Tensor::ones(Shape shape, DType dtype, Device device) {
  Tensor t = empty(std::move(shape), dtype, device);
  t.fill_(1.0);
  return t;
}

Tensor Tensor::full(Shape shape, double value, DType dtype, Device device) {
  Tensor t = empty(std::move(shape), dtype, device);
  t.fill_(value);
  return t;
}

Tensor Tensor::arange(int64_t end, DType dtype, Device device) {
  return arange(0, end, 1, dtype, device);
}

Tensor Tensor::arange(int64_t start, int64_t end, int64_t step, DType dtype, Device device) {
  TESSERACT_CHECK(step != 0, "arange: step must be non-zero");
  const int64_t n = (step > 0) ? std::max<int64_t>(0, (end - start + step - 1) / step)
                               : std::max<int64_t>(0, (start - end - step - 1) / -step);

  // `arange` on CUDA is routed through a CPU scratch tensor + H→D copy
  // rather than launching a dedicated kernel. A CUDA arange kernel
  // lands with the elementwise suite in M2E; until then we prefer
  // this short-circuit because the allocation cost is tiny compared
  // to the `cudaMemcpy` it feeds into. The `copy=false` / same-device
  // short-circuit in `Tensor::to()` means no duplicate copy when the
  // target is CPU.
  if (!device.is_cpu()) {
    return arange(start, end, step, dtype, cpu_device()).to(device);
  }

  Tensor t = empty({n}, dtype, device);
  int64_t v = start;
  dispatch_dtype(dtype, [&]<typename T>() {
    T* p = t.data_ptr<T>();
    for (int64_t i = 0; i < n; ++i) {
      p[i] = static_cast<T>(v);
      v += step;
    }
  });
  return t;
}

Tensor Tensor::from_blob(void* data, Shape shape, DType dtype, Device device) {
  TESSERACT_CHECK(data != nullptr || shape.numel() == 0, "from_blob: null data for non-empty shape");
  TESSERACT_CHECK(dtype_is_implemented(dtype), "dtype {} is not yet implemented",
                  dtype_name(dtype));
  const std::size_t nbytes = static_cast<std::size_t>(shape.numel()) * dtype_size(dtype);
  auto storage = Storage::make_borrowed(data, nbytes, device);
  Shape strides = shape.contiguous_strides();
  return Tensor(std::make_shared<TensorImpl>(std::move(storage), 0, shape, strides, dtype, device));
}

// ============================ Properties =================================== //

void Tensor::ensure_defined(const char* op) const {
  TESSERACT_CHECK(impl_ != nullptr, "Tensor::{}: tensor is undefined (default-constructed)", op);
}

const Shape& Tensor::shape() const {
  ensure_defined("shape");
  return impl_->shape;
}

const Shape& Tensor::strides() const {
  ensure_defined("strides");
  return impl_->strides;
}

DType Tensor::dtype() const {
  ensure_defined("dtype");
  return impl_->dtype;
}

Device Tensor::device() const {
  ensure_defined("device");
  return impl_->device;
}

int64_t Tensor::rank() const {
  ensure_defined("rank");
  return static_cast<int64_t>(impl_->shape.rank());
}

int64_t Tensor::numel() const {
  ensure_defined("numel");
  return impl_->shape.numel();
}

std::size_t Tensor::nbytes() const {
  ensure_defined("nbytes");
  return static_cast<std::size_t>(impl_->shape.numel()) * dtype_size(impl_->dtype);
}

std::size_t Tensor::itemsize() const {
  ensure_defined("itemsize");
  return dtype_size(impl_->dtype);
}

bool Tensor::is_contiguous() const {
  ensure_defined("is_contiguous");
  return impl_->is_contiguous();
}

void* Tensor::raw_data() {
  ensure_defined("raw_data");
  return static_cast<std::byte*>(impl_->storage->data()) + impl_->byte_offset();
}

const void* Tensor::raw_data() const {
  ensure_defined("raw_data");
  return static_cast<const std::byte*>(impl_->storage->data()) + impl_->byte_offset();
}

std::shared_ptr<Storage> Tensor::storage() const {
  ensure_defined("storage");
  return impl_->storage;
}

int64_t Tensor::storage_offset() const {
  ensure_defined("storage_offset");
  return impl_->storage_offset;
}

// ============================ Shape ops ==================================== //

Tensor Tensor::view(Shape new_shape) const {
  ensure_defined("view");
  TESSERACT_CHECK(is_contiguous(),
                  "view() requires a contiguous tensor; use reshape() for non-contiguous inputs");
  TESSERACT_CHECK(new_shape.numel() == numel(),
                  "view: numel mismatch {} -> {}", numel(), new_shape.numel());
  Shape new_strides = new_shape.contiguous_strides();
  return Tensor(std::make_shared<TensorImpl>(impl_->storage, impl_->storage_offset,
                                             std::move(new_shape), std::move(new_strides),
                                             impl_->dtype, impl_->device));
}

Tensor Tensor::reshape(Shape new_shape) const {
  ensure_defined("reshape");
  if (is_contiguous()) return view(std::move(new_shape));
  return contiguous().view(std::move(new_shape));
}

Tensor Tensor::permute(std::span<const int64_t> axes) const {
  ensure_defined("permute");
  const auto r = static_cast<int64_t>(impl_->shape.rank());
  TESSERACT_CHECK(static_cast<int64_t>(axes.size()) == r,
                  "permute: axes size {} != tensor rank {}", axes.size(), r);
  std::array<bool, Shape::kMaxRank> seen{};
  Shape new_shape;
  Shape new_strides;
  new_shape.resize(r);
  new_strides.resize(r);
  for (int64_t i = 0; i < r; ++i) {
    const int64_t a = axes[i];
    TESSERACT_CHECK(a >= 0 && a < r, "permute: axis {} out of range [0,{})", a, r);
    TESSERACT_CHECK(!seen[a], "permute: axis {} appears more than once", a);
    seen[a] = true;
    new_shape[i] = impl_->shape[a];
    new_strides[i] = impl_->strides[a];
  }
  return Tensor(std::make_shared<TensorImpl>(impl_->storage, impl_->storage_offset,
                                             std::move(new_shape), std::move(new_strides),
                                             impl_->dtype, impl_->device));
}

Tensor Tensor::permute(std::initializer_list<int64_t> axes) const {
  return permute(std::span<const int64_t>(axes.begin(), axes.size()));
}

Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t len) const {
  ensure_defined("narrow");
  const int64_t r = rank();
  TESSERACT_CHECK(dim >= 0 && dim < r,
                  "narrow: dim {} out of range [0, {})", dim, r);
  const int64_t dim_size = impl_->shape[dim];
  TESSERACT_CHECK(start >= 0 && len >= 0 && start + len <= dim_size,
                  "narrow: range [{}, {}) out of bounds for dim {} with "
                  "size {}", start, start + len, dim, dim_size);

  Shape new_shape = impl_->shape;
  new_shape[dim] = len;
  Shape new_strides = impl_->strides;

  // Byte-level offset advance: `start * strides[dim] * itemsize`. We do
  // this at element granularity (in units of `strides[dim]`) and let
  // `byte_offset()` do the itemsize multiply when forming a raw pointer.
  const int64_t elem_off =
      impl_->storage_offset + start * impl_->strides[dim];
  return Tensor(std::make_shared<TensorImpl>(impl_->storage, elem_off,
                                             std::move(new_shape),
                                             std::move(new_strides),
                                             impl_->dtype, impl_->device));
}

Tensor Tensor::transpose(int64_t dim_a, int64_t dim_b) const {
  ensure_defined("transpose");
  const int64_t r = rank();
  TESSERACT_CHECK(dim_a >= 0 && dim_a < r && dim_b >= 0 && dim_b < r,
                  "transpose: dims ({},{}) out of range for rank {}", dim_a, dim_b, r);
  std::array<int64_t, Shape::kMaxRank> axes{};
  for (int64_t i = 0; i < r; ++i) axes[static_cast<size_t>(i)] = i;
  std::swap(axes[static_cast<size_t>(dim_a)], axes[static_cast<size_t>(dim_b)]);
  return permute(std::span<const int64_t>(axes.data(), static_cast<size_t>(r)));
}

Tensor Tensor::contiguous() const {
  ensure_defined("contiguous");
  if (is_contiguous()) return *this;

  Tensor out = Tensor::empty(impl_->shape, impl_->dtype, impl_->device);
  const auto& shape = impl_->shape;
  const auto& strides = impl_->strides;

  // CUDA path (M2H): hand the strided→dense copy to the CUDA bridge.
  // `launch_strided_copy` is dtype-generic (dispatches on itemsize),
  // so every dtype the core layer exposes — including Half / BFloat16
  // / Bool / Int8 — flows through the same kernel. A stride of 0 on
  // the source corresponds to a broadcast-like read, which the kernel
  // handles uniformly; our `contiguous()` case always ships non-zero
  // source strides because the forward shape is the tensor's own
  // shape.
  if (impl_->device.is_cuda()) {
    const Shape out_strides = impl_->shape.contiguous_strides();
    Stream s = current_stream(impl_->device);
    // Empty tensors (numel==0) are handled by the launcher's early-out,
    // but we still allocate the output above so callers see a
    // well-formed empty tensor (same semantics as the CPU path).
    cuda::detail::launch_strided_copy(
        impl_->dtype, impl_->device.index,
        static_cast<int>(shape.rank()),
        shape.data(), strides.data(), out_strides.data(),
        this->raw_data(), out.raw_data(),
        s.native_handle());
    return out;
  }

  dispatch_dtype(impl_->dtype, [&]<typename T>() {
    const T* src = this->data_ptr<T>();
    T* dst = out.data_ptr<T>();
    for_each_position(shape, strides, [&](int64_t src_off, int64_t dst_flat) {
      dst[dst_flat] = src[src_off];
    });
  });
  return out;
}

Tensor Tensor::clone() const {
  ensure_defined("clone");
  if (is_contiguous()) {
    Tensor out = Tensor::empty(impl_->shape, impl_->dtype, impl_->device);
    // Same-device byte-blit regardless of whether we're CPU↔CPU or
    // CUDA↔CUDA. Going through the HAL helper lets clone() work on
    // CUDA tensors without a kernel (cudaMemcpy(DeviceToDevice) is a
    // DMA, not a kernel launch).
    Storage::copy_device_bytes(out.raw_data(), out.device(),
                               this->raw_data(), this->device(), nbytes());
    return out;
  }
  return contiguous();
}

void Tensor::fill_(double value) {
  ensure_defined("fill_");
  TESSERACT_CHECK(is_contiguous(), "fill_ currently requires a contiguous tensor");
  const int64_t n = numel();
  if (n == 0) return;

  // CUDA path: route to the M2E fill kernel for dtypes the kernel
  // supports (Float32 / Float64 / Int32 / Int64 / Bool). For Half /
  // BFloat16 the kernel still throws in M2E (those land with M2G's
  // cuBLASLt integration), so we fall back to the pre-M2E host-
  // scratch path to keep `Tensor::ones({..}, Float16, cuda)` working
  // for the FP16 storage round-trip tests. The scratch path is
  // O(N) host memory per call and is NOT a perf target — it exists
  // purely so users can still build Half / BFloat16 tensors on the
  // GPU side while M2G is pending.
  if (impl_->device.is_cuda()) {
    const bool kernel_supports =
        impl_->dtype == DType::Float32 || impl_->dtype == DType::Float64 ||
        impl_->dtype == DType::Int32   || impl_->dtype == DType::Int64 ||
        impl_->dtype == DType::Bool;
    if (kernel_supports) {
      Stream s = current_stream(impl_->device);
      cuda::detail::launch_fill(impl_->dtype, impl_->device.index,
                                static_cast<std::size_t>(n),
                                this->raw_data(), value,
                                s.native_handle());
      return;
    }
    const std::size_t total_bytes = static_cast<std::size_t>(n) * itemsize();
    std::vector<std::byte> scratch(total_bytes);
    dispatch_dtype(impl_->dtype, [&]<typename T>() {
      T* p = reinterpret_cast<T*>(scratch.data());
      const T v = static_cast<T>(value);
      for (int64_t i = 0; i < n; ++i) p[i] = v;
    });
    Storage::copy_device_bytes(this->raw_data(), impl_->device,
                               scratch.data(), cpu_device(), total_bytes);
    return;
  }

  dispatch_dtype(impl_->dtype, [&]<typename T>() {
    T* p = this->data_ptr<T>();
    const T v = static_cast<T>(value);
    for (int64_t i = 0; i < n; ++i) p[i] = v;
  });
}

Tensor Tensor::to(Device target_device) const {
  ensure_defined("to");

  // Same-device: share storage, return a fresh handle. This is the
  // expected PyTorch behavior and lets `.to(device)` be dropped in
  // without allocating when the tensor is already where the caller
  // wants it. Deliberately returns `*this` rather than reconstructing
  // a new TensorImpl — same-device `.to()` must preserve identity so
  // view-based bookkeeping (autograd leaves etc.) stays intact.
  if (target_device == impl_->device) return *this;

  // Materialize a contiguous source layout before the copy. On CPU
  // `contiguous()` either returns self (if already dense) or falls
  // into the CPU-only strided-gather fallback; that is the only
  // supported non-contiguous path in M2D. Going through a dense
  // intermediate also gives us a simple correctness invariant on
  // the destination: strides are always row-major contiguous.
  Tensor src = is_contiguous() ? *this : contiguous();

  Tensor dst = Tensor::empty(src.shape(), src.dtype(), target_device);
  if (dst.numel() > 0) {
    Storage::copy_device_bytes(dst.raw_data(), target_device,
                               src.raw_data(), src.device(), dst.nbytes());
  }
  // Autograd: attach a CopyBackward grad-fn (gradient flows back to the
  // source device) via the registered hook. The hook itself re-checks
  // grad-mode and requires_grad, so this is a no-op for inference tensors.
  if (auto hook = detail::get_to_autograd_hook()) {
    hook(dst, *this);
  }
  return dst;
}

Tensor Tensor::to_async(Device target_device, const Stream& stream) const {
  ensure_defined("to_async");

  // Same-device short-circuit matches `to()`. No allocation, no
  // enqueue — the returned handle shares storage with self and is
  // observable immediately (no `stream.synchronize()` required by
  // the caller, which is what "same-device" should mean).
  if (target_device == impl_->device) return *this;

  // We stage through a contiguous source so strides on the
  // destination are trivially row-major (same invariant as `to()`).
  // A non-contiguous source makes `contiguous()` do a CPU-side
  // gather on the source device — that gather is synchronous;
  // async-ness only kicks in for the actual cross-device transfer
  // below. Documented on the public header so callers aware of the
  // latency pipeline can pre-contiguify.
  Tensor src = is_contiguous() ? *this : contiguous();

  // The destination allocator is the regular default for the target
  // device (e.g. CudaAllocator for a CUDA target). A caller who
  // wants the destination on pinned host memory would call
  // `empty_pinned` + `copy_device_bytes_async` directly; we keep
  // the default behavior aligned with `to()` to avoid surprises.
  Tensor dst = Tensor::empty(src.shape(), src.dtype(), target_device);
  if (dst.numel() > 0) {
    Storage::copy_device_bytes_async(
        dst.raw_data(), target_device,
        src.raw_data(), src.device(),
        dst.nbytes(), stream);
  }
  return dst;
}

void Tensor::move_to_(Device target_device) {
  ensure_defined("move_to_");
  if (impl_->device == target_device) return;

  // Produce the destination buffer via the existing (shape+dtype-preserving,
  // always-contiguous) `to()` path. The result lives in a throwaway
  // TensorImpl we steal fields out of — the important thing is that
  // `moved.impl_->storage` is a brand-new device allocation with the
  // materialized bytes, and we only need its fields (not its identity).
  Tensor moved = this->to(target_device);

  impl_->storage = std::move(moved.impl_->storage);
  impl_->storage_offset = moved.impl_->storage_offset;
  impl_->shape = std::move(moved.impl_->shape);
  impl_->strides = std::move(moved.impl_->strides);
  impl_->dtype = moved.impl_->dtype;
  impl_->device = moved.impl_->device;

  // The grad tensor (if any) is now on the wrong device. Leaf params that
  // are about to run a new backward pass will re-accumulate from scratch
  // anyway; intermediates shouldn't be moved at all (they get a fresh
  // grad_fn in the next forward). Dropping is the safe behaviour.
  if (impl_->autograd_meta) {
    impl_->autograd_meta->grad = Tensor{};
  }
}

// ============================ Pretty-print ================================= //

namespace {

template <typename T>
void append_scalar(std::string& out, T v) {
  if constexpr (std::is_same_v<T, bool>) {
    out += v ? "true" : "false";
  } else if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
    // Half / BFloat16 are user-defined types — fmt can't format them
    // directly but they convert implicitly to float. Going through float
    // also yields a stable textual width regardless of bits width.
    out += fmt::format("{:.4g}", static_cast<float>(v));
  } else if constexpr (std::is_floating_point_v<T>) {
    out += fmt::format("{:.4g}", v);
  } else {
    out += fmt::format("{}", v);
  }
}

}  // namespace

std::string Tensor::to_string(int max_elems) const {
  if (!defined()) return "Tensor(undefined)";

  std::string out = fmt::format("Tensor(shape={}, dtype={}, device={}",
                                impl_->shape.to_string(), dtype_name(impl_->dtype),
                                impl_->device.to_string());
  const int64_t n = numel();
  if (n == 0) {
    out += ", data=[])";
    return out;
  }

  out += ", data=[";
  // Pretty-print always walks a host-side contiguous buffer. For CUDA
  // tensors we materialize a CPU copy first so the dtype-dispatch
  // lambda below can dereference element pointers directly. This is
  // slow for large tensors; that's acceptable because `to_string` is a
  // diagnostic path (clamped by `max_elems`) — users who want raw
  // device views use `data_ptr<T>()` directly after `.to(cpu)`.
  Tensor c = impl_->device.is_cpu() ? contiguous() : this->to(cpu_device());
  const int64_t limit = std::min<int64_t>(n, max_elems);
  dispatch_dtype(impl_->dtype, [&]<typename T>() {
    const T* p = c.data_ptr<T>();
    for (int64_t i = 0; i < limit; ++i) {
      if (i > 0) out += ", ";
      append_scalar(out, p[i]);
    }
  });
  if (limit < n) out += fmt::format(", ... ({} more)", n - limit);
  out += "])";
  return out;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
  return os << t.to_string();
}

// ============================ Autograd accessors ============================ //

bool Tensor::requires_grad() const noexcept {
  if (!impl_) return false;
  const auto* am = impl_->autograd_meta.get();
  return am && (am->requires_grad || am->grad_fn);
}

void Tensor::set_requires_grad(bool value) {
  ensure_defined("set_requires_grad");
  auto& am = impl_->autograd_meta;
  if (!am) {
    if (!value) return;
    am = std::make_unique<AutogradMeta>();
  }
  TESSERACT_CHECK(am->grad_fn == nullptr,
                  "set_requires_grad called on a non-leaf tensor (it has a grad_fn)");
  am->requires_grad = value;
}

const Tensor& Tensor::grad() const {
  ensure_defined("grad");
  static const Tensor kEmpty;
  const auto* am = impl_->autograd_meta.get();
  if (!am) return kEmpty;
  return am->grad;
}

AutogradMeta* Tensor::mutable_autograd_meta() {
  ensure_defined("mutable_autograd_meta");
  if (!impl_->autograd_meta) impl_->autograd_meta = std::make_unique<AutogradMeta>();
  return impl_->autograd_meta.get();
}

const AutogradMeta* Tensor::autograd_meta() const noexcept {
  if (!impl_) return nullptr;
  return impl_->autograd_meta.get();
}

}  // namespace tesseract

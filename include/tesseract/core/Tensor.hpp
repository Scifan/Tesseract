#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Storage.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract {

// Forward declaration; full definition in tesseract/autograd/AutogradMeta.hpp
class AutogradMeta;

// Internal tensor state. Users should never hold a TensorImpl directly; they
// work with the value-type Tensor wrapper below.
//
// Non-copyable, non-movable: shared ownership is provided by the surrounding
// `std::shared_ptr<TensorImpl>` held by every Tensor handle.
struct TensorImpl {
  std::shared_ptr<Storage> storage;
  // Offset into the storage in ELEMENTS (not bytes). Views that do not start
  // at byte 0 set this.
  int64_t storage_offset{0};
  Shape shape;
  Shape strides;
  DType dtype{DType::Float32};
  Device device{};
  // Autograd data is attached lazily; nullptr when requires_grad() == false.
  // Destructor is defined out-of-line so this header does not need the full
  // definition of AutogradMeta.
  std::unique_ptr<AutogradMeta> autograd_meta;

  TensorImpl();
  TensorImpl(std::shared_ptr<Storage> s, int64_t off, Shape sh, Shape st, DType dt, Device dv);
  ~TensorImpl();

  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = delete;
  TensorImpl& operator=(TensorImpl&&) = delete;

  // Byte offset into the raw storage for the (0,0,...,0) element.
  std::size_t byte_offset() const noexcept;

  bool is_contiguous() const noexcept;
};

// Value-type tensor handle: cheap to copy, shared underlying state. A default
// constructed Tensor is "undefined" (defined() == false).
class Tensor {
 public:
  Tensor() = default;

  // Allocate a fresh contiguous tensor with uninitialized content.
  static Tensor empty(Shape shape, DType dtype = DType::Float32, Device device = cpu_device());

  // Wave 2.4 (B-011): allocate a contiguous CPU tensor backed by
  // page-locked host memory (via `cuda::PinnedHostAllocator`). The
  // resulting tensor reports `device() == cpu_device()` — only the
  // underlying byte source is pinned. That's what lets every op /
  // autograd / dispatch path keep working unchanged; the pinning
  // only matters at transfer time (`Tensor::to_async` / cuBLAS
  // staging / SafeTensors loader) where it unlocks real overlap
  // with `cudaMemcpyAsync`. Throws on CPU-only builds with the
  // "rebuild with -DTESSERACT_ENABLE_CUDA=ON" message.
  static Tensor empty_pinned(Shape shape, DType dtype = DType::Float32);

  static Tensor zeros(Shape shape, DType dtype = DType::Float32, Device device = cpu_device());
  static Tensor ones(Shape shape, DType dtype = DType::Float32, Device device = cpu_device());

  // Scalar fill: the value is converted to `dtype`.
  static Tensor full(Shape shape, double value, DType dtype = DType::Float32,
                     Device device = cpu_device());

  // `arange(end)` -> [0, 1, ..., end-1] of dtype Int64 by default.
  static Tensor arange(int64_t end, DType dtype = DType::Int64, Device device = cpu_device());
  static Tensor arange(int64_t start, int64_t end, int64_t step = 1,
                       DType dtype = DType::Int64, Device device = cpu_device());

  // Wrap a user-owned buffer. The caller MUST keep `data` alive for the
  // lifetime of the Tensor (and any view derived from it). No deep copy.
  static Tensor from_blob(void* data, Shape shape, DType dtype, Device device = cpu_device());

  // Construct from an initializer_list for quick testing / examples.
  template <typename T>
  static Tensor from_vector(const std::vector<T>& data, Shape shape);

  // Properties ---------------------------------------------------------------

  bool defined() const noexcept { return static_cast<bool>(impl_); }
  explicit operator bool() const noexcept { return defined(); }

  const Shape& shape() const;
  const Shape& strides() const;
  DType dtype() const;
  Device device() const;

  int64_t rank() const;
  int64_t numel() const;
  std::size_t nbytes() const;
  std::size_t itemsize() const;  // dtype_size

  bool is_contiguous() const;

  // Typed raw-pointer access to the (0,0,...,0) element of the view. Caller is
  // responsible for strides / bounds. Asserts that T matches dtype.
  template <typename T>
  T* data_ptr();
  template <typename T>
  const T* data_ptr() const;

  // Untyped byte pointer to the (0,...,0) element.
  void* raw_data();
  const void* raw_data() const;

  std::shared_ptr<Storage> storage() const;
  int64_t storage_offset() const;

  // Access to the underlying impl; for subsystems that need to attach metadata
  // (e.g. autograd). Prefer the public API whenever possible.
  const std::shared_ptr<TensorImpl>& impl() const noexcept { return impl_; }
  std::shared_ptr<TensorImpl>& impl() noexcept { return impl_; }

  // Shape / layout manipulation ---------------------------------------------

  // view() requires a contiguous underlying layout. The result shares storage.
  Tensor view(Shape new_shape) const;

  // reshape() returns a view when possible, otherwise copies.
  Tensor reshape(Shape new_shape) const;

  // Permute dimensions. `axes` must be a permutation of {0, ..., rank-1}.
  Tensor permute(std::initializer_list<int64_t> axes) const;
  Tensor permute(std::span<const int64_t> axes) const;

  // transpose(a, b) swaps dims a and b.
  Tensor transpose(int64_t dim_a, int64_t dim_b) const;

  // narrow(dim, start, len) returns a view sharing storage whose shape is
  // identical to self except that `dim` is sliced to `[start, start+len)`.
  // The stride along `dim` is preserved; only the storage_offset advances
  // by `start * strides[dim]` bytes. This is the primitive that the
  // KVCache (Wave 2.1) builds its `[B, H, 0..current_len, D_head]` view
  // on top of; it also unlocks future `chunk` / `split` helpers.
  //
  // `len == 0` is allowed and returns an empty tensor with the right
  // shape — handy for the "prefill nothing yet" KVCache case.
  Tensor narrow(int64_t dim, int64_t start, int64_t len) const;

  // Produce a row-major-contiguous copy if self is not contiguous; otherwise
  // returns *this.
  Tensor contiguous() const;

  // Deep copy: always allocates new storage.
  Tensor clone() const;

  // Move the tensor to `target_device`. If the tensor is already on
  // `target_device` the call is a zero-cost no-op and returns `*this`
  // (shallow copy of the handle; the same storage is shared). Otherwise
  // `to()` allocates fresh storage on the target device, performs a
  // synchronous device-aware byte copy, and returns a new tensor
  // owning the destination storage.
  //
  // Non-contiguous inputs are realized through `contiguous()` first so
  // the copy is always a single dense block. Tensors carrying autograd
  // metadata come back without it: `.to(device)` is a pure-data
  // operation in M2; autograd-aware cross-device copy with a
  // `CopyBackward` node is M4 multi-GPU work.
  Tensor to(Device target_device) const;

  // Asynchronous counterpart of `to(device)` (Wave 2.4 / B-011).
  //
  // Allocates the destination on `target_device` (via the default
  // allocator for that device) and enqueues a `cudaMemcpyAsync` on
  // `stream` — the call returns immediately after the copy is
  // submitted. The returned Tensor's storage is LIVE but its
  // contents are NOT yet valid from the caller's thread; the
  // caller must `stream.synchronize()` (or add an `Event` and
  // `wait` on a consumer stream) before reading from the host, and
  // must keep the source Tensor alive until the transfer has
  // completed.
  //
  // Effective async behavior (real overlap with compute on the
  // same device's compute stream) requires the **host** side of
  // the transfer to be pinned:
  //   * `src.to_async(cuda_dev, s)` where `src` came from
  //     `Tensor::empty_pinned` — truly async, GPU DMA engine
  //     streams directly from host DRAM.
  //   * `src.to_async(cuda_dev, s)` where `src` is a plain pageable
  //     tensor — CUDA silently degrades to a sync-equivalent path
  //     (correctness unchanged, overlap lost).
  // Tensor-level validation: source must be contiguous and the
  // stream must live on the CUDA-side device. Same-device copies
  // return `*this` (no allocation, no enqueue), matching the
  // identity semantics of `to()`.
  Tensor to_async(Device target_device, const class Stream& stream) const;

  // In-place device migration used by `nn::Module::to(Device)`. Unlike
  // `to(device)` (which returns a fresh Tensor handle backed by a new
  // TensorImpl), `move_to_` rewrites the fields of *this* TensorImpl to
  // point at the destination storage. Two implications:
  //
  //   1. Every Tensor handle that currently shares this impl (e.g. the
  //      `weight_` member of an `nn::Linear` and the same tensor stored
  //      inside `Module::params_`) observes the device change
  //      simultaneously, which is exactly what lets
  //      `model->to(cuda)` "just work" without every subclass
  //      overriding a `to()` of its own.
  //   2. The autograd metadata object itself is preserved — leaf
  //      flag and `requires_grad` survive — but any stale `.grad`
  //      from a prior CPU step is cleared, since it would be on the
  //      wrong device.
  //
  // No-op (returns self with identity preserved) when the tensor is
  // already on `target_device`.
  void move_to_(Device target_device);

  // Scalar write helpers used by factories / tests. `value` is converted to
  // the tensor's dtype. Tensor must be contiguous.
  void fill_(double value);

  // Pretty-print for diagnostics; NOT meant to be machine-readable.
  std::string to_string(int max_elems = 64) const;

  // Autograd --------------------------------------------------------------- //

  // Returns true iff autograd metadata exists AND the tensor is registered as
  // requiring gradient (either as a leaf or an intermediate with a grad_fn).
  bool requires_grad() const noexcept;

  // Set the leaf-level `requires_grad` flag. Allocates autograd metadata on
  // demand. Only valid on tensors WITHOUT a grad_fn (i.e. leaves); calling on
  // a non-leaf is a programmer error.
  void set_requires_grad(bool value);

  // Returns the accumulated gradient tensor. Undefined if none has been
  // accumulated (e.g. before the first backward pass, or if the tensor never
  // participated in a graph).
  const Tensor& grad() const;

  // Direct mutable access to autograd metadata. Used by ops wiring and the
  // backward engine. Allocates the metadata object on demand.
  AutogradMeta* mutable_autograd_meta();
  const AutogradMeta* autograd_meta() const noexcept;

 private:
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

  std::shared_ptr<TensorImpl> impl_;

  void ensure_defined(const char* op) const;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

// --- Autograd hook for cross-device copy ----------------------------------- //
//
// `Tensor::to(device)` must be differentiable for tensor/data parallelism
// (the backward of a cross-device copy is a copy of the gradient back to the
// source device). The autograd graph (`Node`) lives in the `autograd` layer,
// which depends on `core` — so `core` cannot reference it directly without a
// dependency cycle. Instead `core` exposes a function-pointer hook that the
// autograd/ops layer registers at static-init time; `Tensor::to` invokes it
// (when set, grad is enabled, and `src` requires grad) to attach the
// `CopyBackward` grad-fn to the freshly-copied output.
namespace detail {
using ToAutogradHook = void (*)(Tensor& out, const Tensor& src);
void set_to_autograd_hook(ToAutogradHook hook) noexcept;
ToAutogradHook get_to_autograd_hook() noexcept;
}  // namespace detail

// --- Template definitions -------------------------------------------------- //

template <typename T>
T* Tensor::data_ptr() {
  ensure_defined("data_ptr");
  TESSERACT_CHECK(impl_->dtype == CppTypeToDType<T>::value,
                  "data_ptr<T> dtype mismatch: tensor={}, T={}",
                  dtype_name(impl_->dtype), dtype_name(CppTypeToDType<T>::value));
  return reinterpret_cast<T*>(static_cast<std::byte*>(impl_->storage->data()) +
                              impl_->byte_offset());
}

template <typename T>
const T* Tensor::data_ptr() const {
  ensure_defined("data_ptr");
  TESSERACT_CHECK(impl_->dtype == CppTypeToDType<T>::value,
                  "data_ptr<T> dtype mismatch: tensor={}, T={}",
                  dtype_name(impl_->dtype), dtype_name(CppTypeToDType<T>::value));
  return reinterpret_cast<const T*>(static_cast<const std::byte*>(impl_->storage->data()) +
                                    impl_->byte_offset());
}

template <typename T>
Tensor Tensor::from_vector(const std::vector<T>& data, Shape shape) {
  const DType dt = CppTypeToDType<T>::value;
  Tensor t = Tensor::empty(std::move(shape), dt);
  TESSERACT_CHECK(t.numel() == static_cast<int64_t>(data.size()),
                  "from_vector: data.size()={} does not match shape numel={}",
                  data.size(), t.numel());
  std::memcpy(t.raw_data(), data.data(), data.size() * sizeof(T));
  return t;
}

}  // namespace tesseract

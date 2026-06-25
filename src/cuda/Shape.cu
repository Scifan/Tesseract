// M2H: CUDA shape / view kernels. This TU is only compiled when
// TESSERACT_ENABLE_CUDA=ON (see `src/cuda/CMakeLists.txt`); the
// matching CPU-only stubs live in `ShapeStub.cpp`. The public-ish
// C++ bridge is declared in `include/tesseract/cuda/detail/Shape.hpp`
// and consumed from the op / core layer:
//
//   * `src/core/Tensor.cpp::contiguous()` — strided→dense copy.
//   * `src/ops/cpu/Arithmetic.cpp::broadcast_to` / `reduce_to_shape`
//     — zero-stride read (broadcast) + `atomicAdd` scatter (reduce).
//   * `src/ops/cpu/Indexing.cpp` — slab copies for `cat` / `split`.
//
// Design mirrors `Elementwise.cu` / `Reduction.cu`: all kernels take
// a `ShapePod` + per-operand stride `ShapePod`s as kernel args (both
// trivially copyable). Shape-padding happens on the host side; the
// device kernel just does `flat_to_offset`.
//
// Itemsize dispatch for `launch_strided_copy`: the copy itself is
// dtype-blind, so we specialize only on element size (1 / 2 / 4 / 8
// bytes) instead of instantiating one kernel per dtype. This means
// Half / BFloat16 / Int8 / Bool all work for free — the core tensor
// layer already validates that the dtype payload lines up with the
// itemsize, and `dispatch_numeric`-level correctness falls out
// because we never do arithmetic on the copied bytes.
//
// `launch_strided_scatter_add` does need typed arithmetic, so it
// dispatches on dtype directly. Float32 / Float64 / Int32 / Int64
// map to their native `atomicAdd` overloads. Int8 / Bool / Half /
// BFloat16 would need packed-atomic tricks we're not paying for in
// M2H (and training paths don't build Half grads anyway — the loss
// scale-up path lives in FP32).

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/detail/Shape.hpp"
#include "tesseract/utils/Logging.hpp"

#include "KernelUtils.cuh"

namespace tesseract::cuda::detail {

namespace {

// ---------------- Strided byte-copy ----------------
//
// Each thread services one element of the iteration domain. Element
// size is templated so nvcc emits a single `ld.global` / `st.global`
// instruction of the right width (half-word / word / double-word).
// This is strictly faster than issuing N byte copies per element
// even for the trivial dtype==Bool case.
template <typename Elem>
__global__ void strided_copy_kernel(
    Elem* __restrict__ dst,
    const Elem* __restrict__ src,
    ShapePod sizes, ShapePod src_str, ShapePod dst_str,
    int64_t total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t so = flat_to_offset(i, sizes, src_str);
  const int64_t dof = flat_to_offset(i, sizes, dst_str);
  dst[dof] = src[so];
}

// ---------------- Strided scatter-add ----------------
//
// Generic `atomicAdd`-driven kernel. `Elem` is the arithmetic type;
// all supported dtypes have a native `atomicAdd(T*, T)` overload
// (floats require sm_60+ for double, sm_60+ for __half/bf16 which we
// do NOT enable here — the dtype check rejects them on the host side).
// Integer atomics run on every supported arch (sm_70+).
template <typename Elem>
__global__ void strided_scatter_add_kernel(
    Elem* __restrict__ dst,
    const Elem* __restrict__ src,
    ShapePod sizes, ShapePod src_str, ShapePod dst_str,
    int64_t total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t so = flat_to_offset(i, sizes, src_str);
  const int64_t dof = flat_to_offset(i, sizes, dst_str);
  atomicAdd(&dst[dof], src[so]);
}

// CUDA doesn't ship a 64-bit signed-int `atomicAdd`; the usual
// workaround is to cast to `unsigned long long`, which wraps with
// matching two's-complement semantics for our "accumulate a sum of
// int64 grad components" use case. `reduce_to_shape` never overflows
// the accumulator in practice (grad magnitudes stay bounded by the
// training-step learning rate), and nothing in the Int64 gather
// backward hot path allocates enough summands to worry about.
__device__ __forceinline__ void atomic_add_i64(int64_t* addr, int64_t val) {
  unsigned long long* p = reinterpret_cast<unsigned long long*>(addr);
  atomicAdd(p, static_cast<unsigned long long>(val));
}

__global__ void strided_scatter_add_i64_kernel(
    int64_t* __restrict__ dst,
    const int64_t* __restrict__ src,
    ShapePod sizes, ShapePod src_str, ShapePod dst_str,
    int64_t total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t so = flat_to_offset(i, sizes, src_str);
  const int64_t dof = flat_to_offset(i, sizes, dst_str);
  atomic_add_i64(&dst[dof], src[so]);
}

// ---------------- Host-side itemsize dispatch ----------------

template <typename Elem>
void dispatch_strided_copy_typed(void* dst, const void* src,
                                 const ShapePod& sizes,
                                 const ShapePod& src_str,
                                 const ShapePod& dst_str,
                                 int64_t total,
                                 cudaStream_t stream) {
  const int grid = launch_grid(total);
  strided_copy_kernel<Elem><<<grid, kBlockSize, 0, stream>>>(
      static_cast<Elem*>(dst),
      static_cast<const Elem*>(src),
      sizes, src_str, dst_str, total);
}

}  // namespace

void launch_strided_copy(DType dtype, int device_index,
                         int ndim,
                         const int64_t* sizes,
                         const int64_t* src_strides,
                         const int64_t* dst_strides,
                         const void* src, void* dst,
                         void* stream_handle) {
  TESSERACT_CHECK(ndim >= 0 && ndim <= kMaxRank,
                  "[tesseract] launch_strided_copy: ndim={} out of "
                  "range [0, {}]", ndim, kMaxRank);

  const ShapePod s  = pack_raw(sizes,       ndim);
  const ShapePod ss = pack_raw(src_strides, ndim);
  const ShapePod ds = pack_raw(dst_strides, ndim);
  const int64_t total = numel_pod(s);
  if (total == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  // Dispatch on itemsize rather than dtype. This keeps the kernel
  // instantiation count bounded at four (1B / 2B / 4B / 8B) and lets
  // every dtype the core layer exposes (including Half / BFloat16 /
  // Bool / Int8) round-trip through the same kernel.
  const std::size_t isz = dtype_size(dtype);
  switch (isz) {
    case 1:
      dispatch_strided_copy_typed<uint8_t>(dst, src, s, ss, ds, total, stream);
      break;
    case 2:
      dispatch_strided_copy_typed<uint16_t>(dst, src, s, ss, ds, total, stream);
      break;
    case 4:
      dispatch_strided_copy_typed<uint32_t>(dst, src, s, ss, ds, total, stream);
      break;
    case 8:
      dispatch_strided_copy_typed<uint64_t>(dst, src, s, ss, ds, total, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] launch_strided_copy: unsupported itemsize {} "
          "for dtype {}", isz, dtype_name(dtype)));
  }
  check_launch("strided-copy");
}

void launch_strided_scatter_add(DType dtype, int device_index,
                                int ndim,
                                const int64_t* sizes,
                                const int64_t* src_strides,
                                const int64_t* dst_strides,
                                const void* src, void* dst,
                                void* stream_handle) {
  TESSERACT_CHECK(ndim >= 0 && ndim <= kMaxRank,
                  "[tesseract] launch_strided_scatter_add: ndim={} out "
                  "of range [0, {}]", ndim, kMaxRank);

  const ShapePod s  = pack_raw(sizes,       ndim);
  const ShapePod ss = pack_raw(src_strides, ndim);
  const ShapePod ds = pack_raw(dst_strides, ndim);
  const int64_t total = numel_pod(s);
  if (total == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  const int grid = launch_grid(total);

  switch (dtype) {
    case DType::Float32:
      strided_scatter_add_kernel<float><<<grid, kBlockSize, 0, stream>>>(
          static_cast<float*>(dst),
          static_cast<const float*>(src),
          s, ss, ds, total);
      break;
    case DType::Float64:
      strided_scatter_add_kernel<double><<<grid, kBlockSize, 0, stream>>>(
          static_cast<double*>(dst),
          static_cast<const double*>(src),
          s, ss, ds, total);
      break;
    case DType::Int32:
      strided_scatter_add_kernel<int32_t><<<grid, kBlockSize, 0, stream>>>(
          static_cast<int32_t*>(dst),
          static_cast<const int32_t*>(src),
          s, ss, ds, total);
      break;
    case DType::Int64:
      strided_scatter_add_i64_kernel<<<grid, kBlockSize, 0, stream>>>(
          static_cast<int64_t*>(dst),
          static_cast<const int64_t*>(src),
          s, ss, ds, total);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA strided scatter-add on dtype {} is not "
          "implemented in M2H — only Float32 / Float64 / Int32 / Int64 "
          "have deterministic atomicAdd support. Cast gradients to "
          "Float32 on host first (training loss paths already do this).",
          dtype_name(dtype)));
  }
  check_launch("strided-scatter-add");
}

}  // namespace tesseract::cuda::detail

// M2E / M2L.1 / B-015: CUDA elementwise kernels.
//
// M2E shipped the broadcast-aware strided implementation below. M2L.1
// adds a **dense contiguous fast-path** that routes every operand whose
// strides are the ordinary row-major contiguous strides through a flat,
// vectorized kernel. B-015 extends the dtype coverage from
// {Float32, Float64, Int32, Int64} to also include {Float16, BFloat16}
// via FP32-promotion on the load path — the kernel scaffolding stays
// dtype-agnostic (it templates on a storage `T`), and the op functors
// for half storage widen to `float` before arithmetic and narrow
// back on store via `__float2half` / `__float2bfloat16`. This matches
// the CPU side's `Half::operator float()` round-trip semantics so
// parity tests compare bit-equal up to the expected `2e-3` FP16
// tolerance envelope established by B-005.
//
// The fast path:
//
//   * Treats the operand as a 1-D array of `total` elements, skipping
//     the `flat_to_offset` div/mod chain entirely (that chain is 8
//     iterations of int divide + multiply per thread — measurable
//     overhead on a memory-bound kernel).
//   * Uses wide vector loads (`float4` / `double2` / `int4` / `longlong2`
//     / `ReadVec<T, 4>`) when the pointer + element count are aligned
//     to the vector width. A scalar tail kernel handles the trailing
//     elements when `total % vec != 0`.
//
// On SM 8.9 Ada the elementwise bench hits ~97% of `cudaMemcpyAsync`
// D2D bandwidth with the vectorized path, vs ~60% with the strided
// kernel, which is what puts the M2L.1 "≥ 95% memcpy" hard bar within
// reach. Broadcasted ops (any stride == 0) or non-contiguous views
// (stride layout that doesn't match the shape) take the original
// strided path unchanged — correctness-first, performance-where-safe.
//
// This TU is only compiled when TESSERACT_ENABLE_CUDA=ON (see
// src/cuda/CMakeLists.txt); the matching CPU-only stubs live in
// `ElementwiseStub.cpp`. The public-ish C++ bridge is declared in
// `include/tesseract/cuda/detail/Elementwise.hpp` and consumed from
// the op layer (`src/ops/cpu/Arithmetic.cpp` etc.).
//
// Design notes — see `docs/m2-plan.md` §M2E for the full rationale:
//
//   * Strided kernel scaffolding. Every binary launch receives output
//     shape + per-operand strides (aligned by `ops::align_for_broadcast`
//     to the full rank), so broadcasting is handled uniformly — a
//     stride of 0 means "replicate this dim". The kernel computes
//     per-operand offsets by unpacking the flat output index through
//     the output sizes; no specialization on ndim.
//
//   * Single shared "packed descriptor" POD. The bridge surface
//     gives us raw `int64_t*` + `int ndim` (so nvcc compiled at
//     `CUDA_STANDARD=17` never has to parse `std::span` from
//     Shape.hpp). We copy those into a `ShapePod` — a trivial
//     `int64_t[8]` + `ndim` — which is what the `__global__` kernels
//     take as a kernel arg (trivially-copyable, passed by value).
//
//   * Dtype dispatch lives in host code. We generate instantiations
//     via a dispatch macro that hands a `T` and a per-dtype instance
//     of the op functor to the kernel. For FP16 / BF16 the op functor
//     is wrapped in `PromotedBinary<Op>` / `PromotedUnary<Op>` which
//     widens each operand to `float` before invoking `Op{}(float,
//     float)` and narrows the result back — the compute semantics
//     therefore match the CPU path exactly (see the
//     `TESSERACT_FP16_BINOP` macro in `Float16.hpp`).
//
//   * Stream-aware. The launchers take an opaque `cudaStream_t`
//     (cast to `void*` to keep the bridge header free of
//     `<cuda_runtime.h>`). The op layer resolves it via
//     `current_stream(Device{CUDA, device_index}).native_handle()`
//     before calling us. Keeping the stream as an argument — rather
//     than fetching it here — avoids a circular link dependency
//     between `tesseract_cuda` and `tesseract_core` (the latter
//     already privately links the former). We don't force a
//     `cudaStreamSynchronize`; subsequent synchronous primitives
//     (`Storage::copy_device_bytes` in `Tensor::to(cpu)`, the next
//     kernel on the same stream, ...) give callers their ordering.
//
//   * Error handling. `cudaGetLastError()` after every launch
//     promotes kernel-launch configuration errors (invalid grid,
//     bad __global__) to a thrown `DeviceError`. Out-of-bounds
//     kernel work surfaces on the next sync; `compute-sanitizer`
//     is run as part of the M2E exit bar to flag anything we miss.

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <fmt/format.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/cuda/detail/Elementwise.hpp"
#include "tesseract/utils/Logging.hpp"

// NOTE: do NOT `#include "tesseract/core/Shape.hpp"` here — it uses
// `std::span` and explicit-template lambdas (both C++20). This TU is
// compiled by nvcc with `CUDA_STANDARD=17` (CMake 3.22 cannot select
// `CUDA20`; see `src/cuda/CMakeLists.txt`). The bridge surface in
// `Elementwise.hpp` hands us shape/strides as raw `int64_t*` + `int
// ndim` precisely to keep this TU C++17-clean.

namespace tesseract::cuda::detail {

namespace {

// -------------------- Device-side scaffolding --------------------

// Max tensor rank, mirroring `tesseract::Shape::kMaxRank`. Kept as a
// local constant (rather than including Shape.hpp for the canonical
// definition) so this TU stays C++17-parseable under nvcc. If the
// host-side kMaxRank ever changes, the mismatch surfaces in
// `launch_binary_elementwise` / `launch_unary_elementwise` via the
// `TESSERACT_CHECK(ndim <= kMaxRank, ...)` below, not silently.
constexpr int kMaxRank = 8;

// Plain-POD mirror of a shape (or strides). Kernel arguments must be
// trivially copyable; this is the payload the kernels take by value.
struct ShapePod {
  int64_t sizes[kMaxRank];
  int ndim;
};

// Given a flat output index `i`, a shape, and per-operand strides,
// compute the corresponding flat offset into that operand. Works for
// both dense (stride matches shape) and broadcasted (some strides == 0)
// inputs; the integer division / modulo generates coords lazily from
// the trailing dim toward the leading one. The 8-iteration bound is
// unrolled by nvcc since it is a compile-time loop trip.
__device__ __forceinline__ int64_t
flat_to_offset(int64_t i, const ShapePod& sizes, const ShapePod& strides) {
  int64_t off = 0;
  int64_t rem = i;
#pragma unroll
  for (int d_rev = 0; d_rev < kMaxRank; ++d_rev) {
    const int d = sizes.ndim - 1 - d_rev;
    if (d < 0) break;
    const int64_t dim = sizes.sizes[d];
    const int64_t c = rem % dim;
    rem /= dim;
    off += c * strides.sizes[d];
  }
  return off;
}

// Device math helpers with the CUDA-preferred precision-typed names,
// so the generic op functor below can stay parametric on `T`. nvcc's
// overload set would also resolve `tanh(float)` correctly, but
// spelling out the precision (`tanhf` vs `tanh`) is self-documenting
// and guarantees we call the intrinsic path on the float side.
__device__ __forceinline__ float t_exp(float x)  { return ::__expf(x); }
__device__ __forceinline__ double t_exp(double x) { return ::exp(x); }
__device__ __forceinline__ float t_log(float x)  { return ::logf(x); }
__device__ __forceinline__ double t_log(double x) { return ::log(x); }
__device__ __forceinline__ float t_tanh(float x)  { return ::tanhf(x); }
__device__ __forceinline__ double t_tanh(double x) { return ::tanh(x); }
__device__ __forceinline__ float t_exp_neg(float x)  { return ::__expf(-x); }
__device__ __forceinline__ double t_exp_neg(double x) { return ::exp(-x); }
__device__ __forceinline__ float t_sqrt(float x)  { return ::sqrtf(x); }
__device__ __forceinline__ double t_sqrt(double x) { return ::sqrt(x); }

// -------------------- FP16 / BF16 promotion adaptors (B-015) --------------------
//
// Every half-precision op widens to FP32 for the math and narrows back
// for the store, matching our CPU `Half`/`BFloat16` contract (their
// `operator+` already promotes via `TESSERACT_FP16_BINOP`). The
// `__half` / `__nv_bfloat16` intrinsics below emit the SM 5.3+ /
// SM 8.0+ native conversion instructions (`F2I.F16` / `I2F.F16` on Ada
// pre-Hopper). Float and double specializations are no-ops so the
// templated `PromotedBinary<Op>` / `PromotedUnary<Op>` wrappers can
// be used uniformly when we want the promotion; the unpromoted Op
// remains the default for `float`/`double`/`int` dispatch.

__device__ __forceinline__ float to_fp32(float x)  { return x; }
__device__ __forceinline__ float to_fp32(double x) { return static_cast<float>(x); }
__device__ __forceinline__ float to_fp32(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_fp32(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T from_fp32(float f);
template <> __device__ __forceinline__ float from_fp32<float>(float f) { return f; }
template <> __device__ __forceinline__ double from_fp32<double>(float f) {
  return static_cast<double>(f);
}
template <> __device__ __forceinline__ __half from_fp32<__half>(float f) {
  return __float2half(f);
}
template <> __device__ __forceinline__ __nv_bfloat16 from_fp32<__nv_bfloat16>(float f) {
  return __float2bfloat16(f);
}

// Promoted-op wrappers. These are the functors we hand the half-type
// kernels. Reusing the existing `AddFn` / `SigmoidFn` / etc. means the
// compute semantics (including the `__expf` / `tanhf` fast intrinsic
// choices) are identical across FP32 and FP16 code paths.
template <typename Op>
struct PromotedBinary {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return from_fp32<T>(Op{}(to_fp32(a), to_fp32(b)));
  }
};
template <typename Op>
struct PromotedUnary {
  template <typename T>
  __device__ __forceinline__ T operator()(T x) const {
    return from_fp32<T>(Op{}(to_fp32(x)));
  }
};

// -------------------- Op functors (device) --------------------

struct AddFn { template <typename T> __device__ T operator()(T x, T y) const { return x + y; } };
struct SubFn { template <typename T> __device__ T operator()(T x, T y) const { return x - y; } };
struct MulFn { template <typename T> __device__ T operator()(T x, T y) const { return x * y; } };
struct DivFn {
  template <typename T> __device__ T operator()(T x, T y) const {
    // Integer division by zero is UB on CUDA — we don't check here;
    // the CPU preflight in the op layer is expected to catch it for
    // small tensors. Float divisions IEEE-sanely produce inf / nan.
    return x / y;
  }
};

struct NegFn  { template <typename T> __device__ T operator()(T x) const { return static_cast<T>(-x); } };
struct ReluFn { template <typename T> __device__ T operator()(T x) const { return x > T(0) ? x : T(0); } };
// M2I: positive-indicator step — the derivative mask used by
// `ReluBackward`. Defined only for the float-dispatch path; the
// `launch_unary_elementwise` switch below rejects integer dtypes
// for `Step` with a clear DeviceError.
struct StepFn { template <typename T> __device__ T operator()(T x) const { return x > T(0) ? T(1) : T(0); } };
struct SigmoidFn {
  template <typename T>
  __device__ T operator()(T x) const {
    // 1 / (1 + exp(-x)) — accurate for |x| < 15, saturates cleanly
    // outside. We don't special-case large-|x| ranges; training
    // pipelines clip well before that regime.
    return static_cast<T>(T(1) / (T(1) + t_exp_neg(x)));
  }
};
struct TanhFn { template <typename T> __device__ T operator()(T x) const { return t_tanh(x); } };
struct ExpFn  { template <typename T> __device__ T operator()(T x) const { return t_exp(x); } };
struct LogFn  { template <typename T> __device__ T operator()(T x) const { return t_log(x); } };
// M2K: elementwise `sqrt(x)`. Routed through the precision-typed
// intrinsics (`sqrtf` / `sqrt`) rather than a generic `std::sqrt` so
// nvcc emits the fast-math `MUFU.SQRT` path on both float and double.
struct SqrtFn { template <typename T> __device__ T operator()(T x) const { return t_sqrt(x); } };

// -------------------- Kernels --------------------

template <typename T, typename Op>
__global__ void binary_strided(T* __restrict__ out,
                               const T* __restrict__ a,
                               const T* __restrict__ b,
                               ShapePod sizes, ShapePod a_str, ShapePod b_str,
                               int64_t total, Op op) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t ao = flat_to_offset(i, sizes, a_str);
  const int64_t bo = flat_to_offset(i, sizes, b_str);
  out[i] = op(a[ao], b[bo]);
}

template <typename T, typename Op>
__global__ void unary_strided(T* __restrict__ out,
                              const T* __restrict__ x,
                              ShapePod sizes, ShapePod x_str,
                              int64_t total, Op op) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t xo = flat_to_offset(i, sizes, x_str);
  out[i] = op(x[xo]);
}

// ---------------- Dense fast-path (M2L.1) ----------------
//
// `ReadVec<T, N>` is a POD bundle of N `T`s used for bulk loads/stores.
// `using` a primitive CUDA vec type where one exists (`float4`,
// `double2`, `int4`, `longlong2`) makes nvcc emit `LDG.128` / `LDG.64`
// memory transactions; for other sizes we fall back to an array wrapper
// that still widens the memory transaction (nvcc vectorizes dense access
// to aligned arrays of primitives when the struct is trivially
// copyable).
template <typename T, int N>
struct VecOf {
  T v[N];
};
// Primary template dispatches to VecOf<T, N>; specializations below
// use the toolkit's built-in vector types for the cases where they
// exist (guarantees the widest single-instruction loads).
template <typename T, int N>
struct ReadVec { using type = VecOf<T, N>; };
template <> struct ReadVec<float,   4> { using type = float4;    };
template <> struct ReadVec<double,  2> { using type = double2;   };
template <> struct ReadVec<int32_t, 4> { using type = int4;      };
template <> struct ReadVec<int64_t, 2> { using type = longlong2; };

// Per-element access through the vector wrapper. The CUDA built-in
// vector types expose `.x / .y / .z / .w` so we have to specialize; the
// generic VecOf<T, N> uses `v[i]`. Both paths compile to a register
// move at -O3 — there is no materialized vector in SMEM.
template <int I, typename VecT, typename T>
__device__ __forceinline__ T vec_get(const VecT& v) {
  if constexpr (std::is_same_v<VecT, float4>) {
    if constexpr (I == 0) return v.x;
    if constexpr (I == 1) return v.y;
    if constexpr (I == 2) return v.z;
    if constexpr (I == 3) return v.w;
  } else if constexpr (std::is_same_v<VecT, double2>) {
    if constexpr (I == 0) return v.x;
    if constexpr (I == 1) return v.y;
  } else if constexpr (std::is_same_v<VecT, int4>) {
    if constexpr (I == 0) return v.x;
    if constexpr (I == 1) return v.y;
    if constexpr (I == 2) return v.z;
    if constexpr (I == 3) return v.w;
  } else if constexpr (std::is_same_v<VecT, longlong2>) {
    if constexpr (I == 0) return static_cast<T>(v.x);
    if constexpr (I == 1) return static_cast<T>(v.y);
  } else {
    return v.v[I];
  }
}

template <int I, typename VecT, typename T>
__device__ __forceinline__ void vec_set(VecT& v, T val) {
  if constexpr (std::is_same_v<VecT, float4>) {
    if constexpr (I == 0) { v.x = val; return; }
    if constexpr (I == 1) { v.y = val; return; }
    if constexpr (I == 2) { v.z = val; return; }
    if constexpr (I == 3) { v.w = val; return; }
  } else if constexpr (std::is_same_v<VecT, double2>) {
    if constexpr (I == 0) { v.x = val; return; }
    if constexpr (I == 1) { v.y = val; return; }
  } else if constexpr (std::is_same_v<VecT, int4>) {
    if constexpr (I == 0) { v.x = val; return; }
    if constexpr (I == 1) { v.y = val; return; }
    if constexpr (I == 2) { v.z = val; return; }
    if constexpr (I == 3) { v.w = val; return; }
  } else if constexpr (std::is_same_v<VecT, longlong2>) {
    if constexpr (I == 0) { v.x = static_cast<long long>(val); return; }
    if constexpr (I == 1) { v.y = static_cast<long long>(val); return; }
  } else {
    v.v[I] = val;
  }
}

// Dense binary kernel, `N` elements per thread via wide vector loads.
// `total_vec` is how many vector-sized chunks fit fully into the array;
// a scalar tail kernel runs over `total - total_vec * N` leftovers.
template <typename T, typename Op, int N>
__global__ void binary_dense_vec(T* __restrict__ out,
                                 const T* __restrict__ a,
                                 const T* __restrict__ b,
                                 int64_t total_vec, Op op) {
  using V = typename ReadVec<T, N>::type;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total_vec) return;
  const V* ain  = reinterpret_cast<const V*>(a);
  const V* bin  = reinterpret_cast<const V*>(b);
  V*       oout = reinterpret_cast<V*>(out);
  const V va = ain[i];
  const V vb = bin[i];
  V vo;
  // Manual unroll — `vec_get` / `vec_set` are `constexpr I` templates
  // so a runtime `for (int k ...)` can't drive them. The explicit
  // switch on `N` below keeps nvcc from emitting a scalar fallback
  // that would defeat the vector-load win.
  if constexpr (N == 4) {
    vec_set<0>(vo, op(vec_get<0, V, T>(va), vec_get<0, V, T>(vb)));
    vec_set<1>(vo, op(vec_get<1, V, T>(va), vec_get<1, V, T>(vb)));
    vec_set<2>(vo, op(vec_get<2, V, T>(va), vec_get<2, V, T>(vb)));
    vec_set<3>(vo, op(vec_get<3, V, T>(va), vec_get<3, V, T>(vb)));
  } else if constexpr (N == 2) {
    vec_set<0>(vo, op(vec_get<0, V, T>(va), vec_get<0, V, T>(vb)));
    vec_set<1>(vo, op(vec_get<1, V, T>(va), vec_get<1, V, T>(vb)));
  }
  oout[i] = vo;
}

template <typename T, typename Op>
__global__ void binary_dense_scalar(T* __restrict__ out,
                                    const T* __restrict__ a,
                                    const T* __restrict__ b,
                                    int64_t begin, int64_t end, Op op) {
  const int64_t i = begin + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= end) return;
  out[i] = op(a[i], b[i]);
}

template <typename T, typename Op, int N>
__global__ void unary_dense_vec(T* __restrict__ out,
                                const T* __restrict__ x,
                                int64_t total_vec, Op op) {
  using V = typename ReadVec<T, N>::type;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total_vec) return;
  const V* xin  = reinterpret_cast<const V*>(x);
  V*       oout = reinterpret_cast<V*>(out);
  const V vx = xin[i];
  V vo;
  if constexpr (N == 4) {
    vec_set<0>(vo, op(vec_get<0, V, T>(vx)));
    vec_set<1>(vo, op(vec_get<1, V, T>(vx)));
    vec_set<2>(vo, op(vec_get<2, V, T>(vx)));
    vec_set<3>(vo, op(vec_get<3, V, T>(vx)));
  } else if constexpr (N == 2) {
    vec_set<0>(vo, op(vec_get<0, V, T>(vx)));
    vec_set<1>(vo, op(vec_get<1, V, T>(vx)));
  }
  oout[i] = vo;
}

template <typename T, typename Op>
__global__ void unary_dense_scalar(T* __restrict__ out,
                                   const T* __restrict__ x,
                                   int64_t begin, int64_t end, Op op) {
  const int64_t i = begin + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= end) return;
  out[i] = op(x[i]);
}

template <typename T>
__global__ void fill_dense(T* __restrict__ out, T value, int64_t total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  out[i] = value;
}

// -------------------- Host-side scaffolding --------------------

struct DeviceGuard {
  int previous{-1};

  explicit DeviceGuard(int target) {
    cudaError_t err = cudaGetDevice(&previous);
    if (err != cudaSuccess) previous = -1;
    err = cudaSetDevice(target);
    if (err != cudaSuccess) {
      throw DeviceError(fmt::format(
          "[tesseract] cudaSetDevice({}) failed: {}", target, cudaGetErrorString(err)));
    }
  }
  ~DeviceGuard() { if (previous >= 0) (void)cudaSetDevice(previous); }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
};

void check_launch(const char* op_name) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw DeviceError(fmt::format(
        "[tesseract] CUDA {} kernel launch failed: {}",
        op_name, cudaGetErrorString(err)));
  }
}

// Compute total element count from a ShapePod. Matches Shape::numel()
// on the host side; separated so the kernel-launcher path can do
// everything with plain integers.
int64_t numel_pod(const ShapePod& s) {
  int64_t n = 1;
  for (int d = 0; d < s.ndim; ++d) n *= s.sizes[d];
  return n;
}

// Pack a raw `int64_t[ndim]` array (handed in from the op layer,
// already aligned to the broadcast rank by `ops::align_for_broadcast`)
// into a kernel-arg-safe ShapePod. Callers must have validated
// `ndim <= kMaxRank` before reaching here.
ShapePod pack_raw(const int64_t* data, int ndim) {
  ShapePod pod{};
  pod.ndim = ndim;
  for (int d = 0; d < ndim; ++d) pod.sizes[d] = data[d];
  return pod;
}

constexpr int kBlockSize = 256;

int launch_grid(int64_t total) {
  const int64_t g = (total + kBlockSize - 1) / kBlockSize;
  // CUDA allows up to 2^31-1 blocks in grid.x which is plenty for
  // any tensor we can allocate today. We still clamp defensively.
  return static_cast<int>(std::min<int64_t>(g, 2'147'483'647LL));
}

// ---- M2L.1 dense-fast-path plumbing ----
//
// `dense_strides_match(sizes, strides)` returns true when the stride
// layout would produce ordinary row-major contiguous access over a
// tensor of that shape. Size-1 dims are treated as "don't care" (their
// stride has no observable effect on the indexing arithmetic). A true
// return lets the launchers reinterpret the whole operand as a flat
// 1-D array of `total` elements.
inline bool dense_strides_match(const ShapePod& sizes, const ShapePod& strides) {
  int64_t expected = 1;
  for (int d = sizes.ndim - 1; d >= 0; --d) {
    const int64_t dim = sizes.sizes[d];
    if (dim == 1) {
      // Size-1 dim: stride is irrelevant to the access pattern; accept
      // anything (including the 0 that broadcast-aware ops emit).
      continue;
    }
    if (strides.sizes[d] != expected) return false;
    expected *= dim;
  }
  return true;
}

// A pointer is `N`-wide-aligned iff its byte address is a multiple of
// `N * sizeof(T)`. `reinterpret_cast<uintptr_t>` is well-defined for
// this purpose and nvcc resolves it entirely at host side.
template <typename T>
inline bool is_vec_aligned(const void* p, int N) {
  const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p);
  return (addr % (static_cast<std::uintptr_t>(N) * sizeof(T))) == 0;
}

// Pick the widest vector width we're willing to use for `T`. Matches
// the `ReadVec<T, N>` specializations above.
template <typename T> inline constexpr int dense_vec_width() { return 1; }
template <> inline constexpr int dense_vec_width<float>()    { return 4; }
template <> inline constexpr int dense_vec_width<double>()   { return 2; }
template <> inline constexpr int dense_vec_width<int32_t>()  { return 4; }
template <> inline constexpr int dense_vec_width<int64_t>()  { return 2; }
// B-015: half types vectorize to 8 bytes (LDG.64) — nvcc lowers the
// generic `VecOf<__half, 4>` load to a single 64-bit memory transaction
// once the pointer is 8-byte aligned. Matches the L2/DRAM bandwidth
// envelope that M2L.1's 64 MiB hard bar gates on.
template <> inline constexpr int dense_vec_width<__half>()        { return 4; }
template <> inline constexpr int dense_vec_width<__nv_bfloat16>() { return 4; }

// ----------- Op-code → functor dispatch ---------------

template <typename T, typename Op>
void launch_binary_dense_any(T* out, const T* a, const T* b,
                             int64_t total, Op op_instance,
                             cudaStream_t stream) {
  (void)op_instance;
  constexpr int N = dense_vec_width<T>();
  if constexpr (N > 1) {
    if (is_vec_aligned<T>(out, N) && is_vec_aligned<T>(a, N) &&
        is_vec_aligned<T>(b, N)) {
      const int64_t total_vec = total / N;
      const int64_t tail_beg  = total_vec * N;
      if (total_vec > 0) {
        const int grid = launch_grid(total_vec);
        binary_dense_vec<T, Op, N><<<grid, kBlockSize, 0, stream>>>(
            out, a, b, total_vec, Op{});
      }
      if (tail_beg < total) {
        const int64_t tail = total - tail_beg;
        const int grid = launch_grid(tail);
        binary_dense_scalar<T, Op><<<grid, kBlockSize, 0, stream>>>(
            out, a, b, tail_beg, total, Op{});
      }
      return;
    }
  }
  // Scalar dense fallback — still skips the strided indexing cost.
  const int grid = launch_grid(total);
  binary_dense_scalar<T, Op><<<grid, kBlockSize, 0, stream>>>(
      out, a, b, 0, total, Op{});
}

template <typename T>
void launch_binary_typed(BinaryKind op,
                         T* out, const T* a, const T* b,
                         const ShapePod& sizes,
                         const ShapePod& a_str, const ShapePod& b_str,
                         int64_t total, cudaStream_t stream) {
  // Fast path: when the output shape matches contiguous strides on
  // every operand (no broadcast, no non-contig view), skip the strided
  // kernel and route through the vectorized flat kernel above. Kept
  // behind a strict "both operands dense" check — a broadcasted
  // operand has at least one stride==0 entry and would pass
  // `dense_strides_match` only in pathological cases where every
  // non-trivial dim is size 1.
  const bool dense = dense_strides_match(sizes, a_str) &&
                     dense_strides_match(sizes, b_str);
  if (dense) {
    switch (op) {
      case BinaryKind::Add:
        launch_binary_dense_any<T>(out, a, b, total, AddFn{}, stream); return;
      case BinaryKind::Sub:
        launch_binary_dense_any<T>(out, a, b, total, SubFn{}, stream); return;
      case BinaryKind::Mul:
        launch_binary_dense_any<T>(out, a, b, total, MulFn{}, stream); return;
      case BinaryKind::Div:
        launch_binary_dense_any<T>(out, a, b, total, DivFn{}, stream); return;
    }
  }
  const int grid = launch_grid(total);
  switch (op) {
    case BinaryKind::Add:
      binary_strided<T, AddFn><<<grid, kBlockSize, 0, stream>>>(
          out, a, b, sizes, a_str, b_str, total, AddFn{}); break;
    case BinaryKind::Sub:
      binary_strided<T, SubFn><<<grid, kBlockSize, 0, stream>>>(
          out, a, b, sizes, a_str, b_str, total, SubFn{}); break;
    case BinaryKind::Mul:
      binary_strided<T, MulFn><<<grid, kBlockSize, 0, stream>>>(
          out, a, b, sizes, a_str, b_str, total, MulFn{}); break;
    case BinaryKind::Div:
      binary_strided<T, DivFn><<<grid, kBlockSize, 0, stream>>>(
          out, a, b, sizes, a_str, b_str, total, DivFn{}); break;
  }
}

// B-015: half-dispatch variant of `launch_binary_typed`. Structurally
// identical but uses `PromotedBinary<Op>` so arithmetic happens in
// FP32 and the storage is `__half` / `__nv_bfloat16`.
template <typename T>
void launch_binary_typed_half(BinaryKind op,
                              T* out, const T* a, const T* b,
                              const ShapePod& sizes,
                              const ShapePod& a_str, const ShapePod& b_str,
                              int64_t total, cudaStream_t stream) {
  const bool dense = dense_strides_match(sizes, a_str) &&
                     dense_strides_match(sizes, b_str);
  if (dense) {
    switch (op) {
      case BinaryKind::Add:
        launch_binary_dense_any<T>(out, a, b, total,
                                   PromotedBinary<AddFn>{}, stream);
        return;
      case BinaryKind::Sub:
        launch_binary_dense_any<T>(out, a, b, total,
                                   PromotedBinary<SubFn>{}, stream);
        return;
      case BinaryKind::Mul:
        launch_binary_dense_any<T>(out, a, b, total,
                                   PromotedBinary<MulFn>{}, stream);
        return;
      case BinaryKind::Div:
        launch_binary_dense_any<T>(out, a, b, total,
                                   PromotedBinary<DivFn>{}, stream);
        return;
    }
  }
  const int grid = launch_grid(total);
  switch (op) {
    case BinaryKind::Add:
      binary_strided<T, PromotedBinary<AddFn>><<<grid, kBlockSize, 0, stream>>>(
          out, a, b, sizes, a_str, b_str, total, PromotedBinary<AddFn>{});
      break;
    case BinaryKind::Sub:
      binary_strided<T, PromotedBinary<SubFn>><<<grid, kBlockSize, 0, stream>>>(
          out, a, b, sizes, a_str, b_str, total, PromotedBinary<SubFn>{});
      break;
    case BinaryKind::Mul:
      binary_strided<T, PromotedBinary<MulFn>><<<grid, kBlockSize, 0, stream>>>(
          out, a, b, sizes, a_str, b_str, total, PromotedBinary<MulFn>{});
      break;
    case BinaryKind::Div:
      binary_strided<T, PromotedBinary<DivFn>><<<grid, kBlockSize, 0, stream>>>(
          out, a, b, sizes, a_str, b_str, total, PromotedBinary<DivFn>{});
      break;
  }
}

template <typename T, typename Op>
void launch_unary_dense_any(T* out, const T* x, int64_t total,
                            cudaStream_t stream) {
  constexpr int N = dense_vec_width<T>();
  if constexpr (N > 1) {
    if (is_vec_aligned<T>(out, N) && is_vec_aligned<T>(x, N)) {
      const int64_t total_vec = total / N;
      const int64_t tail_beg  = total_vec * N;
      if (total_vec > 0) {
        const int grid = launch_grid(total_vec);
        unary_dense_vec<T, Op, N><<<grid, kBlockSize, 0, stream>>>(
            out, x, total_vec, Op{});
      }
      if (tail_beg < total) {
        const int64_t tail = total - tail_beg;
        const int grid = launch_grid(tail);
        unary_dense_scalar<T, Op><<<grid, kBlockSize, 0, stream>>>(
            out, x, tail_beg, total, Op{});
      }
      return;
    }
  }
  const int grid = launch_grid(total);
  unary_dense_scalar<T, Op><<<grid, kBlockSize, 0, stream>>>(
      out, x, 0, total, Op{});
}

template <typename T, typename Op>
void launch_unary_kernel(T* out, const T* x, const ShapePod& sizes,
                         const ShapePod& x_str, int64_t total,
                         cudaStream_t stream) {
  // M2L.1 dense-fast-path: every transcendental / ReLU / step / sqrt
  // landed on contiguous inputs in the transformer block and the MNIST
  // path, so short-circuit those to the vectorized flat kernel too.
  // Broadcasts (stride == 0) are impossible for unary ops — the op
  // layer always passes `x_str` equal to `x`'s own strides — but we
  // still gate on `dense_strides_match` defensively in case a caller
  // hands us a permuted view.
  if (dense_strides_match(sizes, x_str)) {
    launch_unary_dense_any<T, Op>(out, x, total, stream);
    return;
  }
  const int grid = launch_grid(total);
  unary_strided<T, Op><<<grid, kBlockSize, 0, stream>>>(
      out, x, sizes, x_str, total, Op{});
}

// Per-dtype unary dispatch helper. Parametric on `T` so we can
// instantiate it from the host-side dtype switch. We can't use a
// C++20 explicit-template lambda here because this TU is nvcc-compiled
// at C++17; a plain free function template is the C++17-portable
// substitute. The `if constexpr` gates on float-only ops keep nvcc from
// instantiating `SigmoidFn<int32_t>` etc. — those would fail to
// compile inside the device functors' `T(1) + t_exp_neg(x)` math.
template <typename T>
void unary_dispatch_typed(UnaryKind op, void* out, const void* x,
                          const ShapePod& sizes, const ShapePod& x_str,
                          int64_t total, cudaStream_t stream) {
  T*        o = static_cast<T*>(out);
  const T*  i = static_cast<const T*>(x);
  switch (op) {
    case UnaryKind::Neg:
      launch_unary_kernel<T, NegFn>(o, i, sizes, x_str, total, stream);
      return;
    case UnaryKind::Relu:
      launch_unary_kernel<T, ReluFn>(o, i, sizes, x_str, total, stream);
      return;
    case UnaryKind::Sigmoid:
      if constexpr (std::is_floating_point_v<T>) {
        launch_unary_kernel<T, SigmoidFn>(o, i, sizes, x_str, total, stream);
      }
      return;
    case UnaryKind::Tanh:
      if constexpr (std::is_floating_point_v<T>) {
        launch_unary_kernel<T, TanhFn>(o, i, sizes, x_str, total, stream);
      }
      return;
    case UnaryKind::Exp:
      if constexpr (std::is_floating_point_v<T>) {
        launch_unary_kernel<T, ExpFn>(o, i, sizes, x_str, total, stream);
      }
      return;
    case UnaryKind::Log:
      if constexpr (std::is_floating_point_v<T>) {
        launch_unary_kernel<T, LogFn>(o, i, sizes, x_str, total, stream);
      }
      return;
    case UnaryKind::Step:
      if constexpr (std::is_floating_point_v<T>) {
        launch_unary_kernel<T, StepFn>(o, i, sizes, x_str, total, stream);
      }
      return;
    case UnaryKind::Sqrt:
      if constexpr (std::is_floating_point_v<T>) {
        launch_unary_kernel<T, SqrtFn>(o, i, sizes, x_str, total, stream);
      }
      return;
  }
}

// B-015: half-dispatch variant of `unary_dispatch_typed`. Every case
// is always emitted (unlike the float/int version above which gates
// float-only ops with `if constexpr`) because half types are
// floating-point by definition — the compute always widens to FP32.
template <typename T>
void unary_dispatch_typed_half(UnaryKind op, void* out, const void* x,
                               const ShapePod& sizes, const ShapePod& x_str,
                               int64_t total, cudaStream_t stream) {
  T*        o = static_cast<T*>(out);
  const T*  i = static_cast<const T*>(x);
  switch (op) {
    case UnaryKind::Neg:
      launch_unary_kernel<T, PromotedUnary<NegFn>>(o, i, sizes, x_str, total, stream);
      return;
    case UnaryKind::Relu:
      launch_unary_kernel<T, PromotedUnary<ReluFn>>(o, i, sizes, x_str, total, stream);
      return;
    case UnaryKind::Sigmoid:
      launch_unary_kernel<T, PromotedUnary<SigmoidFn>>(o, i, sizes, x_str, total, stream);
      return;
    case UnaryKind::Tanh:
      launch_unary_kernel<T, PromotedUnary<TanhFn>>(o, i, sizes, x_str, total, stream);
      return;
    case UnaryKind::Exp:
      launch_unary_kernel<T, PromotedUnary<ExpFn>>(o, i, sizes, x_str, total, stream);
      return;
    case UnaryKind::Log:
      launch_unary_kernel<T, PromotedUnary<LogFn>>(o, i, sizes, x_str, total, stream);
      return;
    case UnaryKind::Step:
      launch_unary_kernel<T, PromotedUnary<StepFn>>(o, i, sizes, x_str, total, stream);
      return;
    case UnaryKind::Sqrt:
      launch_unary_kernel<T, PromotedUnary<SqrtFn>>(o, i, sizes, x_str, total, stream);
      return;
  }
}

}  // namespace

// -------------------- Public launchers --------------------

void launch_binary_elementwise(BinaryKind op, DType dtype, int device_index,
                               int ndim,
                               const int64_t* out_sizes,
                               const int64_t* a_strides,
                               const int64_t* b_strides,
                               void* out, const void* a, const void* b,
                               void* stream_handle) {
  TESSERACT_CHECK(ndim >= 0 && ndim <= kMaxRank,
                  "[tesseract] launch_binary_elementwise: ndim={} out of "
                  "range [0, {}]", ndim, kMaxRank);

  const ShapePod sizes  = pack_raw(out_sizes,  ndim);
  const ShapePod a_str  = pack_raw(a_strides, ndim);
  const ShapePod b_str  = pack_raw(b_strides, ndim);
  const int64_t total   = numel_pod(sizes);
  if (total == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  switch (dtype) {
    case DType::Float32:
      launch_binary_typed<float>(op,
          static_cast<float*>(out),
          static_cast<const float*>(a),
          static_cast<const float*>(b),
          sizes, a_str, b_str, total, stream);
      break;
    case DType::Float64:
      launch_binary_typed<double>(op,
          static_cast<double*>(out),
          static_cast<const double*>(a),
          static_cast<const double*>(b),
          sizes, a_str, b_str, total, stream);
      break;
    case DType::Int32:
      launch_binary_typed<int32_t>(op,
          static_cast<int32_t*>(out),
          static_cast<const int32_t*>(a),
          static_cast<const int32_t*>(b),
          sizes, a_str, b_str, total, stream);
      break;
    case DType::Int64:
      launch_binary_typed<int64_t>(op,
          static_cast<int64_t*>(out),
          static_cast<const int64_t*>(a),
          static_cast<const int64_t*>(b),
          sizes, a_str, b_str, total, stream);
      break;
    case DType::Float16:
      // B-015: FP32-promoted FP16. Our public `Half` struct has the
      // same 2-byte `uint16_t bits` layout as `__half`, so the
      // reinterpret is well-defined.
      launch_binary_typed_half<__half>(op,
          static_cast<__half*>(out),
          static_cast<const __half*>(a),
          static_cast<const __half*>(b),
          sizes, a_str, b_str, total, stream);
      break;
    case DType::BFloat16:
      launch_binary_typed_half<__nv_bfloat16>(op,
          static_cast<__nv_bfloat16*>(out),
          static_cast<const __nv_bfloat16*>(a),
          static_cast<const __nv_bfloat16*>(b),
          sizes, a_str, b_str, total, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA elementwise binary op on dtype {} is not "
          "implemented — only {{Float32, Float64, Int32, Int64, "
          "Float16, BFloat16}} are supported today. Cast to one of "
          "those on the host first.",
          dtype_name(dtype)));
  }
  check_launch("binary-elementwise");
}

void launch_unary_elementwise(UnaryKind op, DType dtype, int device_index,
                              int ndim,
                              const int64_t* shape,
                              const int64_t* x_strides,
                              void* out, const void* x,
                              void* stream_handle) {
  TESSERACT_CHECK(ndim >= 0 && ndim <= kMaxRank,
                  "[tesseract] launch_unary_elementwise: ndim={} out of "
                  "range [0, {}]", ndim, kMaxRank);

  const ShapePod sizes = pack_raw(shape,     ndim);
  const ShapePod x_str = pack_raw(x_strides, ndim);
  const int64_t total  = numel_pod(sizes);
  if (total == 0) return;

  // `Relu` and `Neg` are valid on every numeric dtype; the transcendental
  // unaries plus M2I's `Step` are gated at the op layer (`dispatch_float`
  // on the CPU side) and reach us only for floating dtypes. B-015
  // widens the accepted set from {Float32, Float64} to
  // {Float32, Float64, Float16, BFloat16}.
  const bool needs_float = (op == UnaryKind::Sigmoid || op == UnaryKind::Tanh ||
                            op == UnaryKind::Exp || op == UnaryKind::Log ||
                            op == UnaryKind::Step || op == UnaryKind::Sqrt);
  if (needs_float) {
    TESSERACT_CHECK(dtype == DType::Float32 || dtype == DType::Float64 ||
                    dtype == DType::Float16 || dtype == DType::BFloat16,
                    "[tesseract] CUDA unary op expects a floating-point "
                    "dtype, got {}",
                    dtype_name(dtype));
  }

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  // Dispatch (dtype, op) → kernel instantiation. The nested `switch`
  // inside `unary_dispatch_typed` keeps the instantiation table
  // explicit; nvcc only emits code for the combinations we actually
  // list, which caps our compile-time cost.
  switch (dtype) {
    case DType::Float32:
      unary_dispatch_typed<float>(op, out, x, sizes, x_str, total, stream);
      break;
    case DType::Float64:
      unary_dispatch_typed<double>(op, out, x, sizes, x_str, total, stream);
      break;
    case DType::Int32:
      unary_dispatch_typed<int32_t>(op, out, x, sizes, x_str, total, stream);
      break;
    case DType::Int64:
      unary_dispatch_typed<int64_t>(op, out, x, sizes, x_str, total, stream);
      break;
    case DType::Float16:
      unary_dispatch_typed_half<__half>(op, out, x, sizes, x_str, total, stream);
      break;
    case DType::BFloat16:
      unary_dispatch_typed_half<__nv_bfloat16>(op, out, x, sizes, x_str,
                                               total, stream);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA elementwise unary op on dtype {} is not "
          "implemented — only {{Float32, Float64, Int32, Int64, "
          "Float16, BFloat16}} are supported today. Cast to one of "
          "those on the host first.",
          dtype_name(dtype)));
  }
  check_launch("unary-elementwise");
}

void launch_fill(DType dtype, int device_index, std::size_t nelem,
                 void* out, double value, void* stream_handle) {
  if (nelem == 0) return;
  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  const int grid = launch_grid(static_cast<int64_t>(nelem));

  switch (dtype) {
    case DType::Float32:
      fill_dense<float><<<grid, kBlockSize, 0, stream>>>(
          static_cast<float*>(out), static_cast<float>(value),
          static_cast<int64_t>(nelem));
      break;
    case DType::Float64:
      fill_dense<double><<<grid, kBlockSize, 0, stream>>>(
          static_cast<double*>(out), value,
          static_cast<int64_t>(nelem));
      break;
    case DType::Int32:
      fill_dense<int32_t><<<grid, kBlockSize, 0, stream>>>(
          static_cast<int32_t*>(out), static_cast<int32_t>(value),
          static_cast<int64_t>(nelem));
      break;
    case DType::Int64:
      fill_dense<int64_t><<<grid, kBlockSize, 0, stream>>>(
          static_cast<int64_t*>(out), static_cast<int64_t>(value),
          static_cast<int64_t>(nelem));
      break;
    case DType::Bool:
      // Bool is a 1-byte type in our runtime; compare semantics
      // match `static_cast<bool>(value)` on CPU: non-zero → true.
      fill_dense<uint8_t><<<grid, kBlockSize, 0, stream>>>(
          static_cast<uint8_t*>(out),
          static_cast<uint8_t>(value != 0.0 ? 1 : 0),
          static_cast<int64_t>(nelem));
      break;
    case DType::Float16:
      // B-015: host-side narrow from `double` → `float` → `__half`.
      // Matches CPU `Tensor::fill_(value)` which narrows via
      // `Half(static_cast<float>(value))`.
      fill_dense<__half><<<grid, kBlockSize, 0, stream>>>(
          static_cast<__half*>(out),
          __float2half(static_cast<float>(value)),
          static_cast<int64_t>(nelem));
      break;
    case DType::BFloat16:
      fill_dense<__nv_bfloat16><<<grid, kBlockSize, 0, stream>>>(
          static_cast<__nv_bfloat16*>(out),
          __float2bfloat16(static_cast<float>(value)),
          static_cast<int64_t>(nelem));
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA fill kernel does not support dtype {} — "
          "only {{Float32, Float64, Int32, Int64, Bool, Float16, "
          "BFloat16}} are implemented today. Zero-fill works for every "
          "numeric dtype via Storage::zero_device_bytes.",
          dtype_name(dtype)));
  }
  check_launch("fill");
}

}  // namespace tesseract::cuda::detail

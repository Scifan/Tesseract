// M2G: CUDA matmul backed by cuBLASLt. This TU is only added to the
// `tesseract_cuda` source list when TESSERACT_ENABLE_CUDA=ON; the
// CPU-only throwing stub lives in `MatMulStub.cpp`. The op-layer
// caller in `src/ops/cpu/MatMul.cpp` resolves a CUDA tensor's stream
// + layout and hands us raw pointers + (op, ld) per operand, mirroring
// the elementwise / reduction bridge contract.
//
// Design notes — see `docs/m2-plan.md` §M2G for the full rationale:
//
//   * Why cuBLASLt, not cuBLAS. cuBLASLt is a superset of cuBLAS Ex
//     and is the only path that exposes
//       - explicit heuristic selection per (shape, dtype, compute type),
//       - the row-major layout order (`CUBLASLT_ORDER_ROW`) so we can
//         hand it tensors straight out of `Tensor::empty` without the
//         classical "transpose the whole problem" row↔col trick, and
//       - `CUBLAS_COMPUTE_32F` accumulation for `CUDA_R_16F` /
//         `CUDA_R_16BF` matmul — the Tensor Core path we actually want.
//
//   * Per-device handle cache. `cublasLtCreate` is expensive (100s of
//     µs on first call per driver ctx); we create one handle per
//     device on first use and reuse across every `launch_matmul`. The
//     cache is protected by a `std::once_flag` per slot; allocation
//     after construction is lock-free. Cap at 8 GPUs to match
//     `src/cuda/Allocator.cpp`'s small-array slot table — same board
//     topology assumption.
//
//   * Persistent per-device workspace (M2L.1). `cudaMalloc` of even a
//     4 MiB buffer costs ~30–200 µs per call, which dominates the
//     dispatch budget on 512² FP32 GEMMs. We pin one 4 MiB scratch
//     per device on first touch and reuse across every
//     `launch_matmul`; the buffer never gets freed while the process
//     is alive. Single-stream workloads (our common case) can safely
//     share one workspace; multi-stream users serialize on the
//     cuBLASLt handle naturally via the per-device mutex below.
//     Hopper can benefit from 32 MiB but most of the algo-class
//     uplift is already there at 4 MiB — revisit if Hopper shapes
//     show a shortfall in `bench_cuda_matmul`.
//
//   * Per-call desc / layout cache (M2L.1). The descriptor + three
//     matrix layouts + the heuristic-picked algo depend on
//     (M, N, K, dtype, op_a, op_b, lda, ldb, ldc). Building them is
//     ~40 µs in aggregate, and `cublasLtMatmulAlgoGetHeuristic`
//     itself costs another ~15–40 µs. We memoize the tuple in a
//     per-device `std::unordered_map`, protected by a per-device
//     mutex. The first call for a given shape pays the full cost;
//     every subsequent call is a single map lookup + a hot
//     `cublasLtMatmul`. Entries are kept for the life of the process
//     (no eviction) — a descriptor is a few dozen bytes and a
//     reasonable training loop touches ~O(100) unique shapes.
//
//   * Compute type picker. FP64 uses `CUBLAS_COMPUTE_64F`; everything
//     else (FP32 / FP16 / BF16) uses `CUBLAS_COMPUTE_32F`. That's
//     PyTorch's default too and gives us
//       - FP32 matmul with TF32 Tensor Cores on Ada/Hopper, and
//       - FP16 / BF16 matmul with FP32 accumulation (what every
//         serious training loop does; the "FP16 accumulated in FP16"
//         mode gives ~3 ULP of error per `K` and diverges training
//         even at `K=4096`).
//     The `scale_type` (alpha / beta precision) follows the compute
//     type — FP64 for 64F, FP32 for 32F.

#if !defined(TESSERACT_HAS_CUDA)

// CPU-only build: body lives in `MatMulStub.cpp`. This TU shouldn't be
// compiled at all when TESSERACT_ENABLE_CUDA=OFF (see
// `src/cuda/CMakeLists.txt`), but guard here too so a stray ordering
// change can't double-define `launch_matmul`.
#error "MatMul.cpp should only be compiled with TESSERACT_ENABLE_CUDA=ON"

#else

#include "tesseract/cuda/detail/MatMul.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <fmt/format.h>

#include "tesseract/core/DType.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"  // tesseract::DeviceError
#include "tesseract/utils/Logging.hpp"

namespace tesseract::cuda::detail {

namespace {

// Max number of CUDA devices we keep handle slots for; matches the cap
// in `src/cuda/Allocator.cpp` (single server boards top out at 8× HPC
// GPUs). If we ever target multi-node (NVLink fabric across 16+
// devices) revisit here and in the allocator together.
constexpr int kMaxDevices = 8;

struct LtSlot {
  std::once_flag flag;
  cublasLtHandle_t handle{};
};

std::array<LtSlot, kMaxDevices>& slots() {
  static std::array<LtSlot, kMaxDevices> s;
  return s;
}

// -------------------- Persistent workspace (M2L.1) --------------------
//
// One 4 MiB scratch per device, lazily allocated on first touch.
// Leaked on process exit — cuBLASLt handles itself are leaked the same
// way (no `cublasLtDestroy` on shutdown), so we're not regressing the
// teardown story. Size matches the heuristic ceiling we set below.
constexpr std::size_t kWorkspaceBytes = 4 * 1024 * 1024;

struct WorkspaceSlot {
  std::once_flag flag;
  void* ptr{nullptr};
  std::size_t bytes{0};
};

std::array<WorkspaceSlot, kMaxDevices>& workspaces() {
  static std::array<WorkspaceSlot, kMaxDevices> s;
  return s;
}

// -------------------- Descriptor cache (M2L.1) --------------------
//
// Key uniqueness: any change to (dtype, op_a, op_b, M, N, K, lda, ldb,
// ldc) implies different cuBLASLt descriptor + layouts + heuristic
// pick. Stream is **not** in the key — it's passed per-call to
// `cublasLtMatmul` and we deliberately don't bind the algo to a
// specific stream.
struct CacheKey {
  int dtype;  // `static_cast<int>(DType)` to keep the key trivially hashable
  int op_a, op_b;
  int64_t M, N, K;
  int64_t lda, ldb, ldc;

  bool operator==(const CacheKey& o) const noexcept {
    return dtype == o.dtype && op_a == o.op_a && op_b == o.op_b &&
           M == o.M && N == o.N && K == o.K &&
           lda == o.lda && ldb == o.ldb && ldc == o.ldc;
  }
};

struct CacheKeyHash {
  std::size_t operator()(const CacheKey& k) const noexcept {
    // FNV-1a over the 9 fields. The 64-bit mix keeps collisions
    // negligible at the sizes we care about (< 10 k distinct shapes).
    auto mix = [](std::size_t h, std::size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    };
    std::size_t h = 1469598103934665603ULL;
    h = mix(h, static_cast<std::size_t>(k.dtype));
    h = mix(h, static_cast<std::size_t>(k.op_a));
    h = mix(h, static_cast<std::size_t>(k.op_b));
    h = mix(h, static_cast<std::size_t>(k.M));
    h = mix(h, static_cast<std::size_t>(k.N));
    h = mix(h, static_cast<std::size_t>(k.K));
    h = mix(h, static_cast<std::size_t>(k.lda));
    h = mix(h, static_cast<std::size_t>(k.ldb));
    h = mix(h, static_cast<std::size_t>(k.ldc));
    return h;
  }
};

// Owned bundle of cuBLASLt resources that together specify one matmul
// shape/dtype. RAII-cleaned from the cache's destructor (which itself
// runs at process exit). Copy/move disabled — the map stores pointers
// to heap-allocated entries and we hand out raw references.
struct CacheEntry {
  cublasLtMatmulDesc_t desc{nullptr};
  cublasLtMatrixLayout_t a_layout{nullptr};
  cublasLtMatrixLayout_t b_layout{nullptr};
  cublasLtMatrixLayout_t c_layout{nullptr};
  cublasLtMatmulHeuristicResult_t hr{};
  cublasComputeType_t compute_type{CUBLAS_COMPUTE_32F};
  cudaDataType_t scale_type{CUDA_R_32F};

  CacheEntry() = default;
  ~CacheEntry() {
    if (desc)     (void)cublasLtMatmulDescDestroy(desc);
    if (a_layout) (void)cublasLtMatrixLayoutDestroy(a_layout);
    if (b_layout) (void)cublasLtMatrixLayoutDestroy(b_layout);
    if (c_layout) (void)cublasLtMatrixLayoutDestroy(c_layout);
  }

  CacheEntry(const CacheEntry&) = delete;
  CacheEntry& operator=(const CacheEntry&) = delete;
};

struct CacheSlot {
  std::mutex mu;
  std::unordered_map<CacheKey, std::unique_ptr<CacheEntry>, CacheKeyHash> map;
};

std::array<CacheSlot, kMaxDevices>& caches() {
  static std::array<CacheSlot, kMaxDevices> s;
  return s;
}

const char* cublas_status_str(cublasStatus_t st) {
  switch (st) {
    case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "LICENSE_ERROR";
    default: return "UNKNOWN";
  }
}

[[noreturn]] void throw_cublas(cublasStatus_t st, const char* what) {
  throw tesseract::DeviceError(fmt::format(
      "[tesseract] cuBLASLt {} failed: {} ({})",
      what, cublas_status_str(st), static_cast<int>(st)));
}

void check_cublas(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS) throw_cublas(st, what);
}

[[noreturn]] void throw_cuda(cudaError_t err, const char* what) {
  throw tesseract::DeviceError(fmt::format(
      "[tesseract] CUDA {} failed: {} ({})",
      what, cudaGetErrorString(err), static_cast<int>(err)));
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw_cuda(err, what);
}

// Per-device cuBLASLt handle, created lazily. Keeping one handle per
// device (rather than one global) avoids the `cublasLtSetStream`
// dance that `cudaSetDevice` would impose — each call sets its own
// device before using the handle, and the stream is passed through
// `cublasLtMatmul` directly.
cublasLtHandle_t get_handle(int device_index) {
  TESSERACT_CHECK(device_index >= 0 && device_index < kMaxDevices,
                  "cuBLASLt: device index {} out of supported range [0, {})",
                  device_index, kMaxDevices);
  auto& slot = slots()[device_index];
  std::call_once(slot.flag, [&]() {
    int prev = -1;
    // `cublasLtCreate` reads the current device, so pin before creating.
    check_cuda(cudaGetDevice(&prev), "cudaGetDevice");
    check_cuda(cudaSetDevice(device_index), "cudaSetDevice");
    cublasStatus_t st = cublasLtCreate(&slot.handle);
    // Restore whether or not the create succeeded, so a later call on
    // the previous device isn't left pointing at a changed CTX.
    if (prev >= 0 && prev != device_index) {
      (void)cudaSetDevice(prev);
    }
    check_cublas(st, "cublasLtCreate");
  });
  return slot.handle;
}

cudaDataType_t cuda_type_of(DType dt) {
  switch (dt) {
    case DType::Float32: return CUDA_R_32F;
    case DType::Float64: return CUDA_R_64F;
    case DType::Float16: return CUDA_R_16F;
    case DType::BFloat16: return CUDA_R_16BF;
    default:
      throw tesseract::DeviceError(fmt::format(
          "[tesseract] cuBLASLt: unsupported dtype {} (expected "
          "Float32/Float64/Float16/BFloat16)",
          dtype_name(dt)));
  }
}

cublasComputeType_t compute_type_of(DType dt) {
  // FP64 stays FP64; everything else (FP32 / FP16 / BF16) accumulates in
  // FP32. See the header-comment rationale — matches PyTorch defaults.
  return (dt == DType::Float64) ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
}

cudaDataType_t scale_type_of(cublasComputeType_t ct) {
  return (ct == CUBLAS_COMPUTE_64F) ? CUDA_R_64F : CUDA_R_32F;
}

// RAII pin for the caller's CUDA context. Every `launch_matmul` enters
// the target device, then restores whatever the caller had selected
// on exit — including on exceptional control flow. (The old
// per-call `DescGuard` / `LayoutGuard` / `PrefGuard` /
// `DeviceWorkspace` RAII wrappers are no longer needed now that the
// descriptors live in the per-device cache and the workspace is
// persistent.)
struct DevicePin {
  int prev{-1};
  int target{-1};
  bool restore{false};
  DevicePin(int want) : target(want) {
    check_cuda(cudaGetDevice(&prev), "cudaGetDevice");
    if (prev != target) {
      check_cuda(cudaSetDevice(target), "cudaSetDevice");
      restore = true;
    }
  }
  ~DevicePin() {
    if (restore && prev >= 0) {
      (void)cudaSetDevice(prev);
    }
  }
};

}  // namespace

namespace {

// Resolve (or lazily construct) the workspace slot for `device_index`.
// Runs under `cudaSetDevice` so the allocation lands on the right GPU
// regardless of the caller's current context. Returns `(nullptr, 0)`
// on allocation failure — cuBLASLt accepts a null workspace and falls
// back to a smaller algorithm class automatically.
std::pair<void*, std::size_t> get_workspace(int device_index) {
  auto& w = workspaces()[device_index];
  std::call_once(w.flag, [&]() {
    int prev = -1;
    (void)cudaGetDevice(&prev);
    (void)cudaSetDevice(device_index);
    void* p = nullptr;
    if (cudaMalloc(&p, kWorkspaceBytes) == cudaSuccess) {
      w.ptr   = p;
      w.bytes = kWorkspaceBytes;
    }
    if (prev >= 0 && prev != device_index) (void)cudaSetDevice(prev);
  });
  return {w.ptr, w.bytes};
}

// Build a fresh CacheEntry for the given key. Pays the full cuBLASLt
// setup cost (~50–100 µs on Ada); called exactly once per distinct
// shape. Thread-safety: the caller holds `CacheSlot::mu` so there's no
// interleaving with another builder.
std::unique_ptr<CacheEntry> build_entry(cublasLtHandle_t h,
                                        const CacheKey& k,
                                        std::size_t ws_bytes) {
  auto entry = std::make_unique<CacheEntry>();
  const DType dtype = static_cast<DType>(k.dtype);
  const cudaDataType_t cdt = cuda_type_of(dtype);
  entry->compute_type = compute_type_of(dtype);
  entry->scale_type   = scale_type_of(entry->compute_type);

  check_cublas(cublasLtMatmulDescCreate(&entry->desc, entry->compute_type,
                                        entry->scale_type),
               "MatmulDescCreate");

  const cublasOperation_t opa = (k.op_a != 0) ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t opb = (k.op_b != 0) ? CUBLAS_OP_T : CUBLAS_OP_N;
  check_cublas(cublasLtMatmulDescSetAttribute(
                   entry->desc, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa)),
               "DESC_TRANSA");
  check_cublas(cublasLtMatmulDescSetAttribute(
                   entry->desc, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb)),
               "DESC_TRANSB");

  const uint64_t a_rows = (opa == CUBLAS_OP_N) ? static_cast<uint64_t>(k.M)
                                                : static_cast<uint64_t>(k.K);
  const uint64_t a_cols = (opa == CUBLAS_OP_N) ? static_cast<uint64_t>(k.K)
                                                : static_cast<uint64_t>(k.M);
  const uint64_t b_rows = (opb == CUBLAS_OP_N) ? static_cast<uint64_t>(k.K)
                                                : static_cast<uint64_t>(k.N);
  const uint64_t b_cols = (opb == CUBLAS_OP_N) ? static_cast<uint64_t>(k.N)
                                                : static_cast<uint64_t>(k.K);

  auto mk_layout = [&](cublasLtMatrixLayout_t& out,
                       uint64_t rows, uint64_t cols, int64_t ld) {
    check_cublas(cublasLtMatrixLayoutCreate(&out, cdt, rows, cols, ld),
                 "cublasLtMatrixLayoutCreate");
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    check_cublas(cublasLtMatrixLayoutSetAttribute(
                     out, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
                 "LayoutOrder=ROW");
  };
  mk_layout(entry->a_layout, a_rows, a_cols, k.lda);
  mk_layout(entry->b_layout, b_rows, b_cols, k.ldb);
  mk_layout(entry->c_layout, static_cast<uint64_t>(k.M),
            static_cast<uint64_t>(k.N), k.ldc);

  // Preference object is cheap and only consulted during heuristic
  // selection — destroy it as soon as we have the algo.
  cublasLtMatmulPreference_t pref = nullptr;
  check_cublas(cublasLtMatmulPreferenceCreate(&pref),
               "MatmulPreferenceCreate");
  check_cublas(cublasLtMatmulPreferenceSetAttribute(
                   pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                   &ws_bytes, sizeof(ws_bytes)),
               "PREF_MAX_WORKSPACE_BYTES");

  int returned = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
      h, entry->desc, entry->a_layout, entry->b_layout,
      entry->c_layout, entry->c_layout, pref,
      /*requestedAlgoCount=*/1, &entry->hr, &returned);
  (void)cublasLtMatmulPreferenceDestroy(pref);
  check_cublas(st, "AlgoGetHeuristic");
  TESSERACT_CHECK(returned > 0,
                  "cuBLASLt: no algorithm returned for M={}, N={}, K={}, "
                  "dtype={}, op_a={}, op_b={} — check ld values",
                  k.M, k.N, k.K, dtype_name(dtype), k.op_a, k.op_b);
  return entry;
}

}  // namespace

void* get_cublaslt_handle(int device_index) {
  return static_cast<void*>(get_handle(device_index));
}

void launch_matmul(DType dtype, int device_index,
                   int64_t M, int64_t N, int64_t K,
                   const void* a, MmOp op_a, int64_t lda,
                   const void* b, MmOp op_b, int64_t ldb,
                   void* c, int64_t ldc,
                   void* stream_handle) {
  TESSERACT_CHECK(M > 0 && N > 0 && K > 0,
                  "cuBLASLt: non-positive matmul dims (M={}, N={}, K={})",
                  M, N, K);
  TESSERACT_CHECK(a && b && c,
                  "cuBLASLt: null operand pointer (a={}, b={}, c={})",
                  a, b, c);
  TESSERACT_CHECK(device_index >= 0 && device_index < kMaxDevices,
                  "cuBLASLt: device index {} out of supported range [0, {})",
                  device_index, kMaxDevices);

  DevicePin pin(device_index);

  cublasLtHandle_t h = get_handle(device_index);
  const auto [ws_ptr, ws_bytes] = get_workspace(device_index);

  CacheKey key{
      /*dtype*/ static_cast<int>(dtype),
      /*op_a */ static_cast<int>(op_a == MmOp::Transpose ? 1 : 0),
      /*op_b */ static_cast<int>(op_b == MmOp::Transpose ? 1 : 0),
      /*M*/ M, /*N*/ N, /*K*/ K,
      /*lda*/ lda, /*ldb*/ ldb, /*ldc*/ ldc,
  };

  CacheEntry* entry = nullptr;
  auto& slot = caches()[device_index];
  {
    // Two-phase lookup: shared under the lock just long enough to
    // either find the entry or create it. We hold the lock across
    // `build_entry` so only one thread pays the ~50–100 µs cost per
    // distinct shape, even under concurrent launches — cheap at our
    // call rate (hundreds of launches per second, never thousands).
    std::lock_guard<std::mutex> lock(slot.mu);
    auto it = slot.map.find(key);
    if (it == slot.map.end()) {
      auto built = build_entry(h, key, ws_bytes);
      it = slot.map.emplace(key, std::move(built)).first;
    }
    entry = it->second.get();
  }

  // Alpha / beta in the compute precision. cuBLASLt reads these as raw
  // bytes of `scale_type` size; we have to hold the right variant on
  // the stack. `fp64` is a strict superset of the bits we care about
  // so we never truncate here.
  double alpha_d = 1.0, beta_d = 0.0;
  float  alpha_f = 1.0f, beta_f = 0.0f;
  const void* palpha = (entry->scale_type == CUDA_R_64F)
      ? static_cast<const void*>(&alpha_d)
      : static_cast<const void*>(&alpha_f);
  const void* pbeta = (entry->scale_type == CUDA_R_64F)
      ? static_cast<const void*>(&beta_d)
      : static_cast<const void*>(&beta_f);

  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  check_cublas(cublasLtMatmul(
                   h, entry->desc,
                   palpha, a, entry->a_layout,
                           b, entry->b_layout,
                   pbeta,  c, entry->c_layout,
                           c, entry->c_layout,
                   &entry->hr.algo, ws_ptr, ws_bytes, stream),
               "cublasLtMatmul");
}

}  // namespace tesseract::cuda::detail

#endif  // TESSERACT_HAS_CUDA

#include "tesseract/distributed/NcclCommBackend.hpp"

#include "tesseract/utils/Logging.hpp"

#if defined(TESSERACT_HAS_NCCL)

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/ops/View.hpp"

namespace tesseract::distributed {

namespace {

#define TS_CUDA_CHECK(expr)                                                  \
  do {                                                                       \
    cudaError_t _e = (expr);                                                 \
    if (_e != cudaSuccess) {                                                 \
      TESSERACT_THROW("NCCL backend CUDA error: {} ({})",                    \
                      cudaGetErrorString(_e), #expr);                        \
    }                                                                        \
  } while (0)

#define TS_NCCL_CHECK(expr)                                                  \
  do {                                                                       \
    ncclResult_t _r = (expr);                                               \
    if (_r != ncclSuccess) {                                                 \
      TESSERACT_THROW("NCCL error: {} ({})", ncclGetErrorString(_r), #expr); \
    }                                                                        \
  } while (0)

ncclDataType_t nccl_dtype(DType dt) {
  switch (dt) {
    case DType::Float32:  return ncclFloat32;
    case DType::Float16:  return ncclFloat16;
    case DType::BFloat16: return ncclBfloat16;
    default:
      TESSERACT_THROW("NcclCommBackend: unsupported dtype {} (need "
                      "Float32/Float16/BFloat16)",
                      dtype_name(dt));
  }
}

std::string to_hex(const ncclUniqueId& id) {
  static const char* kHex = "0123456789abcdef";
  const unsigned char* p = reinterpret_cast<const unsigned char*>(&id);
  std::string s;
  s.reserve(sizeof(ncclUniqueId) * 2);
  for (size_t i = 0; i < sizeof(ncclUniqueId); ++i) {
    s.push_back(kHex[p[i] >> 4]);
    s.push_back(kHex[p[i] & 0xF]);
  }
  return s;
}

ncclUniqueId from_hex(const std::string& hex) {
  TESSERACT_CHECK(hex.size() == sizeof(ncclUniqueId) * 2,
                  "NcclCommBackend: unique_id hex length {} != expected {}",
                  hex.size(), sizeof(ncclUniqueId) * 2);
  ncclUniqueId id;
  unsigned char* p = reinterpret_cast<unsigned char*>(&id);
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
  };
  for (size_t i = 0; i < sizeof(ncclUniqueId); ++i) {
    p[i] = static_cast<unsigned char>((nib(hex[2 * i]) << 4) | nib(hex[2 * i + 1]));
  }
  return id;
}

}  // namespace

NcclCommBackend::NcclCommBackend(int rank, int world_size, int device_index,
                                 const std::string& unique_id_hex)
    : rank_(rank),
      world_size_(world_size),
      device_index_(device_index),
      comm_(nullptr),
      stream_(nullptr) {
  TESSERACT_CHECK(world_size >= 1, "NcclCommBackend: world_size must be >= 1");
  TESSERACT_CHECK(rank >= 0 && rank < world_size,
                  "NcclCommBackend: rank {} out of range [0,{})", rank,
                  world_size);
  TS_CUDA_CHECK(cudaSetDevice(device_index_));
  // Collectives are enqueued on the *tensor's current stream* (see
  // `collective_stream`), not a private stream, so they are ordered after the
  // op that produced the input — otherwise NCCL races ahead of the matmul and
  // reads a half-written buffer. `stream_` is unused; kept for ABI.
  stream_ = nullptr;

  ncclUniqueId id = from_hex(unique_id_hex);
  ncclComm_t comm;
  // Collective across all ranks — every process must reach this with the
  // same id and a distinct rank.
  TS_NCCL_CHECK(ncclCommInitRank(&comm, world_size_, id, rank_));
  comm_ = comm;
}

NcclCommBackend::~NcclCommBackend() {
  if (comm_) {
    ncclCommDestroy(static_cast<ncclComm_t>(comm_));
    comm_ = nullptr;
  }
}

namespace {
// The CUDA stream the device's current Tesseract ops run on. Enqueuing the
// collective here orders it after the producing op and before any consumer on
// the same stream — no cross-stream event juggling needed.
cudaStream_t collective_stream(const Device& dev) {
  return static_cast<cudaStream_t>(current_stream(dev).native_handle());
}
}  // namespace

Tensor NcclCommBackend::all_reduce_sum(const Tensor& local) const {
  TESSERACT_CHECK(local.device().is_cuda(),
                  "NcclCommBackend::all_reduce_sum: tensor must be CUDA");
  NoGradGuard nogg;
  Tensor src = local.is_contiguous() ? local : ops::contiguous(local);
  Tensor out = Tensor::empty(src.shape(), src.dtype(), src.device());
  TS_CUDA_CHECK(cudaSetDevice(device_index_));
  cudaStream_t cs = collective_stream(src.device());
  TS_NCCL_CHECK(ncclAllReduce(src.raw_data(), out.raw_data(),
                              static_cast<size_t>(src.numel()),
                              nccl_dtype(src.dtype()), ncclSum,
                              static_cast<ncclComm_t>(comm_), cs));
  TS_CUDA_CHECK(cudaStreamSynchronize(cs));
  return out;
}

Tensor NcclCommBackend::all_gather(const Tensor& local, int64_t dim) const {
  TESSERACT_CHECK(local.device().is_cuda(),
                  "NcclCommBackend::all_gather: tensor must be CUDA");
  NoGradGuard nogg;
  Tensor src = local.is_contiguous() ? local : ops::contiguous(local);
  const int64_t r = src.rank();
  if (dim < 0) dim += r;
  TESSERACT_CHECK(dim >= 0 && dim < r,
                  "NcclCommBackend::all_gather: dim {} out of range", dim);

  // Raw NCCL gather: rank r's `count` elements land at offset r*count, i.e.
  // a leading [world_size, *local_shape] contiguous buffer.
  Shape gshape;
  gshape.push_back(world_size_);
  for (int64_t i = 0; i < r; ++i) gshape.push_back(src.shape()[i]);
  Tensor gathered = Tensor::empty(gshape, src.dtype(), src.device());
  TS_CUDA_CHECK(cudaSetDevice(device_index_));
  cudaStream_t cs = collective_stream(src.device());
  TS_NCCL_CHECK(ncclAllGather(src.raw_data(), gathered.raw_data(),
                              static_cast<size_t>(src.numel()),
                              nccl_dtype(src.dtype()),
                              static_cast<ncclComm_t>(comm_), cs));
  TS_CUDA_CHECK(cudaStreamSynchronize(cs));

  if (dim == 0) {
    // [world, d0, d1, ...] -> [world*d0, d1, ...] is exactly concat along 0.
    Shape out_shape;
    out_shape.push_back(world_size_ * src.shape()[0]);
    for (int64_t i = 1; i < r; ++i) out_shape.push_back(src.shape()[i]);
    return ops::reshape(gathered, out_shape);
  }

  // General dim: move axis 0 (the rank axis) to just before `dim`, then merge
  // it with `dim` so the output is concatenated in rank-major order along it.
  // gathered axes: [world(0), s0(1), ..., s_{r-1}(r)]; target before merge:
  //   [s0, ..., s_{dim-1}, world, s_dim, ..., s_{r-1}].
  std::vector<int64_t> axes;
  for (int64_t i = 1; i <= dim; ++i) axes.push_back(i);  // s0..s_{dim-1}
  axes.push_back(0);                                     // world
  for (int64_t i = dim + 1; i <= r; ++i) axes.push_back(i);  // s_dim..s_{r-1}
  Tensor permuted = ops::contiguous(ops::permute(gathered, axes));
  // Merge (world, s_dim) at position `dim`.
  Shape out_shape;
  for (int64_t i = 0; i < dim; ++i) out_shape.push_back(src.shape()[i]);
  out_shape.push_back(world_size_ * src.shape()[dim]);
  for (int64_t i = dim + 1; i < r; ++i) out_shape.push_back(src.shape()[i]);
  return ops::reshape(permuted, out_shape);
}

void NcclCommBackend::barrier() const {
  Tensor t = Tensor::zeros(Shape({1}), DType::Float32,
                           Device{DeviceType::CUDA, device_index_});
  (void)all_reduce_sum(t);
}

std::string NcclCommBackend::generate_unique_id_hex() {
  ncclUniqueId id;
  TS_NCCL_CHECK(ncclGetUniqueId(&id));
  return to_hex(id);
}

bool NcclCommBackend::available() noexcept { return true; }

}  // namespace tesseract::distributed

#else  // !TESSERACT_HAS_NCCL — stub so CPU/no-NCCL builds still link.

namespace tesseract::distributed {

NcclCommBackend::NcclCommBackend(int, int, int, const std::string&)
    : rank_(0), world_size_(0), device_index_(0), comm_(nullptr),
      stream_(nullptr) {
  TESSERACT_THROW("NcclCommBackend: built without NCCL "
                  "(configure with -DTESSERACT_ENABLE_NCCL=ON)");
}
NcclCommBackend::~NcclCommBackend() = default;
Tensor NcclCommBackend::all_reduce_sum(const Tensor&) const {
  TESSERACT_THROW("NcclCommBackend: built without NCCL");
}
Tensor NcclCommBackend::all_gather(const Tensor&, int64_t) const {
  TESSERACT_THROW("NcclCommBackend: built without NCCL");
}
void NcclCommBackend::barrier() const {
  TESSERACT_THROW("NcclCommBackend: built without NCCL");
}
std::string NcclCommBackend::generate_unique_id_hex() {
  TESSERACT_THROW("NcclCommBackend: built without NCCL");
}
bool NcclCommBackend::available() noexcept { return false; }

}  // namespace tesseract::distributed

#endif  // TESSERACT_HAS_NCCL

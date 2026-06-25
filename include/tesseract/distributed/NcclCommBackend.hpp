#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "tesseract/core/Tensor.hpp"

// M4 Phase 7 — real multi-GPU collectives over NCCL.
//
// Unlike `SimCommBackend` (single process, holds *every* rank's shard in one
// address space), `NcclCommBackend` is the production substrate: **one process
// per GPU**, each owning a single rank and calling collectives on its *local*
// shard. The cross-rank exchange is done by NCCL on the wire/NVLink.
//
// Bootstrap (no MPI): rank 0 generates an `ncclUniqueId`, hex-encodes it, and
// broadcasts it out-of-band (the launcher writes it to a shared file every
// rank reads). Each rank then `ncclCommInitRank(comm, world, id, rank)` after
// `cudaSetDevice(device_index)`.
//
// Built only when `TESSERACT_HAS_NCCL` is defined (-DTESSERACT_ENABLE_NCCL=ON);
// otherwise every method throws so a CPU/no-NCCL build still links.
namespace tesseract::distributed {

class NcclCommBackend {
 public:
  // `unique_id_hex` is the 128-byte ncclUniqueId, hex-encoded, identical on
  // every rank (produced once by `generate_unique_id_hex` on rank 0). This
  // ctor calls `cudaSetDevice(device_index)` and `ncclCommInitRank`, which is
  // a collective across all `world_size` ranks — they must all enter it.
  NcclCommBackend(int rank, int world_size, int device_index,
                  const std::string& unique_id_hex);
  ~NcclCommBackend();

  NcclCommBackend(const NcclCommBackend&) = delete;
  NcclCommBackend& operator=(const NcclCommBackend&) = delete;

  int rank() const noexcept { return rank_; }
  int world_size() const noexcept { return world_size_; }
  int device_index() const noexcept { return device_index_; }

  // Sum all-reduce of the local CUDA tensor; every rank receives the full
  // element-wise sum (same shape/dtype as the input). FP32/FP16/BF16.
  Tensor all_reduce_sum(const Tensor& local) const;

  // All-gather: concatenate every rank's equal-shaped local shard along
  // `dim` (negative indexing allowed). Returns the full tensor every rank
  // holds after the gather. `dim==0` is the raw NCCL layout; other dims are
  // rearranged on-device to the correct concatenation order.
  Tensor all_gather(const Tensor& local, int64_t dim) const;

  // Barrier across all ranks (implemented as a 1-element all-reduce).
  void barrier() const;

  // Rank 0 generates a fresh ncclUniqueId and returns it hex-encoded. The
  // launcher is responsible for broadcasting the string to all ranks.
  static std::string generate_unique_id_hex();

  // True if this build has NCCL compiled in.
  static bool available() noexcept;

 private:
  int rank_;
  int world_size_;
  int device_index_;
  void* comm_;    // ncclComm_t (opaque to non-NCCL TUs)
  void* stream_;  // cudaStream_t dedicated to collectives
};

}  // namespace tesseract::distributed

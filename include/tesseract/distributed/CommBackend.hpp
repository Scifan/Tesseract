#pragma once

#include <vector>

#include "tesseract/core/Tensor.hpp"

// Collective-communication abstraction for tensor/data parallelism
// (M4 Track B3 / B-043).
//
// Tensor parallelism needs exactly two collectives: an **all-reduce (sum)**
// at the row-parallel output, and an **all-gather (concat)** at the
// column-parallel output. We hide both behind `CommBackend` so the parallel
// `nn` modules are written *once* against the interface and run unchanged on:
//
//   * `SimCommBackend` — a single-process simulator that holds every rank's
//     shard in one address space (all-reduce = host-side sum, all-gather =
//     `ops::cat`). Deterministic, dependency-free, and the substrate for the
//     CPU parity tests that pin TP correctness.
//   * (gated tail) a future `NcclCommBackend` behind `TESSERACT_ENABLE_NCCL`
//     that issues `ncclAllReduce` / `ncclAllGather` across real GPUs — the
//     same module code, a different backend object.
//
// This mirrors the `idea.md` §6.1.5 "并行策略作为 IR pass" direction: the
// sharding + collective placement is a *transform* layered over the dense
// model, not a bespoke distributed model class.
namespace tesseract::distributed {

class CommBackend {
 public:
  virtual ~CommBackend() = default;

  // Number of participating ranks.
  virtual int world_size() const = 0;

  // All-reduce with summation. `shards[r]` is rank r's contribution; all
  // shards must share shape/dtype/device. Returns the element-wise sum —
  // the value every rank would observe after a real all-reduce.
  virtual Tensor all_reduce_sum(const std::vector<Tensor>& shards) const = 0;

  // All-gather: concatenate the per-rank `shards` along `dim` (negative
  // indexing allowed). `shards[r]` is rank r's slice; the result is the
  // full tensor every rank would hold after a real all-gather.
  virtual Tensor all_gather(const std::vector<Tensor>& shards,
                            int64_t dim) const = 0;
};

// Single-process, in-memory implementation. Every rank's shard lives in the
// same process, so collectives are plain tensor math: this is the reference
// backend for CPU parity (TP=N output must equal the dense output).
class SimCommBackend : public CommBackend {
 public:
  explicit SimCommBackend(int world_size);

  int world_size() const override { return world_size_; }
  Tensor all_reduce_sum(const std::vector<Tensor>& shards) const override;
  Tensor all_gather(const std::vector<Tensor>& shards,
                    int64_t dim) const override;

 private:
  int world_size_;
};

}  // namespace tesseract::distributed

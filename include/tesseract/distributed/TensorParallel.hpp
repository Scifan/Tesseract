#pragma once

#include <memory>
#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/distributed/CommBackend.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"

// Megatron-style tensor parallelism primitives (M4 Track B3 / B-043).
//
// Two sharding patterns cover every Linear in a transformer:
//
//   * **Column-parallel** — split the weight along its *output* dim. Each
//     rank owns `W_r = W[r*out_p:(r+1)*out_p, :]` and computes
//     `y_r = x @ W_r^T (+ b_r)`. The full output is the concat of the
//     per-rank slices along the feature dim (an all-gather). Used for the
//     q/k/v projections (split heads) and the FFN gate/up projections.
//
//   * **Row-parallel** — split the weight along its *input* dim. Each rank
//     owns `W_r = W[:, r*in_p:(r+1)*in_p]` and a slice of the input
//     `x_r`, computing the partial `y_r = x_r @ W_r^T`. The full output is
//     the sum of the partials (an all-reduce). Used for the attention
//     output projection and the FFN down projection.
//
// The Megatron MLP `down(silu(gate(x)) * up(x))` chains column-parallel
// (gate/up, output kept sharded) → local elementwise → row-parallel (down,
// all-reduce) so the whole block needs exactly **one** collective on the
// forward path. The same structure works for attention.
//
// These modules deliberately mirror dense `nn::Linear` numerics: a TP=N
// sharding of a given dense layer reproduces the dense output (exactly for
// column-parallel; within FP summation-reassociation tolerance for
// row-parallel). `from_dense` builds the sharded module by copying slices of
// an existing `nn::Linear`, which is what the parity tests use.
namespace tesseract::distributed {

// Column-parallel Linear: shards weight rows (output features) across ranks.
class ColumnParallelLinear : public nn::Module {
 public:
  // Fresh-initialized sharded layer. `out_features` must be divisible by
  // `comm->world_size()`. When `gather_output` is true, `forward` concats the
  // per-rank outputs into the full `[..., out_features]` tensor; when false,
  // callers consume the per-rank shards directly via `forward_shards`
  // (the Megatron MLP keeps the hidden sharded between gate/up and down).
  ColumnParallelLinear(int64_t in_features, int64_t out_features,
                       std::shared_ptr<CommBackend> comm, bool use_bias = true,
                       bool gather_output = true, DType dtype = DType::Float32);

  // Per-rank outputs `y_r = x @ W_r^T (+ b_r)` — length `world_size()`.
  std::vector<Tensor> forward_shards(const Tensor& x);

  // Full output `[..., out_features]` via all-gather along the feature dim.
  Tensor forward(const Tensor& x) override;

  // Build a column-parallel layer that reproduces `dense` by copying the
  // matching slices of its weight/bias into per-rank shards.
  static std::shared_ptr<ColumnParallelLinear> from_dense(
      const nn::Linear& dense, std::shared_ptr<CommBackend> comm,
      bool gather_output = true);

  int world_size() const noexcept { return world_size_; }
  bool gather_output() const noexcept { return gather_output_; }

 private:
  int64_t in_features_;
  int64_t out_features_;
  int64_t out_per_rank_;
  int world_size_;
  bool use_bias_;
  bool gather_output_;
  std::shared_ptr<CommBackend> comm_;
  std::vector<Tensor> weight_shards_;  // each [out_per_rank, in_features]
  std::vector<Tensor> bias_shards_;    // each [out_per_rank] (if use_bias)
};

// Row-parallel Linear: shards weight columns (input features) across ranks.
class RowParallelLinear : public nn::Module {
 public:
  // `in_features` must be divisible by `comm->world_size()`. The bias (if
  // any) is a single full `[out_features]` tensor added once after the
  // all-reduce (it is not sharded).
  RowParallelLinear(int64_t in_features, int64_t out_features,
                    std::shared_ptr<CommBackend> comm, bool use_bias = true,
                    DType dtype = DType::Float32);

  // Consume pre-sharded input (one `[..., in_per_rank]` tensor per rank,
  // e.g. the sharded hidden from a column-parallel layer): partial matmul
  // per rank then all-reduce(sum) + bias. This is the no-extra-collective
  // path used inside the Megatron MLP/attention.
  Tensor forward_shards(const std::vector<Tensor>& x_shards);

  // Full input `[..., in_features]`: split along the feature dim into
  // per-rank slices, then `forward_shards`.
  Tensor forward(const Tensor& x) override;

  static std::shared_ptr<RowParallelLinear> from_dense(
      const nn::Linear& dense, std::shared_ptr<CommBackend> comm);

  int world_size() const noexcept { return world_size_; }
  int64_t in_per_rank() const noexcept { return in_per_rank_; }

 private:
  int64_t in_features_;
  int64_t out_features_;
  int64_t in_per_rank_;
  int world_size_;
  bool use_bias_;
  std::shared_ptr<CommBackend> comm_;
  std::vector<Tensor> weight_shards_;  // each [out_features, in_per_rank]
  Tensor bias_;                        // [out_features] (if use_bias)
};

}  // namespace tesseract::distributed

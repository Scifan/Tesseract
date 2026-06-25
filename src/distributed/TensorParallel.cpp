#include "tesseract/distributed/TensorParallel.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Indexing.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::distributed {

namespace {

thread_local uint64_t g_tp_seed = 0x9E3779B97F4A7C15ULL;

void uniform_init(Tensor& t, double bound) {
  uint64_t s = (g_tp_seed ^= (g_tp_seed << 13));
  s ^= s >> 7;
  s ^= s << 17;
  g_tp_seed = s;
  const int64_t n = t.numel();
  dispatch_float(t.dtype(), [&]<typename T>() {
    T* p = t.data_ptr<T>();
    for (int64_t i = 0; i < n; ++i) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      const double u =
          static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFULL) /
          9007199254740992.0;
      p[i] = static_cast<T>((2.0 * u - 1.0) * bound);
    }
  });
}

// y = x @ W^T (+ bias), the dense Linear forward, reused per shard.
Tensor linear_apply(const Tensor& x, const Tensor& weight, const Tensor* bias) {
  Tensor wt = ops::transpose(weight, 0, 1);
  Tensor y = ops::matmul(x, wt);
  if (bias != nullptr) y = ops::add(y, *bias);
  return y;
}

// Copy `src`'s elements into the already-allocated, contiguous `dst` in
// place (preserving `dst`'s storage / registered-parameter aliasing). Used
// by `from_dense` to seed shard weights without re-registering parameters.
void copy_into(Tensor& dst, const Tensor& src) {
  TESSERACT_CHECK(dst.numel() == src.numel(),
                  "copy_into: numel mismatch {} vs {}", dst.numel(),
                  src.numel());
  Tensor s = src.contiguous();
  std::memcpy(dst.raw_data(), s.raw_data(),
              static_cast<std::size_t>(dst.numel()) * dst.itemsize());
}

}  // namespace

// ----------------------------- ColumnParallelLinear ----------------------- //

ColumnParallelLinear::ColumnParallelLinear(int64_t in_features,
                                           int64_t out_features,
                                           std::shared_ptr<CommBackend> comm,
                                           bool use_bias, bool gather_output,
                                           DType dtype)
    : in_features_(in_features),
      out_features_(out_features),
      world_size_(comm ? comm->world_size() : 1),
      use_bias_(use_bias),
      gather_output_(gather_output),
      comm_(std::move(comm)) {
  TESSERACT_CHECK(in_features_ > 0 && out_features_ > 0,
                  "ColumnParallelLinear: features must be positive");
  TESSERACT_CHECK(out_features_ % world_size_ == 0,
                  "ColumnParallelLinear: out_features {} not divisible by "
                  "world_size {}",
                  out_features_, world_size_);
  out_per_rank_ = out_features_ / world_size_;
  const double bound = 1.0 / std::sqrt(static_cast<double>(in_features_));
  for (int r = 0; r < world_size_; ++r) {
    Tensor w = Tensor::empty({out_per_rank_, in_features_}, dtype);
    uniform_init(w, bound);
    register_parameter("shard." + std::to_string(r) + ".weight", w);
    weight_shards_.push_back(w);
    if (use_bias_) {
      Tensor b = Tensor::empty({out_per_rank_}, dtype);
      uniform_init(b, bound);
      register_parameter("shard." + std::to_string(r) + ".bias", b);
      bias_shards_.push_back(b);
    }
  }
}

std::vector<Tensor> ColumnParallelLinear::forward_shards(const Tensor& x) {
  std::vector<Tensor> outs;
  outs.reserve(weight_shards_.size());
  for (int r = 0; r < world_size_; ++r) {
    const Tensor* b = use_bias_ ? &bias_shards_[r] : nullptr;
    outs.push_back(linear_apply(x, weight_shards_[r], b));
  }
  return outs;
}

Tensor ColumnParallelLinear::forward(const Tensor& x) {
  std::vector<Tensor> shards = forward_shards(x);
  if (!gather_output_) {
    // Caller asked for the sharded output but invoked the gathering
    // forward; gather anyway so the contract (full tensor) holds.
    return comm_->all_gather(shards, -1);
  }
  return comm_->all_gather(shards, -1);
}

std::shared_ptr<ColumnParallelLinear> ColumnParallelLinear::from_dense(
    const nn::Linear& dense, std::shared_ptr<CommBackend> comm,
    bool gather_output) {
  const Tensor& w = dense.weight();  // [out, in]
  const int64_t out_features = w.shape()[0];
  const int64_t in_features = w.shape()[1];
  const int world = comm->world_size();
  auto layer = std::make_shared<ColumnParallelLinear>(
      in_features, out_features, comm, dense.has_bias(), gather_output,
      w.dtype());
  const int64_t out_per = out_features / world;
  for (int r = 0; r < world; ++r) {
    // Copy the row-block [r*out_per : (r+1)*out_per, :] into shard r,
    // overwriting the ctor's random init in place (keeps the registered
    // parameter handle aliasing the same storage).
    copy_into(layer->weight_shards_[r], w.narrow(0, r * out_per, out_per));
    if (dense.has_bias()) {
      copy_into(layer->bias_shards_[r],
                dense.bias().narrow(0, r * out_per, out_per));
    }
  }
  return layer;
}

// ------------------------------- RowParallelLinear ------------------------ //

RowParallelLinear::RowParallelLinear(int64_t in_features, int64_t out_features,
                                     std::shared_ptr<CommBackend> comm,
                                     bool use_bias, DType dtype)
    : in_features_(in_features),
      out_features_(out_features),
      world_size_(comm ? comm->world_size() : 1),
      use_bias_(use_bias),
      comm_(std::move(comm)) {
  TESSERACT_CHECK(in_features_ > 0 && out_features_ > 0,
                  "RowParallelLinear: features must be positive");
  TESSERACT_CHECK(in_features_ % world_size_ == 0,
                  "RowParallelLinear: in_features {} not divisible by "
                  "world_size {}",
                  in_features_, world_size_);
  in_per_rank_ = in_features_ / world_size_;
  const double bound = 1.0 / std::sqrt(static_cast<double>(in_features_));
  for (int r = 0; r < world_size_; ++r) {
    Tensor w = Tensor::empty({out_features_, in_per_rank_}, dtype);
    uniform_init(w, bound);
    register_parameter("shard." + std::to_string(r) + ".weight", w);
    weight_shards_.push_back(w);
  }
  if (use_bias_) {
    bias_ = Tensor::empty({out_features_}, dtype);
    uniform_init(bias_, bound);
    register_parameter("bias", bias_);
  }
}

Tensor RowParallelLinear::forward_shards(const std::vector<Tensor>& x_shards) {
  TESSERACT_CHECK(
      static_cast<int>(x_shards.size()) == world_size_,
      "RowParallelLinear::forward_shards: expected {} input shards, got {}",
      world_size_, x_shards.size());
  std::vector<Tensor> partials;
  partials.reserve(x_shards.size());
  for (int r = 0; r < world_size_; ++r) {
    partials.push_back(linear_apply(x_shards[r], weight_shards_[r], nullptr));
  }
  Tensor y = comm_->all_reduce_sum(partials);
  if (use_bias_) y = ops::add(y, bias_);
  return y;
}

Tensor RowParallelLinear::forward(const Tensor& x) {
  const int64_t last = x.rank() - 1;
  // Use the *differentiable* split (SplitChunkBackward) rather than a raw
  // `narrow` view: `narrow` is a pure view with no grad-fn, which silently
  // severs the backward path to the input. With split_with_sizes the input
  // gradient reassembles correctly across shards (TP backward parity).
  std::vector<int64_t> sizes(static_cast<std::size_t>(world_size_),
                             in_per_rank_);
  std::vector<Tensor> x_shards = ops::split_with_sizes(x, sizes, last);
  return forward_shards(x_shards);
}

std::shared_ptr<RowParallelLinear> RowParallelLinear::from_dense(
    const nn::Linear& dense, std::shared_ptr<CommBackend> comm) {
  const Tensor& w = dense.weight();  // [out, in]
  const int64_t out_features = w.shape()[0];
  const int64_t in_features = w.shape()[1];
  const int world = comm->world_size();
  auto layer = std::make_shared<RowParallelLinear>(
      in_features, out_features, comm, dense.has_bias(), w.dtype());
  const int64_t in_per = in_features / world;
  for (int r = 0; r < world; ++r) {
    // Copy the column-block [:, r*in_per : (r+1)*in_per] into shard r.
    copy_into(layer->weight_shards_[r], w.narrow(1, r * in_per, in_per));
  }
  if (dense.has_bias()) {
    copy_into(layer->bias_, dense.bias());
  }
  return layer;
}

}  // namespace tesseract::distributed

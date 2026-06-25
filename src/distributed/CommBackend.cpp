#include "tesseract/distributed/CommBackend.hpp"

#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Indexing.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::distributed {

SimCommBackend::SimCommBackend(int world_size) : world_size_(world_size) {
  TESSERACT_CHECK(world_size_ >= 1,
                  "SimCommBackend: world_size must be >= 1 (got {})",
                  world_size_);
}

Tensor SimCommBackend::all_reduce_sum(
    const std::vector<Tensor>& shards) const {
  TESSERACT_CHECK(
      static_cast<int>(shards.size()) == world_size_,
      "all_reduce_sum: expected {} shards (world_size), got {}",
      world_size_, shards.size());
  TESSERACT_CHECK(!shards.empty(), "all_reduce_sum: empty shard list");
  // Left-to-right summation, matching the rank order. Goes through
  // `ops::add` so the reduction stays autograd-aware (each rank's partial
  // keeps its edge into the graph).
  Tensor acc = shards[0];
  for (std::size_t r = 1; r < shards.size(); ++r) {
    acc = ops::add(acc, shards[r]);
  }
  return acc;
}

Tensor SimCommBackend::all_gather(const std::vector<Tensor>& shards,
                                  int64_t dim) const {
  TESSERACT_CHECK(
      static_cast<int>(shards.size()) == world_size_,
      "all_gather: expected {} shards (world_size), got {}",
      world_size_, shards.size());
  TESSERACT_CHECK(!shards.empty(), "all_gather: empty shard list");
  return ops::cat(shards, dim);
}

}  // namespace tesseract::distributed

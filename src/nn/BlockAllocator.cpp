#include "tesseract/nn/BlockAllocator.hpp"

#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

BlockAllocator::BlockAllocator(int32_t num_blocks)
    : num_blocks_(num_blocks),
      allocated_(static_cast<std::size_t>(num_blocks), false) {
  TESSERACT_CHECK(num_blocks > 0,
                  "BlockAllocator: num_blocks must be positive, got {}",
                  num_blocks);
  // Seed the free-list with every id. Push in descending order so the
  // first `allocate()` returns block 0 — purely cosmetic (keeps the
  // single-request fixture's block ids contiguous + ascending, which
  // makes test assertions readable) since callers may not depend on
  // the concrete id.
  free_list_.reserve(static_cast<std::size_t>(num_blocks));
  for (int32_t id = num_blocks - 1; id >= 0; --id) {
    free_list_.push_back(id);
  }
}

int32_t BlockAllocator::allocate() {
  TESSERACT_CHECK(!free_list_.empty(),
                  "BlockAllocator::allocate: pool exhausted (all {} blocks "
                  "in use) — raise num_blocks or free finished requests",
                  num_blocks_);
  const int32_t id = free_list_.back();
  free_list_.pop_back();
  allocated_[static_cast<std::size_t>(id)] = true;
  return id;
}

void BlockAllocator::free(int32_t id) {
  TESSERACT_CHECK(id >= 0 && id < num_blocks_,
                  "BlockAllocator::free: id {} out of range [0, {})",
                  id, num_blocks_);
  TESSERACT_CHECK(allocated_[static_cast<std::size_t>(id)],
                  "BlockAllocator::free: block {} is already free "
                  "(double-free)", id);
  allocated_[static_cast<std::size_t>(id)] = false;
  free_list_.push_back(id);
}

void BlockAllocator::free_all() noexcept {
  free_list_.clear();
  for (int32_t id = num_blocks_ - 1; id >= 0; --id) {
    allocated_[static_cast<std::size_t>(id)] = false;
    free_list_.push_back(id);
  }
}

}  // namespace tesseract::nn

#pragma once

// Internal CUDA bridge for the Wave 4.5 (B-019b) paged KV-cache gather.
//
// `PagedKVCache` stores K/V in a physical block pool
// `[num_blocks, H, block_size, D_head]` indexed by a per-request block
// table. To feed the existing `ops::attention` (which expects a
// contiguous `[B, H, L, D_head]` prefix) the cache gathers the valid
// prefix out of the scattered blocks. On CPU that's a handful of
// `memcpy`s; on CUDA a naive per-block `cudaMemcpyAsync` loop fires
// thousands of tiny launches per decode step (B·H·ceil(L/block_size))
// and is launch-overhead-bound. This bridge replaces that loop with a
// single element-wise gather kernel that reads the block table from
// device memory.
//
// Layering matches the other `detail/*` bridges: header stays
// C++17-parseable (no CUDA types), entry point takes `const void*` /
// `void*` and an element size so it can be called from a plain `.cpp`.
//
// The gather is a pure bit-move, so it dispatches on element *size*
// (8 / 4 / 2 bytes) rather than the floating dtype — one kernel covers
// Float64 / Float32 / {Float16, BFloat16}.

#include <cstdint>

namespace tesseract::cuda::detail {

// Gather the `[B, H, L, D_head]` prefix out of a paged pool.
//
//   out[b, h, t, d] = pool[ block_table[b, t / block_size], h,
//                           t % block_size, d ]
//
// `block_table` is a device int32 buffer of shape `[B, num_logical]`
// (row-major) where `num_logical = ceil(L / block_size)`. `pool` is the
// physical block pool `[num_blocks, H, block_size, D_head]`. `out` is a
// fresh contiguous `[B, H, L, D_head]` tensor. `elem_size` is the byte
// width of the (floating) element type — 2, 4, or 8.
//
// All pointers must live on `device_index`; the op layer validates the
// shapes and device.
void launch_paged_gather(int device_index, int64_t elem_size,
                         const void* pool, void* out,
                         const int32_t* block_table,
                         int64_t B, int64_t H, int64_t L, int64_t head_dim,
                         int64_t block_size, int64_t num_blocks,
                         int64_t num_logical,
                         void* stream);

}  // namespace tesseract::cuda::detail

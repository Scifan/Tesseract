// Wave 4.5 (B-019b): paged KV-cache gather kernel.
//
// Collapses the per-block `cudaMemcpyAsync` loop in `PagedKVCache`'s
// gather into a single element-wise kernel: one thread per output
// element of the `[B, H, L, D_head]` prefix, each thread following the
// device-resident block table to its source slot in the physical pool
// `[num_blocks, H, block_size, D_head]`.
//
//   idx ∈ [0, B·H·L·D_head):
//     d       = idx % D_head
//     t       = (idx / D_head) % L
//     h       = (idx / (D_head·L)) % H
//     b       =  idx / (D_head·L·H)
//     logical = t / block_size,  slot = t % block_size
//     p       = block_table[b·num_logical + logical]
//     out[idx] = pool[((p·H + h)·block_size + slot)·D_head + d]
//
// The gather is a pure bit-move so we template on element *size* (a POD
// of 2 / 4 / 8 bytes) rather than the floating dtype — one kernel body
// services Float64 / Float32 / {Float16, BFloat16}. An out-of-range
// physical id (corrupt block table) is clamped to block 0 rather than
// reading out of bounds; the op layer guarantees the table is valid, so
// this is just defense in depth that keeps a logic bug from becoming an
// illegal-address crash.

#include "KernelUtils.cuh"

#include "tesseract/cuda/detail/PagedKV.hpp"

namespace tesseract::cuda::detail {

namespace {

template <typename Word>
__global__ void paged_gather_kernel(const Word* __restrict__ pool,
                                    Word* __restrict__ out,
                                    const int32_t* __restrict__ block_table,
                                    int64_t H, int64_t L, int64_t head_dim,
                                    int64_t block_size, int64_t num_blocks,
                                    int64_t num_logical, int64_t total) {
  const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = tid; idx < total; idx += stride) {
    const int64_t d = idx % head_dim;
    int64_t rem     = idx / head_dim;
    const int64_t t = rem % L;
    rem /= L;
    const int64_t h = rem % H;
    const int64_t b = rem / H;

    const int64_t logical = t / block_size;
    const int64_t slot    = t % block_size;
    int64_t p = block_table[b * num_logical + logical];
    if (p < 0 || p >= num_blocks) p = 0;  // defense in depth

    const int64_t src =
        ((p * H + h) * block_size + slot) * head_dim + d;
    out[idx] = pool[src];
  }
}

}  // namespace

void launch_paged_gather(int device_index, int64_t elem_size,
                         const void* pool, void* out,
                         const int32_t* block_table,
                         int64_t B, int64_t H, int64_t L, int64_t head_dim,
                         int64_t block_size, int64_t num_blocks,
                         int64_t num_logical,
                         void* stream_handle) {
  const int64_t total = B * H * L * head_dim;
  if (total == 0) return;

  DeviceGuard g(device_index);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);

  const dim3 grid(launch_grid(total));
  const dim3 block(kBlockSize);

  switch (elem_size) {
    case 8:
      paged_gather_kernel<uint64_t><<<grid, block, 0, stream>>>(
          static_cast<const uint64_t*>(pool), static_cast<uint64_t*>(out),
          block_table, H, L, head_dim, block_size, num_blocks, num_logical,
          total);
      break;
    case 4:
      paged_gather_kernel<uint32_t><<<grid, block, 0, stream>>>(
          static_cast<const uint32_t*>(pool), static_cast<uint32_t*>(out),
          block_table, H, L, head_dim, block_size, num_blocks, num_logical,
          total);
      break;
    case 2:
      paged_gather_kernel<uint16_t><<<grid, block, 0, stream>>>(
          static_cast<const uint16_t*>(pool), static_cast<uint16_t*>(out),
          block_table, H, L, head_dim, block_size, num_blocks, num_logical,
          total);
      break;
    default:
      throw DeviceError(fmt::format(
          "[tesseract] CUDA paged_gather on element size {} is not "
          "supported (2 / 4 / 8 bytes only).", elem_size));
  }
  check_launch("paged_gather_kernel");
}

}  // namespace tesseract::cuda::detail

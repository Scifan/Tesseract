#pragma once

#include <cstdint>
#include <vector>

#include "tesseract/core/Tensor.hpp"

// Autograd-aware concat / slice / index-select / gather operations. All four
// share a common implementation pattern: a custom forward that walks the
// logical index space once, and a backward that relies on the engine's
// leaf-accumulator contract (`ops::add`) so multiple backward slabs sum
// correctly into the parent tensor's gradient. Shapes / dims follow
// PyTorch conventions (negative `dim` allowed, indices are Int64).
namespace tesseract::ops {

// ----------------------------------------------------------------------------
// cat / split
// ----------------------------------------------------------------------------

// Concatenate `tensors` along `dim`. All operands must share the same dtype
// and device and must agree on every dimension except `dim`. `dim` is
// normalized PyTorch-style (`dim < 0` counts from the right). The output
// dtype matches the inputs. `tensors` must be non-empty.
Tensor cat(const std::vector<Tensor>& tensors, int64_t dim);

// Split `src` along `dim` into `sizes.size()` chunks with
// `sizes[i] == out[i].shape()[dim]`. The sum of `sizes` must equal
// `src.shape()[dim]`. Sizes must be non-negative.
std::vector<Tensor> split_with_sizes(const Tensor& src,
                                     const std::vector<int64_t>& sizes,
                                     int64_t dim);

// Split `src` along `dim` into chunks of at most `size` elements (last chunk
// may be smaller if `src.shape()[dim]` is not a multiple of `size`).
std::vector<Tensor> split(const Tensor& src, int64_t size, int64_t dim);

// ----------------------------------------------------------------------------
// index_select / gather
// ----------------------------------------------------------------------------

// PyTorch-style `index_select(src, dim, indices)`: the output has the same
// rank and shape as `src` except along `dim`, where it is replaced by
// `indices.numel()`. `indices` must be a rank-1 Int64 tensor with values
// in `[0, src.shape()[dim])`. The backward is a scatter-add of the gradient
// into the rows identified by `indices`.
Tensor index_select(const Tensor& src, int64_t dim, const Tensor& indices);

// PyTorch-style `gather(src, dim, indices)`: `indices` must have the same
// rank as `src` and `indices.shape()[d] <= src.shape()[d]` for every `d`.
// The output shape equals `indices.shape()`. For every multi-index `I` in
// the output, `out[I] = src[I with axis dim replaced by indices[I]]`.
// The backward scatters grad back element-wise (duplicates sum, matching
// PyTorch / ONNX GatherElements).
Tensor gather(const Tensor& src, int64_t dim, const Tensor& indices);

}  // namespace tesseract::ops

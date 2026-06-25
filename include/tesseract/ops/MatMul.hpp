#pragma once

#include "tesseract/core/Tensor.hpp"

namespace tesseract::ops {

// Matrix multiplication with batched (rank >= 2) operands.
//
//   lhs.shape = [*lhs_batch, M, K]
//   rhs.shape = [*rhs_batch, K, N]
//   out.shape = [*broadcast(lhs_batch, rhs_batch), M, N]
//
// The leading "batch" dims broadcast NumPy/PyTorch-style — in particular,
// one operand can be rank-2 and the other rank>2 and it behaves as if the
// rank-2 operand were repeated across every batch position. Both inputs
// must share the same floating-point dtype (Float32/Float64). Non-
// contiguous inputs are first materialized via `contiguous()`. The
// autograd backward correctly sum-reduces over any broadcasted batch
// axes so `grad_lhs.shape() == lhs.shape()` and likewise for `rhs`.
Tensor matmul(const Tensor& lhs, const Tensor& rhs);

}  // namespace tesseract::ops

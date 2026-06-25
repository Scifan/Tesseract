#pragma once

// Internal CUDA bridge for the fused RMSNorm / LayerNorm kernels (Wave 2).
//
// Same layering convention as `detail/RotaryEmbedding.hpp`: the header
// stays C++17-parseable (no `std::span`, no CUDA types), and the entry
// point takes `const void*` / `void*` with an explicit `DType`
// selector so it can be included from plain `.cpp` translation units.
//
// Scope:
//   * `x` is a row-major contiguous tensor whose last dim is the
//     normalization dim `D`. All leading dims are collapsed into
//     `outer` at the op-layer boundary.
//   * `weight` is a contiguous `[D]` tensor, same dtype as `x`.
//   * `out` is written as a contiguous `[outer, D]` tensor of the
//     same shape and dtype as `x`.
//
// Dtype policy (matches B-015 / B-016 / RoPE):
//   * Float32 / Float64 — compute in the storage type (or FP32 for
//     half-precision storage). The reduction accumulates in the
//     compute type.
//   * Float16 / BFloat16 — load/compute/store through an FP32
//     accumulator. The rsqrt itself runs in FP32 so the dynamic
//     range of FP16 never limits the stats.
//
// Forward-only. Backward continues to flow through the composite
// primitives' autograd nodes (mul/mean/add/sqrt/div/mul). The
// forward-fused speedup alone moves the decode-phase RMSNorm from
// ~5-6 passes over `x` to a single pass, which is the full memory-
// bandwidth win on what is otherwise a memory-bound op.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// Fused forward: `out[i, j] = (x[i, j] * rsqrt(mean_j(x[i, j]^2) + eps)) * weight[j]`.
//
// Launches one block per row (`outer` blocks total) with a fixed
// block width. Each block streams the row twice through global
// memory: once to accumulate the sum-of-squares, once to write the
// normalized-and-scaled output. Weight is read once per thread in the
// second pass. `eps` is taken in `double` regardless of storage
// dtype so the caller never has to cast.
void launch_rms_norm(DType dtype, int device_index,
                     int64_t outer, int64_t D,
                     const void* x,
                     const void* weight,
                     double eps,
                     void* out,
                     void* stream);

// Fused LayerNorm forward:
//   mu    = mean(x[i, :])
//   var   = mean((x[i, :] - mu)^2)                       // biased (ATen default)
//   out[i, j] = ((x[i, j] - mu) * rsqrt(var + eps)) * weight[j] + bias[j]
//
// `bias` may be `nullptr`, in which case the bias term is skipped
// (matches `ops::layer_norm(x, weight, Tensor{}, eps)`). Same
// one-block-per-row launch layout as RMSNorm.
void launch_layer_norm(DType dtype, int device_index,
                       int64_t outer, int64_t D,
                       const void* x,
                       const void* weight,
                       const void* bias,
                       double eps,
                       void* out,
                       void* stream);

}  // namespace tesseract::cuda::detail

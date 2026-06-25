#pragma once

// Internal CUDA bridge for Wave 9 (B-031) KV-cache INT8 quantization.
//
// Layering matches the other `detail/*` bridges: this header stays
// C++17-parseable (no CUDA types), entry points take `const void*` /
// `void*` so they can be called from a plain `.cpp`. The op layer
// (`src/quant/QuantizeKV.cpp`) validates shapes/dtype/device and only
// reaches for these launchers on a CUDA device; CPU-only builds get the
// throwing stubs in `QuantizeKVStub.cpp`.
//
// Both ops operate on a flattened `[rows, D_head]` view: `rows` is the
// product of all-but-the-last dim, `D_head` the last dim. Per-token,
// per-head symmetric INT8 — one FP32 scale per row.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// One thread per row: FP32 absmax over `D_head`, scale = absmax/127
// (1.0 for an all-zero row), then quantize each element (round-nearest-
// even, clamp [-127,127]). `x` is `dtype` (FP32/FP16/BF16); `q` is Int8;
// `scale` is FP32. All device pointers on `device_index`.
void launch_quantize_kv_per_token(DType dtype, int device_index,
                                   int64_t rows, int64_t head_dim,
                                   const void* x, int8_t* q, float* scale,
                                   void* stream);

// One thread per element: `out[i] = q[i] * scale[i / D_head]`, narrowed
// to `dtype` on store.
void launch_dequantize_kv_per_token(DType dtype, int device_index,
                                     int64_t rows, int64_t head_dim,
                                     const int8_t* q, const float* scale,
                                     void* out, void* stream);

}  // namespace tesseract::cuda::detail

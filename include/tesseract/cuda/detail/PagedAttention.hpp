#pragma once

// Internal CUDA bridge for Wave 11 (B-032+) fused paged decode-attention.
//
// Layering matches the other `detail/*` bridges: this header is plain
// C++17 (no CUDA types), the entry point takes `const void*` / `void*`
// so it is callable from a `.cpp`. The op layer (`src/nn/PagedAttention.cpp`)
// validates shapes/dtype/device and only reaches for this launcher on a
// CUDA device; CPU-only builds get the throwing stub in
// `PagedAttentionStub.cpp`.
//
// One CUDA block (a single warp) per `(request, query-head)` pair. The
// warp streams the request's KV prefix (followed via its block table into
// the shared physical pool `[num_blocks, Hkv, block_size, D]`), folds the
// score dot via warp-shuffle, and accumulates the output with an online
// softmax — never materializing the `[1, S_k]` score row. FP32 interior
// math regardless of storage dtype; GQA head `h` reads KV head `h/group`.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

void launch_paged_decode_attention(
    DType dtype, int device_index,
    int64_t A, int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale,
    const void* q, const void* k_pool, const void* v_pool,
    const int32_t* block_tables, const int32_t* lens,
    void* o, void* stream);

// Wave 12 (B-032++): INT8-direct variant. `k_pool`/`v_pool` are `int8_t`
// payloads with per-(block,head,slot) FP32 `k_scale`/`v_scale`; the kernel
// dequantizes `int8 * scale` inline. `dtype` is the FP type of `q` / `o`.
void launch_paged_decode_attention_int8(
    DType dtype, int device_index,
    int64_t A, int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale,
    const void* q, const int8_t* k_pool, const float* k_scale,
    const int8_t* v_pool, const float* v_scale,
    const int32_t* block_tables, const int32_t* lens,
    void* o, void* stream);

// Wave 14 (B-032++++): fused paged PREFILL (S_new > 1). `q` is [A, S, H, D];
// `kv_lens[r]` is request r's total context length *after* appending its S
// new tokens, so query s attends causally to keys `[0, kv_lens[r]-S+s]`.
// Grid is one warp per `(r, s, h)` triple.
void launch_paged_prefill_attention(
    DType dtype, int device_index,
    int64_t A, int64_t S, int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale,
    const void* q, const void* k_pool, const void* v_pool,
    const int32_t* block_tables, const int32_t* kv_lens,
    void* o, void* stream);

void launch_paged_prefill_attention_int8(
    DType dtype, int device_index,
    int64_t A, int64_t S, int64_t H, int64_t Hkv, int64_t D,
    int64_t block_size, int64_t num_blocks, int64_t max_logical,
    int group, float scale,
    const void* q, const int8_t* k_pool, const float* k_scale,
    const int8_t* v_pool, const float* v_scale,
    const int32_t* block_tables, const int32_t* kv_lens,
    void* o, void* stream);

}  // namespace tesseract::cuda::detail

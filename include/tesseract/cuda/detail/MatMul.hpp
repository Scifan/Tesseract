#pragma once

// Internal CUDA bridge for the M2G matmul path. Same `detail/` contract
// as `Elementwise.hpp` / `Reduction.hpp` / `Softmax.hpp` / `Loss.hpp`:
//
//   * Parseable at C++17 — the real implementation lives in
//     `src/cuda/MatMul.cpp`, which (intentionally) is a plain `.cpp`
//     compiled by the host C++ compiler. cuBLASLt ships a C header
//     that's consumable from any C++ dialect; there are no CUDA
//     device-code kernels on the matmul path (cuBLASLt does the heavy
//     lifting), so we don't need `nvcc` for this TU. Keeping it `.cpp`
//     also means the file participates in the host link without
//     pulling in the `CUDA_STANDARD=17` constraint on every downstream
//     header.
//
//   * One `launch_matmul` entry point covers both the rank-2 case and
//     the per-slab batched case. For rank > 2 the op layer
//     (`src/ops/cpu/MatMul.cpp`) walks the broadcasted batch grid and
//     issues one `launch_matmul` call per output slab — same pattern
//     as the CPU GEMM dispatcher. cuBLASLt has strided-batched
//     primitives too, but iterating in C++ keeps the layout-detection
//     logic local to the op layer and avoids a broadcast-stride
//     encoding on the bridge; revisit in M3 if kernel launch latency
//     shows up on the profile for small batched GEMMs.
//
//   * Layout: row-major throughout. We pass the two logical operand
//     ops (`None` / `Transpose`) so the bridge can map to cuBLASLt's
//     `CUBLAS_OP_N` / `CUBLAS_OP_T`; `ld` is the row-stride of the
//     *stored* matrix (i.e. the dense dim of the underlying memory,
//     before any transpose is applied). This matches what the op
//     layer can read directly off `Tensor::strides()` — stride[0] for
//     a row-major `[M,K]` contig, stride[1] for a transposed view.
//
//   * No fallback to a custom kernel when a shape is awkward. Non-
//     plausible layouts (innermost stride neither 1 nor equal to the
//     other-dim size) land on a clean `TESSERACT_CHECK` with a
//     recommendation to `.contiguous()` the operand on host (or wait
//     for M2H strided copy). Supported in M2G: `{Float32, Float64,
//     Float16, BFloat16}`. Mixed-dtype cases land on the op-layer
//     `TESSERACT_CHECK`, same as CPU.

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

// Matches `cublasOperation_t` semantics but keeps cuBLASLt out of the
// public bridge header. The op layer fills this in after inspecting
// the operand's last-two strides.
enum class MmOp : int {
  None = 0,       // use matrix as stored (`op = N`)
  Transpose = 1,  // transpose before multiplying (`op = T`)
};

// Single-slab row-major GEMM:  C[M,N] = A (op_a) * B (op_b), dense output.
//
// `a`, `b`, `c` point into device memory on `device_index`. The
// logical matmul dims are M, N, K (after any implicit transpose).
//
//   * `lda` / `ldb` are the row-stride of the *stored* operand (before
//     op), i.e. the number of elements per row in memory. For a row-
//     major `[rows, cols]` contig tensor this is `cols`.
//   * `ldc` is the row-stride of C (always row-major contig in M2G —
//     the op layer allocates C via `Tensor::empty`, so `ldc == N`).
//
// Supported dtypes: `Float32`, `Float64`, `Float16`, `BFloat16`. FP16
// and BF16 accumulate in FP32 (`CUBLAS_COMPUTE_32F`) so numerical
// parity with the CPU upcast-to-FP32 path is tight (sub-1e-2 on
// 4096² problems). FP64 uses `CUBLAS_COMPUTE_64F` end-to-end.
//
// The call is enqueued on `stream_handle` (the raw `cudaStream_t`
// cast to `void*`); no implicit synchronization. The heuristic for
// algo selection is cached inside cuBLASLt across calls so per-slab
// overhead is amortized in batched matmul.
void launch_matmul(DType dtype, int device_index,
                   int64_t M, int64_t N, int64_t K,
                   const void* a, MmOp op_a, int64_t lda,
                   const void* b, MmOp op_b, int64_t ldb,
                   void* c, int64_t ldc,
                   void* stream_handle);

// Opaque accessor for our per-device cuBLASLt handle. Returned as a
// `void*` so this header stays free of `<cublasLt.h>` — callers that
// need the real `cublasLtHandle_t` cast via
//   `reinterpret_cast<cublasLtHandle_t>(tesseract::cuda::detail::get_cublaslt_handle(dev))`
// Creating a second handle in the same process yields a second,
// independently-tuned heuristic cache — the two handles can pick
// different algorithms for the exact same shape + dtype under some
// call-ordering patterns. The micro-benchmark suite shares this
// handle so the raw-vs-ops comparison is not confounded by algo
// divergence.
void* get_cublaslt_handle(int device_index);

}  // namespace tesseract::cuda::detail

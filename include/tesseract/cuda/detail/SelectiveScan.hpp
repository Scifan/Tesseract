#pragma once

// Internal CUDA bridge for M4 Track A2 (B-039) selective state-space scan.
//
// Layering matches the other `detail/*` bridges: this header is plain C++17
// (no CUDA types), the entry point takes `const void*` / `void*` so it is
// callable from a `.cpp`. The op layer (`src/ops/cpu/SelectiveScan.cpp`)
// validates shapes/dtype/device and only reaches for this launcher on a CUDA
// device; CPU-only builds get the throwing stub in `SelectiveScanStub.cpp`.
//
// One thread per `(batch b, inner channel d)` pair: the thread carries the
// `N`-wide hidden state in registers and runs the sequential `t`-recurrence,
// so threads are independent and the launch is embarrassingly parallel across
// `B·D`. FP32 interior math regardless of storage dtype. Kernel caps
// `d_state N <= 32` (state lives in a per-thread register array).

#include <cstdint>

#include "tesseract/core/DType.hpp"

namespace tesseract::cuda::detail {

void launch_selective_scan(DType dtype, int device_index,
                           int64_t B, int64_t L, int64_t D, int64_t N,
                           const void* u, const void* delta, const void* A,
                           const void* Bm, const void* Cm, const void* Dskip,
                           const void* state_in,  // may be null ⇒ zero init
                           void* y, void* state_out, void* stream);

}  // namespace tesseract::cuda::detail

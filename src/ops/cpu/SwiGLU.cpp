// Wave 4.1 (B-025): fused SwiGLU activation op.
//
// Semantics:
//     out[i] = silu(gate[i]) * up[i]
//            = (gate[i] * sigmoid(gate[i])) * up[i]
//
// Replaces the three-kernel tail (`sigmoid(gate)` + `mul` + `mul`) in
// `nn::FeedForward` — every Llama block's FFN hits this path twice
// (once forward, once in backward via the composite, but the backward
// *does* go through the composite because the fused CUDA kernel is
// forward-only, same convention as `rms_norm`). See the shape and
// dtype contract in `include/tesseract/ops/Activation.hpp`.
//
// Layering (identical to `rms_norm` / `dequantize_matmul_int8`):
//   1. Validate shapes/dtypes/devices at the op-layer boundary.
//   2. Fast path: CUDA + contiguous + no-autograd → `launch_swiglu_silu_gate`.
//   3. Fallback: composite `mul(gate, sigmoid(gate)) * up` — autograd
//      threads through `SigmoidBackward` + `MulBackward` automatically,
//      no custom node.
//   4. Emit a `swiglu_silu_gate` marker into the active `GraphScope`
//      on *both* paths so JIT passes see the op atomically.

#include "tesseract/ops/Activation.hpp"

#include <cmath>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/autograd/Function.hpp"
#include "tesseract/cuda/detail/SwiGLU.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::ops {

namespace {

// CPU reference. One pass over contiguous storage; FP32-accumulated
// on half-precision inputs so numerics match the CUDA fast path to
// the usual `1e-5` FP32 tolerance that the parity tests use. The
// `constexpr` branch for native float/double keeps the code byte-
// identical to the pre-fusion composite on those dtypes.
template <typename T>
void swiglu_silu_gate_cpu_contig(const T* gate, const T* up, T* out,
                                 int64_t numel) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    for (int64_t i = 0; i < numel; ++i) {
      const T g = gate[i];
      const T u = up[i];
      const T sig = static_cast<T>(T{1} / (T{1} + std::exp(-g)));
      out[i] = (g * sig) * u;
    }
  } else {
    for (int64_t i = 0; i < numel; ++i) {
      const float g = static_cast<float>(gate[i]);
      const float u = static_cast<float>(up[i]);
      const float sig = 1.0f / (1.0f + std::exp(-g));
      out[i] = static_cast<T>((g * sig) * u);
    }
  }
}

}  // namespace

Tensor swiglu_silu_gate(const Tensor& gate, const Tensor& up) {
  TESSERACT_CHECK(gate.defined() && up.defined(),
                  "swiglu_silu_gate: gate and up must be defined tensors");
  TESSERACT_CHECK(dtype_is_floating(gate.dtype()),
                  "swiglu_silu_gate: floating-point dtype required, got {}",
                  dtype_name(gate.dtype()));
  TESSERACT_CHECK(gate.dtype() == up.dtype(),
                  "swiglu_silu_gate: gate/up dtype mismatch ({} vs {})",
                  dtype_name(gate.dtype()), dtype_name(up.dtype()));
  TESSERACT_CHECK(gate.device() == up.device(),
                  "swiglu_silu_gate: gate/up device mismatch ({} vs {})",
                  gate.device().to_string(), up.device().to_string());
  TESSERACT_CHECK(gate.shape() == up.shape(),
                  "swiglu_silu_gate: gate/up shape mismatch ({} vs {})",
                  gate.shape().to_string(), up.shape().to_string());

  // Fused CUDA fast path — must satisfy every invariant the kernel
  // assumes (contiguous storage + no autograd). Broadcasting is
  // deliberately not supported here; the caller always feeds us
  // two like-shaped projections, and the composite fallback below
  // handles any other layout for free.
  const bool want_fused_cuda =
      gate.device().is_cuda() &&
      gate.is_contiguous() && up.is_contiguous() &&
      !(is_grad_enabled() && autograd::any_requires_grad(gate, up));

  if (want_fused_cuda) {
    Tensor out = Tensor::empty(gate.shape(), gate.dtype(), gate.device());
    Stream s = current_stream(gate.device());
    cuda::detail::launch_swiglu_silu_gate(
        gate.dtype(), gate.device().index,
        gate.numel(),
        gate.raw_data(), up.raw_data(),
        out.raw_data(), s.native_handle());
    graph::maybe_record("swiglu_silu_gate", {&gate, &up}, {&out});
    return out;
  }

  // CPU fast path: contiguous + same-shape + no-autograd. Still fused
  // (one loop instead of three temporaries) but without any CUDA
  // launch. The CPU path is primarily for CPU-only CI and for
  // platforms where the CUDA backend isn't compiled in.
  const bool want_fused_cpu =
      gate.device().is_cpu() &&
      gate.is_contiguous() && up.is_contiguous() &&
      !(is_grad_enabled() && autograd::any_requires_grad(gate, up));

  if (want_fused_cpu) {
    Tensor out = Tensor::empty(gate.shape(), gate.dtype(), gate.device());
    dispatch_float_with_half(gate.dtype(), [&]<typename T>() {
      swiglu_silu_gate_cpu_contig<T>(
          gate.data_ptr<T>(), up.data_ptr<T>(),
          out.data_ptr<T>(), gate.numel());
    });
    graph::maybe_record("swiglu_silu_gate", {&gate, &up}, {&out});
    return out;
  }

  // Composite fallback. Autograd-active path or non-contiguous input:
  // build the equivalent primitives so `SigmoidBackward` /
  // `MulBackward` carry the gradient through automatically. We
  // deliberately do *not* hand-roll a custom `SwiGLUBackward`:
  //   d/d(gate) out = up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
  //                 = up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
  //   d/d(up)   out = gate * sigmoid(gate)
  // Every term above is already expressible via the composite, and
  // the primitive backward nodes handle broadcasting / reductions
  // uniformly — which is exactly the policy RMSNorm / BatchNorm use.
  Tensor sig       = sigmoid(gate);          // [..., d_ff]
  Tensor silu_gate = mul(gate, sig);          // x * sigmoid(x)
  Tensor out       = mul(silu_gate, up);     // gated product
  graph::maybe_record("swiglu_silu_gate", {&gate, &up}, {&out});
  return out;
}

}  // namespace tesseract::ops

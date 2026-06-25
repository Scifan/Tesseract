#include "tesseract/nn/Linear.hpp"

#include <cmath>
#include <cstdint>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

namespace {

// Deterministic-but-varying seed source so that successive Linear layers use
// different initial weights without explicit per-layer seeding. Bumped every
// call; thread-local to avoid contention.
thread_local uint64_t g_linear_seed = 0xC2B2AE3D27D4EB4FULL;

void uniform_init(Tensor& t, double bound) {
  uint64_t s = (g_linear_seed ^= (g_linear_seed << 13));
  s ^= s >> 7;
  s ^= s << 17;
  g_linear_seed = s;

  const int64_t n = t.numel();
  // `_with_half` so FP16 / BF16 models can be constructed directly. The
  // RNG math stays in fp64 and only the final store down-casts, so the
  // half-precision tables start from the same high-precision source.
  dispatch_float_with_half(t.dtype(), [&]<typename T>() {
    T* p = t.data_ptr<T>();
    for (int64_t i = 0; i < n; ++i) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      const double u = static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0;
      p[i] = static_cast<T>((2.0 * u - 1.0) * bound);
    }
  });
}

}  // namespace

Linear::Linear(int64_t in_features, int64_t out_features, bool use_bias, DType dtype)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {
  TESSERACT_CHECK(in_features > 0 && out_features > 0,
                  "Linear: features must be positive (got {} and {})", in_features, out_features);

  // PyTorch-style default: U(-1/sqrt(in), 1/sqrt(in)).
  const double bound = 1.0 / std::sqrt(static_cast<double>(in_features));

  weight_ = Tensor::empty({out_features, in_features}, dtype);
  uniform_init(weight_, bound);
  register_parameter("weight", weight_);

  if (use_bias_) {
    bias_ = Tensor::empty({out_features}, dtype);
    uniform_init(bias_, bound);
    register_parameter("bias", bias_);
  }
}

Tensor Linear::forward(const Tensor& x) {
  // Accept any rank >= 2 with last dim == in_features. `ops::matmul`
  // already supports batched [..., M, K] @ [K, N] → [..., M, N] via
  // its rank-2 rhs broadcast, and `ops::add`'s NumPy-style broadcast
  // handles the rank-1 bias against the leading dims. This matches
  // PyTorch's `torch.nn.Linear` contract: any leading shape is
  // preserved. Relaxing the rank check in M2K so `[B, S, D_in]`
  // transformer inputs feed Linear without a manual reshape-roundtrip.
  TESSERACT_CHECK(x.rank() >= 2,
                  "Linear::forward: expected rank >= 2 input [..., in_features], got {}",
                  x.shape().to_string());
  const int64_t last = x.shape()[x.rank() - 1];
  TESSERACT_CHECK(last == in_features_,
                  "Linear::forward: input feature dim {} != in_features {}",
                  last, in_features_);
  Tensor wt = ops::transpose(weight_, 0, 1);  // [in, out], keeps autograd edge
  Tensor y  = ops::matmul(x, wt);             // [..., out]
  if (use_bias_) {
    y = ops::add(y, bias_);                   // broadcast [out] across all leading dims
  }
  return y;
}

}  // namespace tesseract::nn

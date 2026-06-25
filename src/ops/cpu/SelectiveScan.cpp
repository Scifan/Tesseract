#include "tesseract/ops/SelectiveScan.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/SelectiveScan.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::ops {

SelectiveScanResult selective_scan(const Tensor& u, const Tensor& delta,
                                   const Tensor& A, const Tensor& B,
                                   const Tensor& C, const Tensor& D,
                                   const Tensor& state_in) {
  TESSERACT_CHECK(u.defined() && delta.defined() && A.defined() &&
                  B.defined() && C.defined() && D.defined(),
                  "selective_scan: u/delta/A/B/C/D must all be defined");
  TESSERACT_CHECK(u.rank() == 3 && delta.rank() == 3,
                  "selective_scan: u and delta must be [B, L, D] (got {} / {})",
                  u.shape().to_string(), delta.shape().to_string());
  const int64_t Bsz = u.shape()[0];
  const int64_t L    = u.shape()[1];
  const int64_t Dim  = u.shape()[2];
  TESSERACT_CHECK(delta.shape() == u.shape(),
                  "selective_scan: delta shape {} != u shape {}",
                  delta.shape().to_string(), u.shape().to_string());
  TESSERACT_CHECK(A.rank() == 2 && A.shape()[0] == Dim,
                  "selective_scan: A must be [D, N] with D={}, got {}", Dim,
                  A.shape().to_string());
  const int64_t N = A.shape()[1];
  TESSERACT_CHECK(B.rank() == 3 && B.shape()[0] == Bsz && B.shape()[1] == L &&
                  B.shape()[2] == N,
                  "selective_scan: B must be [B, L, N], got {}",
                  B.shape().to_string());
  TESSERACT_CHECK(C.shape() == B.shape(),
                  "selective_scan: C shape {} != B shape {}",
                  C.shape().to_string(), B.shape().to_string());
  TESSERACT_CHECK(D.rank() == 1 && D.shape()[0] == Dim,
                  "selective_scan: D must be [D]={}, got {}", Dim,
                  D.shape().to_string());

  const DType dtype = u.dtype();
  TESSERACT_CHECK(delta.dtype() == dtype && A.dtype() == dtype &&
                  B.dtype() == dtype && C.dtype() == dtype && D.dtype() == dtype,
                  "selective_scan: all operands must share dtype");
  TESSERACT_CHECK(u.device() == delta.device() && u.device() == A.device() &&
                  u.device() == B.device() && u.device() == C.device() &&
                  u.device() == D.device(),
                  "selective_scan: all operands must share a device");
  const bool have_state = state_in.defined();
  if (have_state) {
    TESSERACT_CHECK(state_in.rank() == 3 && state_in.shape()[0] == Bsz &&
                    state_in.shape()[1] == Dim && state_in.shape()[2] == N,
                    "selective_scan: state_in must be [B, D, N], got {}",
                    state_in.shape().to_string());
    TESSERACT_CHECK(state_in.dtype() == dtype &&
                    state_in.device() == u.device(),
                    "selective_scan: state_in dtype/device must match u");
  }
  TESSERACT_CHECK(u.is_contiguous() && delta.is_contiguous() &&
                  A.is_contiguous() && B.is_contiguous() &&
                  C.is_contiguous() && D.is_contiguous() &&
                  (!have_state || state_in.is_contiguous()),
                  "selective_scan: all operands must be contiguous");

  Tensor y = Tensor::empty(Shape({Bsz, L, Dim}), dtype, u.device());
  Tensor state_out = Tensor::empty(Shape({Bsz, Dim, N}), dtype, u.device());
  if (Bsz == 0 || L == 0 || Dim == 0) return {y, state_out};

  if (u.device().is_cuda()) {
    TESSERACT_CHECK(N <= 32,
                    "selective_scan: CUDA kernel supports d_state N <= 32 "
                    "(got {})", N);
    Stream s = current_stream(u.device());
    cuda::detail::launch_selective_scan(
        dtype, u.device().index, Bsz, L, Dim, N,
        u.raw_data(), delta.raw_data(), A.raw_data(), B.raw_data(),
        C.raw_data(), D.raw_data(),
        have_state ? state_in.raw_data() : nullptr,
        y.raw_data(), state_out.raw_data(), s.native_handle());
    return {y, state_out};
  }

  dispatch_float(dtype, [&]<typename T>() {
    using Acc = T;
    const T* up = u.data_ptr<T>();
    const T* dp = delta.data_ptr<T>();
    const T* ap = A.data_ptr<T>();
    const T* bp = B.data_ptr<T>();
    const T* cp = C.data_ptr<T>();
    const T* dskip = D.data_ptr<T>();
    const T* sp = have_state ? state_in.data_ptr<T>() : nullptr;
    T* yp = y.data_ptr<T>();
    T* sop = state_out.data_ptr<T>();
    std::vector<Acc> h(static_cast<std::size_t>(N));
    for (int64_t b = 0; b < Bsz; ++b) {
      for (int64_t d = 0; d < Dim; ++d) {
        for (int64_t n = 0; n < N; ++n)
          h[static_cast<std::size_t>(n)] =
              sp ? static_cast<Acc>(sp[(b * Dim + d) * N + n]) : Acc{0};
        for (int64_t t = 0; t < L; ++t) {
          const int64_t bt = b * L + t;
          const Acc dt    = static_cast<Acc>(dp[bt * Dim + d]);
          const Acc u_btd = static_cast<Acc>(up[bt * Dim + d]);
          Acc y_acc = Acc{0};
          for (int64_t n = 0; n < N; ++n) {
            const Acc a  = static_cast<Acc>(ap[d * N + n]);
            const Acc dA = std::exp(dt * a);
            const Acc Bn = static_cast<Acc>(bp[bt * N + n]);
            Acc& hn = h[static_cast<std::size_t>(n)];
            hn = dA * hn + (dt * Bn) * u_btd;
            const Acc Cn = static_cast<Acc>(cp[bt * N + n]);
            y_acc += Cn * hn;
          }
          y_acc += static_cast<Acc>(dskip[d]) * u_btd;
          yp[bt * Dim + d] = static_cast<T>(y_acc);
        }
        for (int64_t n = 0; n < N; ++n)
          sop[(b * Dim + d) * N + n] =
              static_cast<T>(h[static_cast<std::size_t>(n)]);
      }
    }
  });
  return {y, state_out};
}

}  // namespace tesseract::ops

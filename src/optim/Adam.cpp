#include "tesseract/optim/Adam.hpp"

#include <cmath>
#include <cstdint>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/Optim.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::optim {

Adam::Adam(std::vector<Tensor> params, double lr, double beta1, double beta2, double eps)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
  TESSERACT_CHECK(lr_ > 0.0, "Adam: lr must be positive");
  TESSERACT_CHECK(beta1_ >= 0.0 && beta1_ < 1.0, "Adam: beta1 out of range");
  TESSERACT_CHECK(beta2_ >= 0.0 && beta2_ < 1.0, "Adam: beta2 out of range");
  TESSERACT_CHECK(eps_ > 0.0, "Adam: eps must be positive");
  m_.resize(params_.size());
  v_.resize(params_.size());
}

void Adam::step() {
  NoGradGuard nogg;
  ++step_count_;
  const double bc1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
  const double bc2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

  for (std::size_t i = 0; i < params_.size(); ++i) {
    Tensor& p = params_[i];
    const Tensor& g = p.grad();
    if (!g.defined()) continue;

    if (!m_[i].defined()) {
      m_[i] = Tensor::zeros(p.shape(), p.dtype(), p.device());
      v_[i] = Tensor::zeros(p.shape(), p.dtype(), p.device());
    }

    // M2I: the CUDA path drives `launch_adam_step` — a fused
    // elementwise kernel that writes `m`, `v`, `param` in a single
    // pass on the caller's current stream. Grad-device consistency
    // (`g.device() == p.device()`) is a precondition: autograd
    // accumulates into the same device the param lives on, so a
    // mismatch here is a programmer error rather than something the
    // optimizer is expected to paper over.
    if (p.device().is_cuda()) {
      TESSERACT_CHECK(g.device() == p.device(),
                      "Adam::step: grad device ({}) != param device "
                      "({}). Call `model->to(device)` before the first "
                      "backward so the grad is allocated on-device.",
                      g.device().to_string(), p.device().to_string());
      TESSERACT_CHECK(p.is_contiguous() && g.is_contiguous() &&
                          m_[i].is_contiguous() && v_[i].is_contiguous(),
                      "Adam::step: CUDA path expects contiguous param / "
                      "grad / moment tensors (param is_contig={}, grad={}, "
                      "m={}, v={})",
                      p.is_contiguous(), g.is_contiguous(),
                      m_[i].is_contiguous(), v_[i].is_contiguous());
      Stream s = current_stream(p.device());
      cuda::detail::launch_adam_step(
          p.dtype(), p.device().index, p.numel(),
          p.raw_data(), g.raw_data(),
          m_[i].raw_data(), v_[i].raw_data(),
          lr_, beta1_, beta2_, eps_, bc1, bc2,
          s.native_handle());
      continue;
    }

    dispatch_float(p.dtype(), [&]<typename T>() {
      T* pp = p.data_ptr<T>();
      const T* pg = g.data_ptr<T>();
      T* pm = m_[i].data_ptr<T>();
      T* pv = v_[i].data_ptr<T>();
      const int64_t n = p.numel();
      const T b1 = static_cast<T>(beta1_);
      const T b2 = static_cast<T>(beta2_);
      const T lr = static_cast<T>(lr_);
      const T eps = static_cast<T>(eps_);
      const T ibc1 = static_cast<T>(1.0 / bc1);
      const T ibc2 = static_cast<T>(1.0 / bc2);
      for (int64_t k = 0; k < n; ++k) {
        const T gk = pg[k];
        pm[k] = static_cast<T>(b1 * pm[k] + (T{1} - b1) * gk);
        pv[k] = static_cast<T>(b2 * pv[k] + (T{1} - b2) * gk * gk);
        const T m_hat = static_cast<T>(pm[k] * ibc1);
        const T v_hat = static_cast<T>(pv[k] * ibc2);
        pp[k] = static_cast<T>(pp[k] - lr * m_hat / (std::sqrt(v_hat) + eps));
      }
    });
  }
}

}  // namespace tesseract::optim

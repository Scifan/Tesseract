#include "tesseract/optim/SGD.hpp"

#include <cstdint>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::optim {

SGD::SGD(std::vector<Tensor> params, double lr, double momentum)
    : Optimizer(std::move(params)), lr_(lr), momentum_(momentum) {
  TESSERACT_CHECK(lr_ > 0.0, "SGD: lr must be positive (got {})", lr_);
  TESSERACT_CHECK(momentum_ >= 0.0, "SGD: momentum must be non-negative (got {})", momentum_);
  velocity_.resize(params_.size());
}

void SGD::step() {
  NoGradGuard nogg;
  for (std::size_t i = 0; i < params_.size(); ++i) {
    Tensor& p = params_[i];
    const Tensor& g = p.grad();
    if (!g.defined()) continue;  // parameter did not receive a gradient

    TESSERACT_CHECK(g.shape() == p.shape(),
                    "SGD.step: grad shape {} != param shape {}",
                    g.shape().to_string(), p.shape().to_string());
    TESSERACT_CHECK(g.dtype() == p.dtype(),
                    "SGD.step: grad dtype {} != param dtype {}",
                    dtype_name(g.dtype()), dtype_name(p.dtype()));

    dispatch_float(p.dtype(), [&]<typename T>() {
      T* pp = p.data_ptr<T>();
      const T* pg = g.data_ptr<T>();
      const int64_t n = p.numel();
      const T lr = static_cast<T>(lr_);
      if (momentum_ > 0.0) {
        if (!velocity_[i].defined()) {
          velocity_[i] = Tensor::zeros(p.shape(), p.dtype(), p.device());
        }
        T* pv = velocity_[i].data_ptr<T>();
        const T mom = static_cast<T>(momentum_);
        for (int64_t k = 0; k < n; ++k) {
          const T new_v = static_cast<T>(mom * pv[k] + pg[k]);
          pv[k] = new_v;
          pp[k] = static_cast<T>(pp[k] - lr * new_v);
        }
      } else {
        for (int64_t k = 0; k < n; ++k) {
          pp[k] = static_cast<T>(pp[k] - lr * pg[k]);
        }
      }
    });
  }
}

}  // namespace tesseract::optim

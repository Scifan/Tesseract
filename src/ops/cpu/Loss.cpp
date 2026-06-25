#include "tesseract/ops/Loss.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/autograd/Function.hpp"
#include "tesseract/autograd/Node.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/Loss.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::ops {

namespace {

// Forward: numerically stable mean cross-entropy and (when we need autograd)
// the probabilities (softmax(logits)) for the backward pass.
struct CEResult {
  Tensor loss;      // 0-D scalar
  Tensor probs;     // [N, C] softmax values (only populated when requires_grad)
};

CEResult cross_entropy_forward(const Tensor& logits, const Tensor& targets, bool want_probs) {
  TESSERACT_CHECK(logits.defined() && targets.defined(),
                  "cross_entropy_with_logits: operand undefined");
  TESSERACT_CHECK(dtype_is_floating(logits.dtype()),
                  "cross_entropy_with_logits: logits must be floating-point, got {}",
                  dtype_name(logits.dtype()));
  TESSERACT_CHECK(targets.dtype() == DType::Int64,
                  "cross_entropy_with_logits: targets must be Int64, got {}",
                  dtype_name(targets.dtype()));
  TESSERACT_CHECK(logits.rank() == 2, "cross_entropy_with_logits: logits must be rank-2, got {}",
                  logits.shape().to_string());
  TESSERACT_CHECK(targets.rank() == 1, "cross_entropy_with_logits: targets must be rank-1, got {}",
                  targets.shape().to_string());

  const int64_t N = logits.shape()[0];
  const int64_t C = logits.shape()[1];
  TESSERACT_CHECK(targets.shape()[0] == N,
                  "cross_entropy_with_logits: batch mismatch ({} vs {})",
                  targets.shape()[0], N);

  const Tensor lg = logits.is_contiguous() ? logits : logits.contiguous();
  const Tensor tg = targets.is_contiguous() ? targets : targets.contiguous();

  CEResult r;
  r.loss = Tensor::empty({}, logits.dtype(), logits.device());
  if (want_probs) r.probs = Tensor::empty({N, C}, logits.dtype(), logits.device());

  // CUDA path (M2F). We already called `.contiguous()` on both
  // operands above, so `lg` / `tg` are dense rank-2 / rank-1
  // buffers. The fused kernel produces the scalar loss directly,
  // and optionally the [N, C] probs matrix if autograd needs it.
  if (logits.device().is_cuda()) {
    Stream s = current_stream(logits.device());
    cuda::detail::launch_ce_forward(
        logits.dtype(), logits.device().index,
        N, C,
        lg.raw_data(),
        tg.data_ptr<int64_t>(),
        r.loss.raw_data(),
        want_probs ? r.probs.raw_data() : nullptr,
        s.native_handle());
    return r;
  }

  dispatch_float(logits.dtype(), [&]<typename T>() {
    const T* plg = lg.data_ptr<T>();
    const int64_t* ptg = tg.data_ptr<int64_t>();
    T* pp = want_probs ? r.probs.data_ptr<T>() : nullptr;
    T total = T{0};
    for (int64_t n = 0; n < N; ++n) {
      const T* row = plg + n * C;
      const int64_t t = ptg[n];
      TESSERACT_CHECK(t >= 0 && t < C,
                      "cross_entropy_with_logits: target[{}]={} out of range [0,{})", n, t, C);
      T m = std::numeric_limits<T>::lowest();
      for (int64_t c = 0; c < C; ++c) m = row[c] > m ? row[c] : m;
      T s = T{0};
      for (int64_t c = 0; c < C; ++c) s = static_cast<T>(s + std::exp(row[c] - m));
      const T log_z = static_cast<T>(m + std::log(s));
      total = static_cast<T>(total + (log_z - row[t]));
      if (want_probs) {
        const T inv_s = static_cast<T>(T{1} / s);
        for (int64_t c = 0; c < C; ++c) {
          pp[n * C + c] = static_cast<T>(std::exp(row[c] - m) * inv_s);
        }
      }
    }
    *r.loss.data_ptr<T>() = static_cast<T>(total / static_cast<T>(N));
  });
  return r;
}

struct CrossEntropyBackward : Node {
  Tensor probs_saved;   // [N, C]
  Tensor targets_saved; // [N] Int64
  int64_t N{1};
  std::string_view name() const override { return "CrossEntropyBackward"; }
  std::vector<Tensor> apply(const Tensor& g) override {
    NoGradGuard nogg;
    std::vector<Tensor> outs(2);  // [dlogits, dtargets=undef]
    if (!next_edges[0].requires_grad) return outs;
    // dlogits = (probs - one_hot(targets)) * g_scalar / N
    const int64_t C = probs_saved.shape()[1];

    // CUDA path (M2F). The kernel writes every `[n, c]` slot
    // independently, so we allocate `dlogits` fresh rather than
    // cloning `probs_saved` first — saves an N×C device-local
    // memcpy on the autograd hot path.
    if (probs_saved.device().is_cuda()) {
      Tensor dlogits_cu = Tensor::empty({N, C}, probs_saved.dtype(),
                                        probs_saved.device());
      Stream s = current_stream(probs_saved.device());
      cuda::detail::launch_ce_backward(
          probs_saved.dtype(), probs_saved.device().index,
          N, C,
          probs_saved.raw_data(),
          targets_saved.data_ptr<int64_t>(),
          g.raw_data(),
          dlogits_cu.raw_data(),
          s.native_handle());
      outs[0] = dlogits_cu;
      return outs;
    }

    Tensor dlogits = probs_saved.clone();
    const int64_t* pt = targets_saved.data_ptr<int64_t>();
    dispatch_float(probs_saved.dtype(), [&]<typename T>() {
      T* pd = dlogits.data_ptr<T>();
      const T gs = *g.data_ptr<T>();
      const T inv_n = static_cast<T>(T{1} / static_cast<T>(N));
      for (int64_t n = 0; n < N; ++n) {
        pd[n * C + pt[n]] = static_cast<T>(pd[n * C + pt[n]] - T{1});
      }
      const int64_t total = N * C;
      const T scale = static_cast<T>(gs * inv_n);
      for (int64_t i = 0; i < total; ++i) {
        pd[i] = static_cast<T>(pd[i] * scale);
      }
    });
    outs[0] = dlogits;
    return outs;
  }
};

}  // namespace

Tensor cross_entropy_with_logits(const Tensor& logits, const Tensor& targets) {
  const bool want_autograd = is_grad_enabled() && autograd::any_requires_grad(logits);
  CEResult r = cross_entropy_forward(logits, targets, want_autograd);
  if (want_autograd) {
    auto n = std::make_shared<CrossEntropyBackward>();
    n->probs_saved = r.probs;
    n->targets_saved = targets.contiguous();
    n->N = logits.shape()[0];
    n->next_edges = { autograd::edge_for(logits), autograd::edge_for(targets) };
    autograd::attach_grad_fn(r.loss, n);
  }
  graph::maybe_record("cross_entropy_with_logits", {&logits, &targets}, {&r.loss});
  return r.loss;
}

Tensor cross_entropy_with_logits_backward(const Tensor& logits,
                                          const Tensor& targets,
                                          const Tensor& grad) {
  TESSERACT_CHECK(logits.defined() && targets.defined() && grad.defined(),
                  "cross_entropy_with_logits_backward: operand undefined");
  TESSERACT_CHECK(dtype_is_floating(logits.dtype()),
                  "cross_entropy_with_logits_backward: logits must be floating, got {}",
                  dtype_name(logits.dtype()));
  TESSERACT_CHECK(targets.dtype() == DType::Int64,
                  "cross_entropy_with_logits_backward: targets must be Int64, got {}",
                  dtype_name(targets.dtype()));
  TESSERACT_CHECK(grad.dtype() == logits.dtype(),
                  "cross_entropy_with_logits_backward: grad dtype mismatch ({} vs {})",
                  dtype_name(grad.dtype()), dtype_name(logits.dtype()));
  TESSERACT_CHECK(logits.rank() == 2 && targets.rank() == 1,
                  "cross_entropy_with_logits_backward: bad ranks ({} / {})",
                  logits.shape().to_string(), targets.shape().to_string());
  TESSERACT_CHECK(grad.rank() == 0,
                  "cross_entropy_with_logits_backward: grad must be 0-D, got {}",
                  grad.shape().to_string());
  NoGradGuard nogg;
  // Re-derive softmax from logits rather than saving it — we have no tape.
  CEResult r = cross_entropy_forward(logits, targets, /*want_probs=*/true);
  const int64_t N = logits.shape()[0];
  const int64_t C = logits.shape()[1];
  const Tensor tg = targets.is_contiguous() ? targets : targets.contiguous();

  // CUDA path (M2F). Matches the CPU fall-through below; we just
  // hand the recomputed `r.probs` (already on the input's device,
  // thanks to `cross_entropy_forward` above) to the fused backward
  // kernel and let it write dlogits in a single pass.
  if (logits.device().is_cuda()) {
    Tensor dlogits_cu = Tensor::empty({N, C}, logits.dtype(), logits.device());
    Stream s = current_stream(logits.device());
    cuda::detail::launch_ce_backward(
        logits.dtype(), logits.device().index,
        N, C,
        r.probs.raw_data(),
        tg.data_ptr<int64_t>(),
        grad.raw_data(),
        dlogits_cu.raw_data(),
        s.native_handle());
    graph::maybe_record("cross_entropy_with_logits_backward",
                        {&logits, &targets, &grad}, {&dlogits_cu});
    return dlogits_cu;
  }

  Tensor dlogits = r.probs.clone();
  const int64_t* pt = tg.data_ptr<int64_t>();
  dispatch_float(logits.dtype(), [&]<typename T>() {
    T* pd = dlogits.data_ptr<T>();
    const T gs = *grad.data_ptr<T>();
    const T inv_n = static_cast<T>(T{1} / static_cast<T>(N));
    for (int64_t n = 0; n < N; ++n) {
      pd[n * C + pt[n]] = static_cast<T>(pd[n * C + pt[n]] - T{1});
    }
    const int64_t total = N * C;
    const T scale = static_cast<T>(gs * inv_n);
    for (int64_t i = 0; i < total; ++i) {
      pd[i] = static_cast<T>(pd[i] * scale);
    }
  });
  graph::maybe_record("cross_entropy_with_logits_backward",
                      {&logits, &targets, &grad}, {&dlogits});
  return dlogits;
}

}  // namespace tesseract::ops

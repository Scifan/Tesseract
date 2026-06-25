#include "tesseract/nn/MoEFeedForward.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/detail/MoeForward.hpp"
#include "tesseract/cuda/detail/MoeRoute.hpp"
#include "tesseract/nn/FeedForward.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/ModuleList.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Indexing.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/ops/Softmax.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

namespace {

// Build a 0/1 top-k mask over the last axis of `probs`. Computed host-side
// (D→H copy) because there is no device top-k kernel yet; the result is
// uploaded back to `probs.device()`. Tie-break is value-descending then
// index-ascending, so top_k==1 matches argmax's lowest-index convention and
// the mask is deterministic.
template <typename Scalar>
Tensor build_topk_mask_typed(const Tensor& probs_cpu, int64_t k) {
  const int64_t E = probs_cpu.shape()[probs_cpu.rank() - 1];
  const int64_t num_tokens = probs_cpu.numel() / E;
  Tensor mask = Tensor::zeros(probs_cpu.shape(), probs_cpu.dtype(),
                              cpu_device());
  const Scalar* p = probs_cpu.data_ptr<Scalar>();
  Scalar* m = mask.data_ptr<Scalar>();
  std::vector<int64_t> idx(static_cast<std::size_t>(E));
  for (int64_t t = 0; t < num_tokens; ++t) {
    const Scalar* row = p + t * E;
    Scalar* mrow = m + t * E;
    std::iota(idx.begin(), idx.end(), int64_t{0});
    std::partial_sort(
        idx.begin(), idx.begin() + k, idx.end(),
        [&row](int64_t a, int64_t b) {
          if (row[a] != row[b]) return row[a] > row[b];
          return a < b;
        });
    for (int64_t j = 0; j < k; ++j)
      mrow[idx[static_cast<std::size_t>(j)]] = static_cast<Scalar>(1);
  }
  return mask;
}

// Device top-k mask: one kernel computes the 0/1 mask directly on the GPU
// from the (Float32) router probs, avoiding the D→H copy + host partial_sort
// + H→D upload of the generic path. Top-k of softmax(probs) == top-k of
// probs (softmax is monotone), so feeding `probs` as the kernel's logits
// yields the same winner set; the kernel's gates output is discarded here
// (the differentiable gates are still computed via ops on the host graph so
// the router stays trainable). Float32 only — matches the model router dtype.
Tensor build_topk_mask_cuda(const Tensor& probs, int64_t k) {
  Tensor p = probs.contiguous();
  const int64_t E = p.shape()[p.rank() - 1];
  const int64_t T = p.numel() / E;
  Tensor mask = Tensor::empty(p.shape(), DType::Float32, p.device());
  Tensor gates_scratch = Tensor::empty(p.shape(), DType::Float32, p.device());
  Stream s = current_stream(p.device());
  cuda::detail::launch_moe_route(
      p.device().index, T, E, k, p.data_ptr<float>(),
      gates_scratch.data_ptr<float>(), mask.data_ptr<float>(),
      s.native_handle());
  return mask;
}

Tensor build_topk_mask(const Tensor& probs, int64_t k) {
  if (probs.device().is_cuda() && probs.dtype() == DType::Float32) {
    return build_topk_mask_cuda(probs, k);
  }
  Tensor probs_cpu = probs.to(cpu_device()).contiguous();
  Tensor mask_cpu;
  switch (probs_cpu.dtype()) {
    case DType::Float32:
      mask_cpu = build_topk_mask_typed<float>(probs_cpu, k);
      break;
    case DType::Float64:
      mask_cpu = build_topk_mask_typed<double>(probs_cpu, k);
      break;
    default:
      TESSERACT_THROW(
          "MoEFeedForward: router top-k currently supports Float32/Float64 "
          "router output (got {})",
          dtype_name(probs_cpu.dtype()));
  }
  return mask_cpu.to(probs.device());
}

}  // namespace

MoEFeedForward::MoEFeedForward(int64_t d_model, int64_t d_ff,
                               int64_t num_experts, int64_t num_experts_per_tok,
                               bool use_bias, DType dtype)
    : d_model_(d_model), d_ff_(d_ff), num_experts_(num_experts) {
  TESSERACT_CHECK(d_model > 0 && d_ff > 0,
                  "MoEFeedForward: d_model ({}) and d_ff ({}) must be positive",
                  d_model, d_ff);
  TESSERACT_CHECK(num_experts >= 1,
                  "MoEFeedForward: num_experts ({}) must be >= 1", num_experts);
  // 0 (or out-of-range) ⇒ clamp to the full expert set (dense routing).
  top_k_ = (num_experts_per_tok <= 0 || num_experts_per_tok > num_experts)
               ? num_experts
               : num_experts_per_tok;

  router_ = std::make_shared<Linear>(d_model, num_experts, /*use_bias=*/false,
                                     dtype);
  register_module("gate", router_);

  auto experts_holder = std::make_shared<ModuleList>();
  experts_.reserve(static_cast<std::size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    auto expert = std::make_shared<FeedForward>(d_model, d_ff, use_bias, dtype);
    experts_holder->append(expert);
    experts_.push_back(std::move(expert));
  }
  register_module("experts", experts_holder);
}

Tensor MoEFeedForward::forward(const Tensor& x) {
  TESSERACT_CHECK(x.rank() >= 2,
                  "MoEFeedForward::forward: expected rank >= 2 input [..., D], "
                  "got {}", x.shape().to_string());
  TESSERACT_CHECK(x.shape()[x.rank() - 1] == d_model_,
                  "MoEFeedForward::forward: input last dim {} != d_model {}",
                  x.shape()[x.rank() - 1], d_model_);

  const int64_t last = x.rank() - 1;

  Tensor logits = router_->forward(x);                 // [..., E]
  Tensor probs  = ops::softmax(logits, last);          // [..., E]
  Tensor mask   = build_topk_mask(probs, top_k_);      // [..., E] (0/1, const)
  Tensor masked = ops::mul(probs, mask);               // [..., E]
  Tensor denom  = ops::sum(masked, last, /*keepdim=*/true);  // [..., 1]
  Tensor gates  = ops::div(masked, denom);             // [..., E]

  // --- Fully fused GPU inference path (Phase 4) ------------------------------
  // When grad is off, the input is FP32 on CUDA, and every expert is a plain
  // (un-quantized, bias-free) SwiGLU, run the whole layer through one device
  // pipeline: device permute → grouped GEMM (gate/up) → fused SiLU·up →
  // grouped GEMM (down) → scatter-combine. This eliminates the host round-trip
  // and per-expert op overhead that make the generic path slower than dense on
  // GPU. Numerically matches the generic path to TF32 tolerance.
  if (!is_grad_enabled() && x.device().is_cuda() &&
      x.dtype() == DType::Float32) {
    const int64_t Ef = num_experts_, kf = top_k_;
    const int64_t Tf = x.numel() / d_model_;
    std::vector<const float*> wg, wu, wd;
    wg.reserve(static_cast<std::size_t>(Ef));
    wu.reserve(static_cast<std::size_t>(Ef));
    wd.reserve(static_cast<std::size_t>(Ef));
    bool fusable = (Tf > 0);
    for (int64_t e = 0; e < Ef && fusable; ++e) {
      auto& ex = experts_[static_cast<std::size_t>(e)];
      auto g = std::dynamic_pointer_cast<Linear>(ex->gate_proj());
      auto u = std::dynamic_pointer_cast<Linear>(ex->up_proj());
      auto d = std::dynamic_pointer_cast<Linear>(ex->down_proj());
      if (!g || !u || !d || g->has_bias() || u->has_bias() || d->has_bias() ||
          g->weight().dtype() != DType::Float32) {
        fusable = false;
        break;
      }
      wg.push_back(g->weight().data_ptr<float>());
      wu.push_back(u->weight().data_ptr<float>());
      wd.push_back(d->weight().data_ptr<float>());
    }
    if (fusable) {
      Tensor gates_c = gates.contiguous();
      Tensor mask_c = mask.contiguous();
      Tensor y = Tensor::empty(Shape({Tf, d_model_}), DType::Float32,
                               x.device());
      Stream s = current_stream(x.device());
      const bool ok = cuda::detail::launch_moe_grouped_ffn(
          x.device().index, Tf, d_model_, d_ff_, Ef, kf,
          x.contiguous().data_ptr<float>(), gates_c.data_ptr<float>(),
          mask_c.data_ptr<float>(), wg.data(), wu.data(), wd.data(),
          y.data_ptr<float>(), s.native_handle());
      if (ok) return ops::reshape(y, x.shape());
    }
  }

  // Switch load-balancing aux loss: E · Σ_e f_e · P_e.
  //   f_e = mean over tokens of mask[..., e]  (fraction routed, constant)
  //   P_e = mean over tokens of probs[..., e] (differentiable)
  // Collapse all leading axes to a single token axis via reshape so the means
  // are over tokens regardless of input rank.
  {
    const int64_t E = num_experts_;
    const int64_t Tn = x.numel() / d_model_;  // token count
    Shape flat_probs({Tn, E});
    Tensor f = ops::mean(ops::reshape(mask, flat_probs), 0);   // [E]
    Tensor pmean = ops::mean(ops::reshape(probs, flat_probs), 0);  // [E]
    Tensor prod = ops::mul(f, pmean);                         // [E]
    aux_loss_ = ops::mul(ops::sum(prod),
                         Tensor::full(Shape({}), static_cast<double>(E),
                                      prod.dtype(), prod.device()));
  }

  // --- Sparse dispatch / combine (B-038+) -----------------------------------
  // Instead of running every expert on every token and masking (dense, O(E)
  // expert-FLOPs), permute tokens into per-expert groups, run each expert only
  // on its routed rows (O(k) expert-FLOPs total — Σ_e n_e == T·k), then
  // un-permute and sum each token's k contributions. Numerically identical to
  // the dense path (each token's output is the same gate-weighted sum of the
  // same expert outputs, accumulated in ascending-expert order), but the heavy
  // FFN compute scales with the *active* experts, not the full set.
  const int64_t E = num_experts_;
  const int64_t k = top_k_;
  const int64_t T = x.numel() / d_model_;  // flattened token count

  if (T == 0) return ops::reshape(x, x.shape());

  // Host-side routing bookkeeping from the 0/1 mask. Exactly k ones per row.
  Tensor mask_cpu = mask.to(cpu_device()).contiguous();
  std::vector<int64_t> expert_of(static_cast<std::size_t>(T * k));  // [T*k] asc-e
  struct Asn { int64_t e; int64_t t; int64_t pos_major; };
  std::vector<Asn> asn;
  asn.reserve(static_cast<std::size_t>(T * k));
  dispatch_float(mask_cpu.dtype(), [&]<typename S>() {
    const S* m = mask_cpu.data_ptr<S>();
    for (int64_t t = 0; t < T; ++t) {
      int64_t j = 0;
      for (int64_t e = 0; e < E; ++e) {
        if (m[t * E + e] != S{0}) {
          const int64_t pos = t * k + j;
          expert_of[static_cast<std::size_t>(pos)] = e;
          asn.push_back({e, t, pos});
          ++j;
        }
      }
    }
  });

  // Group assignments by expert (stable ⇒ within an expert, tokens stay in
  // ascending order, matching the dense accumulation order).
  std::stable_sort(asn.begin(), asn.end(),
                   [](const Asn& a, const Asn& b) { return a.e < b.e; });
  std::vector<int64_t> perm_src(asn.size());      // expert-grouped → token id
  std::vector<int64_t> gather_back(asn.size());   // token-major pos → grouped i
  std::vector<int64_t> counts(static_cast<std::size_t>(E), 0);
  for (std::size_t i = 0; i < asn.size(); ++i) {
    perm_src[i] = asn[i].t;
    gather_back[static_cast<std::size_t>(asn[i].pos_major)] =
        static_cast<int64_t>(i);
    counts[static_cast<std::size_t>(asn[i].e)] += 1;
  }

  auto to_index = [&](const std::vector<int64_t>& v, Shape shp) {
    Tensor t = Tensor::empty(std::move(shp), DType::Int64, cpu_device());
    std::copy(v.begin(), v.end(), t.data_ptr<int64_t>());
    return t.to(x.device());
  };
  Tensor perm_src_t    = to_index(perm_src, Shape({T * k}));
  Tensor gather_back_t = to_index(gather_back, Shape({T * k}));
  Tensor expert_of_t   = to_index(expert_of, Shape({T, k}));

  // Permute tokens into expert-contiguous order, run each expert on its slice.
  Tensor x_flat = ops::reshape(x, Shape({T, d_model_}));        // [T, D]
  Tensor x_perm = ops::index_select(x_flat, 0, perm_src_t);     // [T*k, D]
  std::vector<Tensor> parts;
  parts.reserve(static_cast<std::size_t>(E));
  int64_t off = 0;
  for (int64_t e = 0; e < E; ++e) {
    const int64_t c = counts[static_cast<std::size_t>(e)];
    if (c > 0) {
      Tensor slice = x_perm.narrow(0, off, c);                  // [n_e, D]
      parts.push_back(experts_[static_cast<std::size_t>(e)]->forward(slice));
    }
    off += c;
  }
  Tensor y_perm = ops::cat(parts, 0);                           // [T*k, D]

  // Un-permute back to token-major (k contiguous rows per token), weight by the
  // (differentiable) gates gathered from the router, and sum the k slots.
  Tensor y_back   = ops::index_select(y_perm, 0, gather_back_t);   // [T*k, D]
  Tensor y_tk     = ops::reshape(y_back, Shape({T, k, d_model_})); // [T, k, D]
  Tensor gates_2d = ops::reshape(gates, Shape({T, E}));            // [T, E]
  Tensor gates_tk = ops::gather(gates_2d, 1, expert_of_t);        // [T, k]
  Tensor weighted = ops::mul(y_tk, ops::reshape(gates_tk, Shape({T, k, 1})));
  Tensor out_flat = ops::sum(weighted, 1, /*keepdim=*/false);     // [T, D]
  return ops::reshape(out_flat, x.shape());
}

}  // namespace tesseract::nn

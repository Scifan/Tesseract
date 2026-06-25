#include "tesseract/ops/Attention.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <vector>

#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/detail/FusedAttention.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Softmax.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::ops {

namespace {

// Build an [S_q, S_k] additive causal mask on `device`/`dtype`:
//
//   m[i, j] = 0         if j <= i
//           = -infinity if j >  i
//
// We materialize on the host then push to the target device in a single
// async byte copy. For the reference SDPA the cost is O(S_q · S_k) per
// call, dwarfed by the O(S_q · S_k · D) attention matmul; a fused FA3
// kernel (M2L) will generate this on-device as part of the score
// accumulator rather than as a separate allocation. Float32 is used for
// all dtypes (`Half`, `BFloat16`, `Float64`) because the -inf constant
// round-trips through `Tensor::to(dtype=...)` cleanly (FP32 -inf
// converts exactly to the target dtype's -inf on both IEEE halves and
// `double`).
Tensor build_causal_mask(int64_t s_q, int64_t s_k, DType dtype, Device device) {
  // The -inf sentinel: using `std::numeric_limits<float>::lowest()` would
  // round to a finite value after softmax; -inf is what we need for the
  // exp() to produce an exact 0.
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const std::size_t n = static_cast<std::size_t>(s_q) * static_cast<std::size_t>(s_k);
  std::vector<float> host(n, 0.0f);
  for (int64_t i = 0; i < s_q; ++i) {
    for (int64_t j = i + 1; j < s_k; ++j) {
      host[static_cast<std::size_t>(i) * static_cast<std::size_t>(s_k) + static_cast<std::size_t>(j)] = neg_inf;
    }
  }
  Tensor m_cpu = Tensor::from_vector(host, {s_q, s_k});
  // Cast to the input dtype on CPU first if needed (cheaper than
  // allocating the target dtype directly and losing the inf on
  // narrow-dtype construction).
  Tensor m_cast = (dtype == DType::Float32) ? m_cpu
                                            : Tensor::empty({s_q, s_k}, dtype, cpu_device());
  if (dtype != DType::Float32) {
    // Hand-copy with cast so -inf is preserved (from_vector doesn't
    // know about -inf pass-through for Half).
    const float* src = m_cpu.data_ptr<float>();
    switch (dtype) {
      case DType::Float64: {
        double* dst = m_cast.data_ptr<double>();
        for (std::size_t i = 0; i < n; ++i) {
          dst[i] = std::isinf(src[i]) ? -std::numeric_limits<double>::infinity()
                                      : static_cast<double>(src[i]);
        }
        break;
      }
      case DType::Float16: {
        Half* dst = m_cast.data_ptr<Half>();
        const Half hinf = Half(-std::numeric_limits<float>::infinity());
        for (std::size_t i = 0; i < n; ++i) {
          dst[i] = std::isinf(src[i]) ? hinf : Half(src[i]);
        }
        break;
      }
      case DType::BFloat16: {
        BFloat16* dst = m_cast.data_ptr<BFloat16>();
        const BFloat16 binf = BFloat16(-std::numeric_limits<float>::infinity());
        for (std::size_t i = 0; i < n; ++i) {
          dst[i] = std::isinf(src[i]) ? binf : BFloat16(src[i]);
        }
        break;
      }
      default:
        TESSERACT_CHECK(false, "attention: unsupported floating dtype {} for causal mask",
                        dtype_name(dtype));
    }
  }
  return device.is_cpu() ? m_cast : m_cast.to(device);
}

}  // namespace

Tensor attention(const Tensor& q,
                 const Tensor& k,
                 const Tensor& v,
                 const Tensor& mask,
                 bool causal,
                 double dropout_p) {
  TESSERACT_CHECK(q.defined() && k.defined() && v.defined(),
                  "attention: q, k, v must all be defined tensors");
  TESSERACT_CHECK(dropout_p == 0.0,
                  "attention: dropout_p != 0 is not implemented in M2J "
                  "(landed with the RNG HAL in M3 / the fused FA3 kernel in M2L)");
  TESSERACT_CHECK(dtype_is_floating(q.dtype()),
                  "attention: floating-point dtype required, got {}", dtype_name(q.dtype()));
  TESSERACT_CHECK(q.dtype() == k.dtype() && k.dtype() == v.dtype(),
                  "attention: q/k/v dtype mismatch ({}, {}, {})",
                  dtype_name(q.dtype()), dtype_name(k.dtype()), dtype_name(v.dtype()));
  TESSERACT_CHECK(q.device() == k.device() && k.device() == v.device(),
                  "attention: q/k/v device mismatch ({}, {}, {})",
                  q.device().to_string(), k.device().to_string(), v.device().to_string());

  const std::size_t qr = q.shape().rank();
  const std::size_t kr = k.shape().rank();
  const std::size_t vr = v.shape().rank();
  TESSERACT_CHECK(qr >= 2 && kr >= 2 && vr >= 2,
                  "attention: q/k/v must each be rank >= 2 (got {}, {}, {})",
                  q.shape().to_string(), k.shape().to_string(), v.shape().to_string());

  const int64_t s_q = q.shape()[qr - 2];
  const int64_t d_q = q.shape()[qr - 1];
  const int64_t s_k = k.shape()[kr - 2];
  const int64_t d_k = k.shape()[kr - 1];
  const int64_t s_v = v.shape()[vr - 2];
  const int64_t d_v = v.shape()[vr - 1];
  TESSERACT_CHECK(d_q == d_k,
                  "attention: head_dim mismatch between q ({}) and k ({})",
                  d_q, d_k);
  TESSERACT_CHECK(s_k == s_v,
                  "attention: key seq-len ({}) must match value seq-len ({})",
                  s_k, s_v);
  if (causal) {
    TESSERACT_CHECK(s_q == s_k,
                    "attention: causal=true requires S_q ({}) == S_k ({})", s_q, s_k);
  }
  (void)d_v;  // sanity-check only; used as part of the output shape via matmul.

  // ------------------------------------------------------------------
  // Wave 4.2 (B-024): fused FlashAttention-2 fast path on CUDA.
  //
  // Preconditions (anything else falls through to the composite):
  //   * CUDA device, matching index on all three inputs.
  //   * Row-major contiguous Q / K / V.
  //   * No external `mask` tensor. Arbitrary additive masks defeat
  //     the memory-bound premise of the fused kernel — the
  //     composite path still supports them via `add(scores, mask)`.
  //   * Head dim `D <= 128` and `D_v == D` (MHA constraint today).
  //   * Rank-equal shapes with matching leading dims on all three
  //     inputs — i.e. `q.shape[:-2] == k.shape[:-2] == v.shape[:-2]`.
  //     Broadcast across batch / head is not implemented in the
  //     fused path; the composite handles it for free via matmul.
  //   * Either autograd is disabled globally (`is_grad_enabled()`
  //     is false) or none of Q / K / V carries a grad requirement.
  //     The fused kernel is forward-only; backward flows through
  //     the composite primitives below, same convention as
  //     `launch_rms_norm` / `launch_swiglu_silu_gate`.
  //
  // The dispatch emits the same `attention` GraphScope marker as the
  // composite path so downstream MLIR pattern-matchers see an
  // identical call shape regardless of which kernel actually ran.
  const bool leading_dims_equal =
      qr == kr && kr == vr && [&]() {
        for (std::size_t i = 0; i + 2 < qr; ++i) {
          if (q.shape()[i] != k.shape()[i]) return false;
          if (q.shape()[i] != v.shape()[i]) return false;
        }
        return true;
      }();

  const bool fused_dtype =
      q.dtype() == DType::Float32 ||
      q.dtype() == DType::Float16 ||
      q.dtype() == DType::BFloat16;

  // Shape gating. The CUDA-core fused kernel beats the cuBLASLt
  // tensor-core composite only when the composite's launch-count +
  // score-tensor overhead dominates its GEMM time. Empirically
  // (benchmarks/bench_cuda_fused_attention.cpp on Ada SM 8.9):
  //
  //   * S_q ≤ 8 and B·H ≥ 64: fused wins (decode + chunked-decode).
  //     The composite pays 3 launches + a `[S_q, S_k]` score
  //     materialization; at these shapes the GEMMs are cuBLASLt
  //     GEMVs whose tensor-core advantage doesn't matter.
  //   * Larger S_q (prefill): composite wins because the Q·Kᵀ and
  //     probs·V matmuls run through FP16 tensor cores with ~4× the
  //     effective FLOP throughput of the fused kernel's CUDA-core
  //     FP32 accumulation. Closing that gap is the WMMA / mma.sync
  //     follow-up explicitly deferred in `docs/backlog.md` (B-024).
  int64_t bh = 1;
  for (std::size_t i = 0; i + 2 < qr; ++i) bh *= q.shape()[i];
  // Pure single-query decode (`s_q == 1`) always prefers the fused
  // split-K kernel: it collapses the composite's per-head score/softmax/
  // output GEMVs (B·H tiny cuBLASLt GEMVs + a softmax launch + a score
  // HBM round trip) into two memory-bound kernels, and the split-K grid
  // (BH·num_splits blocks) fills every SM even when B·H is small (e.g.
  // a single-batch 32-head GQA model, bh=32). For chunked decode
  // (2 ≤ s_q ≤ 8) keep the bh≥64 gate — there the composite's score
  // tensor amortizes and tensor-core GEMMs start to pay off.
  const bool decode_like = (s_q == 1) || (s_q <= 8 && bh >= 64);

  // Short causal prefill also prefers the fused kernel. At small S_q the
  // composite path is dominated by fixed overhead — 6 kernel launches
  // (scale, QKᵀ, mask-add, softmax, probs·V) plus an `[S_q, S_k]` score
  // matrix written to and re-read from HBM — none of which shrinks with
  // S_q, while its tensor-core FLOP advantage only matters once S_q is
  // large. The fused kernel collapses all of that into one launch with
  // the causal mask generated on-chip. Empirically (Dh=64) it wins up to
  // a few hundred query rows; beyond `kPrefillFusedMax` the composite's
  // FP16 tensor cores overtake the fused kernel's CUDA-core accumulation,
  // so we keep the composite there (and for the rectangular chunked-
  // prefill case, which arrives with a mask and is excluded here).
  constexpr int64_t kPrefillFusedMax = 256;
  const bool prefill_fused = causal && !mask.defined() && s_q == s_k &&
                             s_q > 1 && s_q <= kPrefillFusedMax;

  // Escape hatch for the fused-kernel unit tests and bench suites:
  // the shape gate above is tuned for *production* routing (which
  // favors composite on prefill where tensor cores dominate), but
  // the fused kernel must still exercise its full shape coverage
  // under ctest. Setting `TESSERACT_FORCE_FUSED_ATTENTION=1` forces
  // the fused path whenever the *other* preconditions (dtype,
  // contiguity, D ≤ 128, no mask, no autograd) are met. This knob
  // is strictly for testing/benchmarking — production code must
  // not depend on it.
  static const bool kForceFusedEnv = []() {
    const char* v = std::getenv("TESSERACT_FORCE_FUSED_ATTENTION");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();

  const bool fused_cuda_eligible =
      q.device().is_cuda() &&
      fused_dtype &&
      !mask.defined() &&
      qr >= 3 &&
      leading_dims_equal &&
      d_q <= 128 && d_q == d_v &&
      q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
      (decode_like || prefill_fused || kForceFusedEnv) &&
      !(is_grad_enabled() &&
        (q.requires_grad() || k.requires_grad() || v.requires_grad()));

  if (fused_cuda_eligible) {
    // Flatten all leading dims into a single `B*H` axis; the kernel
    // treats Q / K / V as `[B*H, S, D]`. We produce `O` with the
    // original shape so downstream consumers are unaffected.
    std::vector<int64_t> out_dims(qr);
    for (std::size_t i = 0; i < qr; ++i) out_dims[i] = q.shape()[i];
    out_dims[qr - 1] = d_v;  // [..., S_q, D_v]
    Tensor out = Tensor::empty(Shape(out_dims), q.dtype(), q.device());

    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d_q)));

    Stream s = current_stream(q.device());
    cuda::detail::launch_fused_attention(
        q.dtype(), q.device().index,
        bh, 1, /*H_kv=*/0, s_q, s_k, d_q,
        scale, causal,
        q.raw_data(), k.raw_data(), v.raw_data(),
        out.raw_data(), s.native_handle());

    std::vector<const Tensor*> ins{&q, &k, &v};
    graph::maybe_record("attention", std::move(ins), {&out},
                        {{"causal", causal},
                         {"dropout_p", dropout_p},
                         {"fused", true}});
    return out;
  }
  // ------------------------------------------------------------------

  // Scale Q by 1/√d before the matmul. Multiplying the smaller operand
  // (O(B·H·S_q·D) elements) costs less than broadcasting across the
  // dense [..., S_q, S_k] score matrix. 0-D scalar tensor broadcasts
  // against any rank under `ops::mul`'s NumPy-style broadcast rules.
  const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(d_q));
  Tensor scale = Tensor::full({}, inv_sqrt_d, q.dtype(), q.device());
  Tensor q_scaled = mul(q, scale);

  // scores = Q_scaled · Kᵀ  ([..., S_q, S_k])
  Tensor k_t = transpose(k, static_cast<int64_t>(kr) - 2, static_cast<int64_t>(kr) - 1);
  Tensor scores = matmul(q_scaled, k_t);

  // Additive mask (optional). `ops::add` already broadcasts.
  if (mask.defined()) {
    TESSERACT_CHECK(mask.dtype() == q.dtype(),
                    "attention: mask dtype {} must match q dtype {}",
                    dtype_name(mask.dtype()), dtype_name(q.dtype()));
    TESSERACT_CHECK(mask.device() == q.device(),
                    "attention: mask device {} must match q device {}",
                    mask.device().to_string(), q.device().to_string());
    scores = add(scores, mask);
  }
  if (causal) {
    Tensor causal_mask = build_causal_mask(s_q, s_k, q.dtype(), q.device());
    scores = add(scores, causal_mask);
  }

  // probs = softmax(scores, dim=-1)
  Tensor probs = softmax(scores, static_cast<int64_t>(scores.rank()) - 1);

  // out = probs · V  ([..., S_q, D_v])
  Tensor out = matmul(probs, v);

  // The composite path already emits primitive ops into the active
  // GraphScope (matmul / softmax / add / mul each call `maybe_record`
  // inside). We still emit a single `attention` marker op at the tail
  // so a future MLIR pattern-matcher (M2L) can detect the intent and
  // collapse the primitive chain into a fused `tesseract.attention`
  // before codegen. The marker is metadata-only — it doesn't re-run
  // forward, and autograd flows through the inner primitives.
  std::vector<const Tensor*> ins{&q, &k, &v};
  if (mask.defined()) ins.push_back(&mask);
  graph::maybe_record("attention", std::move(ins), {&out},
                      {{"causal", causal},
                       {"dropout_p", dropout_p}});
  return out;
}

Tensor decode_attention_gqa(const Tensor& q, const Tensor& k, const Tensor& v,
                            bool causal) {
  TESSERACT_CHECK(q.defined() && k.defined() && v.defined(),
                  "decode_attention_gqa: q/k/v must be defined");
  TESSERACT_CHECK(q.rank() == 4 && k.rank() == 4 && v.rank() == 4,
                  "decode_attention_gqa: expected rank-4 [B,H,S,D], got q={} k={} v={}",
                  q.shape().to_string(), k.shape().to_string(), v.shape().to_string());
  const int64_t B   = q.shape()[0];
  const int64_t H   = q.shape()[1];
  const int64_t S_q = q.shape()[2];
  const int64_t D   = q.shape()[3];
  const int64_t Hkv = k.shape()[1];
  const int64_t S_k = k.shape()[2];
  TESSERACT_CHECK(S_q == 1,
                  "decode_attention_gqa: only single-query decode (S_q==1) "
                  "is supported, got S_q={}", S_q);
  TESSERACT_CHECK(k.shape()[0] == B && v.shape()[0] == B &&
                      v.shape()[1] == Hkv && v.shape()[2] == S_k &&
                      k.shape()[3] == D && v.shape()[3] == D,
                  "decode_attention_gqa: q={} k={} v={} shape mismatch",
                  q.shape().to_string(), k.shape().to_string(),
                  v.shape().to_string());
  TESSERACT_CHECK(Hkv > 0 && H % Hkv == 0,
                  "decode_attention_gqa: H ({}) must be a multiple of Hkv ({})",
                  H, Hkv);

  const bool fused_dtype = q.dtype() == DType::Float32 ||
                           q.dtype() == DType::Float16 ||
                           q.dtype() == DType::BFloat16;
  const bool fused_ok =
      q.device().is_cuda() && fused_dtype && D <= 128 &&
      q.dtype() == k.dtype() && q.dtype() == v.dtype() &&
      q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
      !(is_grad_enabled() &&
        (q.requires_grad() || k.requires_grad() || v.requires_grad()));

  if (fused_ok) {
    // K/V keep their Hkv heads — the kernel maps query head h to KV head
    // h/(H/Hkv) internally, so no `repeat_kv` materialization is needed.
    Tensor out = Tensor::empty(Shape({B, H, S_q, D}), q.dtype(), q.device());
    const float scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
    Stream s = current_stream(q.device());
    cuda::detail::launch_fused_attention(
        q.dtype(), q.device().index,
        B, H, Hkv, S_q, S_k, D, scale, causal,
        q.raw_data(), k.raw_data(), v.raw_data(),
        out.raw_data(), s.native_handle());
    graph::maybe_record("attention", {&q, &k, &v}, {&out},
                        {{"causal", causal}, {"dropout_p", 0.0}, {"fused", true}});
    return out;
  }

  // Fallback (CPU, FP64, non-contiguous, or autograd-on): expand K/V to H
  // heads with the same kv0,kv0,…,kv1,… interleave the kernel assumes, then
  // run the generic composite attention.
  const int64_t group = H / Hkv;
  auto expand = [&](const Tensor& t) -> Tensor {
    if (group == 1) return t;
    Tensor r = ops::reshape(t, Shape({B, Hkv, 1, S_k, D}));
    r = ops::broadcast_to(r, Shape({B, Hkv, group, S_k, D}));
    return ops::reshape(r, Shape({B, Hkv * group, S_k, D}));
  };
  return attention(q, expand(k), expand(v), Tensor{}, causal, 0.0);
}

Tensor prefill_attention_gqa(const Tensor& q, const Tensor& k, const Tensor& v,
                             bool causal) {
  TESSERACT_CHECK(q.defined() && k.defined() && v.defined(),
                  "prefill_attention_gqa: q/k/v must be defined");
  TESSERACT_CHECK(q.rank() == 4 && k.rank() == 4 && v.rank() == 4,
                  "prefill_attention_gqa: expected rank-4 [B,H,S,D], got "
                  "q={} k={} v={}", q.shape().to_string(),
                  k.shape().to_string(), v.shape().to_string());
  const int64_t B   = q.shape()[0];
  const int64_t H   = q.shape()[1];
  const int64_t S_q = q.shape()[2];
  const int64_t D   = q.shape()[3];
  const int64_t Hkv = k.shape()[1];
  const int64_t S_k = k.shape()[2];
  TESSERACT_CHECK(k.shape()[0] == B && v.shape()[0] == B &&
                      v.shape()[1] == Hkv && v.shape()[2] == S_k &&
                      k.shape()[3] == D && v.shape()[3] == D,
                  "prefill_attention_gqa: q={} k={} v={} shape mismatch",
                  q.shape().to_string(), k.shape().to_string(),
                  v.shape().to_string());
  TESSERACT_CHECK(Hkv > 0 && H % Hkv == 0,
                  "prefill_attention_gqa: H ({}) must be a multiple of Hkv ({})",
                  H, Hkv);

  const bool fused_dtype = q.dtype() == DType::Float32 ||
                           q.dtype() == DType::Float16 ||
                           q.dtype() == DType::BFloat16;
  // The square-causal FlashAttention kernel encodes the mask as `j > q`,
  // which assumes the new queries sit at positions [0, S_q) — i.e. the
  // full-prompt prefill (S_q == S_k). Chunked prefill (S_q < S_k) needs
  // the rectangular mask and stays on the composite path.
  const bool fused_ok =
      q.device().is_cuda() && fused_dtype && D <= 128 && S_q == S_k &&
      q.dtype() == k.dtype() && q.dtype() == v.dtype() &&
      q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
      !(is_grad_enabled() &&
        (q.requires_grad() || k.requires_grad() || v.requires_grad()));

  if (fused_ok) {
    Tensor out = Tensor::empty(Shape({B, H, S_q, D}), q.dtype(), q.device());
    const float scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
    Stream s = current_stream(q.device());
    cuda::detail::launch_fused_attention(
        q.dtype(), q.device().index,
        B, H, Hkv, S_q, S_k, D, scale, causal,
        q.raw_data(), k.raw_data(), v.raw_data(),
        out.raw_data(), s.native_handle());
    graph::maybe_record("attention", {&q, &k, &v}, {&out},
                        {{"causal", causal}, {"dropout_p", 0.0}, {"fused", true}});
    return out;
  }

  // Fallback: expand K/V to H heads (kv0,kv0,…,kv1,… interleave) and run
  // the composite attention with the same causal flag.
  const int64_t group = H / Hkv;
  auto expand = [&](const Tensor& t) -> Tensor {
    if (group == 1) return t;
    Tensor r = ops::reshape(t, Shape({B, Hkv, 1, S_k, D}));
    r = ops::broadcast_to(r, Shape({B, Hkv, group, S_k, D}));
    return ops::reshape(r, Shape({B, Hkv * group, S_k, D}));
  };
  return attention(q, expand(k), expand(v), Tensor{}, causal, 0.0);
}

Tensor prefill_attention_gqa_bshd(const Tensor& q, const Tensor& k,
                                  const Tensor& v, bool causal) {
  // B-024c fast path: same square-causal GQA prefill as
  // `prefill_attention_gqa`, but it reads Q (the contiguous permuted view)
  // and K/V (the KV-cache `[B,Hkv,S,D]` narrows — which are *not* contiguous
  // along the head/batch dims) in place via element-strides, and writes its
  // output in the **BSHD** `[B, S_q, H, D]` layout. Returning BSHD lets
  // `forward_step` flatten straight to `[B, S_q, d_model]` for `o_proj`
  // without the `[B,H,S,D] → [B,S,H,D]` output transpose, and skipping the
  // K/V `contiguous()` removes those copies too. Together these are the
  // largest non-GEMM prefill cost (backlog B-024c). Returns an undefined
  // Tensor when the fused CUDA path is not eligible — the caller then falls
  // back to the contiguous `prefill_attention_gqa` + transpose.
  if (!(q.defined() && k.defined() && v.defined())) return Tensor{};
  if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4) return Tensor{};

  const int64_t B   = q.shape()[0];
  const int64_t H   = q.shape()[1];
  const int64_t S_q = q.shape()[2];
  const int64_t D   = q.shape()[3];
  const int64_t Hkv = k.shape()[1];
  const int64_t S_k = k.shape()[2];

  const bool fused_dtype = q.dtype() == DType::Float32 ||
                           q.dtype() == DType::Float16 ||
                           q.dtype() == DType::BFloat16;
  // Inner head-dim must be unit-stride for both K and V (the kernel stages
  // `[...] + d` with d contiguous); the seq/head/batch strides are arbitrary.
  const auto qs = q.strides();
  const auto ks = k.strides();
  const auto vs = v.strides();
  const bool inner_contig =
      qs[3] == 1 && ks[3] == 1 && vs[3] == 1 &&
      ks[0] == Hkv * ks[1] && vs[0] == Hkv * vs[1] && ks[1] == vs[1] &&
      ks[2] == vs[2];
  const bool fused_ok =
      q.device().is_cuda() && fused_dtype && D <= 128 && S_q == S_k &&
      Hkv > 0 && H % Hkv == 0 && q.dtype() == k.dtype() &&
      q.dtype() == v.dtype() && inner_contig &&
      k.shape()[0] == B && v.shape()[0] == B && v.shape()[1] == Hkv &&
      v.shape()[2] == S_k && k.shape()[3] == D && v.shape()[3] == D &&
      !(is_grad_enabled() &&
        (q.requires_grad() || k.requires_grad() || v.requires_grad()));
  if (!fused_ok) return Tensor{};

  // Output in BSHD [B, S_q, H, D] (row-major contiguous): seq-stride H*D,
  // head-stride D, batch-stride S_q*H*D.
  Tensor out = Tensor::empty(Shape({B, S_q, H, D}), q.dtype(), q.device());
  const int64_t strides[9] = {
      qs[0], qs[1], qs[2],            // Q: [B,H,S,D] view strides
      ks[0], ks[1], ks[2],            // K (== V layout): [B,Hkv,S,D] narrow
      S_q * H * D, D, H * D,          // O: BSHD [B,S,H,D]
  };
  const float scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
  Stream s = current_stream(q.device());
  cuda::detail::launch_fused_attention(
      q.dtype(), q.device().index, B, H, Hkv, S_q, S_k, D, scale, causal,
      q.raw_data(), k.raw_data(), v.raw_data(), out.raw_data(),
      s.native_handle(), strides);
  graph::maybe_record(
      "attention", {&q, &k, &v}, {&out},
      {{"causal", causal}, {"dropout_p", 0.0}, {"fused", true}});
  return out;
}

}  // namespace tesseract::ops

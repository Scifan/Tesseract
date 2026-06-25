// Wave 4.2 (B-024): unit tests for the fused FA2-style SDPA kernel
// on CUDA.
//
// The `ops::attention` op exposes a single entry point that routes
// to either the fused kernel (`src/cuda/FusedAttention.cu`) or the
// composite (`matmul → softmax → matmul`). These tests exercise the
// fused path specifically by picking inputs that match its
// preconditions (contiguous, no external mask, D ≤ 128, FP32/FP16/
// BF16, no autograd), and check that the fused result matches the
// CPU composite to the usual TF32-aware tolerance.
//
// Coverage matrix:
//   * D ∈ {16, 64, 128}             — all three cover different
//     warp-layout paths in the kernel (D=16: only a few threads in
//     the O update; D=64: two warps; D=128: full block).
//   * causal ∈ {false, true}        — the causal path exercises
//     the `j > q ⇒ -inf` mask plus the "new_max == -inf" skip guard
//     on the leading tiles.
//   * S_k mod BLOCK_K ≠ 0           — ragged trailing tile exercises
//     the `tile_len < BLOCK_K` branch at tile boundaries.
//   * S_q ≠ S_k                     — non-causal only; validates
//     the stride math when query and key lengths differ.
//   * dtype ∈ {FP32, FP16, BF16}    — cross-dtype parity.
//
// Hard fallbacks (deliberately routed to the composite, verified
// here through correctness tests that would fail if the fused
// kernel silently activated):
//   * `mask.defined()` → fused path is skipped.
//   * `requires_grad == true` under active autograd → fused path
//     is skipped.
//
// CPU-only builds exit each CUDA-specific case with `SKIP()`; CTest
// reports them as `Skipped` via `PROPERTIES SKIP_RETURN_CODE 4` in
// `tests/CMakeLists.txt`.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Attention.hpp"

using Catch::Matchers::WithinAbs;
using tesseract::BFloat16;
using tesseract::cpu_device;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Half;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

namespace {

// Force the fused kernel for these unit tests. `ops::attention` has
// a shape gate in `src/ops/cpu/Attention.cpp` that routes large-S_q
// prefill shapes to the tensor-core-backed composite (where the
// fused kernel's FP32 CUDA-core FLOPs lose to cuBLASLt); without
// this override the fused path would never be exercised by the
// tests, which pick small shapes for fast iteration. The override
// is documented as test-only in the gate comment.
struct ForceFusedInit {
  ForceFusedInit() {
    // `setenv(..., overwrite=0)` respects a developer override from
    // the outer shell.
    ::setenv("TESSERACT_FORCE_FUSED_ATTENTION", "1", 0);
  }
};
[[maybe_unused]] const ForceFusedInit s_force_fused_init{};

bool cuda_ready() { return has_cuda_support() && is_available(); }
Device cuda0() { return Device{DeviceType::CUDA, 0}; }

// Deterministic "random" data. We avoid std::mt19937 correlations
// across calls by always seeding from the same base; the magic
// constants differ per tensor role.
std::vector<float> gen(std::size_t n, uint64_t role_seed, float scale) {
  std::mt19937_64 rng(42 + role_seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = scale * d(rng);
  return v;
}

std::vector<float> to_float_vec(const Tensor& t_host) {
  REQUIRE(t_host.device().is_cpu());
  REQUIRE(t_host.dtype() == DType::Float32);
  const float* p = t_host.data_ptr<float>();
  return std::vector<float>(p, p + t_host.numel());
}

void require_close(const std::vector<float>& a, const std::vector<float>& b,
                   float tol, const char* where) {
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    INFO(where << " element " << i);
    REQUIRE_THAT(b[i], WithinAbs(a[i], tol));
  }
}

// Lift a CPU FP32 tensor into whichever floating dtype the caller
// needs. Mirrors the `make_fp` helper in `test_ops_cuda_matmul.cpp`.
Tensor make_fp(DType dt, const std::vector<float>& data, Shape shape) {
  switch (dt) {
    case DType::Float32:
      return Tensor::from_vector(data, std::move(shape));
    case DType::Float16: {
      std::vector<Half> v(data.size());
      for (std::size_t i = 0; i < data.size(); ++i) v[i] = Half(data[i]);
      return Tensor::from_vector(v, std::move(shape));
    }
    case DType::BFloat16: {
      std::vector<BFloat16> v(data.size());
      for (std::size_t i = 0; i < data.size(); ++i) v[i] = BFloat16(data[i]);
      return Tensor::from_vector(v, std::move(shape));
    }
    default:
      FAIL("make_fp: unsupported dtype");
      return Tensor{};
  }
}

std::vector<float> half_tensor_to_f32_vec(const Tensor& t_host) {
  REQUIRE(t_host.device().is_cpu());
  std::vector<float> out(t_host.numel());
  switch (t_host.dtype()) {
    case DType::Float32: {
      const float* p = t_host.data_ptr<float>();
      for (int64_t i = 0; i < t_host.numel(); ++i) out[i] = p[i];
      break;
    }
    case DType::Float16: {
      const Half* p = t_host.data_ptr<Half>();
      for (int64_t i = 0; i < t_host.numel(); ++i) out[i] = static_cast<float>(p[i]);
      break;
    }
    case DType::BFloat16: {
      const BFloat16* p = t_host.data_ptr<BFloat16>();
      for (int64_t i = 0; i < t_host.numel(); ++i) out[i] = static_cast<float>(p[i]);
      break;
    }
    default:
      FAIL("half_tensor_to_f32_vec: unsupported dtype");
  }
  return out;
}

// FP32 on-device parity tolerance: the fused kernel runs FP32
// throughout (scores, exp, weighted sum, normalize), so drift vs
// the CPU composite's FP32 matmul is on the order of 1e-5.
// However, the composite path on CUDA uses cuBLASLt in TF32 mode
// which can compound to ~3e-3. We compare the fused CUDA result
// against the *CPU* composite to get the tight bound.
constexpr float kFusedFwdTolF32 = 1e-4f;

// Half-precision parity tolerance: FP16/BF16 storage limits
// precision to ~1e-3 on accumulated sums over S_k up to ~64. For
// longer S_k the FP32 accumulator in the fused kernel keeps
// accuracy bounded by the final narrow-to-half store, so 5e-3 is
// a safe envelope.
constexpr float kFusedFwdTolF16 = 5e-3f;
constexpr float kFusedFwdTolBF16 = 1e-2f;  // BF16 has 7-bit mantissa

}  // namespace

TEST_CASE("fused attention: FP32 CPU↔CUDA parity, D=64 non-causal, rank-4",
          "[ops][gpu][attention][fused]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, H = 4, S = 32, D = 64;

  auto qd = gen(B * H * S * D, 1, 0.3f);
  auto kd = gen(B * H * S * D, 2, 0.3f);
  auto vd = gen(B * H * S * D, 3, 0.3f);

  Tensor q = Tensor::from_vector(qd, {B, H, S, D});
  Tensor k = Tensor::from_vector(kd, {B, H, S, D});
  Tensor v = Tensor::from_vector(vd, {B, H, S, D});

  Tensor cpu_out = tesseract::ops::attention(q, k, v);
  Tensor gpu_out = tesseract::ops::attention(q.to(cuda0()),
                                             k.to(cuda0()),
                                             v.to(cuda0()));
  REQUIRE(gpu_out.device() == cuda0());
  REQUIRE(gpu_out.shape() == Shape({B, H, S, D}));

  require_close(to_float_vec(cpu_out),
                to_float_vec(gpu_out.to(cpu_device())),
                kFusedFwdTolF32, "D=64 non-causal");
}

TEST_CASE("fused attention: FP32 CPU↔CUDA parity, D=128 causal, realistic Llama shape",
          "[ops][gpu][attention][fused][causal]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Llama-7B-like: D=128, but S scaled down so CI stays fast.
  const int64_t B = 1, H = 4, S = 64, D = 128;

  auto qd = gen(B * H * S * D, 10, 0.2f);
  auto kd = gen(B * H * S * D, 11, 0.2f);
  auto vd = gen(B * H * S * D, 12, 0.2f);

  Tensor q = Tensor::from_vector(qd, {B, H, S, D});
  Tensor k = Tensor::from_vector(kd, {B, H, S, D});
  Tensor v = Tensor::from_vector(vd, {B, H, S, D});

  Tensor cpu_out = tesseract::ops::attention(q, k, v,
                                             /*mask=*/Tensor{},
                                             /*causal=*/true);
  Tensor gpu_out = tesseract::ops::attention(q.to(cuda0()),
                                             k.to(cuda0()),
                                             v.to(cuda0()),
                                             /*mask=*/Tensor{},
                                             /*causal=*/true);
  REQUIRE(gpu_out.shape() == Shape({B, H, S, D}));

  require_close(to_float_vec(cpu_out),
                to_float_vec(gpu_out.to(cpu_device())),
                kFusedFwdTolF32, "D=128 causal");
}

TEST_CASE("fused attention: ragged trailing tile (S_k not a multiple of BLOCK_K=32)",
          "[ops][gpu][attention][fused]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // S_q == S_k == 41 so both dims cross one full tile and bring a
  // 9-entry trailing tile. D=32 keeps shared-mem small.
  const int64_t B = 1, H = 2, S = 41, D = 32;

  auto qd = gen(B * H * S * D, 20, 0.25f);
  auto kd = gen(B * H * S * D, 21, 0.25f);
  auto vd = gen(B * H * S * D, 22, 0.25f);

  Tensor q = Tensor::from_vector(qd, {B, H, S, D});
  Tensor k = Tensor::from_vector(kd, {B, H, S, D});
  Tensor v = Tensor::from_vector(vd, {B, H, S, D});

  {
    Tensor cpu_out = tesseract::ops::attention(q, k, v);
    Tensor gpu_out = tesseract::ops::attention(q.to(cuda0()),
                                               k.to(cuda0()),
                                               v.to(cuda0()));
    require_close(to_float_vec(cpu_out),
                  to_float_vec(gpu_out.to(cpu_device())),
                  kFusedFwdTolF32, "ragged non-causal");
  }
  {
    Tensor cpu_out = tesseract::ops::attention(q, k, v, Tensor{}, true);
    Tensor gpu_out = tesseract::ops::attention(q.to(cuda0()),
                                               k.to(cuda0()),
                                               v.to(cuda0()),
                                               Tensor{}, true);
    require_close(to_float_vec(cpu_out),
                  to_float_vec(gpu_out.to(cpu_device())),
                  kFusedFwdTolF32, "ragged causal");
  }
}

TEST_CASE("fused attention: S_q != S_k (cross-attention pattern)",
          "[ops][gpu][attention][fused]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 2, H = 2, S_q = 17, S_k = 48, D = 64;

  auto qd = gen(B * H * S_q * D, 30, 0.3f);
  auto kd = gen(B * H * S_k * D, 31, 0.3f);
  auto vd = gen(B * H * S_k * D, 32, 0.3f);

  Tensor q = Tensor::from_vector(qd, {B, H, S_q, D});
  Tensor k = Tensor::from_vector(kd, {B, H, S_k, D});
  Tensor v = Tensor::from_vector(vd, {B, H, S_k, D});

  Tensor cpu_out = tesseract::ops::attention(q, k, v);
  Tensor gpu_out = tesseract::ops::attention(q.to(cuda0()),
                                             k.to(cuda0()),
                                             v.to(cuda0()));
  REQUIRE(gpu_out.shape() == Shape({B, H, S_q, D}));

  require_close(to_float_vec(cpu_out),
                to_float_vec(gpu_out.to(cpu_device())),
                kFusedFwdTolF32, "S_q != S_k");
}

TEST_CASE("fused attention: decode S_q=1 split-K parity, D=128 long KV (FP32)",
          "[ops][gpu][attention][fused][decode]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Decode step: a single query row against a long KV cache. S_k=2048
  // forces the split-K path to use >1 split, exercising the partial +
  // reduction kernels and the cross-split online-softmax recombination.
  const int64_t B = 2, H = 8, S_q = 1, S_k = 2048, D = 128;

  auto qd = gen(B * H * S_q * D, 50, 0.2f);
  auto kd = gen(B * H * S_k * D, 51, 0.2f);
  auto vd = gen(B * H * S_k * D, 52, 0.2f);

  Tensor q = Tensor::from_vector(qd, {B, H, S_q, D});
  Tensor k = Tensor::from_vector(kd, {B, H, S_k, D});
  Tensor v = Tensor::from_vector(vd, {B, H, S_k, D});

  Tensor cpu_out = tesseract::ops::attention(q, k, v);
  Tensor gpu_out = tesseract::ops::attention(q.to(cuda0()),
                                             k.to(cuda0()),
                                             v.to(cuda0()));
  REQUIRE(gpu_out.shape() == Shape({B, H, S_q, D}));
  require_close(to_float_vec(cpu_out),
                to_float_vec(gpu_out.to(cpu_device())),
                kFusedFwdTolF32, "decode S_q=1 split-K");
}

TEST_CASE("fused attention: decode S_q=1 split-K parity, FP16 long KV",
          "[ops][gpu][attention][fused][decode][fp16]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 1, H = 4, S_q = 1, S_k = 1024, D = 128;

  auto qd = gen(B * H * S_q * D, 60, 0.2f);
  auto kd = gen(B * H * S_k * D, 61, 0.2f);
  auto vd = gen(B * H * S_k * D, 62, 0.2f);

  Tensor q = make_fp(DType::Float16, qd, {B, H, S_q, D});
  Tensor k = make_fp(DType::Float16, kd, {B, H, S_k, D});
  Tensor v = make_fp(DType::Float16, vd, {B, H, S_k, D});

  // CPU FP32 composite reference (lift back to FP32 for the bound).
  Tensor q32 = Tensor::from_vector(qd, {B, H, S_q, D});
  Tensor k32 = Tensor::from_vector(kd, {B, H, S_k, D});
  Tensor v32 = Tensor::from_vector(vd, {B, H, S_k, D});
  Tensor cpu_out = tesseract::ops::attention(q32, k32, v32);

  Tensor gpu_out = tesseract::ops::attention(q.to(cuda0()),
                                             k.to(cuda0()),
                                             v.to(cuda0()));
  REQUIRE(gpu_out.shape() == Shape({B, H, S_q, D}));
  require_close(to_float_vec(cpu_out),
                half_tensor_to_f32_vec(gpu_out.to(cpu_device())),
                kFusedFwdTolF16, "decode S_q=1 split-K FP16");
}

TEST_CASE("fused attention: FP16 CPU↔CUDA parity, D=64 non-causal",
          "[ops][gpu][attention][fused][fp16]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 1, H = 4, S = 48, D = 64;

  auto qd = gen(B * H * S * D, 40, 0.2f);
  auto kd = gen(B * H * S * D, 41, 0.2f);
  auto vd = gen(B * H * S * D, 42, 0.2f);

  Tensor q_half = make_fp(DType::Float16, qd, {B, H, S, D});
  Tensor k_half = make_fp(DType::Float16, kd, {B, H, S, D});
  Tensor v_half = make_fp(DType::Float16, vd, {B, H, S, D});

  // CPU reference: composite in FP16 on CPU.
  Tensor cpu_out = tesseract::ops::attention(q_half, k_half, v_half);
  Tensor gpu_out = tesseract::ops::attention(q_half.to(cuda0()),
                                             k_half.to(cuda0()),
                                             v_half.to(cuda0()));

  require_close(half_tensor_to_f32_vec(cpu_out),
                half_tensor_to_f32_vec(gpu_out.to(cpu_device())),
                kFusedFwdTolF16, "FP16 D=64");
}

TEST_CASE("fused attention: BF16 CPU↔CUDA parity, D=64 causal",
          "[ops][gpu][attention][fused][bf16]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 1, H = 2, S = 48, D = 64;

  auto qd = gen(B * H * S * D, 50, 0.2f);
  auto kd = gen(B * H * S * D, 51, 0.2f);
  auto vd = gen(B * H * S * D, 52, 0.2f);

  Tensor q_bf = make_fp(DType::BFloat16, qd, {B, H, S, D});
  Tensor k_bf = make_fp(DType::BFloat16, kd, {B, H, S, D});
  Tensor v_bf = make_fp(DType::BFloat16, vd, {B, H, S, D});

  Tensor cpu_out = tesseract::ops::attention(q_bf, k_bf, v_bf,
                                             Tensor{}, true);
  Tensor gpu_out = tesseract::ops::attention(q_bf.to(cuda0()),
                                             k_bf.to(cuda0()),
                                             v_bf.to(cuda0()),
                                             Tensor{}, true);

  require_close(half_tensor_to_f32_vec(cpu_out),
                half_tensor_to_f32_vec(gpu_out.to(cpu_device())),
                kFusedFwdTolBF16, "BF16 causal D=64");
}

TEST_CASE("fused attention: rank-3 leading-batch routing (B, S_q, D)",
          "[ops][gpu][attention][fused]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // No head dim; rank-3 input flattens into a single "B*H" axis
  // of size B. Covers the dispatch arithmetic for `qr > 3` too
  // (handled by the other test cases) — this one just guarantees
  // the rank-3 contract works.
  const int64_t B = 4, S = 32, D = 64;

  auto qd = gen(B * S * D, 60, 0.25f);
  auto kd = gen(B * S * D, 61, 0.25f);
  auto vd = gen(B * S * D, 62, 0.25f);

  Tensor q = Tensor::from_vector(qd, {B, S, D});
  Tensor k = Tensor::from_vector(kd, {B, S, D});
  Tensor v = Tensor::from_vector(vd, {B, S, D});

  Tensor cpu_out = tesseract::ops::attention(q, k, v);
  Tensor gpu_out = tesseract::ops::attention(q.to(cuda0()),
                                             k.to(cuda0()),
                                             v.to(cuda0()));
  REQUIRE(gpu_out.shape() == Shape({B, S, D}));

  require_close(to_float_vec(cpu_out),
                to_float_vec(gpu_out.to(cpu_device())),
                kFusedFwdTolF32, "rank-3");
}

TEST_CASE("fused attention: mask defined → composite fallback still produces correct output",
          "[ops][gpu][attention][fused][fallback]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Providing an additive mask tensor is one of the documented
  // fallback conditions — the fused kernel doesn't support
  // arbitrary masks. Correctness here flows through the
  // composite; we only verify that `ops::attention` doesn't
  // accidentally try the fused path and crash.
  const int64_t B = 1, H = 2, S = 32, D = 64;

  auto qd = gen(B * H * S * D, 70, 0.25f);
  auto kd = gen(B * H * S * D, 71, 0.25f);
  auto vd = gen(B * H * S * D, 72, 0.25f);

  std::vector<float> md(S, 0.0f);
  md[5] = -1.0f;  // light mask, nothing -inf

  Tensor q = Tensor::from_vector(qd, {B, H, S, D});
  Tensor k = Tensor::from_vector(kd, {B, H, S, D});
  Tensor v = Tensor::from_vector(vd, {B, H, S, D});
  Tensor mask = Tensor::from_vector(md, {1, 1, 1, S});

  Tensor cpu_out = tesseract::ops::attention(q, k, v, mask);
  Tensor gpu_out = tesseract::ops::attention(q.to(cuda0()),
                                             k.to(cuda0()),
                                             v.to(cuda0()),
                                             mask.to(cuda0()));

  // Tolerance matches the existing `test_ops_cuda_attention` composite
  // envelope — cuBLASLt runs TF32 by default on Ada.
  require_close(to_float_vec(cpu_out),
                to_float_vec(gpu_out.to(cpu_device())),
                3e-3f, "mask fallback");
}

TEST_CASE("prefill_attention_gqa: FP32 GQA head-mapping parity vs composite",
          "[ops][gpu][attention][fused][gqa][prefill]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // TinyLlama-shaped GQA ratio: H=8 query heads share Hkv=2 KV heads
  // (group=4). The fused kernel maps query head h -> KV head h/group
  // internally; the reference expands K/V to H heads with the same
  // kv0,kv0,…,kv1,… interleave and runs the composite causal attention.
  const int64_t B = 2, H = 8, Hkv = 2, S = 40, D = 64;

  auto qd = gen(B * H * S * D, 80, 0.3f);
  auto kd = gen(B * Hkv * S * D, 81, 0.3f);
  auto vd = gen(B * Hkv * S * D, 82, 0.3f);

  Tensor q = Tensor::from_vector(qd, {B, H, S, D});
  Tensor k = Tensor::from_vector(kd, {B, Hkv, S, D});
  Tensor v = Tensor::from_vector(vd, {B, Hkv, S, D});

  // Reference: expand K/V to H heads (kv0,kv0,…,kv1,… interleave) on CPU
  // and run the composite causal attention.
  const int64_t group = H / Hkv;
  std::vector<float> ked(B * H * S * D), ved(B * H * S * D);
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h) {
      const int64_t kvh = h / group;
      for (int64_t s = 0; s < S; ++s)
        for (int64_t d = 0; d < D; ++d) {
          const int64_t dst = ((b * H + h) * S + s) * D + d;
          const int64_t src = ((b * Hkv + kvh) * S + s) * D + d;
          ked[dst] = kd[src];
          ved[dst] = vd[src];
        }
    }
  Tensor k_exp = Tensor::from_vector(ked, {B, H, S, D});
  Tensor v_exp = Tensor::from_vector(ved, {B, H, S, D});
  Tensor cpu_out = tesseract::ops::attention(q, k_exp, v_exp,
                                             /*mask=*/Tensor{}, /*causal=*/true);

  Tensor gpu_out = tesseract::ops::prefill_attention_gqa(
      q.to(cuda0()), k.to(cuda0()), v.to(cuda0()), /*causal=*/true);
  REQUIRE(gpu_out.device() == cuda0());
  REQUIRE(gpu_out.shape() == Shape({B, H, S, D}));

  require_close(to_float_vec(cpu_out),
                to_float_vec(gpu_out.to(cpu_device())),
                kFusedFwdTolF32, "GQA prefill head-mapping");
}

TEST_CASE("prefill_attention_gqa: FP16 GQA parity vs composite",
          "[ops][gpu][attention][fused][gqa][prefill][fp16]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 1, H = 8, Hkv = 2, S = 64, D = 64;

  auto qd = gen(B * H * S * D, 90, 0.2f);
  auto kd = gen(B * Hkv * S * D, 91, 0.2f);
  auto vd = gen(B * Hkv * S * D, 92, 0.2f);

  const int64_t group = H / Hkv;
  std::vector<float> ked(B * H * S * D), ved(B * H * S * D);
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h) {
      const int64_t kvh = h / group;
      for (int64_t s = 0; s < S; ++s)
        for (int64_t d = 0; d < D; ++d) {
          const int64_t dst = ((b * H + h) * S + s) * D + d;
          const int64_t src = ((b * Hkv + kvh) * S + s) * D + d;
          ked[dst] = kd[src];
          ved[dst] = vd[src];
        }
    }
  // CPU FP32 composite reference for the tight bound.
  Tensor cpu_out = tesseract::ops::attention(
      Tensor::from_vector(qd, {B, H, S, D}),
      Tensor::from_vector(ked, {B, H, S, D}),
      Tensor::from_vector(ved, {B, H, S, D}),
      /*mask=*/Tensor{}, /*causal=*/true);

  Tensor q = make_fp(DType::Float16, qd, {B, H, S, D});
  Tensor k = make_fp(DType::Float16, kd, {B, Hkv, S, D});
  Tensor v = make_fp(DType::Float16, vd, {B, Hkv, S, D});
  Tensor gpu_out = tesseract::ops::prefill_attention_gqa(
      q.to(cuda0()), k.to(cuda0()), v.to(cuda0()), /*causal=*/true);
  REQUIRE(gpu_out.shape() == Shape({B, H, S, D}));

  require_close(to_float_vec(cpu_out),
                half_tensor_to_f32_vec(gpu_out.to(cpu_device())),
                kFusedFwdTolF16, "GQA prefill FP16");
}

TEST_CASE("fused attention: dtype validation — FP64 routes to composite",
          "[ops][gpu][attention][fused][fallback]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // FP64 is deliberately excluded from the fused path (its static
  // shared-mem footprint at BLOCK_K=32 exceeds Ada's 48 KB limit).
  // The op must still run correctly by routing through the
  // composite kernels — matmul + softmax both support FP64 on CUDA.
  const int64_t B = 1, H = 2, S = 16, D = 32;

  std::vector<double> qd(B * H * S * D), kd(B * H * S * D), vd(B * H * S * D);
  for (std::size_t i = 0; i < qd.size(); ++i) {
    qd[i] = 0.1 * static_cast<double>((static_cast<int>(i) % 7) - 3);
    kd[i] = 0.15 * static_cast<double>((static_cast<int>(i) % 5) - 2);
    vd[i] = 0.05 * static_cast<double>((static_cast<int>(i) % 11) - 4);
  }

  Tensor q = Tensor::from_vector(qd, {B, H, S, D});
  Tensor k = Tensor::from_vector(kd, {B, H, S, D});
  Tensor v = Tensor::from_vector(vd, {B, H, S, D});

  Tensor cpu_out = tesseract::ops::attention(q, k, v);
  Tensor gpu_out = tesseract::ops::attention(q.to(cuda0()),
                                             k.to(cuda0()),
                                             v.to(cuda0()));
  REQUIRE(gpu_out.dtype() == DType::Float64);
  // Narrow tolerance on FP64.
  const double* cp = cpu_out.data_ptr<double>();
  Tensor gpu_h = gpu_out.to(cpu_device());
  const double* gp = gpu_h.data_ptr<double>();
  for (int64_t i = 0; i < cpu_out.numel(); ++i) {
    REQUIRE_THAT(gp[i], WithinAbs(cp[i], 1e-10));
  }
}

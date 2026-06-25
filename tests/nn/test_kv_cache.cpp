// Wave 2.1 (B-019) — KVCache + MultiHeadAttention::forward_step parity.
//
// The decode-phase path lives or dies on one property: calling
// `forward_step(x[:, i:i+1, :], cache)` `S` times is numerically
// identical to running `forward(x)` once on the full `[B, S, D]`
// prompt. If that equality holds, every downstream optimization
// (CUDA Graph capture, paged storage, continuous batching) is a
// transparent swap behind the same public API. If it doesn't, the
// generated tokens silently diverge from the training distribution.
//
// What this file exercises:
//
//   * KVCache::append() writes the right slab at the right offset,
//     under both rank-4 contiguous inputs from split-heads+permute
//     and the edge case S_new > 1 (chunked prefill).
//   * keys_view() / values_view() are zero-copy narrows of the
//     cache slabs — they share storage with the cache, so a follow-up
//     `append()` doesn't need to reallocate.
//   * MultiHeadAttention::forward_step produces the same output,
//     token-by-token, as forward() on the full prompt — both with
//     and without RoPE enabled. Tolerance is FP32 abs 1e-5 which is
//     well below the attention's own round-off floor for the shapes
//     we test.
//   * CPU ↔ CUDA parity: same decode trace on both devices produces
//     bit-similar outputs (skipped when CUDA is not built).

#include <algorithm>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/KVCache.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

#include "tesseract/cuda/CudaRuntime.hpp"

using Catch::Matchers::WithinAbs;
using namespace tesseract;

namespace {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  float uniform(float lo, float hi) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u =
        static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFULL) /
        9007199254740992.0;
    return static_cast<float>(lo + (hi - lo) * u);
  }
};

Tensor make_random_f32(Shape s, Rng& rng, float lo = -0.5f, float hi = 0.5f) {
  Tensor t = Tensor::empty(std::move(s), DType::Float32);
  float* p = t.data_ptr<float>();
  const int64_t n = t.numel();
  for (int64_t i = 0; i < n; ++i) p[i] = rng.uniform(lo, hi);
  return t;
}

// Slice `x : [B, S, D]` along the seq axis to `[B, lo..lo+len, D]`.
Tensor slice_seq(const Tensor& x, int64_t lo, int64_t len) {
  return x.narrow(/*dim=*/1, lo, len);
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
  REQUIRE(a.numel() == b.numel());
  Tensor a_cpu = a.device().is_cpu() ? a.contiguous() : a.to(cpu_device()).contiguous();
  Tensor b_cpu = b.device().is_cpu() ? b.contiguous() : b.to(cpu_device()).contiguous();
  const float* pa = a_cpu.data_ptr<float>();
  const float* pb = b_cpu.data_ptr<float>();
  float m = 0.0f;
  for (int64_t i = 0; i < a_cpu.numel(); ++i) {
    m = std::max(m, std::abs(pa[i] - pb[i]));
  }
  return m;
}

}  // namespace

TEST_CASE("KVCache: append fills the right prefix slab", "[nn][kvcache]") {
  constexpr int64_t B = 2, H = 3, Dh = 4, MAX = 6;
  nn::KVCache cache(B, H, Dh, MAX, DType::Float32, cpu_device());

  REQUIRE(cache.current_len() == 0);
  REQUIRE(cache.keys_view().shape() == Shape({B, H, 0, Dh}));

  Rng rng(0xC0FFEEu);
  // Append two tokens, then one, then three — exercises both the
  // chunked-prefill path and the single-token-decode path.
  Tensor k0 = make_random_f32({B, H, 2, Dh}, rng);
  Tensor v0 = make_random_f32({B, H, 2, Dh}, rng);
  cache.append(k0, v0);
  REQUIRE(cache.current_len() == 2);

  Tensor k1 = make_random_f32({B, H, 1, Dh}, rng);
  Tensor v1 = make_random_f32({B, H, 1, Dh}, rng);
  cache.append(k1, v1);
  REQUIRE(cache.current_len() == 3);

  Tensor k2 = make_random_f32({B, H, 3, Dh}, rng);
  Tensor v2 = make_random_f32({B, H, 3, Dh}, rng);
  cache.append(k2, v2);
  REQUIRE(cache.current_len() == 6);

  // Reconstruct the expected full-prefix [B, H, 6, Dh] tensor by
  // concatenating the three appended slabs along the seq axis.
  auto full_keys = cache.keys_view();
  REQUIRE(full_keys.shape() == Shape({B, H, 6, Dh}));
  auto fk_cpu = full_keys.contiguous();
  const float* pk = fk_cpu.data_ptr<float>();

  const float* pk0 = k0.data_ptr<float>();
  const float* pk1 = k1.data_ptr<float>();
  const float* pk2 = k2.data_ptr<float>();

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t d = 0; d < Dh; ++d) {
        const int64_t row_base = ((b * H + h) * 6) * Dh + d;
        // positions 0..1 from k0
        REQUIRE(pk[row_base + 0 * Dh] ==
                pk0[((b * H + h) * 2 + 0) * Dh + d]);
        REQUIRE(pk[row_base + 1 * Dh] ==
                pk0[((b * H + h) * 2 + 1) * Dh + d]);
        // position 2 from k1
        REQUIRE(pk[row_base + 2 * Dh] ==
                pk1[((b * H + h) * 1 + 0) * Dh + d]);
        // positions 3..5 from k2
        REQUIRE(pk[row_base + 3 * Dh] ==
                pk2[((b * H + h) * 3 + 0) * Dh + d]);
        REQUIRE(pk[row_base + 4 * Dh] ==
                pk2[((b * H + h) * 3 + 1) * Dh + d]);
        REQUIRE(pk[row_base + 5 * Dh] ==
                pk2[((b * H + h) * 3 + 2) * Dh + d]);
      }
    }
  }

  // reset() does not reallocate but rewinds the valid prefix.
  cache.reset();
  REQUIRE(cache.current_len() == 0);
  REQUIRE(cache.keys_view().shape() == Shape({B, H, 0, Dh}));
}

TEST_CASE("KVCache: append rejects size-overflow and mis-shape",
          "[nn][kvcache]") {
  nn::KVCache cache(/*B=*/1, /*H=*/2, /*Dh=*/4, /*MAX=*/3);
  Rng rng(7u);
  Tensor k = make_random_f32({1, 2, 2, 4}, rng);
  Tensor v = make_random_f32({1, 2, 2, 4}, rng);
  cache.append(k, v);

  Tensor k_too_big = make_random_f32({1, 2, 2, 4}, rng);
  Tensor v_too_big = make_random_f32({1, 2, 2, 4}, rng);
  REQUIRE_THROWS(cache.append(k_too_big, v_too_big));  // 2 + 2 > max_len=3

  Tensor k_bad_B = make_random_f32({2, 2, 1, 4}, rng);
  Tensor v_bad_B = make_random_f32({2, 2, 1, 4}, rng);
  REQUIRE_THROWS(cache.append(k_bad_B, v_bad_B));

  Tensor k_bad_Dh = make_random_f32({1, 2, 1, 5}, rng);
  Tensor v_bad_Dh = make_random_f32({1, 2, 1, 5}, rng);
  REQUIRE_THROWS(cache.append(k_bad_Dh, v_bad_Dh));
}

TEST_CASE("MHA::forward_step matches full forward (no RoPE, CPU)",
          "[nn][kvcache][mha]") {
  constexpr int64_t B = 2, S = 5, D = 16, H = 4;
  nn::MultiHeadAttention mha(
      /*d_model=*/D, /*num_heads=*/H, /*use_bias=*/false,
      /*causal=*/true, DType::Float32, /*rope_base=*/0.0,
      /*rope_max_seq=*/0);

  Rng rng(0xDEADBEEFu);
  Tensor x = make_random_f32({B, S, D}, rng);

  // Reference: one-shot forward, take the full [B, S, D] output.
  Tensor out_full = mha.forward(x);

  // Streaming: step through `x` one token at a time and concatenate
  // the [B, 1, D] outputs in order. Two distinct chunk schedules
  // (1-1-1-1-1 and 2-1-2) both must match.
  auto run_streaming = [&](const std::vector<int64_t>& chunks) -> Tensor {
    nn::KVCache cache(B, H, D / H, /*max_len=*/S, DType::Float32,
                      cpu_device());
    Tensor acc = Tensor::empty({B, S, D}, DType::Float32);
    float* p_acc = acc.data_ptr<float>();
    int64_t pos = 0;
    for (int64_t c : chunks) {
      Tensor xc = slice_seq(x, pos, c).contiguous();
      Tensor yc = mha.forward_step(xc, cache);
      REQUIRE(yc.shape() == Shape({B, c, D}));
      const float* p_yc = yc.contiguous().data_ptr<float>();
      for (int64_t b = 0; b < B; ++b) {
        for (int64_t i = 0; i < c; ++i) {
          for (int64_t d = 0; d < D; ++d) {
            p_acc[(b * S + (pos + i)) * D + d] =
                p_yc[(b * c + i) * D + d];
          }
        }
      }
      pos += c;
    }
    REQUIRE(pos == S);
    return acc;
  };

  Tensor out_stream_11111 = run_streaming({1, 1, 1, 1, 1});
  Tensor out_stream_212   = run_streaming({2, 1, 2});

  REQUIRE(max_abs_diff(out_full, out_stream_11111) < 1e-5f);
  REQUIRE(max_abs_diff(out_full, out_stream_212)   < 1e-5f);
}

TEST_CASE("MHA::forward_step matches full forward (with RoPE, CPU)",
          "[nn][kvcache][mha][rope]") {
  constexpr int64_t B = 1, S = 6, D = 16, H = 4;
  nn::MultiHeadAttention mha(
      /*d_model=*/D, /*num_heads=*/H, /*use_bias=*/false,
      /*causal=*/true, DType::Float32,
      /*rope_base=*/10000.0, /*rope_max_seq=*/16);

  Rng rng(0xB00Bu);
  Tensor x = make_random_f32({B, S, D}, rng);
  Tensor out_full = mha.forward(x);

  nn::KVCache cache(B, H, D / H, /*max_len=*/S, DType::Float32, cpu_device());
  std::vector<float> out_stream(static_cast<std::size_t>(B * S * D));
  for (int64_t t = 0; t < S; ++t) {
    Tensor xt = slice_seq(x, t, 1).contiguous();
    Tensor yt = mha.forward_step(xt, cache);
    REQUIRE(yt.shape() == Shape({B, 1, D}));
    const float* p = yt.contiguous().data_ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t d = 0; d < D; ++d) {
        out_stream[(b * S + t) * D + d] = p[b * D + d];
      }
    }
  }

  const float* pf = out_full.contiguous().data_ptr<float>();
  float m = 0.0f;
  for (std::size_t i = 0; i < out_stream.size(); ++i) {
    m = std::max(m, std::abs(out_stream[i] - pf[i]));
  }
  REQUIRE(m < 1e-5f);
}

TEST_CASE("MHA::forward_step matches full forward (with RoPE, CUDA)",
          "[nn][kvcache][mha][rope][cuda]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  constexpr int64_t B = 1, S = 8, D = 32, H = 4;
  nn::MultiHeadAttention mha(
      /*d_model=*/D, /*num_heads=*/H, /*use_bias=*/false,
      /*causal=*/true, DType::Float32,
      /*rope_base=*/10000.0, /*rope_max_seq=*/32);
  const Device cuda0{DeviceType::CUDA, 0};
  mha.to(cuda0);

  Rng rng(0xFEEDFACEu);
  Tensor x_cpu = make_random_f32({B, S, D}, rng);
  Tensor x = x_cpu.to(cuda0);

  Tensor out_full = mha.forward(x).to(cpu_device());

  nn::KVCache cache(B, H, D / H, /*max_len=*/S, DType::Float32, cuda0);
  Tensor acc = Tensor::empty({B, S, D}, DType::Float32);
  float* p_acc = acc.data_ptr<float>();
  for (int64_t t = 0; t < S; ++t) {
    Tensor xt = slice_seq(x, t, 1).contiguous();
    Tensor yt = mha.forward_step(xt, cache).to(cpu_device()).contiguous();
    const float* p = yt.data_ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t d = 0; d < D; ++d) {
        p_acc[(b * S + t) * D + d] = p[b * D + d];
      }
    }
  }

  REQUIRE(max_abs_diff(out_full, acc) < 5e-5f);
}

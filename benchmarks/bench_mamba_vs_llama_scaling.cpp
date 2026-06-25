// M4 Track A2 (B-039) — the architectural-differentiation metric for SSMs:
// decode-step latency and resident state *as a function of context length L*.
//
// The whole point of a selective SSM (Mamba) over attention is asymptotics:
//   * attention decode: per-step cost O(L) (score against the full KV prefix)
//     and KV memory O(L) (store K/V for every past token);
//   * SSM decode:       per-step cost O(1) and state memory O(1) (a fixed-width
//     recurrent state, independent of L).
//
// Every existing Tesseract bench is a single-shape micro-bench; none shows this
// scaling. This one sweeps L and emits, per runtime, the per-step decode
// latency and the resident KV/state bytes, so the O(L)-vs-O(1) gap is a
// *measured* curve rather than a claim. CPU-only and informational (no hard
// bar) — see docs/design/mamba-scaling.md.
//
// Output (one line per (runtime, L)):
//   [bench] <runtime> ssm_scaling  L=<..>  ms_per_step=<..>  decode_tok_s=<..>
//           state_mib=<..>
//
// Both models share d_model / layers / vocab so the embed + lm_head per-step
// work is identical and cancels — the only moving part is attention vs SSM.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/models/MambaModel.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using tesseract::DType;
using tesseract::Device;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::cpu_device;

constexpr int64_t kDModel = 256;
constexpr int64_t kLayers = 4;
constexpr int64_t kVocab = 4096;
constexpr int64_t kHeads = 8;          // Llama: head_dim = 32, MHA (kv == q)
constexpr int64_t kDState = 16;        // Mamba SSM state width N
constexpr int64_t kDConv = 4;
constexpr int64_t kExpand = 2;         // d_inner = expand * d_model = 512
constexpr int64_t kDecodeSteps = 24;   // timed steps per length
constexpr double kF32 = 4.0;           // bytes/elem (Float32)

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

Tensor token(int64_t id) {
  Tensor t = Tensor::empty({1, 1}, DType::Int64, cpu_device());
  t.data_ptr<int64_t>()[0] = id;
  return t;
}

Tensor tokens_row(int64_t len) {
  Tensor t = Tensor::empty({1, len}, DType::Int64, cpu_device());
  int64_t* p = t.data_ptr<int64_t>();
  for (int64_t i = 0; i < len; ++i) p[i] = (i * 7 + 1) % kVocab;
  return t;
}

void bench_llama(const std::vector<int64_t>& lengths) {
  using namespace tesseract::models;
  LlamaConfig cfg;
  cfg.vocab_size = kVocab;
  cfg.hidden_size = kDModel;
  cfg.num_hidden_layers = kLayers;
  cfg.num_attention_heads = kHeads;
  cfg.num_key_value_heads = kHeads;
  cfg.intermediate_size = 688;
  cfg.max_position_embeddings = lengths.back() + kDecodeSteps + 8;
  cfg.dtype = DType::Float32;
  LlamaModel model(cfg);
  const int64_t head_dim = kDModel / kHeads;

  for (int64_t L : lengths) {
    auto caches = model.make_kv_caches(/*batch=*/1,
                                       /*max_len=*/L + kDecodeSteps + 2);
    (void)model.forward_step(tokens_row(L), caches);  // one-shot prefill to L
    (void)model.forward_step(token(123), caches);     // warm one decode step

    const auto t0 = Clock::now();
    for (int64_t s = 0; s < kDecodeSteps; ++s)
      (void)model.forward_step(token((s * 13 + 5) % kVocab), caches);
    const double secs = seconds_since(t0);

    const double ms_per_step = 1e3 * secs / kDecodeSteps;
    const double tok_s = kDecodeSteps / secs;
    // Resident KV at length ~L: K and V, all layers, all kv-heads.
    const double kv_mib =
        2.0 * kLayers * kHeads * head_dim * static_cast<double>(L) * kF32 /
        (1024.0 * 1024.0);
    std::printf(
        "[bench] llama  ssm_scaling  L=%lld  ms_per_step=%.4f  "
        "decode_tok_s=%.1f  state_mib=%.3f\n",
        static_cast<long long>(L), ms_per_step, tok_s, kv_mib);
  }
}

void bench_mamba(const std::vector<int64_t>& lengths) {
  using namespace tesseract::models;
  MambaConfig cfg;
  cfg.vocab_size = kVocab;
  cfg.hidden_size = kDModel;
  cfg.num_hidden_layers = kLayers;
  cfg.d_state = kDState;
  cfg.d_conv = kDConv;
  cfg.expand = kExpand;
  cfg.dtype = DType::Float32;
  MambaModel model(cfg);
  const int64_t d_inner = kExpand * kDModel;

  for (int64_t L : lengths) {
    auto caches = model.make_state_caches(/*batch=*/1);
    for (int64_t i = 0; i < L; ++i)              // prefill: O(1) state per step
      (void)model.forward_step(token((i * 7 + 1) % kVocab), caches);
    (void)model.forward_step(token(123), caches);  // warm one decode step

    const auto t0 = Clock::now();
    for (int64_t s = 0; s < kDecodeSteps; ++s)
      (void)model.forward_step(token((s * 13 + 5) % kVocab), caches);
    const double secs = seconds_since(t0);

    const double ms_per_step = 1e3 * secs / kDecodeSteps;
    const double tok_s = kDecodeSteps / secs;
    // Resident state: conv_state [(d_conv-1) x d_inner] + ssm_state
    // [d_inner x d_state], all layers — constant in L.
    const double state_mib =
        kLayers * d_inner * ((kDConv - 1) + kDState) * kF32 /
        (1024.0 * 1024.0);
    std::printf(
        "[bench] mamba  ssm_scaling  L=%lld  ms_per_step=%.4f  "
        "decode_tok_s=%.1f  state_mib=%.3f\n",
        static_cast<long long>(L), ms_per_step, tok_s, state_mib);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int64_t> lengths = {128, 256, 512, 1024, 2048};
  if (argc > 1) {
    lengths.clear();
    for (int i = 1; i < argc; ++i) lengths.push_back(std::atoll(argv[i]));
  }

  std::printf("# Mamba (O(1) decode/state) vs Llama (O(L) decode/KV) — CPU\n");
  std::printf("# d_model=%lld layers=%lld vocab=%lld | llama heads=%lld | "
              "mamba d_state=%lld expand=%lld\n",
              static_cast<long long>(kDModel), static_cast<long long>(kLayers),
              static_cast<long long>(kVocab), static_cast<long long>(kHeads),
              static_cast<long long>(kDState), static_cast<long long>(kExpand));
  bench_llama(lengths);
  bench_mamba(lengths);
  return 0;
}

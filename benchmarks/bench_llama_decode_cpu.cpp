// M4 cross-cutting (B-046) — end-to-end CPU decode throughput for
// `tesseract::models::LlamaModel`, in a form directly comparable to
// `llama.cpp`'s `llama-bench` (tokens/second, greedy decode).
//
// Why: every one of the 16 existing benches is *internal* (Tesseract-vs-
// Tesseract). `idea.md` §6.2/§8.5 calls for alignment against an external
// reference. CPU + llama.cpp is the most fairly-alignable pair (no Hopper /
// vLLM dependency). This binary emits the Tesseract side of that comparison;
// `scripts/bench_vs_llama_cpp.sh` drives the llama.cpp side and tabulates the
// gap. See docs/design/external-benchmark.md.
//
// Output: a machine-readable line per configuration:
//   [bench] tesseract llama_decode_cpu  cfg=<...>  prefill_tok_s=<..>  decode_tok_s=<..>
//
// This bench is informational (always exits 0); it has no hard perf bar — the
// external comparison is about *recording* the gap, not gating CI on a number
// that depends on the machine and on llama.cpp's build flags.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "tesseract/core/Device.hpp"
#include "tesseract/models/Llama.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

tesseract::models::LlamaConfig small_config() {
  // A deliberately small, llama.cpp-comparable architecture so the run is
  // fast and the comparison is about the runtime, not the weights. Mirror a
  // tiny Llama: 4 layers, d_model 256, 8 heads, GQA off.
  tesseract::models::LlamaConfig c;
  c.vocab_size = 4096;
  c.hidden_size = 256;
  c.num_hidden_layers = 4;
  c.num_attention_heads = 8;
  c.num_key_value_heads = 8;
  c.intermediate_size = 688;  // ~2.7x, SwiGLU
  c.max_position_embeddings = 1024;
  c.dtype = tesseract::DType::Float32;
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace tesseract;
  using namespace tesseract::models;

  int64_t prompt_len = 32;
  int64_t new_tokens = 64;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--prompt" && i + 1 < argc) prompt_len = std::atoll(argv[++i]);
    else if (a == "--gen" && i + 1 < argc) new_tokens = std::atoll(argv[++i]);
  }

  const LlamaConfig cfg = small_config();
  LlamaModel model(cfg);  // random-init; throughput is weight-independent.

  std::vector<int32_t> prompt;
  prompt.reserve(prompt_len);
  for (int64_t i = 0; i < prompt_len; ++i) {
    prompt.push_back(static_cast<int32_t>((i * 7 + 1) % cfg.vocab_size));
  }

  // Warm-up generate (prime allocators / caches) — not timed.
  {
    LlamaModel::GenerateConfig warm;
    warm.max_new_tokens = 4;
    (void)model.generate(prompt, warm);
  }

  // Timed full generate (prefill of `prompt_len` + decode of `new_tokens`).
  LlamaModel::GenerateConfig gc;
  gc.max_new_tokens = new_tokens;
  const auto t0 = Clock::now();
  std::vector<int32_t> out = model.generate(prompt, gc);
  const double total_s = seconds_since(t0);

  // Approximate split: prefill cost ≈ one forward over the prompt. We
  // measure it separately so the decode tok/s isn't diluted by prefill.
  const auto tp = Clock::now();
  {
    LlamaModel::GenerateConfig only_prefill;
    only_prefill.max_new_tokens = 1;
    (void)model.generate(prompt, only_prefill);
  }
  const double prefill_s = seconds_since(tp);

  const int64_t generated = static_cast<int64_t>(out.size()) - prompt_len;
  const double decode_s = total_s - prefill_s > 1e-9 ? total_s - prefill_s
                                                     : total_s;
  const double decode_tok_s =
      generated > 0 ? static_cast<double>(generated) / decode_s : 0.0;
  const double prefill_tok_s =
      prefill_s > 1e-9 ? static_cast<double>(prompt_len) / prefill_s : 0.0;

  std::printf(
      "[bench] tesseract llama_decode_cpu  "
      "cfg=L%lld_d%lld_h%lld_v%lld  prompt=%lld  gen=%lld  "
      "prefill_tok_s=%.2f  decode_tok_s=%.2f  total_s=%.4f\n",
      static_cast<long long>(cfg.num_hidden_layers),
      static_cast<long long>(cfg.hidden_size),
      static_cast<long long>(cfg.num_attention_heads),
      static_cast<long long>(cfg.vocab_size),
      static_cast<long long>(prompt_len), static_cast<long long>(generated),
      prefill_tok_s, decode_tok_s, total_s);
  return 0;
}

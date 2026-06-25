// M4 Phase 9 — bench: full-model Llama serving latency (TTFT / TPOT) on CUDA.
//
// This is the Tesseract side of the vLLM head-to-head. It mirrors
// `bench/external/vllm_serving.py` exactly so the two numbers are
// apples-to-apples on the *same* model:
//
//   * TTFT — time-to-first-token: one prefill of a `prompt_len`-token prompt
//            through every layer's KV cache (the latency the user waits before
//            the first output token).
//   * TPOT — time-per-output-token: steady-state per-step decode latency,
//            measured as the mean over `gen_len` single-token `forward_step`s
//            after the prefill. decode_tok_s = 1000 / TPOT_ms.
//
// Token *selection* (argmax / sampling) is excluded on both sides: it is a
// negligible host-or-device reduction in either engine and would otherwise
// add an unfair D2H to the Tesseract path. What we compare is the pure model
// forward latency, which is what the matmul-bound decode step is dominated by.
//
// Usage (load the real TinyLlama-1.1B that vLLM also serves):
//   CUDA_VISIBLE_DEVICES=0 ./bench_cuda_llama_serving \
//       --config   /path/to/tinyllama/config.json \
//       --safetensors /path/to/tinyllama/model.safetensors \
//       --prompt-len 128 --gen-len 128 --reps 10
//
// `--synthetic` builds a random-weight TinyLlama-shape model instead (no
// file needed) — useful for a shape-only latency sanity check.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/cuda/CudaGraph.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/nn/KVCache.hpp"
#include "tesseract/quant/Scheme.hpp"

#include "cuda_bench_util.hpp"

namespace bench = tesseract::bench;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::NoGradGuard;
using tesseract::Tensor;
using tesseract::models::LlamaConfig;
using tesseract::models::LlamaModel;
using Clock = std::chrono::steady_clock;

namespace {

Tensor make_tokens(const std::vector<int64_t>& ids, int64_t batch, int64_t seq,
                   const Device& dev) {
  Tensor host = Tensor::empty({batch, seq}, DType::Int64, tesseract::cpu_device());
  std::memcpy(host.raw_data(), ids.data(), ids.size() * sizeof(int64_t));
  return dev.is_cpu() ? host : host.to(dev);
}

double ms_since(const Clock::time_point& t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  std::string config_path, safetensors_path;
  int64_t prompt_len = 128, gen_len = 128, reps = 10;
  bool synthetic = false;
  bool force_fp32 = false;
  bool force_fp16 = false;
  bool quantize_int8 = false;
  bool use_cudagraph = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--config") config_path = next();
    else if (a == "--safetensors") safetensors_path = next();
    else if (a == "--prompt-len") prompt_len = std::atoll(next().c_str());
    else if (a == "--gen-len") gen_len = std::atoll(next().c_str());
    else if (a == "--reps") reps = std::atoll(next().c_str());
    else if (a == "--synthetic") synthetic = true;
    else if (a == "--fp32") force_fp32 = true;
    else if (a == "--fp16") force_fp16 = true;
    else if (a == "--int8") quantize_int8 = true;
    else if (a == "--cudagraph") use_cudagraph = true;
    else { std::fprintf(stderr, "unknown arg '%s'\n", a.c_str()); return 2; }
  }

  if (bench::visible_cuda_devices() == 0) {
    std::printf("[bench_cuda_llama_serving] no CUDA device — skipping.\n");
    return bench::kExitNoCuda;
  }
  bench::print_banner("bench_cuda_llama_serving");

  // Resolve config: real config.json, else a TinyLlama-1.1B shape.
  LlamaConfig cfg;
  if (!config_path.empty()) {
    cfg = LlamaConfig::from_json_file(config_path);
  } else {
    cfg.vocab_size = 32000;
    cfg.hidden_size = 2048;
    cfg.num_hidden_layers = 22;
    cfg.num_attention_heads = 32;
    cfg.num_key_value_heads = 4;
    cfg.intermediate_size = 5632;
    cfg.max_position_embeddings = 2048;
    cfg.rope_theta = 10000.0;
    cfg.dtype = DType::Float32;
  }
  if (force_fp32) cfg.dtype = DType::Float32;
  if (force_fp16) cfg.dtype = DType::Float16;

  const Device cuda0{DeviceType::CUDA, 0};
  NoGradGuard no_grad;

  std::shared_ptr<LlamaModel> model;
  if (synthetic || safetensors_path.empty()) {
    model = std::make_shared<LlamaModel>(cfg);
    std::printf("  weights  : synthetic (random init)\n");
  } else {
    model = LlamaModel::from_pretrained(safetensors_path, cfg);
    std::printf("  weights  : %s\n", safetensors_path.c_str());
  }
  if (quantize_int8) {
    model->quantize_(tesseract::quant::Scheme::int8_symmetric());
    std::printf("  quantize : INT8 symmetric (all Linear projections)\n");
  }
  model->to(cuda0);
  model->eval();

  std::printf("  config   : vocab=%ld hidden=%ld layers=%ld H=%ld KV=%ld d_ff=%ld dtype=%s\n",
              long(cfg.vocab_size), long(cfg.hidden_size),
              long(cfg.num_hidden_layers), long(cfg.num_attention_heads),
              long(cfg.kv_heads()), long(cfg.intermediate_size),
              cfg.dtype == DType::Float16 ? "fp16" : "fp32");
  std::printf("  workload : prompt_len=%ld gen_len=%ld reps=%ld\n\n",
              long(prompt_len), long(gen_len), long(reps));

  const int64_t max_len = prompt_len + gen_len + 8;

  std::vector<int64_t> prompt_ids(static_cast<std::size_t>(prompt_len));
  for (int64_t i = 0; i < prompt_len; ++i)
    prompt_ids[static_cast<std::size_t>(i)] = (i * 131 + 7) % cfg.vocab_size;
  const std::vector<int64_t> step_id{42 % cfg.vocab_size};

  // Warmup (build kernels / allocator / cuBLASLt descriptors).
  {
    auto caches = model->make_kv_caches(1, max_len);
    Tensor p = make_tokens(prompt_ids, 1, prompt_len, cuda0);
    (void)model->forward_step(p, caches);
    Tensor s = make_tokens(step_id, 1, 1, cuda0);
    for (int i = 0; i < 3; ++i) (void)model->forward_step(s, caches);
    cudaDeviceSynchronize();
  }

  // ---- TTFT: prefill latency, averaged over `reps` fresh caches ----
  //
  // Two paths mirror the decode measurement below:
  //   * eager     — one fresh-cache `forward_step([1, prompt_len])` per rep.
  //   * cudagraph — capture the whole prefill step once, then replay it.
  //                 Prefill is a fixed [1, prompt_len] shape here, so it
  //                 graph-captures the same way vLLM captures its prefill
  //                 buckets; replay collapses the ~hundreds of per-layer
  //                 kernel launches into one graph launch.
  double ttft_sum = 0.0, ttft_min = 1e30, ttft_max = 0.0;
  if (use_cudagraph) {
    bench::BenchStream bench_stream;
    const tesseract::Stream& stream = bench_stream.stream();
    auto caches = model->make_kv_caches(1, max_len);
    Tensor p = make_tokens(prompt_ids, 1, prompt_len, cuda0);
    auto reset = [&]() { for (auto& c : caches) c->set_current_len(0); };

    tesseract::cuda::CudaGraph graph(0);
    reset();
    graph.capture(stream, [&]() {
      NoGradGuard nogg;
      reset();
      (void)model->forward_step(p, caches);
    });
    for (int64_t r = 0; r < reps; ++r) {
      reset();
      cudaDeviceSynchronize();
      auto t0 = Clock::now();
      graph.launch(stream);
      cudaDeviceSynchronize();
      double dt = ms_since(t0);
      ttft_sum += dt;
      ttft_min = std::min(ttft_min, dt);
      ttft_max = std::max(ttft_max, dt);
    }
  } else {
    for (int64_t r = 0; r < reps; ++r) {
      auto caches = model->make_kv_caches(1, max_len);
      Tensor p = make_tokens(prompt_ids, 1, prompt_len, cuda0);
      cudaDeviceSynchronize();
      auto t0 = Clock::now();
      (void)model->forward_step(p, caches);
      cudaDeviceSynchronize();
      double dt = ms_since(t0);
      ttft_sum += dt;
      ttft_min = std::min(ttft_min, dt);
      ttft_max = std::max(ttft_max, dt);
    }
  }
  const double ttft_mean = ttft_sum / static_cast<double>(reps);

  // ---- TPOT: steady-state per-token decode, averaged over `reps` ----
  // Two paths:
  //   * eager      — one `forward_step([1,1])` per token (every op a launch).
  //   * cudagraph  — capture the whole decode step into one `cudaGraphExec_t`
  //                  and replay it, collapsing the ~hundreds of per-layer
  //                  kernel launches into a single graph launch. This is the
  //                  same technique vLLM uses, and it removes the host launch
  //                  overhead that dominates small-model decode.
  double tpot_sum = 0.0, tpot_min = 1e30, tpot_max = 0.0;
  if (use_cudagraph) {
    bench::BenchStream bench_stream;  // StreamGuard: all ops queue on this stream
    const tesseract::Stream& stream = bench_stream.stream();
    for (int64_t r = 0; r < reps; ++r) {
      auto caches = model->make_kv_caches(1, max_len);
      Tensor p = make_tokens(prompt_ids, 1, prompt_len, cuda0);
      (void)model->forward_step(p, caches);  // prefill (not timed)
      const int64_t pos = prompt_len;
      Tensor s = make_tokens(step_id, 1, 1, cuda0);
      auto reset = [&]() { for (auto& c : caches) c->set_current_len(pos); };

      tesseract::cuda::CudaGraph graph(0);
      reset();
      graph.capture(stream, [&]() {
        NoGradGuard nogg;
        reset();
        (void)model->forward_step(s, caches);
      });

      cudaDeviceSynchronize();
      auto t0 = Clock::now();
      for (int64_t i = 0; i < gen_len; ++i) graph.launch(stream);
      cudaDeviceSynchronize();
      double per_tok = ms_since(t0) / static_cast<double>(gen_len);
      tpot_sum += per_tok;
      tpot_min = std::min(tpot_min, per_tok);
      tpot_max = std::max(tpot_max, per_tok);
    }
  } else {
    for (int64_t r = 0; r < reps; ++r) {
      auto caches = model->make_kv_caches(1, max_len);
      Tensor p = make_tokens(prompt_ids, 1, prompt_len, cuda0);
      (void)model->forward_step(p, caches);  // prefill (not timed)
      Tensor s = make_tokens(step_id, 1, 1, cuda0);
      cudaDeviceSynchronize();
      auto t0 = Clock::now();
      for (int64_t i = 0; i < gen_len; ++i)
        (void)model->forward_step(s, caches);
      cudaDeviceSynchronize();
      double per_tok = ms_since(t0) / static_cast<double>(gen_len);
      tpot_sum += per_tok;
      tpot_min = std::min(tpot_min, per_tok);
      tpot_max = std::max(tpot_max, per_tok);
    }
  }
  const double tpot_mean = tpot_sum / static_cast<double>(reps);
  const double decode_tok_s = 1000.0 / tpot_mean;

  std::printf("  metric            mean       min       max\n");
  std::printf("  ----------------------------------------------\n");
  std::printf("  TTFT (ms)   %9.3f %9.3f %9.3f\n", ttft_mean, ttft_min, ttft_max);
  std::printf("  TPOT (ms)   %9.3f %9.3f %9.3f\n", tpot_mean, tpot_min, tpot_max);
  std::printf("  decode tok/s %8.1f\n\n", decode_tok_s);

  std::printf("  JSON {\"engine\":\"tesseract\",\"ttft_ms\":%.3f,\"tpot_ms\":%.4f,"
              "\"decode_tok_s\":%.2f,\"prompt_len\":%ld,\"gen_len\":%ld}\n",
              ttft_mean, tpot_mean, decode_tok_s, long(prompt_len), long(gen_len));
  return bench::kExitOk;
}

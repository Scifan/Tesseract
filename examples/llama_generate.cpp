// Wave 16 (B-033) — real Hugging Face checkpoint end-to-end generation.
//
// The smallest "actually load a real model and talk to it" binary. Unlike
// `llama_infer` (which needs the architecture passed on the command line
// and folds prompt ids `% vocab_size` so it can also run a tiny synthetic
// model), this demo is built for a genuine HF checkpoint directory:
//
//   * reads the model architecture straight from `config.json`
//     (`LlamaConfig::from_json_file`) — no hand-entered --hidden/--layers;
//   * loads the weights from the directory's `*.safetensors`;
//   * loads `tokenizer.json` (byte-level BPE) and encodes the prompt with
//     the REAL vocab (no `% vocab_size` corruption);
//   * streams the decoded continuation token-by-token to stdout.
//
// Usage:
//   tesseract_llama_generate --model-dir /path/to/hf/checkpoint
//       --prompt "The capital of France is" --max-new-tokens 64
//   tesseract_llama_generate --config config.json --safetensors model.safetensors
//       --tokenizer tokenizer.json --prompt "..." [--device cuda] [--kv-int8]
//       [--sample --temperature 0.8 --top-k 40 --top-p 0.95 --seed 7]

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/io/BpeTokenizer.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/models/Sampler.hpp"
#include "tesseract/models/StructuredDecoding.hpp"
#include "tesseract/utils/Logging.hpp"

#include <memory>
#include <optional>

namespace fs = std::filesystem;

namespace {

// Greedy argmax over the vocab axis of the last position of a [1, S, V]
// logits tensor — mirrors LlamaModel::generate / the scheduler exactly.
int32_t argmax_last(const tesseract::Tensor& logits) {
  using namespace tesseract;
  const Tensor host = logits.device().is_cpu() ? logits : logits.to(cpu_device());
  const int64_t S = host.shape()[1];
  const int64_t V = host.shape()[2];
  int32_t best = 0;
  dispatch_float_with_half(host.dtype(), [&]<typename T>() {
    const T* p = host.data_ptr<T>() + (S - 1) * V;
    double best_val = -std::numeric_limits<double>::infinity();
    for (int64_t v = 0; v < V; ++v) {
      const double val = static_cast<double>(p[v]);
      if (val > best_val) { best_val = val; best = static_cast<int32_t>(v); }
    }
  });
  return best;
}

// Host FP32 copy of the last logits row — feeds the Sampler.
std::vector<float> last_row(const tesseract::Tensor& logits) {
  using namespace tesseract;
  const Tensor host = logits.device().is_cpu() ? logits : logits.to(cpu_device());
  const int64_t S = host.shape()[1];
  const int64_t V = host.shape()[2];
  std::vector<float> row(static_cast<std::size_t>(V));
  dispatch_float_with_half(host.dtype(), [&]<typename T>() {
    const T* p = host.data_ptr<T>() + (S - 1) * V;
    for (int64_t v = 0; v < V; ++v)
      row[static_cast<std::size_t>(v)] = static_cast<float>(p[v]);
  });
  return row;
}

// Pick the single existing *.safetensors in `dir` (prefers `model.safetensors`).
std::string find_safetensors(const std::string& dir) {
  const fs::path preferred = fs::path(dir) / "model.safetensors";
  if (fs::exists(preferred)) return preferred.string();
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.is_regular_file() && e.path().extension() == ".safetensors")
      return e.path().string();
  }
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  using namespace tesseract;
  using tesseract::models::LlamaConfig;
  using tesseract::models::LlamaModel;

  std::string model_dir, config_path, safetensors_path, tokenizer_path, prompt;
  std::string grammar_regex;
  std::string device_cli = "cpu";
  int64_t max_new_tokens = 64;
  bool do_sample = false, kv_int8 = false, stream = true;
  double temperature = 1.0, top_p = 1.0, repetition_penalty = 1.0;
  int64_t top_k = 0;
  uint64_t seed = 0;
  int32_t eos_override = -2;  // -2 ⇒ use config/tokenizer default

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* flag) {
      if (i + 1 >= argc) { std::cerr << "error: " << flag << " needs an argument\n"; std::exit(1); }
      return std::string(argv[++i]);
    };
    if (a == "--model-dir")          model_dir = need("--model-dir");
    else if (a == "--config")        config_path = need("--config");
    else if (a == "--safetensors")   safetensors_path = need("--safetensors");
    else if (a == "--tokenizer")     tokenizer_path = need("--tokenizer");
    else if (a == "--prompt")        prompt = need("--prompt");
    else if (a == "--device")        device_cli = need("--device");
    else if (a == "--max-new-tokens") max_new_tokens = std::atoll(need("--max-new-tokens").c_str());
    else if (a == "--sample")        do_sample = true;
    else if (a == "--temperature")   temperature = std::atof(need("--temperature").c_str());
    else if (a == "--top-k")         top_k = std::atoll(need("--top-k").c_str());
    else if (a == "--top-p")         top_p = std::atof(need("--top-p").c_str());
    else if (a == "--repetition-penalty") repetition_penalty = std::atof(need("--repetition-penalty").c_str());
    else if (a == "--seed")          seed = static_cast<uint64_t>(std::atoll(need("--seed").c_str()));
    else if (a == "--eos")           eos_override = static_cast<int32_t>(std::atoll(need("--eos").c_str()));
    else if (a == "--kv-int8")       kv_int8 = true;
    else if (a == "--grammar-regex") grammar_regex = need("--grammar-regex");
    else if (a == "--no-stream")     stream = false;
    else if (a == "--help" || a == "-h") {
      std::cout <<
        "Usage: " << argv[0] << " --model-dir DIR --prompt TEXT [options]\n"
        "   or: " << argv[0] << " --config c.json --safetensors m.safetensors\n"
        "                       --tokenizer t.json --prompt TEXT [options]\n"
        "Options: --device cpu|cuda  --max-new-tokens N  --kv-int8  --no-stream\n"
        "         --sample --temperature F --top-k N --top-p F\n"
        "         --repetition-penalty F --seed N  --eos ID\n"
        "         --grammar-regex PATTERN  (constrain output to a regex)\n";
      return 0;
    } else { std::cerr << "error: unknown arg '" << a << "'\n"; return 1; }
  }

  // Resolve the three artifact paths from --model-dir when given.
  if (!model_dir.empty()) {
    if (config_path.empty())      config_path = (fs::path(model_dir) / "config.json").string();
    if (tokenizer_path.empty())   tokenizer_path = (fs::path(model_dir) / "tokenizer.json").string();
    if (safetensors_path.empty()) safetensors_path = find_safetensors(model_dir);
  }
  if (config_path.empty() || safetensors_path.empty() || tokenizer_path.empty() ||
      prompt.empty()) {
    std::cerr << "error: need a config.json, *.safetensors, tokenizer.json, and "
                 "--prompt (pass --model-dir or the three paths). See --help.\n";
    return 1;
  }

  Device dev = cpu_device();
  if (device_cli == "cuda") dev = Device{DeviceType::CUDA, 0};
  else if (device_cli != "cpu") { std::cerr << "error: unknown --device\n"; return 1; }

  NoGradGuard no_grad;

  std::cout << "[llama_generate] config   = " << config_path << "\n";
  LlamaConfig cfg = LlamaConfig::from_json_file(config_path);
  std::cout << "[llama_generate] arch     = vocab " << cfg.vocab_size
            << ", hidden " << cfg.hidden_size << ", layers " << cfg.num_hidden_layers
            << ", heads " << cfg.num_attention_heads << "/" << cfg.kv_heads()
            << " (Q/KV), ff " << cfg.intermediate_size
            << ", dtype " << static_cast<int>(cfg.dtype) << "\n";

  std::cout << "[llama_generate] weights  = " << safetensors_path << "\n";
  auto model = LlamaModel::from_pretrained(safetensors_path, cfg);
  model->to(dev);
  std::cout << "[llama_generate] loaded " << model->named_parameters().size()
            << " params on " << dev.to_string() << "\n";

  std::cout << "[llama_generate] tokenizer= " << tokenizer_path << "\n";
  auto tk = io::BpeTokenizer::from_file(tokenizer_path);
  auto ids = tk.encode(prompt, /*add_special=*/true);  // REAL ids, no mod
  TESSERACT_CHECK(!ids.empty(), "llama_generate: prompt encoded to 0 tokens");
  for (int32_t id : ids)
    TESSERACT_CHECK(id >= 0 && id < cfg.vocab_size,
                    "llama_generate: token id {} outside model vocab [0,{}) — "
                    "tokenizer/checkpoint mismatch?", id, cfg.vocab_size);

  int32_t eos = eos_override != -2 ? eos_override
                : (cfg.eos_token_id >= 0 ? cfg.eos_token_id : tk.eos_token_id());
  std::cout << "[llama_generate] prompt   = \"" << prompt << "\" ("
            << ids.size() << " tokens), eos=" << eos
            << (kv_int8 ? ", kv=int8" : ", kv=fp")
            << (do_sample ? ", sampling" : ", greedy") << "\n";
  std::cout << "[llama_generate] ---- generation ----\n" << prompt << std::flush;

  // KV caches (FP or INT8) for one sequence over the whole sequence length.
  const int64_t max_len = static_cast<int64_t>(ids.size()) + max_new_tokens;
  std::vector<std::shared_ptr<nn::KVCacheBase>> caches;
  if (kv_int8) {
    caches = model->make_quantized_kv_caches(/*batch=*/1, max_len);
  } else {
    auto fp = model->make_kv_caches(/*batch=*/1, max_len);
    caches.assign(fp.begin(), fp.end());
  }

  auto make_tokens = [&](const int32_t* p, int64_t n) {
    Tensor t = Tensor::empty({1, n}, DType::Int64, cpu_device());
    int64_t* d = t.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(p[i]);
    return dev.is_cpu() ? t : t.to(dev);
  };

  models::Sampler sampler(
      models::SamplingParams{temperature, static_cast<int>(top_k), top_p,
                             repetition_penalty},
      seed);

  // Optional structured generation (Wave 17): compile the regex into a
  // byte automaton, build a vocab-aware constraint, and mask logits every
  // step so the output is grammar-valid by construction.
  std::unique_ptr<models::RegexAutomaton> automaton;
  std::optional<models::GrammarConstraint> constraint;
  if (!grammar_regex.empty()) {
    automaton = std::make_unique<models::RegexAutomaton>(
        models::RegexAutomaton::compile(grammar_regex));
    constraint.emplace(
        models::GrammarConstraint::from_tokenizer(*automaton, tk));
    constraint->reset();
    std::cout << "[llama_generate] grammar  = /" << grammar_regex << "/\n";
  }

  // Prefill the prompt in one step, then decode one token at a time.
  Tensor logits = model->forward_step(make_tokens(ids.data(), static_cast<int64_t>(ids.size())),
                                      caches);
  std::vector<int32_t> generated;
  std::string printed;  // decoded text already shown (for stream deltas)
  const auto prompt_count = ids.size();

  for (int64_t step = 0; step < max_new_tokens; ++step) {
    int32_t next;
    if (constraint) {
      // Mask first, then pick over the survivors (greedy or sampled).
      auto row = last_row(logits);
      constraint->apply(std::span<float>(row.data(), row.size()));
      if (do_sample) {
        next = sampler.sample(std::span<const float>(row.data(), row.size()),
                              std::span<const int32_t>(ids.data(), ids.size()));
      } else {
        int32_t best = 0;
        float best_val = -std::numeric_limits<float>::infinity();
        for (std::size_t v = 0; v < row.size(); ++v)
          if (row[v] > best_val) { best_val = row[v]; best = static_cast<int32_t>(v); }
        next = best;
      }
      constraint->accept(next);
    } else if (do_sample) {
      const auto row = last_row(logits);
      next = sampler.sample(std::span<const float>(row.data(), row.size()),
                            std::span<const int32_t>(ids.data(), ids.size()));
    } else {
      next = argmax_last(logits);
    }
    ids.push_back(next);
    generated.push_back(next);
    if (eos >= 0 && next == eos) break;

    if (stream) {
      // Decode the full generated suffix and print only the new bytes
      // (byte-level BPE can split a UTF-8 char across tokens, so we
      // re-decode the suffix each step rather than per-token).
      const std::string text =
          tk.decode(std::span<const int32_t>(generated.data(), generated.size()),
                    /*skip_special=*/true);
      if (text.size() > printed.size()) {
        std::cout << text.substr(printed.size()) << std::flush;
        printed = text;
      }
    }

    if (step + 1 < max_new_tokens) {
      logits = model->forward_step(make_tokens(&next, 1), caches);
    }
  }

  if (!stream) {
    std::cout << tk.decode(std::span<const int32_t>(generated.data(), generated.size()),
                           /*skip_special=*/true);
  }
  std::cout << "\n[llama_generate] ---- done ("
            << (ids.size() - prompt_count) << " new tokens) ----\n";
  return 0;
}

// Full-stack Llama forward demo — builds a complete `LlamaModel`, loads
// weights from a safetensors file (or, when `--synthetic` is passed,
// exercises the randomly-initialized model end-to-end), and prints the
// top-k logits for a random token sequence.
//
// The emphasis here is the *assembly*:
//
//   tokens  →  embed_tokens  →  N × TransformerBlock (RoPE + RMSNorm + SwiGLU)
//                                     ↓
//                                final RMSNorm
//                                     ↓
//                                  lm_head
//                                     ↓
//                                  logits
//
// Companion test (`tests/models/test_llama_parity.cpp`) pins byte-exact
// loader behavior; this binary is the smallest runnable thing that
// exercises the full stack on either CPU or CUDA.
//
// Usage:
//   ./examples/tesseract_llama_infer --synthetic
//   ./examples/tesseract_llama_infer --safetensors path/to/model.safetensors
//       --vocab 32000 --hidden 4096 --layers 32 --heads 32 --ff 11008

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/io/BpeTokenizer.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/utils/Logging.hpp"

namespace {

int64_t count_finite(const tesseract::Tensor& host) {
  const float* p = host.data_ptr<float>();
  int64_t ok = 0;
  for (int64_t i = 0; i < host.numel(); ++i) {
    if (std::isfinite(p[i])) ++ok;
  }
  return ok;
}

// Find the top-k indices of the last row of a [B, S, V] logits tensor
// (i.e. the predictions for the *final* position of the first batch).
std::vector<std::pair<float, int64_t>> topk_last(const tesseract::Tensor& logits,
                                                 int64_t k) {
  const int64_t S = logits.shape()[1];
  const int64_t V = logits.shape()[2];
  const float* p = logits.data_ptr<float>() + (S - 1) * V;  // b=0, s=S-1
  std::vector<std::pair<float, int64_t>> all;
  all.reserve(static_cast<std::size_t>(V));
  for (int64_t i = 0; i < V; ++i) {
    all.emplace_back(p[i], i);
  }
  std::partial_sort(all.begin(), all.begin() + k, all.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
  all.resize(static_cast<std::size_t>(k));
  return all;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace tesseract;
  using tesseract::models::LlamaConfig;
  using tesseract::models::LlamaModel;

  std::string device_cli = "cpu";
  std::string safetensors_path;
  std::string tokenizer_path;
  std::string prompt;
  bool synthetic = false;
  bool do_generate = false;
  int64_t max_new_tokens = 32;
  bool do_sample = false;
  double temperature = 1.0;
  int64_t top_k = 0;
  double top_p = 1.0;
  double repetition_penalty = 1.0;
  uint64_t seed = 0;
  int64_t batch = 1;
  int64_t seq   = 8;
  int64_t topk  = 5;

  // Default tiny shape so `--synthetic` runs in well under a second.
  LlamaConfig cfg;
  cfg.vocab_size              = 64;
  cfg.hidden_size             = 32;
  cfg.num_hidden_layers       = 2;
  cfg.num_attention_heads     = 4;
  cfg.intermediate_size       = 64;
  cfg.max_position_embeddings = 128;
  cfg.rope_theta              = 10000.0;
  cfg.rms_norm_eps            = 1e-5;
  cfg.tie_word_embeddings     = true;
  cfg.dtype                   = DType::Float32;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* flag) {
      if (i + 1 >= argc) {
        std::cerr << "error: " << flag << " requires an argument\n";
        std::exit(1);
      }
      return std::string(argv[++i]);
    };
    if (a == "--device")            device_cli       = need("--device");
    else if (a == "--synthetic")    synthetic        = true;
    else if (a == "--safetensors")  safetensors_path = need("--safetensors");
    else if (a == "--tokenizer")    tokenizer_path   = need("--tokenizer");
    else if (a == "--prompt")       prompt           = need("--prompt");
    else if (a == "--generate")     do_generate      = true;
    else if (a == "--max-new-tokens") max_new_tokens = std::atoll(need("--max-new-tokens").c_str());
    else if (a == "--sample")       do_sample        = true;
    else if (a == "--temperature")  temperature      = std::atof(need("--temperature").c_str());
    else if (a == "--top-k")        top_k            = std::atoll(need("--top-k").c_str());
    else if (a == "--top-p")        top_p            = std::atof(need("--top-p").c_str());
    else if (a == "--repetition-penalty") repetition_penalty = std::atof(need("--repetition-penalty").c_str());
    else if (a == "--seed")         seed             = static_cast<uint64_t>(std::atoll(need("--seed").c_str()));
    else if (a == "--vocab")        cfg.vocab_size        = std::atoll(need("--vocab").c_str());
    else if (a == "--hidden")       cfg.hidden_size       = std::atoll(need("--hidden").c_str());
    else if (a == "--layers")       cfg.num_hidden_layers = std::atoll(need("--layers").c_str());
    else if (a == "--heads")        cfg.num_attention_heads = std::atoll(need("--heads").c_str());
    else if (a == "--kv-heads")     cfg.num_key_value_heads = std::atoll(need("--kv-heads").c_str());
    else if (a == "--ff")           cfg.intermediate_size = std::atoll(need("--ff").c_str());
    else if (a == "--max-seq")      cfg.max_position_embeddings = std::atoll(need("--max-seq").c_str());
    else if (a == "--rope-theta")   cfg.rope_theta        = std::atof(need("--rope-theta").c_str());
    else if (a == "--tie")          cfg.tie_word_embeddings = (need("--tie") != "false");
    else if (a == "--batch")        batch                 = std::atoll(need("--batch").c_str());
    else if (a == "--seq")          seq                   = std::atoll(need("--seq").c_str());
    else if (a == "--topk")         topk                  = std::atoll(need("--topk").c_str());
    else if (a == "--help" || a == "-h") {
      std::cout <<
        "Usage: " << argv[0] << " [--device cpu|cuda] [--synthetic | --safetensors path]\n"
        "                      [--tokenizer tokenizer.json --prompt \"text\"]\n"
        "                      [--generate [--max-new-tokens N]]\n"
        "                      [--sample [--temperature F] [--top-k N] [--top-p F]\n"
        "                                [--repetition-penalty F] [--seed N]]\n"
        "                      [--vocab N] [--hidden N] [--layers N] [--heads N] [--kv-heads N] [--ff N]\n"
        "                      [--max-seq N] [--rope-theta F] [--tie true|false]\n"
        "                      [--batch N] [--seq N] [--topk N]\n";
      return 0;
    } else {
      std::cerr << "error: unknown arg '" << a << "' (see --help)\n";
      return 1;
    }
  }

  if (!synthetic && safetensors_path.empty()) {
    std::cerr << "error: pass either --synthetic or --safetensors <path>\n";
    return 1;
  }

  Device run_device = cpu_device();
  if (device_cli == "cuda") {
    run_device = Device{DeviceType::CUDA, 0};
  } else if (device_cli != "cpu") {
    std::cerr << "error: unknown --device '" << device_cli << "'\n";
    return 1;
  }

  std::cout << "[llama_infer] config:\n"
            << "  device              = " << run_device.to_string() << "\n"
            << "  vocab_size          = " << cfg.vocab_size << "\n"
            << "  hidden_size         = " << cfg.hidden_size << "\n"
            << "  num_hidden_layers   = " << cfg.num_hidden_layers << "\n"
            << "  num_attention_heads = " << cfg.num_attention_heads << "\n"
            << "  num_key_value_heads = " << cfg.kv_heads() << "\n"
            << "  intermediate_size   = " << cfg.intermediate_size << "\n"
            << "  rope_theta          = " << cfg.rope_theta << "\n"
            << "  tie_embeddings      = " << (cfg.tie_word_embeddings ? "true" : "false") << "\n"
            << "  batch x seq         = " << batch << " x " << seq << "\n";

  NoGradGuard no_grad;
  std::shared_ptr<LlamaModel> model;
  if (synthetic) {
    model = std::make_shared<LlamaModel>(cfg);
    std::cout << "[llama_infer] using synthetic (random-init) weights\n";
  } else {
    std::cout << "[llama_infer] loading " << safetensors_path << "\n";
    model = LlamaModel::from_pretrained(safetensors_path, cfg);
    std::cout << "[llama_infer] loaded "
              << model->named_parameters().size() << " parameters\n";
  }
  model->to(run_device);

  // Autoregressive generation path (Wave 5 / B-027). Encode the prompt with
  // the byte-level BPE tokenizer, greedily decode `--max-new-tokens` steps
  // through the per-layer KV cache, and print the decoded continuation.
  if (do_generate) {
    if (tokenizer_path.empty() || prompt.empty()) {
      std::cerr << "error: --generate requires --tokenizer and --prompt\n";
      return 1;
    }
    auto tk = tesseract::io::BpeTokenizer::from_file(tokenizer_path);
    auto prompt_ids = tk.encode(prompt, /*add_special=*/true);
    // Map ids into the model's vocab so the demo also runs against the tiny
    // synthetic model (whose vocab is far smaller than a real checkpoint's).
    for (auto& id : prompt_ids) {
      id = static_cast<int32_t>(static_cast<int64_t>(id) % cfg.vocab_size);
    }
    std::cout << "[llama_infer] prompt = \"" << prompt << "\"\n"
              << "[llama_infer] prompt tokens = " << prompt_ids.size()
              << ", max_new_tokens = " << max_new_tokens << "\n";

    LlamaModel::GenerateConfig gc;
    gc.max_new_tokens = max_new_tokens;
    gc.eos_token_id = tk.eos_token_id() >= 0
                          ? static_cast<int32_t>(tk.eos_token_id() % cfg.vocab_size)
                          : -1;
    gc.do_sample = do_sample;
    gc.seed = seed;
    gc.sampling.temperature = temperature;
    gc.sampling.top_k = static_cast<int>(top_k);
    gc.sampling.top_p = top_p;
    gc.sampling.repetition_penalty = repetition_penalty;
    std::cout << "[llama_infer] decoding = "
              << (do_sample ? "sampling" : "greedy");
    if (do_sample) {
      std::cout << " (T=" << temperature << ", top_k=" << top_k
                << ", top_p=" << top_p << ", rep_pen=" << repetition_penalty
                << ", seed=" << seed << ")";
    }
    std::cout << "\n";
    const auto out_ids = model->generate(prompt_ids, gc);

    std::cout << "[llama_infer] generated " << (out_ids.size() - prompt_ids.size())
              << " new tokens (full sequence = " << out_ids.size() << "):\n  ";
    for (int32_t id : out_ids) std::cout << id << " ";
    std::cout << "\n[llama_infer] decoded text = \""
              << tk.decode(std::span<const int32_t>(out_ids.data(), out_ids.size()),
                           /*skip_special=*/true)
              << "\"\n[llama_infer] done.\n";
    return 0;
  }

  // Build the token ids. If a `--tokenizer` + `--prompt` were given, encode
  // the prompt with the byte-level BPE tokenizer (B-018) for a genuine
  // end-to-end run; otherwise fall back to random ids. The encoded ids are
  // taken mod vocab_size so the demo also runs against the tiny synthetic
  // model (whose vocab is far smaller than a real checkpoint's).
  std::unique_ptr<tesseract::io::BpeTokenizer> tokenizer;
  std::vector<int64_t> tok;
  if (!tokenizer_path.empty()) {
    tokenizer = std::make_unique<tesseract::io::BpeTokenizer>(
        tesseract::io::BpeTokenizer::from_file(tokenizer_path));
    std::cout << "[llama_infer] tokenizer vocab_size = "
              << tokenizer->vocab_size() << "\n";
    const auto ids = tokenizer->encode(prompt, /*add_special=*/true);
    std::cout << "[llama_infer] prompt = \"" << prompt << "\"\n"
              << "[llama_infer] encoded " << ids.size() << " tokens: ";
    for (int32_t id : ids) std::cout << id << " ";
    std::cout << "\n[llama_infer] decoded = \""
              << tokenizer->decode(
                     std::span<const int32_t>(ids.data(), ids.size()), true)
              << "\"\n";
    batch = 1;
    seq = static_cast<int64_t>(ids.size());
    TESSERACT_CHECK(seq > 0, "llama_infer: empty prompt produced no tokens");
    tok.reserve(ids.size());
    for (int32_t id : ids) {
      tok.push_back(static_cast<int64_t>(static_cast<int64_t>(id) % cfg.vocab_size));
    }
  } else {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> vd(0, cfg.vocab_size - 1);
    tok.resize(static_cast<std::size_t>(batch * seq));
    for (auto& t : tok) t = vd(rng);
  }
  Tensor tokens_host = Tensor::empty({batch, seq}, DType::Int64, cpu_device());
  std::memcpy(tokens_host.raw_data(), tok.data(), tok.size() * sizeof(int64_t));
  Tensor tokens = run_device.is_cpu() ? tokens_host : tokens_host.to(run_device);

  Tensor logits = model->forward(tokens);
  std::cout << "[llama_infer] logits.shape = " << logits.shape().to_string() << "\n";

  // Move results back to host for sanity + top-k.
  const Tensor logits_host = run_device.is_cpu() ? logits : logits.to(cpu_device());
  const int64_t finite = count_finite(logits_host);
  std::cout << "[llama_infer] finite = " << finite << " / " << logits_host.numel() << "\n";

  auto top = topk_last(logits_host, std::min<int64_t>(topk, cfg.vocab_size));
  std::cout << "[llama_infer] top-" << top.size() << " for b=0, last position:\n";
  for (const auto& [score, idx] : top) {
    std::cout << "  id=" << idx << "  logit=" << score << "\n";
  }
  std::cout << "[llama_infer] done.\n";
  return 0;
}

// Continuous-batching serving demo — Wave 7 (B-029).
//
// Submits several prompts to the `ContinuousBatchingScheduler` at once
// and drives the engine to completion over a single shared paged KV
// pool. Demonstrates the three things continuous batching adds over the
// one-shot `llama_infer --generate` loop:
//
//   * dynamic admission up to a batch cap (requests beyond the cap wait
//     and are admitted as running ones finish);
//   * a SHARED block pool whose residency tracks live tokens across all
//     requests, with blocks recycled the instant a request finishes;
//   * per-request sampling configs (each prompt can have its own
//     temperature / seed) decoded concurrently.
//
// Synthetic model by default so it runs without a checkpoint; pass
// `--tokenizer tokenizer.json` to encode/decode real text.
//
// Usage:
//   ./examples/tesseract_llama_serve --synthetic [--device cpu|cuda]
//       [--tokenizer t.json] [--max-new-tokens N] [--max-batch B]
//       [--sample [--temperature F] [--seed N]]
//       [--prompts "alpha|beta|gamma"]

#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "tesseract/core/Device.hpp"
#include "tesseract/io/BpeTokenizer.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/models/Scheduler.hpp"
#include "tesseract/utils/Logging.hpp"

using namespace tesseract;

namespace {

std::vector<std::string> split_pipe(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == '|') { out.push_back(cur); cur.clear(); }
    else cur.push_back(c);
  }
  out.push_back(cur);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string device_str = "cpu";
  std::string tokenizer_path;
  std::string prompts_str = "the quick brown fox|hello world|machine learning|a b c";
  int64_t max_new_tokens = 12;
  int64_t max_batch = 2;
  bool do_sample = false;
  double temperature = 0.8;
  uint64_t seed = 0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      TESSERACT_CHECK(i + 1 < argc, "missing value for {}", a);
      return argv[++i];
    };
    if (a == "--device") device_str = val();
    else if (a == "--tokenizer") tokenizer_path = val();
    else if (a == "--prompts") prompts_str = val();
    else if (a == "--max-new-tokens") max_new_tokens = std::stoll(val());
    else if (a == "--max-batch") max_batch = std::stoll(val());
    else if (a == "--sample") do_sample = true;
    else if (a == "--temperature") temperature = std::stod(val());
    else if (a == "--seed") seed = static_cast<uint64_t>(std::stoll(val()));
    else if (a == "--synthetic") { /* default */ }
    else { std::cerr << "unknown flag: " << a << "\n"; return 2; }
  }

  const Device dev = device_str == "cuda" ? Device{DeviceType::CUDA, 0} : cpu_device();

  models::LlamaConfig cfg;
  cfg.vocab_size = 256;
  cfg.hidden_size = 64;
  cfg.num_hidden_layers = 4;
  cfg.num_attention_heads = 8;
  cfg.intermediate_size = 128;
  cfg.max_position_embeddings = 512;
  cfg.tie_word_embeddings = false;
  auto model = std::make_shared<models::LlamaModel>(cfg);
  model->to(dev);

  std::unique_ptr<io::BpeTokenizer> tok;
  if (!tokenizer_path.empty()) {
    tok = std::make_unique<io::BpeTokenizer>(io::BpeTokenizer::from_file(tokenizer_path));
  }

  const std::vector<std::string> texts = split_pipe(prompts_str);

  // Encode each prompt → token ids (clamp into the synthetic vocab).
  auto encode = [&](const std::string& text) -> std::vector<int32_t> {
    std::vector<int32_t> ids;
    if (tok) {
      for (int32_t id : tok->encode(text, /*add_special_tokens=*/false))
        ids.push_back(static_cast<int32_t>(id % cfg.vocab_size));
    } else {
      for (unsigned char c : text) ids.push_back(static_cast<int32_t>(c % cfg.vocab_size));
    }
    if (ids.empty()) ids.push_back(1);
    return ids;
  };

  models::EngineConfig ec;
  ec.block_size = 16;
  ec.num_blocks = 512;
  ec.max_seq_len = 256;
  ec.max_batch_size = max_batch;
  models::ContinuousBatchingScheduler sched(model, ec);

  std::cout << "[llama_serve] device=" << device_str
            << " requests=" << texts.size()
            << " max_batch=" << max_batch
            << " decoding=" << (do_sample ? "sampling" : "greedy") << "\n";

  std::vector<models::RequestId> ids;
  std::vector<std::vector<int32_t>> prompts;
  for (std::size_t i = 0; i < texts.size(); ++i) {
    models::LlamaModel::GenerateConfig g;
    g.max_new_tokens = max_new_tokens;
    g.do_sample = do_sample;
    g.seed = seed + i;  // distinct seed per request
    g.sampling.temperature = temperature;
    g.sampling.top_k = 40;
    g.sampling.top_p = 0.95;
    auto ids_in = encode(texts[i]);
    prompts.push_back(ids_in);
    ids.push_back(sched.add_request(ids_in, g));
  }

  // Drive the engine, reporting scheduler occupancy each tick.
  int tick = 0;
  while (sched.has_pending()) {
    sched.step();
    std::cout << "  tick " << tick++
              << ": running=" << sched.num_running()
              << " waiting=" << sched.num_waiting()
              << " blocks=" << sched.allocated_blocks() << "/" << ec.num_blocks
              << "\n";
  }

  std::cout << "[llama_serve] all requests finished; blocks reclaimed="
            << (sched.allocated_blocks() == 0 ? "yes" : "no") << "\n";

  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto& out = sched.result(ids[i]);
    const std::size_t plen = prompts[i].size();
    std::cout << "  req " << ids[i] << " [\"" << texts[i] << "\"] -> +"
              << (out.size() - plen) << " tokens: ";
    for (std::size_t j = plen; j < out.size(); ++j) std::cout << out[j] << " ";
    if (tok) {
      std::vector<int32_t> gen(out.begin() + static_cast<long>(plen), out.end());
      std::cout << " | \"" << tok->decode(std::span<const int32_t>(gen.data(), gen.size()))
                << "\"";
    }
    std::cout << "\n";
  }
  std::cout << "[llama_serve] done.\n";
  return 0;
}

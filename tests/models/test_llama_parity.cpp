// End-to-end test for `tesseract::models::LlamaModel`:
//
//   1. Construct a tiny but complete LlamaConfig (vocab=32, d_model=16,
//      num_heads=4, d_ff=32, num_layers=2) — small enough to fit in a few
//      KB of synthetic weights but structurally identical to a real Llama.
//
//   2. Build a matching "HF-style" safetensors file in a tmp path: one
//      tensor per HF key (input_layernorm, self_attn.{q,k,v,o}_proj,
//      post_attention_layernorm, mlp.{gate,up,down}_proj, model.norm,
//      embed_tokens) filled with deterministic but non-trivial values.
//
//   3. Call `LlamaModel::from_pretrained(path, config)` and verify:
//        a. `named_parameters()` sees every leaf, with the right shapes.
//        b. Every parameter's bytes exactly match the bytes we wrote to
//           the safetensors file (loader is lossless).
//        c. The forward pass on a small token sequence produces finite
//           logits of shape [B, S, vocab_size].
//        d. Tied embeddings: when `tie_word_embeddings=true` and
//           `lm_head.weight` is absent from the file, the loader copies
//           `embed_tokens.weight` into `lm_head.weight`.
//
//   4. Exercise the HF-name translator independently.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/utils/Logging.hpp"

namespace fs = std::filesystem;
using tesseract::DType;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::models::LlamaConfig;
using tesseract::models::LlamaModel;

namespace {

// Build a safetensors file with FP32 tensors for a given config. If
// `emit_lm_head` is false, the file does NOT contain `lm_head.weight`
// (simulates HF's tied-embedding format).
//
// Returns the file path and a map from HF name to the raw float payload
// we actually wrote (so the test can `memcmp` vs the loaded params).
struct Fixture {
  fs::path path;
  std::unordered_map<std::string, std::vector<float>> payload;
};

void fill_deterministic(std::vector<float>& v, std::string_view seed_tag) {
  // Hash the tag into a 64-bit seed so every tensor gets a distinct pattern
  // without cross-tensor accidental similarity (which would let a loader
  // bug go undetected).
  uint64_t s = 0xDEADBEEFCAFEBABEULL;
  for (char c : seed_tag) s = s * 1099511628211ULL ^ static_cast<uint8_t>(c);

  std::mt19937_64 rng(s);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  for (auto& x : v) x = d(rng);
}

std::vector<std::byte> float_bytes(const std::vector<float>& v) {
  std::vector<std::byte> out(v.size() * sizeof(float));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

struct RawEntry {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<std::byte> bytes;
};

std::vector<std::byte> build_safetensors_blob(const std::vector<RawEntry>& entries) {
  std::string json = "{";
  std::size_t running = 0;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (i) json += ',';
    const auto& e = entries[i];
    json += '"'; json += e.name; json += "\":{\"dtype\":\"F32\",\"shape\":[";
    for (std::size_t k = 0; k < e.shape.size(); ++k) {
      if (k) json += ',';
      json += std::to_string(e.shape[k]);
    }
    json += "],\"data_offsets\":[";
    json += std::to_string(running);
    json += ',';
    json += std::to_string(running + e.bytes.size());
    json += "]}";
    running += e.bytes.size();
  }
  json += '}';

  const uint64_t n = static_cast<uint64_t>(json.size());
  std::vector<std::byte> out(8);
  std::memcpy(out.data(), &n, 8);
  for (char c : json) out.push_back(static_cast<std::byte>(c));
  for (const auto& e : entries) {
    out.insert(out.end(), e.bytes.begin(), e.bytes.end());
  }
  return out;
}

Fixture write_llama_fixture(const LlamaConfig& cfg, bool emit_lm_head,
                             const std::string& tag) {
  Fixture fx;
  auto dir = fs::temp_directory_path() / "tesseract_llama_tests";
  fs::create_directories(dir);
  fx.path = dir / (tag + "_" + std::to_string(::getpid()) + ".safetensors");

  const int64_t V = cfg.vocab_size;
  const int64_t D = cfg.hidden_size;
  const int64_t F = cfg.intermediate_size;

  std::vector<RawEntry> entries;
  auto add_to = [&](std::string name, std::vector<int64_t> shape) {
    std::vector<float> data;
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    data.resize(static_cast<std::size_t>(n));
    fill_deterministic(data, name);
    fx.payload[name] = data;
    entries.push_back({std::move(name), std::move(shape), float_bytes(data)});
  };

  add_to("model.embed_tokens.weight", {V, D});
  for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    const std::string pfx = "model.layers." + std::to_string(i) + ".";
    add_to(pfx + "input_layernorm.weight", {D});
    add_to(pfx + "self_attn.q_proj.weight", {D, D});
    add_to(pfx + "self_attn.k_proj.weight", {D, D});
    add_to(pfx + "self_attn.v_proj.weight", {D, D});
    add_to(pfx + "self_attn.o_proj.weight", {D, D});
    add_to(pfx + "post_attention_layernorm.weight", {D});
    add_to(pfx + "mlp.gate_proj.weight", {F, D});
    add_to(pfx + "mlp.up_proj.weight",   {F, D});
    add_to(pfx + "mlp.down_proj.weight", {D, F});
  }
  add_to("model.norm.weight", {D});
  if (emit_lm_head) {
    add_to("lm_head.weight", {V, D});
  }

  auto blob = build_safetensors_blob(entries);
  std::ofstream f(fx.path, std::ios::binary | std::ios::trunc);
  REQUIRE(f.good());
  f.write(reinterpret_cast<const char*>(blob.data()),
          static_cast<std::streamsize>(blob.size()));
  REQUIRE(f.good());
  return fx;
}

LlamaConfig tiny_config(bool tie) {
  LlamaConfig c;
  c.vocab_size              = 32;
  c.hidden_size             = 16;
  c.num_hidden_layers       = 2;
  c.num_attention_heads     = 4;     // d_head = 4
  c.intermediate_size       = 32;
  c.max_position_embeddings = 64;
  c.rope_theta              = 10000.0;
  c.rms_norm_eps            = 1e-5;
  c.tie_word_embeddings     = tie;
  c.dtype                   = DType::Float32;
  return c;
}

}  // namespace

TEST_CASE("LlamaModel: HF-name translator covers every block sub-module") {
  using tesseract::models::llama_local_to_hf_name;

  REQUIRE(llama_local_to_hf_name("embed_tokens.weight") ==
          "model.embed_tokens.weight");
  REQUIRE(llama_local_to_hf_name("layers.0.norm_1.weight") ==
          "model.layers.0.input_layernorm.weight");
  REQUIRE(llama_local_to_hf_name("layers.7.attn.q_proj.weight") ==
          "model.layers.7.self_attn.q_proj.weight");
  REQUIRE(llama_local_to_hf_name("layers.7.attn.k_proj.weight") ==
          "model.layers.7.self_attn.k_proj.weight");
  REQUIRE(llama_local_to_hf_name("layers.7.attn.v_proj.weight") ==
          "model.layers.7.self_attn.v_proj.weight");
  REQUIRE(llama_local_to_hf_name("layers.7.attn.o_proj.weight") ==
          "model.layers.7.self_attn.o_proj.weight");
  REQUIRE(llama_local_to_hf_name("layers.7.norm_2.weight") ==
          "model.layers.7.post_attention_layernorm.weight");
  REQUIRE(llama_local_to_hf_name("layers.7.ffn.gate_proj.weight") ==
          "model.layers.7.mlp.gate_proj.weight");
  REQUIRE(llama_local_to_hf_name("layers.7.ffn.up_proj.weight") ==
          "model.layers.7.mlp.up_proj.weight");
  REQUIRE(llama_local_to_hf_name("layers.7.ffn.down_proj.weight") ==
          "model.layers.7.mlp.down_proj.weight");
  REQUIRE(llama_local_to_hf_name("norm.weight") == "model.norm.weight");
  REQUIRE(llama_local_to_hf_name("lm_head.weight") == "lm_head.weight");
}

TEST_CASE("LlamaModel: named_parameters surfaces the full HF-shaped tree") {
  LlamaConfig cfg = tiny_config(/*tie=*/false);
  LlamaModel m(cfg);
  auto nps = m.named_parameters();

  // Expected per-layer parameter names (9 per block):
  //   norm_1.weight
  //   attn.q_proj.weight
  //   attn.k_proj.weight
  //   attn.v_proj.weight
  //   attn.o_proj.weight
  //   norm_2.weight
  //   ffn.gate_proj.weight
  //   ffn.up_proj.weight
  //   ffn.down_proj.weight
  const int64_t per_layer = 9;
  const int64_t expected = 1 /* embed */ + per_layer * cfg.num_hidden_layers +
                           1 /* norm */ + 1 /* lm_head */;
  REQUIRE(static_cast<int64_t>(nps.size()) == expected);

  // Spot-check a few qualified names exist.
  const auto contains = [&](std::string_view want) {
    return std::any_of(nps.begin(), nps.end(),
                       [&](const auto& p) { return p.first == want; });
  };
  REQUIRE(contains("embed_tokens.weight"));
  REQUIRE(contains("layers.0.attn.q_proj.weight"));
  REQUIRE(contains("layers.1.ffn.down_proj.weight"));
  REQUIRE(contains("norm.weight"));
  REQUIRE(contains("lm_head.weight"));
}

TEST_CASE("LlamaModel::from_pretrained loads every parameter byte-exact (no tie)") {
  LlamaConfig cfg = tiny_config(/*tie=*/false);
  auto fx = write_llama_fixture(cfg, /*emit_lm_head=*/true, "no_tie");

  auto model = LlamaModel::from_pretrained(fx.path.string(), cfg);
  auto nps = model->named_parameters();

  for (const auto& [local, param] : nps) {
    const std::string hf = tesseract::models::llama_local_to_hf_name(local);
    const auto it = fx.payload.find(hf);
    REQUIRE(it != fx.payload.end());
    const auto& expect = it->second;
    REQUIRE(param.numel() == static_cast<int64_t>(expect.size()));
    const float* p = param.data_ptr<float>();
    for (std::size_t i = 0; i < expect.size(); ++i) {
      REQUIRE(p[i] == expect[i]);
    }
  }

  fs::remove(fx.path);
}

TEST_CASE("LlamaModel::from_pretrained ties lm_head to embed_tokens when requested") {
  LlamaConfig cfg = tiny_config(/*tie=*/true);
  auto fx = write_llama_fixture(cfg, /*emit_lm_head=*/false, "tie");

  auto model = LlamaModel::from_pretrained(fx.path.string(), cfg);
  // Pre-quantize, lm_head is always an FP `nn::Linear`; downcast so
  // we can peek at the weight tensor directly. Wave 3.3 widened the
  // slot type to `shared_ptr<nn::Module>` to allow post-quantize
  // swaps.
  auto fp_head = std::dynamic_pointer_cast<tesseract::nn::Linear>(model->lm_head());
  REQUIRE(fp_head != nullptr);
  const auto& head = fp_head->weight();
  const auto& emb  = model->embed_tokens()->weight();
  REQUIRE(head.shape() == emb.shape());
  const float* hp = head.data_ptr<float>();
  const float* ep = emb.data_ptr<float>();
  for (int64_t i = 0; i < head.numel(); ++i) {
    REQUIRE(hp[i] == ep[i]);
  }

  fs::remove(fx.path);
}

TEST_CASE("LlamaModel::forward on loaded weights produces finite logits of expected shape") {
  LlamaConfig cfg = tiny_config(/*tie=*/false);
  auto fx = write_llama_fixture(cfg, /*emit_lm_head=*/true, "fwd");

  auto model = LlamaModel::from_pretrained(fx.path.string(), cfg);
  const int64_t B = 2;
  const int64_t S = 5;

  // Random valid token ids.
  std::vector<int64_t> tok(B * S);
  std::mt19937_64 rng(0xC0FFEE);
  std::uniform_int_distribution<int64_t> d(0, cfg.vocab_size - 1);
  for (auto& t : tok) t = d(rng);

  Tensor toks = Tensor::empty({B, S}, DType::Int64);
  std::memcpy(toks.raw_data(), tok.data(), tok.size() * sizeof(int64_t));

  Tensor logits = model->forward(toks);
  REQUIRE(logits.shape() == Shape({B, S, cfg.vocab_size}));
  const float* p = logits.data_ptr<float>();
  int64_t finite = 0;
  for (int64_t i = 0; i < logits.numel(); ++i) {
    if (std::isfinite(p[i])) ++finite;
  }
  REQUIRE(finite == logits.numel());

  fs::remove(fx.path);
}

TEST_CASE("LlamaModel::load_safetensors rejects missing required tensors") {
  LlamaConfig cfg = tiny_config(/*tie=*/false);
  auto fx = write_llama_fixture(cfg, /*emit_lm_head=*/false, "missing_head");

  // With tie=false, lm_head.weight is required; its absence must raise.
  LlamaModel m(cfg);
  REQUIRE_THROWS(m.load_safetensors(fx.path.string()));

  fs::remove(fx.path);
}

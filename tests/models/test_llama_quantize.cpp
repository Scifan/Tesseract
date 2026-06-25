// Wave 3.3 (B-021) — `LlamaModel::quantize_` walker parity.
//
// This TU drives the full INT8 / INT4 quantization pipeline end-to-end
// on a synthetic 2-layer Llama, validating:
//
//   1. The walker swaps every `nn::Linear` projection inside every
//      TransformerBlock's MHA / FFN, plus the top-level `lm_head`.
//      Exercise via `Module::named_parameters()` + `named_buffers()`
//      before/after quantize — the `weight` entries for each
//      projection should disappear from params and the `q_weight` /
//      `weight_scale` entries should appear among buffers under the
//      same dotted prefix.
//
//   2. **INT8 symmetric**: per-token top-1 argmax of the INT8
//      logits matches the FP32 logits for ≥95% of tokens in a
//      `[B=4, S=128]` (512-token) batch. This is the "INT8 top-1
//      logit rank vs. FP32" bar from the B-021 DoD.
//
//   3. **INT4 group-symmetric** (group_size=32): per-token top-5
//      overlap between INT4 and FP32 is ≥ 4/5 on average across
//      512 tokens. Also the B-021 DoD bar.
//
// Both quantized models share weights with the FP model (loaded
// from the same safetensors fixture), so any divergence has to
// come from the quantization-error budget.
//
// The test intentionally skips CUDA: the walker itself is a module-
// tree edit that lives in host code, and the per-op CPU↔CUDA parity
// for `dequantize_matmul_int8` / `_int4_group` is already covered
// by tests/nn/test_nn_quantized_linear.cpp and
// tests/nn/test_nn_quantized_linear_int4.cpp. A CUDA end-to-end
// parity run would only re-check what those TUs already pin.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/nn/FeedForward.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"
#include "tesseract/nn/QuantizedLinear.hpp"
#include "tesseract/nn/QuantizedLinearInt4G.hpp"
#include "tesseract/nn/TransformerBlock.hpp"
#include "tesseract/quant/Scheme.hpp"

namespace fs = std::filesystem;
using tesseract::DType;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::models::LlamaConfig;
using tesseract::models::LlamaModel;
using tesseract::nn::FeedForward;
using tesseract::nn::Linear;
using tesseract::nn::MultiHeadAttention;
using tesseract::nn::QuantizedLinear;
using tesseract::nn::QuantizedLinearInt4G;
using tesseract::quant::Scheme;

namespace {

// ---------------------------------------------------------------------------
// Fixture: build a deterministic safetensors blob for a test-sized Llama.
// Parallels the helpers in test_llama_parity.cpp but pared down to what
// this TU needs. Kept local (anonymous namespace) so the two TUs can
// evolve independently.
// ---------------------------------------------------------------------------

struct RawEntry {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<std::byte> bytes;
};

void fill_deterministic(std::vector<float>& v, std::string_view seed_tag) {
  uint64_t s = 0xDEADBEEFCAFEBABEULL;
  for (char c : seed_tag) s = s * 1099511628211ULL ^ static_cast<uint8_t>(c);
  std::mt19937_64 rng(s);
  // Slightly tighter range than the parity TU's `[-0.5, 0.5]` because
  // INT8 rounding is comparatively more brittle in the tails than in
  // the center of the distribution — we want the random weights to
  // exercise that center so the quantization-error budget is a fair
  // reflection of in-distribution weights, not outlier-stress-test
  // weights.
  std::uniform_real_distribution<float> d(-0.2f, 0.2f);
  for (auto& x : v) x = d(rng);
}

std::vector<std::byte> float_bytes(const std::vector<float>& v) {
  std::vector<std::byte> out(v.size() * sizeof(float));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

std::vector<std::byte> build_safetensors_blob(const std::vector<RawEntry>& es) {
  std::string json = "{";
  std::size_t off = 0;
  for (std::size_t i = 0; i < es.size(); ++i) {
    if (i) json += ',';
    const auto& e = es[i];
    json += '"'; json += e.name; json += "\":{\"dtype\":\"F32\",\"shape\":[";
    for (std::size_t k = 0; k < e.shape.size(); ++k) {
      if (k) json += ',';
      json += std::to_string(e.shape[k]);
    }
    json += "],\"data_offsets\":[";
    json += std::to_string(off);
    json += ',';
    json += std::to_string(off + e.bytes.size());
    json += "]}";
    off += e.bytes.size();
  }
  json += '}';
  const uint64_t n = static_cast<uint64_t>(json.size());
  std::vector<std::byte> out(8);
  std::memcpy(out.data(), &n, 8);
  for (char c : json) out.push_back(static_cast<std::byte>(c));
  for (const auto& e : es) {
    out.insert(out.end(), e.bytes.begin(), e.bytes.end());
  }
  return out;
}

fs::path write_fixture(const LlamaConfig& cfg, const std::string& tag) {
  auto dir = fs::temp_directory_path() / "tesseract_llama_quant_tests";
  fs::create_directories(dir);
  auto path = dir / (tag + "_" + std::to_string(::getpid()) + ".safetensors");

  const int64_t V = cfg.vocab_size;
  const int64_t D = cfg.hidden_size;
  const int64_t F = cfg.intermediate_size;

  std::vector<RawEntry> entries;
  auto add = [&](std::string name, std::vector<int64_t> shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::vector<float> data(static_cast<std::size_t>(n));
    fill_deterministic(data, name);
    entries.push_back({std::move(name), std::move(shape), float_bytes(data)});
  };

  add("model.embed_tokens.weight", {V, D});
  for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    const std::string pfx = "model.layers." + std::to_string(i) + ".";
    add(pfx + "input_layernorm.weight", {D});
    add(pfx + "self_attn.q_proj.weight", {D, D});
    add(pfx + "self_attn.k_proj.weight", {D, D});
    add(pfx + "self_attn.v_proj.weight", {D, D});
    add(pfx + "self_attn.o_proj.weight", {D, D});
    add(pfx + "post_attention_layernorm.weight", {D});
    add(pfx + "mlp.gate_proj.weight", {F, D});
    add(pfx + "mlp.up_proj.weight",   {F, D});
    add(pfx + "mlp.down_proj.weight", {D, F});
  }
  add("model.norm.weight", {D});
  add("lm_head.weight", {V, D});

  auto blob = build_safetensors_blob(entries);
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  REQUIRE(f.good());
  f.write(reinterpret_cast<const char*>(blob.data()),
          static_cast<std::streamsize>(blob.size()));
  REQUIRE(f.good());
  return path;
}

LlamaConfig tiny_config() {
  // Sized so the three in-features dims the INT4 packer sees
  // (`hidden=64`, `hidden=64`, `d_ff=128`) are all multiples of the
  // test's `group_size=32`. vocab=128 gives a large enough logit
  // space for top-5 to be meaningful (top-5 on vocab=5 would be
  // trivially always-perfect).
  LlamaConfig c;
  c.vocab_size              = 128;
  c.hidden_size             = 64;
  c.num_hidden_layers       = 2;
  c.num_attention_heads     = 4;    // d_head = 16
  c.intermediate_size       = 128;
  c.max_position_embeddings = 256;
  c.rope_theta              = 10000.0;
  c.rms_norm_eps            = 1e-5;
  c.tie_word_embeddings     = false;
  c.dtype                   = DType::Float32;
  return c;
}

// Load weights into a freshly-constructed model. Factored so the test
// can instantiate three identical models (FP, to-be-INT8, to-be-INT4)
// from one fixture.
std::shared_ptr<LlamaModel> load_model(const LlamaConfig& cfg,
                                       const fs::path& fixture) {
  auto m = std::make_shared<LlamaModel>(cfg);
  (void)m->load_safetensors(fixture.string());
  return m;
}

// Build a random but valid [B, S] token tensor.
Tensor build_tokens(int64_t B, int64_t S, int64_t vocab, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> d(0, vocab - 1);
  std::vector<int64_t> tok(static_cast<std::size_t>(B * S));
  for (auto& t : tok) t = d(rng);
  Tensor toks = Tensor::empty({B, S}, DType::Int64);
  std::memcpy(toks.raw_data(), tok.data(), tok.size() * sizeof(int64_t));
  return toks;
}

// Index of the largest value in `row`. Ties broken by picking the
// lowest index — matches NumPy's `argmax` semantics so if a reader
// cross-checks vs. a Python baseline the numbers agree.
int64_t argmax_row(const float* row, int64_t n) {
  int64_t best = 0;
  float best_v = row[0];
  for (int64_t i = 1; i < n; ++i) {
    if (row[i] > best_v) {
      best_v = row[i];
      best = i;
    }
  }
  return best;
}

// Indices of the 5 largest values in `row`, returned in descending-
// logit order. Uses `std::partial_sort_copy` through an index array
// so we avoid an O(n log n) full sort for what is always 5 slots.
std::vector<int64_t> top5_row(const float* row, int64_t n) {
  std::vector<int64_t> idx(static_cast<std::size_t>(n));
  for (int64_t i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
  const int64_t k = std::min<int64_t>(5, n);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int64_t a, int64_t b) {
                      if (row[a] != row[b]) return row[a] > row[b];
                      return a < b;
                    });
  idx.resize(static_cast<std::size_t>(k));
  return idx;
}

}  // namespace

// ---------------------------------------------------------------------------
// Structural check: walker swaps projections under the right slot names.
// ---------------------------------------------------------------------------

TEST_CASE("LlamaModel::quantize_(INT8): module-tree structure after walk") {
  LlamaConfig cfg = tiny_config();
  auto fixture = write_fixture(cfg, "struct_int8");
  auto model = load_model(cfg, fixture);

  // Pre-quantize every projection + lm_head is a plain `Linear`.
  for (auto& block : model->layers()) {
    auto attn = block->attn();
    REQUIRE(std::dynamic_pointer_cast<Linear>(attn->q_proj()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<Linear>(attn->k_proj()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<Linear>(attn->v_proj()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<Linear>(attn->o_proj()) != nullptr);
    auto ffn = block->ffn();
    REQUIRE(std::dynamic_pointer_cast<Linear>(ffn->gate_proj()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<Linear>(ffn->up_proj())   != nullptr);
    REQUIRE(std::dynamic_pointer_cast<Linear>(ffn->down_proj()) != nullptr);
  }
  REQUIRE(std::dynamic_pointer_cast<Linear>(model->lm_head()) != nullptr);

  model->quantize_(Scheme::int8_symmetric());

  // Post-quantize every one of those seven-per-block + lm_head slots
  // is a QuantizedLinear. The walker must be total on Linear children.
  for (auto& block : model->layers()) {
    auto attn = block->attn();
    REQUIRE(std::dynamic_pointer_cast<QuantizedLinear>(attn->q_proj()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<QuantizedLinear>(attn->k_proj()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<QuantizedLinear>(attn->v_proj()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<QuantizedLinear>(attn->o_proj()) != nullptr);
    auto ffn = block->ffn();
    REQUIRE(std::dynamic_pointer_cast<QuantizedLinear>(ffn->gate_proj()) != nullptr);
    REQUIRE(std::dynamic_pointer_cast<QuantizedLinear>(ffn->up_proj())   != nullptr);
    REQUIRE(std::dynamic_pointer_cast<QuantizedLinear>(ffn->down_proj()) != nullptr);
  }
  REQUIRE(std::dynamic_pointer_cast<QuantizedLinear>(model->lm_head()) != nullptr);

  // `named_buffers()` now exposes `q_weight` + `weight_scale` under
  // every old projection prefix, which is how a downstream checkpoint
  // saver finds them without hard-coding the module tree.
  std::unordered_set<std::string> buf_names;
  for (auto& [name, _] : model->named_buffers()) buf_names.insert(name);

  auto has = [&](const std::string& n) {
    return buf_names.find(n) != buf_names.end();
  };
  REQUIRE(has("layers.0.attn.q_proj.q_weight"));
  REQUIRE(has("layers.0.attn.q_proj.weight_scale"));
  REQUIRE(has("layers.1.ffn.down_proj.q_weight"));
  REQUIRE(has("layers.1.ffn.down_proj.weight_scale"));
  REQUIRE(has("lm_head.q_weight"));
  REQUIRE(has("lm_head.weight_scale"));

  // Conversely the FP `.weight` keys for those same projections
  // must be GONE from `named_parameters()` — the whole point of
  // quantization is that the FP weight is no longer a trainable
  // leaf. Embedding + RMSNorm + norm stay in params (walker leaves
  // them alone by design).
  std::unordered_set<std::string> param_names;
  for (auto& [name, _] : model->named_parameters()) param_names.insert(name);
  REQUIRE(param_names.count("layers.0.attn.q_proj.weight") == 0);
  REQUIRE(param_names.count("layers.1.ffn.down_proj.weight") == 0);
  REQUIRE(param_names.count("lm_head.weight") == 0);
  REQUIRE(param_names.count("embed_tokens.weight") == 1);
  REQUIRE(param_names.count("norm.weight") == 1);

  fs::remove(fixture);
}

// ---------------------------------------------------------------------------
// INT8 end-to-end parity: top-1 argmax per token matches FP32 for >=95%
// of tokens in a 512-token batch.
// ---------------------------------------------------------------------------

TEST_CASE("LlamaModel::quantize_(INT8): top-1 logit rank matches FP32") {
  LlamaConfig cfg = tiny_config();
  auto fixture = write_fixture(cfg, "int8_parity");

  auto fp = load_model(cfg, fixture);
  auto q  = load_model(cfg, fixture);
  q->quantize_(Scheme::int8_symmetric());

  const int64_t B = 4;
  const int64_t S = 128;
  REQUIRE(B * S == 512);  // B-021 DoD fixture size
  Tensor toks = build_tokens(B, S, cfg.vocab_size, /*seed=*/0xABCD1234ULL);

  Tensor fp_logits = fp->forward(toks);    // [B, S, V]
  Tensor q_logits  = q->forward(toks);
  REQUIRE(fp_logits.shape() == q_logits.shape());
  REQUIRE(fp_logits.shape() == Shape({B, S, cfg.vocab_size}));

  const float* fp_p = fp_logits.data_ptr<float>();
  const float* q_p  = q_logits.data_ptr<float>();
  const int64_t V = cfg.vocab_size;

  int64_t agree = 0;
  int64_t total = 0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      const float* fp_row = fp_p + (b * S + s) * V;
      const float* q_row  = q_p  + (b * S + s) * V;
      if (argmax_row(fp_row, V) == argmax_row(q_row, V)) ++agree;
      ++total;
    }
  }

  // INT8 symmetric per-channel gives very tight top-1 fidelity —
  // reference GPTQ / AWQ / llama.cpp Q8_0 papers report ≥ 99% on
  // vocab-large models. On this 2-layer 64-dim toy we loosen the
  // bar to 95% (B-021 DoD). Below that the quantizer or the walker
  // has a bug.
  const double ratio = static_cast<double>(agree) / static_cast<double>(total);
  REQUIRE(ratio >= 0.95);

  fs::remove(fixture);
}

// ---------------------------------------------------------------------------
// INT4 end-to-end parity: average top-5 overlap >= 4/5 across 512 tokens.
// ---------------------------------------------------------------------------

TEST_CASE("LlamaModel::quantize_(INT4, group_size=32): top-5 overlap >= 4/5 "
          "across 512 tokens") {
  LlamaConfig cfg = tiny_config();
  auto fixture = write_fixture(cfg, "int4_parity");

  auto fp = load_model(cfg, fixture);
  auto q  = load_model(cfg, fixture);
  // group_size=32 divides every in_features this model has:
  // hidden=64 (Q/K/V/O/gate/up projections + lm_head) and d_ff=128
  // (down projection). It also matches the "keep the overhead low
  // without sacrificing accuracy" sweet spot used by the B-021
  // tests earlier in the stack.
  q->quantize_(Scheme::int4_group_symmetric(/*group_size=*/32));

  const int64_t B = 4;
  const int64_t S = 128;
  Tensor toks = build_tokens(B, S, cfg.vocab_size, /*seed=*/0x5EED5EEDULL);

  Tensor fp_logits = fp->forward(toks);
  Tensor q_logits  = q->forward(toks);
  REQUIRE(fp_logits.shape() == q_logits.shape());

  const float* fp_p = fp_logits.data_ptr<float>();
  const float* q_p  = q_logits.data_ptr<float>();
  const int64_t V = cfg.vocab_size;

  int64_t overlap_sum = 0;   // total agreeing entries across all rows
  int64_t rows = 0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      const float* fp_row = fp_p + (b * S + s) * V;
      const float* q_row  = q_p  + (b * S + s) * V;
      auto fp5 = top5_row(fp_row, V);
      auto q5  = top5_row(q_row, V);
      std::unordered_set<int64_t> fp_set(fp5.begin(), fp5.end());
      int64_t hits = 0;
      for (int64_t idx : q5) {
        if (fp_set.count(idx)) ++hits;
      }
      overlap_sum += hits;
      ++rows;
    }
  }

  // Mean top-5 overlap per row. B-021 DoD bar is ≥ 4/5 on average
  // across the 512-token fixture.
  const double mean_overlap = static_cast<double>(overlap_sum) /
                              (5.0 * static_cast<double>(rows));
  REQUIRE(mean_overlap >= 0.80);

  fs::remove(fixture);
}

// ---------------------------------------------------------------------------
// Walker is idempotent: a second call on an already-quantized model is
// a no-op (modules stay the same shared_ptr).
// ---------------------------------------------------------------------------

TEST_CASE("LlamaModel::quantize_(INT8): second call is a no-op") {
  LlamaConfig cfg = tiny_config();
  auto fixture = write_fixture(cfg, "idem");
  auto model = load_model(cfg, fixture);

  model->quantize_(Scheme::int8_symmetric());
  auto before = model->layers()[0]->attn()->q_proj();
  auto lm_before = model->lm_head();
  // Second call. `try_swap` early-exits because the `dynamic_pointer_cast<Linear>`
  // returns null for a QuantizedLinear, so the slot is untouched.
  model->quantize_(Scheme::int8_symmetric());
  auto after = model->layers()[0]->attn()->q_proj();
  auto lm_after = model->lm_head();

  REQUIRE(before.get() == after.get());
  REQUIRE(lm_before.get() == lm_after.get());

  fs::remove(fixture);
}

// Wave 10 (B-032) — compute-batched decode parity tests.
//
// `LlamaModel::forward_step_batched` folds the active set's decode into
// batched matmuls: embedding, the four attention projections, the FFN,
// and the LM head run **once** over all A stacked sequences, while each
// sequence's attention threads through its own (batch-1) KV cache at its
// own length. The correctness contract is exact: row r of the batched
// output must equal `forward_step` driven on sequence r alone, fed the
// same tokens, with its own cache. Every GEMM row is independent of the
// others, so on CPU this is bit-identical; on CUDA it agrees within a
// small float tolerance (batched vs single GEMM may pick different algos).
//
// What this pins down:
//   * batched decode == per-request decode across mixed prompt lengths
//     (so caches sit at different current_len — the ragged case);
//   * GQA (num_kv_heads < num_heads) batched decode parity;
//   * chunked-prefill batched parity (S_new > 1);
//   * A == 1 degenerates to plain forward_step;
//   * shape / cache-count validation.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/models/Llama.hpp"

using namespace tesseract;
using tesseract::models::LlamaConfig;
using tesseract::models::LlamaModel;

namespace {

LlamaConfig tiny_config(int64_t kv_heads = 0) {
  LlamaConfig c;
  c.vocab_size              = 48;
  c.hidden_size             = 32;
  c.num_hidden_layers       = 2;
  c.num_attention_heads     = 4;
  c.num_key_value_heads     = kv_heads;  // 0 ⇒ plain MHA
  c.intermediate_size       = 64;
  c.max_position_embeddings = 64;
  c.rope_theta              = 10000.0;
  c.rms_norm_eps            = 1e-5;
  c.tie_word_embeddings     = false;
  c.dtype                   = DType::Float32;
  return c;
}

Tensor make_tokens(const std::vector<int32_t>& ids, Device dev) {
  Tensor t = Tensor::empty({1, static_cast<int64_t>(ids.size())}, DType::Int64,
                           cpu_device());
  int64_t* p = t.data_ptr<int64_t>();
  for (std::size_t i = 0; i < ids.size(); ++i) p[i] = ids[i];
  return dev.is_cpu() ? t : t.to(dev);
}

// Stack one token per sequence into a [A, 1] Int64 tensor.
Tensor make_batched_tokens(const std::vector<int32_t>& ids, Device dev) {
  Tensor t = Tensor::empty({static_cast<int64_t>(ids.size()), 1}, DType::Int64,
                           cpu_device());
  int64_t* p = t.data_ptr<int64_t>();
  for (std::size_t i = 0; i < ids.size(); ++i) p[i] = ids[i];
  return dev.is_cpu() ? t : t.to(dev);
}

// Per-layer KVCacheBase caches for one sequence (contiguous caches upcast).
std::vector<std::shared_ptr<nn::KVCacheBase>> base_caches(
    const std::shared_ptr<LlamaModel>& model, int64_t max_len) {
  auto kv = model->make_kv_caches(/*batch=*/1, max_len);
  return std::vector<std::shared_ptr<nn::KVCacheBase>>(kv.begin(), kv.end());
}

// Max abs diff between row (b, S-1) of `a` [A,Sa,V] and (0, S-1) of `b`
// [1,Sb,V]. Both reduced to host floats first.
float row_max_abs_diff(const Tensor& a, int64_t a_b, const Tensor& b) {
  const Tensor ah = a.device().is_cpu() ? a.contiguous() : a.to(cpu_device()).contiguous();
  const Tensor bh = b.device().is_cpu() ? b.contiguous() : b.to(cpu_device()).contiguous();
  const int64_t Sa = ah.shape()[1], V = ah.shape()[2];
  const int64_t Sb = bh.shape()[1];
  const float* pa = ah.data_ptr<float>() + (a_b * Sa + (Sa - 1)) * V;
  const float* pb = bh.data_ptr<float>() + (Sb - 1) * V;
  float m = 0.0f;
  for (int64_t v = 0; v < V; ++v) m = std::max(m, std::abs(pa[v] - pb[v]));
  return m;
}

int32_t argmax_row(const Tensor& logits, int64_t b) {
  const Tensor h = logits.device().is_cpu() ? logits.contiguous()
                                            : logits.to(cpu_device()).contiguous();
  const int64_t S = h.shape()[1], V = h.shape()[2];
  const float* p = h.data_ptr<float>() + (b * S + (S - 1)) * V;
  int32_t best = 0;
  float bv = p[0];
  for (int64_t v = 1; v < V; ++v)
    if (p[v] > bv) { bv = p[v]; best = static_cast<int32_t>(v); }
  return best;
}

// Drive A sequences of different lengths through decode two ways — per
// request (`forward_step`) and batched (`forward_step_batched`) — feeding
// identical tokens, and assert the logits agree at every step.
void run_parity(Device dev, int64_t kv_heads) {
  auto model = std::make_shared<LlamaModel>(tiny_config(kv_heads));
  model->to(dev);
  const float tol = dev.is_cpu() ? 0.0f : 3e-3f;
  const int64_t max_len = 48;

  const std::vector<std::vector<int32_t>> prompts = {
      {3, 17, 42, 8}, {25, 1}, {39, 7, 12, 5, 30, 2}, {9},
  };
  const int64_t A = static_cast<int64_t>(prompts.size());

  // Per-request reference caches and the batched caches — separate slabs,
  // both prefilled identically one sequence at a time (prefill is always
  // per-request; only steady-state decode batches).
  std::vector<std::vector<std::shared_ptr<nn::KVCacheBase>>> ref(A), bat(A);
  std::vector<int32_t> cur(static_cast<std::size_t>(A));
  for (int64_t r = 0; r < A; ++r) {
    ref[r] = base_caches(model, max_len);
    bat[r] = base_caches(model, max_len);
    Tensor toks = make_tokens(prompts[r], dev);
    model->forward_step(toks, ref[r]);
    Tensor lb = model->forward_step(toks, bat[r]);
    cur[r] = argmax_row(lb, 0);
  }

  // Lockstep decode: feed `cur` to both paths, compare, advance greedily.
  for (int step = 0; step < 12; ++step) {
    Tensor bt = make_batched_tokens(cur, dev);            // [A, 1]
    Tensor lb = model->forward_step_batched(bt, bat);     // [A, 1, V]
    REQUIRE(lb.shape()[0] == A);
    REQUIRE(lb.shape()[1] == 1);

    for (int64_t r = 0; r < A; ++r) {
      Tensor single = model->forward_step(make_tokens({cur[r]}, dev), ref[r]);
      INFO("kv_heads=" << kv_heads << " step=" << step << " seq=" << r);
      REQUIRE(row_max_abs_diff(lb, r, single) <= tol);
      REQUIRE(argmax_row(lb, r) == argmax_row(single, 0));
      cur[r] = argmax_row(single, 0);
    }
  }
}

}  // namespace

TEST_CASE("Batched decode matches per-request decode (CPU, MHA)",
          "[models][llama][batched]") {
  run_parity(cpu_device(), /*kv_heads=*/0);
}

TEST_CASE("Batched decode matches per-request decode (CPU, GQA)",
          "[models][llama][batched]") {
  run_parity(cpu_device(), /*kv_heads=*/2);
}

TEST_CASE("Batched decode matches per-request decode (CUDA)",
          "[models][llama][batched][cuda]") {
  if (cuda::device_count() <= 0) {
    SKIP("CUDA not available");
    return;
  }
  const Device dev{DeviceType::CUDA, 0};
  run_parity(dev, /*kv_heads=*/0);
  run_parity(dev, /*kv_heads=*/2);
}

TEST_CASE("Batched decode chunked prefill matches per-request (S_new > 1)",
          "[models][llama][batched]") {
  auto model = std::make_shared<LlamaModel>(tiny_config(/*kv_heads=*/2));
  const Device dev = cpu_device();

  // Two sequences of the SAME length so they share one [A, S] slab.
  const std::vector<std::vector<int32_t>> prompts = {{3, 17, 42, 8},
                                                     {25, 1, 39, 7}};
  const int64_t A = 2, S = 4;

  std::vector<std::vector<std::shared_ptr<nn::KVCacheBase>>> bat(A);
  std::vector<Tensor> ref(A);
  for (int64_t r = 0; r < A; ++r) {
    bat[r] = base_caches(model, /*max_len=*/16);
    auto rc = base_caches(model, /*max_len=*/16);
    ref[r] = model->forward_step(make_tokens(prompts[r], dev), rc);  // [1,S,V]
  }

  // Batched prefill: stack the two prompts into [A, S].
  Tensor toks = Tensor::empty({A, S}, DType::Int64, cpu_device());
  int64_t* p = toks.data_ptr<int64_t>();
  for (int64_t r = 0; r < A; ++r)
    for (int64_t s = 0; s < S; ++s)
      p[r * S + s] = prompts[static_cast<std::size_t>(r)][static_cast<std::size_t>(s)];

  Tensor lb = model->forward_step_batched(toks, bat);   // [A, S, V]
  REQUIRE(lb.shape()[0] == A);
  REQUIRE(lb.shape()[1] == S);
  for (int64_t r = 0; r < A; ++r) {
    INFO("chunked prefill seq " << r);
    REQUIRE(row_max_abs_diff(lb, r, ref[r]) == 0.0f);
  }
}

TEST_CASE("Batched decode with A==1 equals forward_step",
          "[models][llama][batched]") {
  auto model = std::make_shared<LlamaModel>(tiny_config());
  const Device dev = cpu_device();
  const std::vector<int32_t> prompt = {5, 11, 30, 2};

  auto a = base_caches(model, 16);
  auto b = base_caches(model, 16);
  Tensor toks = make_tokens(prompt, dev);
  Tensor single = model->forward_step(toks, a);

  std::vector<std::vector<std::shared_ptr<nn::KVCacheBase>>> bat = {b};
  Tensor batched = model->forward_step_batched(toks, bat);  // [1, S, V]
  REQUIRE(row_max_abs_diff(batched, 0, single) == 0.0f);
}

TEST_CASE("Batched decode validates shapes and cache counts",
          "[models][llama][batched]") {
  auto model = std::make_shared<LlamaModel>(tiny_config());
  const Device dev = cpu_device();

  std::vector<std::vector<std::shared_ptr<nn::KVCacheBase>>> two = {
      base_caches(model, 16), base_caches(model, 16)};

  // tokens batch (2) disagrees with cache-set count (1).
  std::vector<std::vector<std::shared_ptr<nn::KVCacheBase>>> one = {
      base_caches(model, 16)};
  Tensor toks2 = make_batched_tokens({1, 2}, dev);   // [2, 1]
  REQUIRE_THROWS(model->forward_step_batched(toks2, one));

  // wrong rank (rank-1 tokens).
  Tensor bad = make_tokens({1, 2}, dev);             // [1, 2] ok rank but...
  Tensor rank1 = bad.reshape(Shape({2}));            // [2]
  REQUIRE_THROWS(model->forward_step_batched(rank1, two));

  // a sequence missing a layer cache.
  auto short_set = two;
  short_set[0].pop_back();
  REQUIRE_THROWS(model->forward_step_batched(toks2, short_set));
}

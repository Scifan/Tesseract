// M4 Track B2 (B-042) — single-device LLM training smoke.
//
// The LLM stack historically only ran inference. This test exercises the
// *training* path end-to-end: forward over `LlamaModel`, next-token
// cross-entropy, reverse-mode autograd (`Engine::backward`), and an Adam
// step — and asserts the loss of a fixed synthetic batch collapses toward
// zero (the model memorizes an in-capacity batch). A working forward +
// backward + optimizer MUST be able to drive this to ~0; if it can't, the
// autograd wiring through the transformer training path is broken.

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/optim/Adam.hpp"

using namespace tesseract;
using tesseract::models::LlamaConfig;
using tesseract::models::LlamaModel;

namespace {

LlamaConfig tiny_train_config() {
  LlamaConfig c;
  c.vocab_size = 48;
  c.hidden_size = 32;
  c.num_hidden_layers = 2;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 4;
  c.intermediate_size = 64;
  c.max_position_embeddings = 64;
  c.dtype = DType::Float32;
  return c;
}

}  // namespace

TEST_CASE("LlamaModel: next-token training overfits a fixed batch",
          "[models][llama][train]") {
  const LlamaConfig cfg = tiny_train_config();
  const int64_t B = 2;
  const int64_t S = 12;
  const int64_t flat = B * (S - 1);

  auto model = std::make_shared<LlamaModel>(cfg);
  optim::Adam opt(model->parameters(), /*lr=*/3e-3);

  // Fixed seeded batch; inputs = tokens[:, :-1], targets = tokens[:, 1:].
  auto inputs = Tensor::empty({B, S - 1}, DType::Int64);
  auto targets = Tensor::empty(Shape(std::vector<int64_t>{flat}), DType::Int64);
  {
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int64_t> dist(0, cfg.vocab_size - 1);
    int64_t* in = inputs.data_ptr<int64_t>();
    int64_t* tg = targets.data_ptr<int64_t>();
    for (int64_t b = 0; b < B; ++b) {
      int64_t prev = dist(rng);
      for (int64_t t = 0; t < S - 1; ++t) {
        in[b * (S - 1) + t] = prev;
        const int64_t nxt = dist(rng);
        tg[b * (S - 1) + t] = nxt;
        prev = nxt;
      }
    }
  }

  double first_loss = 0.0;
  double last_loss = 0.0;
  const int kSteps = 120;
  for (int step = 0; step < kSteps; ++step) {
    opt.zero_grad();
    Tensor logits = model->forward(inputs);  // [B, S-1, V]
    Tensor flat_logits =
        ops::reshape(logits, Shape({flat, cfg.vocab_size}));
    Tensor loss = ops::cross_entropy_with_logits(flat_logits, targets);
    Engine::backward(loss);
    opt.step();
    last_loss = static_cast<double>(*loss.data_ptr<float>());
    if (step == 0) first_loss = last_loss;
  }

  INFO("first_loss=" << first_loss << " last_loss=" << last_loss);
  // Started near ln(vocab); must drop a lot and reach near-zero.
  REQUIRE(last_loss < first_loss);
  REQUIRE(last_loss < 0.1);
}

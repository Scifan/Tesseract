// Single-GPU (or CPU) LLM training loop — M4 Track B2 / B-042.
//
// The LLM stack has only ever run *inference* (generate / KV-cache decode).
// This example closes the hidden prerequisite for distributed *training*:
// it trains a `tesseract::models::LlamaModel` with a next-token
// cross-entropy objective and an Adam optimizer, and shows the loss
// collapsing toward zero as the model memorizes a fixed synthetic batch.
//
// Why "overfit a fixed batch"? It is the cleanest possible convergence
// signal: a correctly wired forward + backward + optimizer MUST be able to
// drive the training loss of a fixed, small, in-capacity batch to ~0. If it
// can't, autograd or the optimizer is broken. This doubles as the B-042
// smoke test (`--max-steps` + `--target-loss` make it CI-fast and assertable).
//
// Usage:
//   ./tesseract_llama_train [--steps N] [--lr F] [--device cpu|cuda]
//                           [--batch B] [--seq S] [--target-loss F]
//
// Exit code is non-zero if the final loss fails to drop below --target-loss
// (default off: <=0 disables the gate), so CI can assert convergence.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/optim/Adam.hpp"

namespace {

// Build a fixed [B, S] Int64 token matrix from a seeded RNG. The same seed
// always yields the same batch, so the run is fully reproducible and the
// "memorize a fixed batch" framing is exact.
tesseract::Tensor make_fixed_tokens(int64_t batch, int64_t seq,
                                    int64_t vocab, uint64_t seed) {
  auto t = tesseract::Tensor::empty({batch, seq}, tesseract::DType::Int64);
  int64_t* p = t.data_ptr<int64_t>();
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> dist(0, vocab - 1);
  for (int64_t i = 0; i < batch * seq; ++i) p[i] = dist(rng);
  return t;
}

// Slice the contiguous token matrix `[B, S]` along the sequence axis into
// `[B, end-begin]`. Plain host copy — these are integer model *inputs*, not
// graph tensors, so no autograd is involved.
tesseract::Tensor seq_slice(const tesseract::Tensor& tokens,
                            int64_t begin, int64_t end) {
  const int64_t B = tokens.shape()[0];
  const int64_t S = tokens.shape()[1];
  const int64_t W = end - begin;
  auto out = tesseract::Tensor::empty({B, W}, tesseract::DType::Int64);
  const int64_t* src = tokens.data_ptr<int64_t>();
  int64_t* dst = out.data_ptr<int64_t>();
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t w = 0; w < W; ++w) {
      dst[b * W + w] = src[b * S + (begin + w)];
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace tesseract;
  using namespace tesseract::models;

  int64_t steps = 100;
  double lr = 3e-3;
  int64_t batch = 2;
  int64_t seq = 16;
  double target_loss = 0.0;  // <=0 disables the convergence gate.
  std::string device_cli = "cpu";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](int64_t def) -> int64_t {
      return (i + 1 < argc) ? std::atoll(argv[++i]) : def;
    };
    if (a == "--steps") steps = next(steps);
    else if (a == "--batch") batch = next(batch);
    else if (a == "--seq") seq = next(seq);
    else if (a == "--lr" && i + 1 < argc) lr = std::atof(argv[++i]);
    else if (a == "--target-loss" && i + 1 < argc) target_loss = std::atof(argv[++i]);
    else if (a == "--device" && i + 1 < argc) device_cli = argv[++i];
    else {
      std::cerr << "Usage: " << argv[0]
                << " [--steps N] [--lr F] [--batch B] [--seq S]"
                << " [--target-loss F] [--device cpu|cuda]\n";
      return 1;
    }
  }

  Device run_device = cpu_device();
  if (device_cli == "cuda") {
    run_device = Device{DeviceType::CUDA, 0};
  } else if (device_cli != "cpu") {
    std::cerr << "error: unknown --device '" << device_cli << "'\n";
    return 1;
  }

  // A tiny in-capacity Llama: small enough that a fixed batch is trivially
  // memorizable on CPU in a fraction of a second, yet exercising the full
  // RMSNorm → RoPE-MHA → SwiGLU → lm_head training path.
  LlamaConfig cfg;
  cfg.vocab_size = 64;
  cfg.hidden_size = 64;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 4;
  cfg.intermediate_size = 128;
  cfg.max_position_embeddings = 128;
  cfg.dtype = DType::Float32;

  auto model = std::make_shared<LlamaModel>(cfg);
  model->to(run_device);

  optim::Adam opt(model->parameters(), lr);

  // Fixed batch + shifted next-token targets:
  //   inputs  = tokens[:, :-1]   targets = tokens[:, 1:]
  const Tensor tokens = make_fixed_tokens(batch, seq, cfg.vocab_size, 1234);
  Tensor inputs = seq_slice(tokens, 0, seq - 1);          // [B, S-1]
  const int64_t flat = batch * (seq - 1);
  // Flattened next-token targets directly (avoids a 1-D reshape): for each
  // (b, t) the label is tokens[b, t+1].
  Tensor targets = Tensor::empty(Shape(std::vector<int64_t>{flat}),
                                 DType::Int64);
  {
    const int64_t* src = tokens.data_ptr<int64_t>();
    int64_t* dst = targets.data_ptr<int64_t>();
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t t = 0; t < seq - 1; ++t) {
        dst[b * (seq - 1) + t] = src[b * seq + (t + 1)];
      }
    }
  }
  if (!run_device.is_cpu()) {
    inputs = inputs.to(run_device);
    targets = targets.to(run_device);
  }

  std::cout << "[llama_train] device=" << run_device.to_string()
            << " layers=" << cfg.num_hidden_layers
            << " d_model=" << cfg.hidden_size
            << " batch=" << batch << " seq=" << seq
            << " steps=" << steps << " lr=" << lr << "\n";

  double first_loss = 0.0;
  double last_loss = 0.0;
  for (int64_t step = 0; step < steps; ++step) {
    opt.zero_grad();
    Tensor logits = model->forward(inputs);                 // [B, S-1, V]
    Tensor flat_logits = ops::reshape(
        logits, Shape({flat, cfg.vocab_size}));  // [B*(S-1), V]
    Tensor loss = ops::cross_entropy_with_logits(flat_logits, targets);
    Engine::backward(loss);
    opt.step();

    const Tensor loss_host =
        run_device.is_cpu() ? loss : loss.to(cpu_device());
    last_loss = static_cast<double>(*loss_host.data_ptr<float>());
    if (step == 0) first_loss = last_loss;
    if (step == 0 || step + 1 == steps || (step + 1) % 10 == 0) {
      std::cout << "  step " << (step + 1) << "/" << steps
                << "  loss=" << last_loss << "\n";
    }
  }

  std::cout << "[llama_train] loss: " << first_loss << " -> " << last_loss
            << "  (drop " << (first_loss - last_loss) << ")\n";

  if (last_loss >= first_loss) {
    std::cerr << "error: training loss did not decrease\n";
    return 2;
  }
  if (target_loss > 0.0 && last_loss > target_loss) {
    std::cerr << "error: final loss " << last_loss
              << " did not reach target " << target_loss << "\n";
    return 3;
  }
  std::cout << "[llama_train] OK\n";
  return 0;
}

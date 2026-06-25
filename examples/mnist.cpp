// Minimal MNIST trainer: fetches MNIST via scripts/fetch_mnist.sh, builds a
// two-layer MLP, trains a few epochs, prints loss/accuracy.
//
// Usage:
//   ./examples/mnist data/mnist
//   ./examples/mnist data/mnist --mode graph   # (ENABLE_MLIR=ON only)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Autograd.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/graph/Interpreter.hpp"
#include "tesseract/nn/Activation.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Loss.hpp"
#include "tesseract/nn/Sequential.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/optim/Adam.hpp"
#include "tesseract/utils/Logging.hpp"

#ifdef TESSERACT_ENABLE_MLIR
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include "tesseract/ir/Emit.hpp"
#include "tesseract/ir/JitEngine.hpp"
#include "tesseract/ir/Passes.hpp"
#include "tesseract/ir/TesseractDialect.h"
#endif

namespace {

uint32_t read_u32_be(std::ifstream& f) {
  uint8_t b[4];
  f.read(reinterpret_cast<char*>(b), 4);
  return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
         (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
}

tesseract::Tensor load_images(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) TESSERACT_THROW("Cannot open MNIST image file: {}", path);
  const uint32_t magic = read_u32_be(f);
  TESSERACT_CHECK(magic == 2051, "Bad MNIST image magic: {}", magic);
  const uint32_t n = read_u32_be(f);
  const uint32_t rows = read_u32_be(f);
  const uint32_t cols = read_u32_be(f);
  const int64_t pixels = static_cast<int64_t>(rows) * cols;

  std::vector<uint8_t> buf(static_cast<std::size_t>(n) * pixels);
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  TESSERACT_CHECK(f.gcount() == static_cast<std::streamsize>(buf.size()),
                  "Short read on MNIST images");

  auto t = tesseract::Tensor::empty({n, pixels}, tesseract::DType::Float32);
  float* p = t.data_ptr<float>();
  for (std::size_t i = 0; i < buf.size(); ++i) {
    p[i] = static_cast<float>(buf[i]) / 255.0f;
  }
  return t;
}

tesseract::Tensor load_labels(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) TESSERACT_THROW("Cannot open MNIST label file: {}", path);
  const uint32_t magic = read_u32_be(f);
  TESSERACT_CHECK(magic == 2049, "Bad MNIST label magic: {}", magic);
  const uint32_t n = read_u32_be(f);

  std::vector<uint8_t> buf(n);
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));

  auto t = tesseract::Tensor::empty({n}, tesseract::DType::Int64);
  int64_t* p = t.data_ptr<int64_t>();
  for (std::size_t i = 0; i < buf.size(); ++i) p[i] = static_cast<int64_t>(buf[i]);
  return t;
}

double accuracy(const tesseract::Tensor& logits, const tesseract::Tensor& targets) {
  const int64_t N = logits.shape()[0];
  const int64_t C = logits.shape()[1];
  const float* pl = logits.data_ptr<float>();
  const int64_t* pt = targets.data_ptr<int64_t>();
  int64_t correct = 0;
  for (int64_t n = 0; n < N; ++n) {
    int64_t best = 0;
    float bv = pl[n * C + 0];
    for (int64_t c = 1; c < C; ++c) {
      if (pl[n * C + c] > bv) { bv = pl[n * C + c]; best = c; }
    }
    if (best == pt[n]) ++correct;
  }
  return static_cast<double>(correct) / static_cast<double>(N);
}

tesseract::Tensor slice_rows(const tesseract::Tensor& t, int64_t start, int64_t end) {
  // Contiguous row slice for Float32[N,D] or Int64[N]. Implemented manually
  // because we haven't shipped an `index` op in M0.
  const int64_t rows = end - start;
  std::vector<int64_t> dims;
  for (int64_t d = 0; d < t.rank(); ++d) dims.push_back(t.shape()[d]);
  dims[0] = rows;
  tesseract::Shape new_shape(dims.begin(), dims.end());
  auto out = tesseract::Tensor::empty(new_shape, t.dtype());
  const std::size_t itemsize = t.itemsize();
  const std::size_t row_bytes = static_cast<std::size_t>(t.numel() / t.shape()[0]) * itemsize;
  std::memcpy(out.raw_data(),
              static_cast<const std::byte*>(t.raw_data()) + start * row_bytes,
              rows * row_bytes);
  return out;
}

// --- Graph mode (M1I stage-2): real training loop over the graph IR. --- //
//
// Strategy:
//   1. Capture a single forward pass on a warm-up mini-batch through a
//      `graph::GraphScope`. The recorded graph is the shape-specialized
//      "MNIST MLP" program.
//   2. Run `graph::build_backward` to extend the graph with reverse-mode
//      AD. The result lists one cotangent input (seed grad of the loss)
//      and one extra output per parameter (dL/dparam).
//   3. For each training step, call `graph::run(g, bindings)` to execute
//      the extended graph — the interpreter dispatches to the eager CPU
//      kernels. We copy the resulting per-param gradients into each
//      parameter's `.grad` and then step the Adam optimizer exactly as
//      in eager mode.
//
// When `--mode graph` is combined with `--dump-ir` (and MLIR is enabled
// at build time), we also emit the forward graph as MLIR and apply the
// `--tesseract-backward --convert-tesseract-to-linalg` pipeline so the
// user can eyeball the lowered IR alongside training output. The MLIR
// path is purely informational at M1I.2; execution still flows through
// the C++ interpreter.
// `engine` selects the execution backend for the captured + post-AD
// graph. "interp" runs the C++ graph interpreter (stage-1); "mlir"
// builds a `ir::JitEngine` once and dispatches every training step
// through the MLIR ExecutionEngine (stage-2). Numerical parity between
// the two is checked by `test_graph_jit_parity`.
enum class GraphEngine { kInterpreter, kMlir };

void run_graph_mode_training(tesseract::Tensor train_x,
                             tesseract::Tensor train_y,
                             tesseract::Tensor test_x,
                             tesseract::Tensor test_y,
                             int epochs,
                             int max_steps,
                             bool dump_ir,
                             GraphEngine engine) {
  using namespace tesseract;

  auto fc1  = std::make_shared<nn::Linear>(784, 128);
  auto relu = std::make_shared<nn::ReLU>();
  auto fc2  = std::make_shared<nn::Linear>(128, 10);
  auto model = std::make_shared<nn::Sequential>(
      std::initializer_list<std::shared_ptr<nn::Module>>{fc1, relu, fc2});

  optim::Adam opt(model->parameters(), /*lr=*/1e-3);
  constexpr int kBatch = 64;

  // --- 1. Warm-up: capture one forward pass into a Graph. --- //
  graph::Graph captured;
  graph::ValueId x_id = graph::kInvalidValueId;
  graph::ValueId y_id = graph::kInvalidValueId;
  std::vector<graph::ValueId> param_ids;
  {
    graph::GraphScope scope;
    Tensor warm_x = slice_rows(train_x, 0, kBatch);
    Tensor warm_y = slice_rows(train_y, 0, kBatch);
    x_id = graph::bind_input(warm_x, "x");
    y_id = graph::bind_input(warm_y, "targets");
    const std::array<const char*, 4> pnames = {
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
    const std::array<Tensor, 4> params = {
        fc1->weight(), fc1->bias(), fc2->weight(), fc2->bias()};
    for (std::size_t i = 0; i < params.size(); ++i) {
      param_ids.push_back(graph::bind_param(params[i], pnames[i]));
    }

    Tensor logits = model->forward(warm_x);
    Tensor loss   = ops::cross_entropy_with_logits(logits, warm_y);
    graph::mark_output(loss);
    std::cout << "[graph] captured " << scope.graph().num_ops() << " ops, "
              << scope.graph().num_values() << " values, "
              << scope.graph().params().size() << " params\n";
    captured = std::move(scope.graph_);
  }

  // --- 2. Build backward and collect the post-AD output layout. --- //
  graph::BackwardResult bwd = graph::build_backward(captured);
  TESSERACT_CHECK(bwd.cotangents.size() == 1,
                  "[graph] expected one cotangent (the loss seed), got {}",
                  bwd.cotangents.size());
  TESSERACT_CHECK(bwd.dparams.size() == param_ids.size(),
                  "[graph] dparam count {} != param count {}",
                  bwd.dparams.size(), param_ids.size());
  std::cout << "[graph] after backward: " << captured.num_ops() << " ops, "
            << captured.outputs().size() << " outputs\n";

#ifdef TESSERACT_ENABLE_MLIR
  if (dump_ir) {
    // Re-capture a tiny slice and drive it through the MLIR pipeline
    // purely for visual inspection. We keep this separate from the
    // interpreter's graph to avoid mixing M1I.1 and M1I.2 concerns.
    mlir::MLIRContext ctx;
    ctx.loadDialect<ir::TesseractDialect>();
    graph::Graph ir_graph;
    {
      graph::GraphScope scope;
      Tensor warm_x = slice_rows(train_x, 0, kBatch);
      Tensor warm_y = slice_rows(train_y, 0, kBatch);
      graph::bind_input(warm_x, "x");
      graph::bind_input(warm_y, "targets");
      graph::bind_param(fc1->weight(), "fc1.weight");
      graph::bind_param(fc1->bias(),   "fc1.bias");
      graph::bind_param(fc2->weight(), "fc2.weight");
      graph::bind_param(fc2->bias(),   "fc2.bias");
      Tensor logits = model->forward(warm_x);
      Tensor loss   = ops::cross_entropy_with_logits(logits, warm_y);
      graph::mark_output(loss);
      ir_graph = std::move(scope.graph_);
    }
    auto module = ir::emit_mlir(ctx, ir_graph);
    mlir::PassManager pm(&ctx);
    pm.addPass(ir::createBackwardPass());
    pm.addPass(ir::createConvertTesseractToLinalgPass());
    if (mlir::failed(pm.run(*module)) ||
        mlir::failed(mlir::verify(*module))) {
      std::cerr << "[graph] MLIR dump pipeline failed to verify\n";
    } else {
      std::string ir_text;
      llvm::raw_string_ostream os(ir_text);
      module->print(os);
      std::cout << "---------------- lowered IR ----------------\n"
                << ir_text
                << "\n--------------------------------------------\n";
    }
  }
#else
  (void)dump_ir;
#endif

  // --- 3. Training loop through the selected execution engine. --- //
  //
  // For `engine == kMlir` we build `JitEngine` *once* against the post-
  // backward graph — the shape-specialized IR is stable across steps —
  // and reuse it for every mini-batch. Params are rebound each step
  // because Adam's in-place updates preserve storage but the bind map
  // needs to see the current tensor (contiguous, correct dtype).
#ifdef TESSERACT_ENABLE_MLIR
  // `jit` is only referenced inside `#ifdef TESSERACT_ENABLE_MLIR`
  // below; moving the declaration under the same gate keeps the
  // MLIR-disabled CUDA build (which doesn't link the `ir::` headers)
  // from touching an unknown type name.
  std::unique_ptr<ir::JitEngine> jit;
  if (engine == GraphEngine::kMlir) {
    ir::JitEngineOptions jit_opts;
    jit_opts.dump_ir = dump_ir;
    jit = std::make_unique<ir::JitEngine>(captured, std::move(jit_opts));
    std::cout << "[graph] MLIR JIT engine built (compiles once, reused "
                 "across steps)\n";
  }
#else
  if (engine == GraphEngine::kMlir) {
    TESSERACT_THROW(
        "--engine mlir requires -DTESSERACT_ENABLE_MLIR=ON at build time");
  }
#endif

  const int64_t N = train_x.shape()[0];
  std::vector<int64_t> idx(static_cast<std::size_t>(N));
  std::iota(idx.begin(), idx.end(), 0);
  std::mt19937_64 rng(42);

  Tensor loss_seed = Tensor::ones({}, DType::Float32);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(idx.begin(), idx.end(), rng);
    double running_loss = 0.0;
    int64_t steps = 0;

    for (int64_t i = 0; i + kBatch <= N; i += kBatch) {
      auto batch_x = Tensor::empty({kBatch, 784}, DType::Float32);
      auto batch_y = Tensor::empty({kBatch}, DType::Int64);
      const float*  src_x = train_x.data_ptr<float>();
      const int64_t* src_y = train_y.data_ptr<int64_t>();
      float* dst_x = batch_x.data_ptr<float>();
      int64_t* dst_y = batch_y.data_ptr<int64_t>();
      for (int b = 0; b < kBatch; ++b) {
        const int64_t r = idx[i + b];
        std::memcpy(dst_x + b * 784, src_x + r * 784, 784 * sizeof(float));
        dst_y[b] = src_y[r];
      }

      std::unordered_map<graph::ValueId, Tensor> bind;
      bind.emplace(x_id, batch_x);
      bind.emplace(y_id, batch_y);
      // Params come from the model — identical shapes every step; Adam
      // updates the underlying storage in place.
      bind.emplace(param_ids[0], fc1->weight());
      bind.emplace(param_ids[1], fc1->bias());
      bind.emplace(param_ids[2], fc2->weight());
      bind.emplace(param_ids[3], fc2->bias());
      bind.emplace(bwd.cotangents[0], loss_seed);

      std::vector<Tensor> outs;
#ifdef TESSERACT_ENABLE_MLIR
      if (engine == GraphEngine::kMlir) {
        outs = jit->invoke(bind);
      } else {
        outs = graph::run(captured, bind);
      }
#else
      outs = graph::run(captured, bind);
#endif
      // outs[0] = loss; outs[1..4] = dparams in declaration order.
      opt.zero_grad();
      std::array<Tensor, 4> param_handles = {
          fc1->weight(), fc1->bias(), fc2->weight(), fc2->bias()};
      for (std::size_t p = 0; p < param_handles.size(); ++p) {
        auto* am = param_handles[p].mutable_autograd_meta();
        am->grad = outs[1 + p];
      }
      opt.step();
      running_loss += static_cast<double>(*outs[0].data_ptr<float>());
      ++steps;
      if (steps % 200 == 0) {
        std::cout << "  [graph] epoch " << epoch << " step " << steps
                  << "  avg_loss=" << running_loss / steps << "\n";
      }
      if (max_steps > 0 && steps >= max_steps) {
        std::cout << "  [graph] --max-steps " << max_steps
                  << " reached; avg_loss=" << running_loss / steps << "\n";
        break;
      }
    }

    NoGradGuard nogg;
    double correct = 0.0;
    int64_t total = 0;
    constexpr int kEvalBatch = 512;
    for (int64_t i = 0; i < test_x.shape()[0]; i += kEvalBatch) {
      const int64_t end = std::min<int64_t>(i + kEvalBatch, test_x.shape()[0]);
      Tensor xb = slice_rows(test_x, i, end);
      Tensor yb = slice_rows(test_y, i, end);
      Tensor logits = model->forward(xb);
      correct += accuracy(logits, yb) * (end - i);
      total += (end - i);
    }
    std::cout << "[graph] epoch " << epoch << "  test_acc="
              << correct / total << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  using namespace tesseract;
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mnist_data_dir>"
              << " [--epochs N] [--max-steps N] [--mode eager|graph]"
              << " [--engine interp|mlir] [--dump-ir]"
              << " [--device cpu|cuda]\n";
    return 1;
  }
  const std::string dir = argv[1];
  int epochs_cli = 1;
  // --max-steps N caps the inner training loop to N batches per epoch
  // (<= 0 = unlimited). Exists purely for CI: lets B-006's smoke test
  // exercise the full eager / graph pipelines without burning seconds on
  // a real 937-batch epoch. Does not affect the eval pass.
  int max_steps_cli = 0;
  std::string mode = "eager";
  std::string engine_cli = "interp";
  std::string device_cli = "cpu";
  bool dump_ir = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--epochs" && i + 1 < argc) {
      epochs_cli = std::atoi(argv[++i]);
    } else if (a == "--max-steps" && i + 1 < argc) {
      max_steps_cli = std::atoi(argv[++i]);
    } else if (a == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (a == "--engine" && i + 1 < argc) {
      engine_cli = argv[++i];
    } else if (a == "--device" && i + 1 < argc) {
      device_cli = argv[++i];
    } else if (a == "--dump-ir") {
      dump_ir = true;
    }
  }

  // M2I: eager training loop now runs on either CPU or CUDA. The
  // graph-mode path stays CPU-only until M2 closes out the MLIR
  // stage-2 device story (tracked separately under the M3 lowering
  // plan); passing `--device cuda --mode graph` is therefore rejected
  // explicitly instead of silently running on CPU.
  Device run_device = cpu_device();
  if (device_cli == "cuda") {
    run_device = Device{DeviceType::CUDA, 0};
  } else if (device_cli != "cpu") {
    std::cerr << "error: unknown --device '" << device_cli
              << "'. Expected 'cpu' or 'cuda'.\n";
    return 1;
  }

  std::cout << "Loading MNIST from " << dir << " ...\n";
  Tensor train_x = load_images(dir + "/train-images-idx3-ubyte");
  Tensor train_y = load_labels(dir + "/train-labels-idx1-ubyte");
  Tensor test_x  = load_images(dir + "/t10k-images-idx3-ubyte");
  Tensor test_y  = load_labels(dir + "/t10k-labels-idx1-ubyte");
  std::cout << "train: " << train_x.shape().to_string()
            << "  test: "  << test_x.shape().to_string() << "\n";

  if (mode == "graph" && !run_device.is_cpu()) {
    std::cerr << "error: --mode graph currently only supports "
                 "--device cpu (the graph interpreter + MLIR JIT stays "
                 "CPU-only until the M3 device-aware lowering lands).\n";
    return 1;
  }

  if (mode == "graph") {
#ifndef TESSERACT_ENABLE_MLIR
    if (dump_ir) {
      std::cerr << "warning: --dump-ir ignored: build without "
                   "-DTESSERACT_ENABLE_MLIR=ON.\n";
      dump_ir = false;
    }
    if (engine_cli == "mlir") {
      std::cerr << "error: --engine mlir requires "
                   "-DTESSERACT_ENABLE_MLIR=ON.\n";
      return 1;
    }
#endif
    GraphEngine engine = GraphEngine::kInterpreter;
    if (engine_cli == "mlir") engine = GraphEngine::kMlir;
    else if (engine_cli != "interp") {
      std::cerr << "error: unknown --engine '" << engine_cli
                << "'. Expected 'interp' or 'mlir'.\n";
      return 1;
    }
    run_graph_mode_training(train_x, train_y, test_x, test_y,
                            epochs_cli, max_steps_cli, dump_ir, engine);
    return 0;
  } else if (mode != "eager") {
    std::cerr << "error: unknown --mode '" << mode
              << "'. Expected 'eager' or 'graph'.\n";
    return 1;
  }

  auto fc1  = std::make_shared<nn::Linear>(784, 128);
  auto relu = std::make_shared<nn::ReLU>();
  auto fc2  = std::make_shared<nn::Linear>(128, 10);
  auto model = std::make_shared<nn::Sequential>(
      std::initializer_list<std::shared_ptr<nn::Module>>{fc1, relu, fc2});

  // M2I: migrate all parameters to the requested device BEFORE Adam
  // is constructed. `Module::to()` mutates each TensorImpl in place,
  // so the handles Adam subsequently copies into its `params_` vector
  // already point at the on-device storage; the optimizer's moment
  // buffers then allocate on the same device on first `step()`.
  model->to(run_device);

  optim::Adam opt(model->parameters(), /*lr=*/1e-3);
  if (!run_device.is_cpu()) {
    std::cout << "[eager] training on device " << run_device.to_string()
              << "\n";
  }

  const int kEpochs = epochs_cli;
  constexpr int kBatch = 64;
  const int64_t N = train_x.shape()[0];
  std::vector<int64_t> idx(static_cast<std::size_t>(N));
  std::iota(idx.begin(), idx.end(), 0);
  std::mt19937_64 rng(42);

  for (int epoch = 0; epoch < kEpochs; ++epoch) {
    std::shuffle(idx.begin(), idx.end(), rng);
    double running_loss = 0.0;
    int64_t steps = 0;
    for (int64_t i = 0; i + kBatch <= N; i += kBatch) {
      // Gather batch rows by index (simple, not fast).
      auto batch_x = Tensor::empty({kBatch, 784}, DType::Float32);
      auto batch_y = Tensor::empty({kBatch}, DType::Int64);
      const float*  src_x = train_x.data_ptr<float>();
      const int64_t* src_y = train_y.data_ptr<int64_t>();
      float* dst_x = batch_x.data_ptr<float>();
      int64_t* dst_y = batch_y.data_ptr<int64_t>();
      for (int b = 0; b < kBatch; ++b) {
        const int64_t r = idx[i + b];
        std::memcpy(dst_x + b * 784, src_x + r * 784, 784 * sizeof(float));
        dst_y[b] = src_y[r];
      }

      if (!run_device.is_cpu()) {
        batch_x = batch_x.to(run_device);
        batch_y = batch_y.to(run_device);
      }

      opt.zero_grad();
      Tensor logits = model->forward(batch_x);
      Tensor loss = ops::cross_entropy_with_logits(logits, batch_y);
      Engine::backward(loss);
      opt.step();
      // Scalar loss readback is host-side; bounce through `.to(cpu)`
      // for CUDA runs so the printed curve stays valid.
      const Tensor loss_host = run_device.is_cpu() ? loss : loss.to(cpu_device());
      running_loss += static_cast<double>(*loss_host.data_ptr<float>());
      ++steps;
      if (steps % 200 == 0) {
        std::cout << "  epoch " << epoch << " step " << steps
                  << "  avg_loss=" << running_loss / steps << "\n";
      }
      if (max_steps_cli > 0 && steps >= max_steps_cli) {
        std::cout << "  --max-steps " << max_steps_cli
                  << " reached; avg_loss=" << running_loss / steps << "\n";
        break;
      }
    }

    // Evaluate on test set (no grad).
    NoGradGuard nogg;
    double correct = 0.0;
    int64_t total = 0;
    constexpr int kEvalBatch = 512;
    for (int64_t i = 0; i < test_x.shape()[0]; i += kEvalBatch) {
      const int64_t end = std::min<int64_t>(i + kEvalBatch, test_x.shape()[0]);
      Tensor xb = slice_rows(test_x, i, end);
      Tensor yb = slice_rows(test_y, i, end);
      if (!run_device.is_cpu()) {
        xb = xb.to(run_device);
      }
      Tensor logits = model->forward(xb);
      // `accuracy(...)` walks `data_ptr<float>()` on host, so CUDA
      // logits take one final hop through `.to(cpu)` here. Keep
      // labels on host the whole time — they never crossed the
      // device boundary in the eval pass.
      const Tensor logits_host = run_device.is_cpu()
                                     ? logits
                                     : logits.to(cpu_device());
      correct += accuracy(logits_host, yb) * (end - i);
      total += (end - i);
    }
    std::cout << "epoch " << epoch << "  test_acc=" << correct / total << "\n";
  }
  return 0;
}

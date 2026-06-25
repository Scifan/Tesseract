// M1J / M1I.2.b micro-benchmark: eager vs graph-interpreter vs MLIR JIT
// for a small MLP.
//
// Compares the following execution paths on the same model + batch:
//   * eager-forward         — a plain `model->forward(x)` under NoGrad.
//   * graph-forward (interp)— capture the forward once into `graph::Graph`,
//                             then feed batches through `graph::run`.
//   * graph-forward (mlir)  — same captured graph, but executed by the
//                             MLIR ExecutionEngine via `ir::JitEngine`.
//                             Only compiled when `TESSERACT_ENABLE_MLIR`
//                             is defined (i.e. the build enabled MLIR).
//   * eager-train-step      — forward + cross-entropy + `Engine::backward`
//                             + `optim::Adam::step()`, the M0 training loop.
//   * graph-train-step (interp) — capture forward + `graph::build_backward`
//                             once, then per step run the extended graph
//                             through `graph::run` and hand the resulting
//                             gradients to Adam.
//   * graph-train-step (mlir)   — same, but the post-backward graph is
//                             JITed and dispatched through ExecutionEngine.
//
// The numbers we care about:
//   1. Forward parity. Graph-interp dispatches to the same eager kernels
//      and should land within noise of eager. Any wild regression is a
//      signal that interpreter overhead (map lookups, dynamic dispatch)
//      has crept up.
//   2. JIT speedup. The JIT emits a single fused function that hits
//      memrefs directly — we expect it to beat both eager and the
//      interpreter on larger batches where the per-op Python-style
//      dispatch cost of the interpreter is amortized.
//   3. Compile time. JIT `ctor` time is reported separately so the
//      capture + lower + LLVM codegen cost is visible and not folded
//      into per-step latency.
//
// Output is plain text so it composes into `make bench`-style pipelines.
// Run as: ./benchmarks/bench_graph_vs_eager

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Autograd.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/graph/Interpreter.hpp"
#include "tesseract/nn/Activation.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Sequential.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/optim/Adam.hpp"

#ifdef TESSERACT_ENABLE_MLIR
#include "tesseract/ir/JitEngine.hpp"
#endif

namespace {

using Clock = std::chrono::steady_clock;
using namespace tesseract;

double secs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

void fill_random(Tensor& t, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  float* p = t.data_ptr<float>();
  const int64_t n = t.numel();
  for (int64_t i = 0; i < n; ++i) p[i] = dist(rng);
}

void fill_random_labels(Tensor& t, int64_t num_classes, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> dist(0, num_classes - 1);
  int64_t* p = t.data_ptr<int64_t>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
}

struct Mlp {
  std::shared_ptr<nn::Linear>    fc1;
  std::shared_ptr<nn::Linear>    fc2;
  std::shared_ptr<nn::Sequential> seq;
};

Mlp make_mlp(int in_features, int hidden, int out_classes) {
  auto fc1  = std::make_shared<nn::Linear>(in_features, hidden);
  auto relu = std::make_shared<nn::ReLU>();
  auto fc2  = std::make_shared<nn::Linear>(hidden, out_classes);
  auto seq  = std::make_shared<nn::Sequential>(
      std::initializer_list<std::shared_ptr<nn::Module>>{fc1, relu, fc2});
  return {fc1, fc2, seq};
}

struct Case {
  const char* label;
  int batch;
  int in_features;
  int hidden;
  int out_classes;
  int warmup;
  int iters;
};

// --- Forward-only benchmarks --------------------------------------------- //

double bench_eager_forward(Mlp& m, const Tensor& x, int warmup, int iters) {
  NoGradGuard nogg;
  for (int i = 0; i < warmup; ++i) {
    auto y = m.seq->forward(x);
    (void)y;
  }
  const auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    auto y = m.seq->forward(x);
    (void)y;
  }
  return secs(t0, Clock::now()) / iters;
}

struct GraphForward {
  graph::Graph graph;
  graph::ValueId x_id;
  double capture_secs;
};

GraphForward capture_forward(Mlp& m, const Tensor& warm_x) {
  const auto t0 = Clock::now();
  GraphForward gf;
  graph::GraphScope scope;
  gf.x_id = graph::bind_input(warm_x, "x");
  graph::bind_param(m.fc1->weight(), "fc1.weight");
  graph::bind_param(m.fc1->bias(),   "fc1.bias");
  graph::bind_param(m.fc2->weight(), "fc2.weight");
  graph::bind_param(m.fc2->bias(),   "fc2.bias");
  auto y = m.seq->forward(warm_x);
  graph::mark_output(y);
  gf.graph = std::move(scope.graph_);
  gf.capture_secs = secs(t0, Clock::now());
  return gf;
}

double bench_graph_forward(const GraphForward& gf, Mlp& m, const Tensor& x,
                           int warmup, int iters) {
  const auto run = [&]() {
    std::unordered_map<graph::ValueId, Tensor> bind;
    bind.reserve(8);
    bind.emplace(gf.x_id, x);
    // Params are deterministic in declaration order; the first four
    // non-input values are W1, b1, W2, b2.
    int param_idx = 0;
    const std::array<Tensor, 4> params = {
        m.fc1->weight(), m.fc1->bias(),
        m.fc2->weight(), m.fc2->bias()};
    for (const auto& pid : gf.graph.params()) {
      bind.emplace(pid, params[param_idx++]);
    }
    auto outs = graph::run(gf.graph, bind);
    (void)outs;
  };
  for (int i = 0; i < warmup; ++i) run();
  const auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) run();
  return secs(t0, Clock::now()) / iters;
}

#ifdef TESSERACT_ENABLE_MLIR
// Wraps `capture + JitEngine` so the per-step benchmark measures pure
// invoke time and the (much higher) one-shot build cost surfaces in its
// own column. We deliberately reuse the same `GraphForward` the
// interpreter benchmark produced — the JIT uses the captured graph
// directly and shares the parameter-id ordering.
struct JitForward {
  std::unique_ptr<ir::JitEngine> engine;
  double build_secs = 0.0;
};

JitForward build_jit_forward(const GraphForward& gf) {
  const auto t0 = Clock::now();
  JitForward jf;
  // Benchmarks want the JIT's best-case steady-state throughput, so we
  // crank the LLVM opt level past the `ir::JitEngine` default of 0
  // (which is tuned for quick capture→run turnaround rather than peak
  // runtime). Build time goes up and surfaces in the `jit-build`
  // column; per-iteration numbers get closer to what a production
  // `model.compile()` flow would see.
  ir::JitEngine::Options opts;
  opts.opt_level = 3;
  jf.engine = std::make_unique<ir::JitEngine>(gf.graph, opts);
  jf.build_secs = secs(t0, Clock::now());
  return jf;
}

double bench_jit_forward(const GraphForward& gf, const JitForward& jf,
                         Mlp& m, const Tensor& x, int warmup, int iters) {
  const std::array<Tensor, 4> params = {
      m.fc1->weight(), m.fc1->bias(),
      m.fc2->weight(), m.fc2->bias()};
  const auto run = [&]() {
    std::unordered_map<graph::ValueId, Tensor> bind;
    bind.reserve(8);
    bind.emplace(gf.x_id, x);
    int param_idx = 0;
    for (const auto& pid : gf.graph.params()) {
      bind.emplace(pid, params[param_idx++]);
    }
    auto outs = jf.engine->invoke(bind);
    (void)outs;
  };
  for (int i = 0; i < warmup; ++i) run();
  const auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) run();
  return secs(t0, Clock::now()) / iters;
}
#endif  // TESSERACT_ENABLE_MLIR

// --- Full training-step benchmarks --------------------------------------- //

double bench_eager_train_step(Mlp& m, const Tensor& x, const Tensor& y,
                              int warmup, int iters) {
  optim::Adam opt(m.seq->parameters(), /*lr=*/1e-3);
  for (int i = 0; i < warmup; ++i) {
    opt.zero_grad();
    Tensor logits = m.seq->forward(x);
    Tensor loss = ops::cross_entropy_with_logits(logits, y);
    Engine::backward(loss);
    opt.step();
  }
  const auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    opt.zero_grad();
    Tensor logits = m.seq->forward(x);
    Tensor loss = ops::cross_entropy_with_logits(logits, y);
    Engine::backward(loss);
    opt.step();
  }
  return secs(t0, Clock::now()) / iters;
}

struct GraphStep {
  graph::Graph graph;
  graph::ValueId x_id;
  graph::ValueId y_id;
  std::array<graph::ValueId, 4> pids;
  graph::ValueId grad_seed_id;
  double capture_secs;
  double build_backward_secs;
};

GraphStep capture_train_step(Mlp& m, const Tensor& warm_x,
                             const Tensor& warm_y) {
  GraphStep gs;
  const auto tc0 = Clock::now();
  {
    graph::GraphScope scope;
    gs.x_id = graph::bind_input(warm_x, "x");
    gs.y_id = graph::bind_input(warm_y, "targets");
    const std::array<Tensor, 4> params = {
        m.fc1->weight(), m.fc1->bias(),
        m.fc2->weight(), m.fc2->bias()};
    const std::array<const char*, 4> names = {
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
    for (std::size_t i = 0; i < params.size(); ++i) {
      gs.pids[i] = graph::bind_param(params[i], names[i]);
    }
    Tensor logits = m.seq->forward(warm_x);
    Tensor loss = ops::cross_entropy_with_logits(logits, warm_y);
    graph::mark_output(loss);
    gs.graph = std::move(scope.graph_);
  }
  gs.capture_secs = secs(tc0, Clock::now());

  const auto tb0 = Clock::now();
  auto bwd = graph::build_backward(gs.graph);
  gs.build_backward_secs = secs(tb0, Clock::now());
  gs.grad_seed_id = bwd.cotangents.at(0);
  return gs;
}

double bench_graph_train_step(const GraphStep& gs, Mlp& m,
                              const Tensor& x, const Tensor& y,
                              int warmup, int iters) {
  optim::Adam opt(m.seq->parameters(), /*lr=*/1e-3);
  Tensor loss_seed = Tensor::ones({}, DType::Float32);
  const auto run = [&]() {
    std::unordered_map<graph::ValueId, Tensor> bind;
    bind.reserve(8);
    bind.emplace(gs.x_id, x);
    bind.emplace(gs.y_id, y);
    bind.emplace(gs.pids[0], m.fc1->weight());
    bind.emplace(gs.pids[1], m.fc1->bias());
    bind.emplace(gs.pids[2], m.fc2->weight());
    bind.emplace(gs.pids[3], m.fc2->bias());
    bind.emplace(gs.grad_seed_id, loss_seed);
    auto outs = graph::run(gs.graph, bind);

    opt.zero_grad();
    std::array<Tensor, 4> handles = {
        m.fc1->weight(), m.fc1->bias(),
        m.fc2->weight(), m.fc2->bias()};
    for (std::size_t p = 0; p < handles.size(); ++p) {
      auto* am = handles[p].mutable_autograd_meta();
      am->grad = outs[1 + p];
    }
    opt.step();
  };
  for (int i = 0; i < warmup; ++i) run();
  const auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) run();
  return secs(t0, Clock::now()) / iters;
}

#ifdef TESSERACT_ENABLE_MLIR
struct JitStep {
  std::unique_ptr<ir::JitEngine> engine;
  double build_secs = 0.0;
};

JitStep build_jit_step(const GraphStep& gs) {
  const auto t0 = Clock::now();
  JitStep js;
  ir::JitEngine::Options opts;
  opts.opt_level = 3;  // See note in build_jit_forward.
  js.engine = std::make_unique<ir::JitEngine>(gs.graph, opts);
  js.build_secs = secs(t0, Clock::now());
  return js;
}

double bench_jit_train_step(const GraphStep& gs, const JitStep& js, Mlp& m,
                            const Tensor& x, const Tensor& y,
                            int warmup, int iters) {
  optim::Adam opt(m.seq->parameters(), /*lr=*/1e-3);
  Tensor loss_seed = Tensor::ones({}, DType::Float32);
  const auto run = [&]() {
    std::unordered_map<graph::ValueId, Tensor> bind;
    bind.reserve(8);
    bind.emplace(gs.x_id, x);
    bind.emplace(gs.y_id, y);
    bind.emplace(gs.pids[0], m.fc1->weight());
    bind.emplace(gs.pids[1], m.fc1->bias());
    bind.emplace(gs.pids[2], m.fc2->weight());
    bind.emplace(gs.pids[3], m.fc2->bias());
    bind.emplace(gs.grad_seed_id, loss_seed);
    auto outs = js.engine->invoke(bind);

    opt.zero_grad();
    std::array<Tensor, 4> handles = {
        m.fc1->weight(), m.fc1->bias(),
        m.fc2->weight(), m.fc2->bias()};
    for (std::size_t p = 0; p < handles.size(); ++p) {
      auto* am = handles[p].mutable_autograd_meta();
      am->grad = outs[1 + p];
    }
    opt.step();
  };
  for (int i = 0; i < warmup; ++i) run();
  const auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) run();
  return secs(t0, Clock::now()) / iters;
}
#endif  // TESSERACT_ENABLE_MLIR

void run_forward_case(const Case& c) {
  Mlp m = make_mlp(c.in_features, c.hidden, c.out_classes);
  auto x = Tensor::empty({c.batch, c.in_features}, DType::Float32);
  fill_random(x, 0xABCDEF);

  const double eager_ms = bench_eager_forward(m, x, c.warmup, c.iters) * 1e3;

  GraphForward gf = capture_forward(m, x);
  const double graph_ms = bench_graph_forward(gf, m, x, c.warmup, c.iters) * 1e3;

#ifdef TESSERACT_ENABLE_MLIR
  JitForward jf = build_jit_forward(gf);
  const double jit_ms = bench_jit_forward(gf, jf, m, x, c.warmup, c.iters) * 1e3;
  std::printf("%-10s  B=%-3d  D=%d->%d->%d  "
              "eager=%6.3f ms  interp=%6.3f ms (%4.2fx)  "
              "jit=%6.3f ms (%4.2fx)  "
              "(capture=%5.2f ms, jit-build=%6.1f ms)\n",
              c.label, c.batch, c.in_features, c.hidden, c.out_classes,
              eager_ms,
              graph_ms, graph_ms / eager_ms,
              jit_ms,   jit_ms   / eager_ms,
              gf.capture_secs * 1e3, jf.build_secs * 1e3);
#else
  std::printf("%-10s  B=%-3d  D=%d->%d->%d  eager=%6.3f ms  "
              "interp=%6.3f ms  (interp/eager=%5.2fx, capture=%5.2f ms)\n",
              c.label, c.batch, c.in_features, c.hidden, c.out_classes,
              eager_ms, graph_ms, graph_ms / eager_ms,
              gf.capture_secs * 1e3);
#endif
}

void run_step_case(const Case& c) {
  // Fresh models — each path gets its own parameters so stateful
  // optimizers don't leak between runs.
  Mlp me = make_mlp(c.in_features, c.hidden, c.out_classes);
  Mlp mg = make_mlp(c.in_features, c.hidden, c.out_classes);
#ifdef TESSERACT_ENABLE_MLIR
  Mlp mj = make_mlp(c.in_features, c.hidden, c.out_classes);
#endif
  auto x = Tensor::empty({c.batch, c.in_features}, DType::Float32);
  fill_random(x, 0xABCDEF);
  auto y = Tensor::empty({c.batch}, DType::Int64);
  fill_random_labels(y, c.out_classes, 0x123456);

  const double eager_ms = bench_eager_train_step(me, x, y, c.warmup, c.iters) * 1e3;

  GraphStep gs = capture_train_step(mg, x, y);
  const double graph_ms = bench_graph_train_step(gs, mg, x, y, c.warmup, c.iters) * 1e3;

#ifdef TESSERACT_ENABLE_MLIR
  // The JIT needs its own capture because `build_backward` mutates the
  // graph; reusing `gs.graph` would be fine but we keep a separate Mlp
  // so the two paths train independent copies of the model (stateful
  // Adam state otherwise bleeds between runs and skews per-step timing
  // after the first few iterations).
  GraphStep gsj = capture_train_step(mj, x, y);
  JitStep js = build_jit_step(gsj);
  const double jit_ms = bench_jit_train_step(gsj, js, mj, x, y,
                                             c.warmup, c.iters) * 1e3;
  std::printf("%-10s  B=%-3d  D=%d->%d->%d  "
              "eager=%6.3f ms  interp=%6.3f ms (%4.2fx)  "
              "jit=%6.3f ms (%4.2fx)  "
              "(capture=%5.2f ms, build_bwd=%5.2f ms, "
              "jit-build=%6.1f ms)\n",
              c.label, c.batch, c.in_features, c.hidden, c.out_classes,
              eager_ms,
              graph_ms, graph_ms / eager_ms,
              jit_ms,   jit_ms   / eager_ms,
              gs.capture_secs * 1e3, gs.build_backward_secs * 1e3,
              js.build_secs * 1e3);
#else
  std::printf("%-10s  B=%-3d  D=%d->%d->%d  eager=%6.3f ms  "
              "interp=%6.3f ms  (interp/eager=%5.2fx, capture=%5.2f ms, "
              "build_bwd=%5.2f ms)\n",
              c.label, c.batch, c.in_features, c.hidden, c.out_classes,
              eager_ms, graph_ms, graph_ms / eager_ms,
              gs.capture_secs * 1e3, gs.build_backward_secs * 1e3);
#endif
}

}  // namespace

int main() {
  const Case cases[] = {
      // Matches examples/mnist.cpp exactly.
      {"mnist",     64, 784, 128,  10, 5, 200},
      // Smaller: shape-registration + dispatch overhead dominates.
      {"tiny",      32,  16,  32,   8, 5, 400},
      // Wider hidden: per-op work grows, interpreter overhead fades.
      {"wide",      64, 512, 512, 128, 3, 100},
  };

  std::printf("========== forward-only ==========\n");
  for (const auto& c : cases) run_forward_case(c);

  std::printf("\n========== train step (fwd + bwd + Adam) ==========\n");
  for (const auto& c : cases) run_step_case(c);
  return 0;
}

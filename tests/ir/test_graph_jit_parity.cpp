// M1I.2.b-Phase-2 — full-training parity between the graph interpreter
// and the MLIR JIT engine.
//
// The Phase-1 test suite (`test_ir_jit_forward.cpp`) covers individual
// lowerings on short synthetic graphs. This test closes the loop on the
// roadmap's M1I.2.b success criterion: a complete training step
// (forward + `graph::build_backward` + Adam step) should produce
// bit-parallel numerics whether it's executed by the C++ graph
// interpreter or by `ir::JitEngine`. The test reuses the exact Gaussian-
// mixture dataset + MLP + batch schedule from
// `tests/graph/test_graph_train_parity.cpp` so any drift between the
// engines shows up as a clear delta relative to the interpreter's
// already-validated loss curve.
//
// Strategy:
//   1. Build a 2-layer MLP + ReLU and snapshot its initial parameters.
//   2. Run N training steps through the graph interpreter and record
//      each step's loss.
//   3. Restore the snapshot, rebuild Adam, rebuild `JitEngine` against
//      the (re-captured) post-backward graph, and run the same schedule.
//   4. Assert the two loss curves agree to within 1e-4 absolute (the
//      JIT's linalg.matmul / linalg.reduce loop nest picks a slightly
//      different accumulation order than the hand-written eager kernel;
//      the drift is deterministic but non-zero in float32).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include "tesseract/autograd/AutogradMeta.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Autograd.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/graph/Interpreter.hpp"
#include "tesseract/ir/JitEngine.hpp"
#include "tesseract/nn/Activation.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Sequential.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/optim/Adam.hpp"

using namespace tesseract;

namespace {

struct Dataset {
  Tensor x;
  Tensor y;
};

Dataset make_gaussian_mixture(int per_class, uint64_t seed) {
  constexpr int C = 3;
  const int N = C * per_class;
  auto x = Tensor::empty({N, 2}, DType::Float32);
  auto y = Tensor::empty({N}, DType::Int64);
  float* px = x.data_ptr<float>();
  int64_t* py = y.data_ptr<int64_t>();

  std::mt19937_64 rng(seed);
  std::normal_distribution<float> noise(0.0f, 0.35f);
  const float centers[C][2] = {{-2.0f, -2.0f}, {2.0f, 0.0f}, {0.0f, 2.5f}};
  int idx = 0;
  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < per_class; ++i) {
      px[2 * idx + 0] = centers[c][0] + noise(rng);
      px[2 * idx + 1] = centers[c][1] + noise(rng);
      py[idx] = c;
      ++idx;
    }
  }
  return {std::move(x), std::move(y)};
}

struct Batch {
  Tensor x;
  Tensor y;
};

std::vector<Batch> make_batch_schedule(const Dataset& ds, int batch,
                                       int num_batches, uint64_t seed) {
  const int64_t N = ds.x.shape()[0];
  std::vector<int64_t> idx(static_cast<std::size_t>(N));
  std::iota(idx.begin(), idx.end(), 0);
  std::mt19937_64 rng(seed);

  std::vector<Batch> out;
  out.reserve(num_batches);
  const float* sx = ds.x.data_ptr<float>();
  const int64_t* sy = ds.y.data_ptr<int64_t>();
  int64_t cursor = N;
  for (int k = 0; k < num_batches; ++k) {
    if (cursor + batch > N) {
      std::shuffle(idx.begin(), idx.end(), rng);
      cursor = 0;
    }
    auto bx = Tensor::empty({batch, 2}, DType::Float32);
    auto by = Tensor::empty({batch}, DType::Int64);
    float* dx = bx.data_ptr<float>();
    int64_t* dy = by.data_ptr<int64_t>();
    for (int b = 0; b < batch; ++b) {
      const int64_t r = idx[cursor + b];
      dx[2 * b + 0] = sx[2 * r + 0];
      dx[2 * b + 1] = sx[2 * r + 1];
      dy[b] = sy[r];
    }
    cursor += batch;
    out.push_back({std::move(bx), std::move(by)});
  }
  return out;
}

struct Mlp {
  std::shared_ptr<nn::Linear> fc1;
  std::shared_ptr<nn::ReLU> relu;
  std::shared_ptr<nn::Linear> fc2;
  std::shared_ptr<nn::Sequential> seq;
};

Mlp make_mlp(int in_features, int hidden, int out_classes) {
  auto fc1 = std::make_shared<nn::Linear>(in_features, hidden);
  auto relu = std::make_shared<nn::ReLU>();
  auto fc2 = std::make_shared<nn::Linear>(hidden, out_classes);
  auto seq = std::make_shared<nn::Sequential>(
      std::initializer_list<std::shared_ptr<nn::Module>>{fc1, relu, fc2});
  return {fc1, relu, fc2, seq};
}

std::array<Tensor, 4> snapshot_params(const Mlp& m) {
  return {m.fc1->weight().clone(), m.fc1->bias().clone(),
          m.fc2->weight().clone(), m.fc2->bias().clone()};
}

void copy_into(Tensor& dst, const Tensor& src) {
  REQUIRE(dst.shape() == src.shape());
  REQUIRE(dst.dtype() == src.dtype());
  std::memcpy(dst.raw_data(), src.raw_data(),
              static_cast<std::size_t>(dst.numel()) * dst.itemsize());
  if (auto* am = dst.mutable_autograd_meta()) {
    am->grad = Tensor{};
  }
}

void restore_params(Mlp& m, const std::array<Tensor, 4>& snap) {
  Tensor w1 = m.fc1->weight();
  Tensor b1 = m.fc1->bias();
  Tensor w2 = m.fc2->weight();
  Tensor b2 = m.fc2->bias();
  copy_into(w1, snap[0]);
  copy_into(b1, snap[1]);
  copy_into(w2, snap[2]);
  copy_into(b2, snap[3]);
}

// Capture the forward + backward graph and return it together with the
// value ids the caller needs to bind every step.
struct CapturedProgram {
  graph::Graph g;
  graph::ValueId x_id;
  graph::ValueId y_id;
  std::array<graph::ValueId, 4> param_ids;
  graph::BackwardResult bwd;
};

CapturedProgram capture_and_backward(Mlp& m, const Batch& warm) {
  CapturedProgram prog;
  {
    graph::GraphScope scope;
    prog.x_id = graph::bind_input(warm.x, "x");
    prog.y_id = graph::bind_input(warm.y, "targets");
    const std::array<Tensor, 4> params = {
        m.fc1->weight(), m.fc1->bias(),
        m.fc2->weight(), m.fc2->bias()};
    const std::array<const char*, 4> pnames = {
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
    for (std::size_t i = 0; i < params.size(); ++i) {
      prog.param_ids[i] = graph::bind_param(params[i], pnames[i]);
    }
    Tensor logits = m.seq->forward(warm.x);
    Tensor loss = ops::cross_entropy_with_logits(logits, warm.y);
    graph::mark_output(loss);
    prog.g = std::move(scope.graph_);
  }
  prog.bwd = graph::build_backward(prog.g);
  REQUIRE(prog.bwd.cotangents.size() == 1);
  REQUIRE(prog.bwd.dparams.size() == prog.param_ids.size());
  return prog;
}

// Generic training loop parametrised by a `run_fn` that turns a
// per-step binding into the output list. We reuse this for both the
// interpreter and the JIT paths to keep the optimizer / bookkeeping
// code identical between them.
template <typename RunFn>
std::vector<double> train_with(Mlp& m, const CapturedProgram& prog,
                               const std::vector<Batch>& sched, double lr,
                               RunFn&& run_fn) {
  optim::Adam opt(m.seq->parameters(), lr);
  Tensor loss_seed = Tensor::ones({}, DType::Float32);

  std::vector<double> losses;
  losses.reserve(sched.size());
  for (const auto& b : sched) {
    std::unordered_map<graph::ValueId, Tensor> bind;
    bind.emplace(prog.x_id, b.x);
    bind.emplace(prog.y_id, b.y);
    bind.emplace(prog.param_ids[0], m.fc1->weight());
    bind.emplace(prog.param_ids[1], m.fc1->bias());
    bind.emplace(prog.param_ids[2], m.fc2->weight());
    bind.emplace(prog.param_ids[3], m.fc2->bias());
    bind.emplace(prog.bwd.cotangents[0], loss_seed);

    auto outs = run_fn(bind);
    REQUIRE(outs.size() == 1 + prog.param_ids.size());

    opt.zero_grad();
    std::array<Tensor, 4> handles = {m.fc1->weight(), m.fc1->bias(),
                                      m.fc2->weight(), m.fc2->bias()};
    for (std::size_t p = 0; p < handles.size(); ++p) {
      auto* am = handles[p].mutable_autograd_meta();
      am->grad = outs[1 + p];
    }
    opt.step();
    // `outs[0]` is a `Tensor` whose type is dependent on `RunFn`'s
    // deduced return, so we need `.template` to disambiguate.
    losses.push_back(
        static_cast<double>(*outs[0].template data_ptr<float>()));
  }
  return losses;
}

}  // namespace

TEST_CASE("jit: training loop matches graph interpreter loss curve",
          "[ir][jit][parity]") {
  auto ds = make_gaussian_mixture(/*per_class=*/64, /*seed=*/17);
  constexpr int kBatch = 32;
  constexpr int kSteps = 40;
  constexpr double kLr = 5e-2;
  auto sched = make_batch_schedule(ds, kBatch, kSteps, /*seed=*/11);

  // --- 1. Interpreter pass (reference). --- //
  Mlp m = make_mlp(/*in=*/2, /*hidden=*/16, /*out=*/3);
  const auto snap = snapshot_params(m);

  CapturedProgram interp_prog = capture_and_backward(m, sched.front());
  const auto interp_losses = train_with(
      m, interp_prog, sched, kLr,
      [&](const std::unordered_map<graph::ValueId, Tensor>& bind) {
        return graph::run(interp_prog.g, bind);
      });

  // --- 2. Rewind the model, rebuild the program (`build_backward`
  //        mutates the input graph irreversibly), and run the same
  //        schedule through the JIT. --- //
  restore_params(m, snap);
  CapturedProgram jit_prog = capture_and_backward(m, sched.front());
  ir::JitEngine jit(jit_prog.g);

  const auto jit_losses = train_with(
      m, jit_prog, sched, kLr,
      [&](const std::unordered_map<graph::ValueId, Tensor>& bind) {
        return jit.invoke(bind);
      });

  REQUIRE(interp_losses.size() == jit_losses.size());
  double max_abs_delta = 0.0;
  for (std::size_t i = 0; i < interp_losses.size(); ++i) {
    const double d = std::abs(interp_losses[i] - jit_losses[i]);
    max_abs_delta = std::max(max_abs_delta, d);
    INFO("step " << i << "  interp=" << interp_losses[i]
                 << "  jit=" << jit_losses[i] << "  |Δ|=" << d);
    // The two paths run the same arithmetic but through different loop
    // nests (a hand-written C++ kernel vs. a linalg.generic expanded to
    // scf.for). Reduction order can reshuffle in float32, so 1e-4 is
    // the practical upper bound — still tight enough to catch any real
    // numerical bug (e.g. wrong broadcast semantics or a stale param
    // binding).
    REQUIRE(d < 1e-4);
  }
  INFO("max |interp - jit| over " << interp_losses.size()
                                  << " steps = " << max_abs_delta);

  // Sanity: training actually converged under both engines.
  REQUIRE(interp_losses.back() < interp_losses.front() * 0.5);
  REQUIRE(jit_losses.back() < jit_losses.front() * 0.5);
}

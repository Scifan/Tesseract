// M1I.2 — end-to-end parity test: graph-mode training (capture +
// `graph::build_backward` + `graph::run` + Adam) must reproduce eager
// mode's loss curve bit-for-bit on a shape-specialized MLP.
//
// Why this exists: the graph interpreter is already unit-tested for
// gradient correctness (tests/graph/test_graph_autograd.cpp), but the
// real question the roadmap asks is whether a full *training loop*
// through the captured graph produces the same numerics as eager. This
// test answers yes/no on a tiny synthetic classification problem so it
// runs under ctest in under a second.
//
// Strategy:
//   1. Build a 2-layer MLP + ReLU and snapshot its initial parameter
//      tensors via `Tensor::clone()`.
//   2. Drive an eager training loop over a fixed sequence of batches
//      (identical seed + identical shuffle order) and record the
//      per-step loss.
//   3. Restore parameters from the snapshot, rebuild the optimizer
//      (so Adam moments reset), capture the forward graph, extend it
//      with `graph::build_backward`, and drive the same sequence of
//      batches through `graph::run`. Record the per-step loss again.
//   4. Assert the two loss curves match to within a tight numerical
//      tolerance. On CPU with deterministic ops the match is
//      essentially bit-exact (< 1e-6).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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

using namespace tesseract;

namespace {

// Small, well-separated 3-class Gaussian mixture. Identical to the
// nn/test_mnist_smoke.cpp dataset so readers can cross-reference; we
// duplicate it here rather than adding a shared test fixture because
// the two files live in independent Catch2 executables.
struct Dataset {
  Tensor x;  // [N, 2] float
  Tensor y;  // [N]    int64
};

Dataset make_gaussian_mixture(int per_class, uint64_t seed) {
  constexpr int C = 3;
  const int N = C * per_class;
  auto x = Tensor::empty({N, 2}, DType::Float32);
  auto y = Tensor::empty({N}, DType::Int64);
  float*   px = x.data_ptr<float>();
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

// Precompute the full shuffled batch schedule so that eager and graph
// paths consume exactly the same (batch_x, batch_y) pairs in exactly the
// same order. Returns a flat list of batches.
struct Batch {
  Tensor x;  // [B, 2]
  Tensor y;  // [B]
};

std::vector<Batch> make_batch_schedule(const Dataset& ds, int batch,
                                       int num_batches, uint64_t seed) {
  const int64_t N = ds.x.shape()[0];
  std::vector<int64_t> idx(static_cast<std::size_t>(N));
  std::iota(idx.begin(), idx.end(), 0);
  std::mt19937_64 rng(seed);

  std::vector<Batch> out;
  out.reserve(num_batches);
  const float*   sx = ds.x.data_ptr<float>();
  const int64_t* sy = ds.y.data_ptr<int64_t>();
  int64_t cursor = N;  // force reshuffle on first use
  for (int k = 0; k < num_batches; ++k) {
    if (cursor + batch > N) {
      std::shuffle(idx.begin(), idx.end(), rng);
      cursor = 0;
    }
    auto bx = Tensor::empty({batch, 2}, DType::Float32);
    auto by = Tensor::empty({batch}, DType::Int64);
    float*   dx = bx.data_ptr<float>();
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

// Copy-assign a tensor's underlying storage from `src`. Both tensors
// must be contiguous, same shape and dtype. We use this to reset a
// model's parameters to a saved snapshot before re-running training.
void copy_into(Tensor& dst, const Tensor& src) {
  REQUIRE(dst.shape() == src.shape());
  REQUIRE(dst.dtype() == src.dtype());
  std::memcpy(dst.raw_data(), src.raw_data(),
              static_cast<std::size_t>(dst.numel()) * dst.itemsize());
  // Blow away any stale gradient tape / .grad attached to dst: we are
  // about to feed it into a fresh Adam run.
  if (auto* am = dst.mutable_autograd_meta()) {
    am->grad = Tensor{};
  }
}

struct Mlp {
  std::shared_ptr<nn::Linear>    fc1;
  std::shared_ptr<nn::ReLU>      relu;
  std::shared_ptr<nn::Linear>    fc2;
  std::shared_ptr<nn::Sequential> seq;
};

Mlp make_mlp(int in_features, int hidden, int out_classes) {
  auto fc1  = std::make_shared<nn::Linear>(in_features, hidden);
  auto relu = std::make_shared<nn::ReLU>();
  auto fc2  = std::make_shared<nn::Linear>(hidden, out_classes);
  auto seq  = std::make_shared<nn::Sequential>(
      std::initializer_list<std::shared_ptr<nn::Module>>{fc1, relu, fc2});
  return {fc1, relu, fc2, seq};
}

std::array<Tensor, 4> snapshot_params(const Mlp& m) {
  return {m.fc1->weight().clone(), m.fc1->bias().clone(),
          m.fc2->weight().clone(), m.fc2->bias().clone()};
}

void restore_params(Mlp& m, const std::array<Tensor, 4>& snap) {
  // `weight()` / `bias()` return `const Tensor&`; we take a copy of the
  // Tensor handle so we can call non-const methods on it. The copy still
  // shares storage with the module's parameter, so writes land in the
  // right buffer.
  Tensor w1 = m.fc1->weight();
  Tensor b1 = m.fc1->bias();
  Tensor w2 = m.fc2->weight();
  Tensor b2 = m.fc2->bias();
  copy_into(w1, snap[0]);
  copy_into(b1, snap[1]);
  copy_into(w2, snap[2]);
  copy_into(b2, snap[3]);
}

// Run `num_steps` eager training steps and return the per-step loss
// sequence. Mutates `m`'s parameters in place.
std::vector<double> train_eager(Mlp& m, const std::vector<Batch>& sched,
                                double lr) {
  optim::Adam opt(m.seq->parameters(), lr);
  std::vector<double> losses;
  losses.reserve(sched.size());
  for (const auto& b : sched) {
    opt.zero_grad();
    Tensor logits = m.seq->forward(b.x);
    Tensor loss = ops::cross_entropy_with_logits(logits, b.y);
    Engine::backward(loss);
    opt.step();
    losses.push_back(static_cast<double>(*loss.data_ptr<float>()));
  }
  return losses;
}

// Capture a forward graph on a warm-up batch shaped like the real
// batches, extend with backward, then run `num_steps` through the
// interpreter. Mutates `m`'s parameters via Adam.
std::vector<double> train_graph(Mlp& m, const std::vector<Batch>& sched,
                                double lr) {
  REQUIRE_FALSE(sched.empty());

  graph::Graph captured;
  graph::ValueId x_id = graph::kInvalidValueId;
  graph::ValueId y_id = graph::kInvalidValueId;
  std::array<graph::ValueId, 4> pids{};
  {
    graph::GraphScope scope;
    const Batch& warm = sched.front();
    x_id = graph::bind_input(warm.x, "x");
    y_id = graph::bind_input(warm.y, "targets");
    const std::array<Tensor, 4> params = {
        m.fc1->weight(), m.fc1->bias(),
        m.fc2->weight(), m.fc2->bias()};
    const std::array<const char*, 4> pnames = {
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
    for (std::size_t i = 0; i < params.size(); ++i) {
      pids[i] = graph::bind_param(params[i], pnames[i]);
    }
    Tensor logits = m.seq->forward(warm.x);
    Tensor loss = ops::cross_entropy_with_logits(logits, warm.y);
    graph::mark_output(loss);
    captured = std::move(scope.graph_);
  }

  graph::BackwardResult bwd = graph::build_backward(captured);
  REQUIRE(bwd.cotangents.size() == 1);
  REQUIRE(bwd.dparams.size() == pids.size());

  // Reset Adam state because the warm-up forward left the module's
  // autograd tape populated; starting from a clean optimizer makes the
  // two runs start identically.
  optim::Adam opt(m.seq->parameters(), lr);

  Tensor loss_seed = Tensor::ones({}, DType::Float32);
  std::vector<double> losses;
  losses.reserve(sched.size());
  for (const auto& b : sched) {
    std::unordered_map<graph::ValueId, Tensor> bind;
    bind.emplace(x_id, b.x);
    bind.emplace(y_id, b.y);
    bind.emplace(pids[0], m.fc1->weight());
    bind.emplace(pids[1], m.fc1->bias());
    bind.emplace(pids[2], m.fc2->weight());
    bind.emplace(pids[3], m.fc2->bias());
    bind.emplace(bwd.cotangents[0], loss_seed);

    auto outs = graph::run(captured, bind);
    REQUIRE(outs.size() == 1 + pids.size());

    opt.zero_grad();
    std::array<Tensor, 4> handles = {
        m.fc1->weight(), m.fc1->bias(),
        m.fc2->weight(), m.fc2->bias()};
    for (std::size_t p = 0; p < handles.size(); ++p) {
      auto* am = handles[p].mutable_autograd_meta();
      am->grad = outs[1 + p];
    }
    opt.step();
    losses.push_back(static_cast<double>(*outs[0].data_ptr<float>()));
  }
  return losses;
}

}  // namespace

TEST_CASE("graph-mode training reproduces eager loss curve",
          "[graph][m1i]") {
  auto ds = make_gaussian_mixture(/*per_class=*/64, /*seed=*/17);
  constexpr int kBatch = 32;
  constexpr int kSteps = 40;
  constexpr double kLr = 5e-2;
  auto sched = make_batch_schedule(ds, kBatch, kSteps, /*seed=*/11);

  Mlp m = make_mlp(/*in=*/2, /*hidden=*/16, /*out=*/3);
  const auto snap = snapshot_params(m);

  const std::vector<double> eager_losses = train_eager(m, sched, kLr);

  // Rewind the model to its pre-training state and re-run the same
  // schedule through the graph interpreter.
  restore_params(m, snap);
  const std::vector<double> graph_losses = train_graph(m, sched, kLr);

  REQUIRE(eager_losses.size() == graph_losses.size());
  double max_abs_delta = 0.0;
  for (std::size_t i = 0; i < eager_losses.size(); ++i) {
    const double d = std::abs(eager_losses[i] - graph_losses[i]);
    max_abs_delta = std::max(max_abs_delta, d);
    INFO("step " << i << "  eager=" << eager_losses[i]
         << "  graph=" << graph_losses[i] << "  |Δ|=" << d);
    // The two paths go through the same kernels; drift can only come
    // from floating-point associativity differences (gradient
    // accumulation order). 1e-5 is already 10x looser than what we
    // observed locally.
    REQUIRE(d < 1e-5);
  }
  INFO("max |eager - graph| over " << eager_losses.size()
       << " steps = " << max_abs_delta);

  // Sanity: training actually made progress under both modes.
  REQUIRE(eager_losses.back() < eager_losses.front() * 0.5);
  REQUIRE(graph_losses.back() < graph_losses.front() * 0.5);
}

// M4 Track B3 (B-043) — tensor-parallelism CPU parity.
//
// A TP=N sharding of a dense layer must reproduce the dense output under the
// single-process `SimCommBackend`:
//
//   * column-parallel (split output features) + all-gather  -> exact
//   * row-parallel (split input features) + all-reduce(sum) -> within FP
//     summation-reassociation tolerance
//   * a Megatron SwiGLU MLP (column gate/up -> local silu*up -> row down,
//     one all-reduce) -> matches the dense MLP
//
// These pin the correctness of the sharding transform before a real NCCL
// backend (gated tail) is introduced — the parallel modules are unchanged
// between the simulator and NCCL, only the `CommBackend` differs.

#include <cmath>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/distributed/CommBackend.hpp"
#include "tesseract/distributed/TensorParallel.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Indexing.hpp"
#include "tesseract/ops/Reduction.hpp"

using namespace tesseract;
using namespace tesseract::distributed;

namespace {

Tensor random_input(int64_t rows, int64_t cols, uint64_t seed) {
  auto x = Tensor::empty({rows, cols}, DType::Float32);
  float* p = x.data_ptr<float>();
  uint64_t s = seed;
  for (int64_t i = 0; i < rows * cols; ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u =
        static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFULL) /
        9007199254740992.0;
    p[i] = static_cast<float>(2.0 * u - 1.0);
  }
  return x;
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
  REQUIRE(a.numel() == b.numel());
  const Tensor ac = a.contiguous();
  const Tensor bc = b.contiguous();
  const float* pa = ac.data_ptr<float>();
  const float* pb = bc.data_ptr<float>();
  double m = 0.0;
  for (int64_t i = 0; i < ac.numel(); ++i) {
    m = std::max(m, std::abs(static_cast<double>(pa[i] - pb[i])));
  }
  return m;
}

}  // namespace

TEST_CASE("ColumnParallelLinear: TP=2 all-gather equals dense (exact)",
          "[distributed][tp]") {
  const int64_t in = 16, out = 24;
  nn::Linear dense(in, out, /*use_bias=*/true);
  Tensor x = random_input(5, in, 0xABCDEF01);

  auto comm = std::make_shared<SimCommBackend>(2);
  auto col = ColumnParallelLinear::from_dense(dense, comm,
                                              /*gather_output=*/true);

  Tensor y_dense = dense.forward(x);
  Tensor y_tp = col->forward(x);
  REQUIRE(y_tp.shape() == y_dense.shape());
  // Output features partition cleanly: each block is an independent matmul,
  // so concat is bit-identical to the dense forward.
  REQUIRE(max_abs_diff(y_tp, y_dense) == 0.0);
}

TEST_CASE("RowParallelLinear: TP=2 all-reduce equals dense (FP tol)",
          "[distributed][tp]") {
  const int64_t in = 16, out = 12;
  nn::Linear dense(in, out, /*use_bias=*/true);
  Tensor x = random_input(5, in, 0x12345678);

  auto comm = std::make_shared<SimCommBackend>(2);
  auto row = RowParallelLinear::from_dense(dense, comm);

  Tensor y_dense = dense.forward(x);
  Tensor y_tp = row->forward(x);
  REQUIRE(y_tp.shape() == y_dense.shape());
  // Partial-sum reassociation across ranks can perturb the last ULPs.
  REQUIRE(max_abs_diff(y_tp, y_dense) < 1e-5);
}

TEST_CASE("Megatron SwiGLU MLP: TP=2 matches dense (one all-reduce)",
          "[distributed][tp]") {
  const int64_t d_model = 16, d_ff = 32;
  nn::Linear gate(d_model, d_ff, /*use_bias=*/false);
  nn::Linear up(d_model, d_ff, /*use_bias=*/false);
  nn::Linear down(d_ff, d_model, /*use_bias=*/false);

  Tensor x = random_input(4, d_model, 0xDEADBEEF);

  // Dense reference: down(silu(gate(x)) * up(x)).
  Tensor h_dense =
      down.forward(ops::swiglu_silu_gate(gate.forward(x), up.forward(x)));

  // TP=2: gate/up are column-parallel (output kept sharded), the elementwise
  // SwiGLU runs per-rank on matching shards, down is row-parallel consuming
  // the sharded hidden directly -> exactly one all-reduce.
  auto comm = std::make_shared<SimCommBackend>(2);
  auto gate_cp = ColumnParallelLinear::from_dense(gate, comm,
                                                  /*gather_output=*/false);
  auto up_cp = ColumnParallelLinear::from_dense(up, comm,
                                                /*gather_output=*/false);
  auto down_rp = RowParallelLinear::from_dense(down, comm);

  std::vector<Tensor> gate_sh = gate_cp->forward_shards(x);
  std::vector<Tensor> up_sh = up_cp->forward_shards(x);
  std::vector<Tensor> hidden_sh;
  hidden_sh.reserve(gate_sh.size());
  for (std::size_t r = 0; r < gate_sh.size(); ++r) {
    hidden_sh.push_back(ops::swiglu_silu_gate(gate_sh[r], up_sh[r]));
  }
  Tensor h_tp = down_rp->forward_shards(hidden_sh);

  REQUIRE(h_tp.shape() == h_dense.shape());
  REQUIRE(max_abs_diff(h_tp, h_dense) < 1e-5);
}

TEST_CASE("ColumnParallelLinear: TP=3 all-gather equals dense (exact)",
          "[distributed][tp]") {
  const int64_t in = 16, out = 24;  // 24 % 3 == 0
  nn::Linear dense(in, out, /*use_bias=*/true);
  Tensor x = random_input(5, in, 0x33333333);

  auto comm = std::make_shared<SimCommBackend>(3);
  auto col = ColumnParallelLinear::from_dense(dense, comm,
                                              /*gather_output=*/true);

  Tensor y_dense = dense.forward(x);
  Tensor y_tp = col->forward(x);
  REQUIRE(y_tp.shape() == y_dense.shape());
  REQUIRE(max_abs_diff(y_tp, y_dense) == 0.0);
}

TEST_CASE("Megatron SwiGLU MLP: TP=3 matches dense (one all-reduce)",
          "[distributed][tp]") {
  const int64_t d_model = 24, d_ff = 36;  // both % 3 == 0
  nn::Linear gate(d_model, d_ff, /*use_bias=*/false);
  nn::Linear up(d_model, d_ff, /*use_bias=*/false);
  nn::Linear down(d_ff, d_model, /*use_bias=*/false);

  Tensor x = random_input(4, d_model, 0xCAFEF00D);

  Tensor h_dense =
      down.forward(ops::swiglu_silu_gate(gate.forward(x), up.forward(x)));

  auto comm = std::make_shared<SimCommBackend>(3);
  auto gate_cp = ColumnParallelLinear::from_dense(gate, comm,
                                                  /*gather_output=*/false);
  auto up_cp = ColumnParallelLinear::from_dense(up, comm,
                                                /*gather_output=*/false);
  auto down_rp = RowParallelLinear::from_dense(down, comm);

  std::vector<Tensor> gate_sh = gate_cp->forward_shards(x);
  std::vector<Tensor> up_sh = up_cp->forward_shards(x);
  std::vector<Tensor> hidden_sh;
  hidden_sh.reserve(gate_sh.size());
  for (std::size_t r = 0; r < gate_sh.size(); ++r) {
    hidden_sh.push_back(ops::swiglu_silu_gate(gate_sh[r], up_sh[r]));
  }
  Tensor h_tp = down_rp->forward_shards(hidden_sh);

  REQUIRE(h_tp.shape() == h_dense.shape());
  REQUIRE(max_abs_diff(h_tp, h_dense) < 1e-5);
}

// Backward parity: a TP-sharded layer must produce gradients whose
// re-assembly equals the dense layer's gradient. This pins that the sharding
// transform is correct on the *backward* path, not just forward — the missing
// exit-bar item from issue.md B3.
TEST_CASE("ColumnParallelLinear: TP=2 backward grads reassemble to dense",
          "[distributed][tp]") {
  const int64_t in = 16, out = 24;
  nn::Linear dense(in, out, /*use_bias=*/false);
  Tensor x = random_input(5, in, 0x0BADF00D);

  auto comm = std::make_shared<SimCommBackend>(2);
  auto col = ColumnParallelLinear::from_dense(dense, comm,
                                              /*gather_output=*/true);

  // Dense reference grad.
  Tensor y_dense = dense.forward(x);
  Tensor loss_dense = ops::sum(y_dense);
  Engine::backward(loss_dense);
  Tensor g_dense = dense.weight().grad().contiguous();  // [out, in]

  // TP forward/backward; each shard's grad is [out/2, in]. Concatenated along
  // the output dim they must equal the dense weight grad (column-parallel
  // splits output features, so gradients partition exactly).
  Tensor y_tp = col->forward(x);
  Tensor loss_tp = ops::sum(y_tp);
  Engine::backward(loss_tp);
  std::vector<Tensor> shard_grads;
  for (const auto& [name, p] : col->named_parameters()) {
    if (name.find("weight") != std::string::npos) {
      shard_grads.push_back(p.grad());
    }
  }
  REQUIRE(static_cast<int>(shard_grads.size()) == 2);
  Tensor g_tp = ops::cat(shard_grads, 0).contiguous();
  REQUIRE(g_tp.shape() == g_dense.shape());
  REQUIRE(max_abs_diff(g_tp, g_dense) == 0.0);
}

TEST_CASE("RowParallelLinear: TP=2 backward grads reassemble to dense",
          "[distributed][tp]") {
  const int64_t in = 16, out = 12;
  nn::Linear dense(in, out, /*use_bias=*/false);
  Tensor x = random_input(5, in, 0xFEEDBEEF);

  auto comm = std::make_shared<SimCommBackend>(2);
  auto row = RowParallelLinear::from_dense(dense, comm);

  Tensor y_dense = dense.forward(x);
  Tensor loss_dense = ops::sum(y_dense);
  Engine::backward(loss_dense);
  Tensor g_dense = dense.weight().grad().contiguous();  // [out, in]

  Tensor y_tp = row->forward(x);
  Tensor loss_tp = ops::sum(y_tp);
  Engine::backward(loss_tp);
  // Row-parallel splits input features, so shard r holds [out, in/2] and the
  // grads concatenate along the input dim.
  std::vector<Tensor> shard_grads;
  for (const auto& [name, p] : row->named_parameters()) {
    if (name.find("weight") != std::string::npos) {
      shard_grads.push_back(p.grad());
    }
  }
  REQUIRE(static_cast<int>(shard_grads.size()) == 2);
  Tensor g_tp = ops::cat(shard_grads, 1).contiguous();
  REQUIRE(g_tp.shape() == g_dense.shape());
  // Backward reuses the same matmuls as forward; partition is exact here.
  REQUIRE(max_abs_diff(g_tp, g_dense) < 1e-5);
}

namespace {

// Collect a TP module's weight / bias shard grads in rank order (the module
// registers them rank-major, mirroring the dense->shard split).
void collect_shard_grads(const nn::Module& m, std::vector<Tensor>& weights,
                         std::vector<Tensor>& biases) {
  for (const auto& [name, p] : m.named_parameters()) {
    if (name.find("weight") != std::string::npos) weights.push_back(p.grad());
    else if (name.find("bias") != std::string::npos) biases.push_back(p.grad());
  }
}

}  // namespace

// ---- TP=3 backward parity: weight + bias + INPUT gradients ----------------
// Extends the TP=2 weight-only checks to TP=3 and additionally pins the bias
// gradient and the gradient flowing back to the (replicated) input — the full
// backward contract the NCCL backend must also satisfy.
TEST_CASE("ColumnParallelLinear: TP=3 backward weight+bias+input grads",
          "[distributed][tp]") {
  const int64_t in = 24, out = 36;  // 36 % 3 == 0
  nn::Linear dense(in, out, /*use_bias=*/true);

  // Identical data, independent leaves so grads don't cross-contaminate.
  Tensor xd = random_input(5, in, 0x51515151);
  Tensor xt = random_input(5, in, 0x51515151);
  xd.set_requires_grad(true);
  xt.set_requires_grad(true);

  Tensor y_dense = dense.forward(xd);
  Engine::backward(ops::sum(y_dense));
  Tensor gw_dense = dense.weight().grad().contiguous();  // [out, in]
  Tensor gb_dense = dense.bias().grad().contiguous();    // [out]
  Tensor gx_dense = xd.grad().contiguous();              // [5, in]

  auto comm = std::make_shared<SimCommBackend>(3);
  auto col = ColumnParallelLinear::from_dense(dense, comm,
                                              /*gather_output=*/true);
  Tensor y_tp = col->forward(xt);
  Engine::backward(ops::sum(y_tp));

  std::vector<Tensor> w_sh, b_sh;
  collect_shard_grads(*col, w_sh, b_sh);
  REQUIRE(static_cast<int>(w_sh.size()) == 3);
  REQUIRE(static_cast<int>(b_sh.size()) == 3);
  // Column-parallel splits output features: weights/bias concat along dim 0.
  Tensor gw_tp = ops::cat(w_sh, 0).contiguous();
  Tensor gb_tp = ops::cat(b_sh, 0).contiguous();
  REQUIRE(gw_tp.shape() == gw_dense.shape());
  REQUIRE(max_abs_diff(gw_tp, gw_dense) == 0.0);
  REQUIRE(gb_tp.shape() == gb_dense.shape());
  REQUIRE(max_abs_diff(gb_tp, gb_dense) == 0.0);
  // Input grad is summed over the gathered output shards -> matches dense
  // within FP reassociation.
  REQUIRE(xt.grad().shape() == gx_dense.shape());
  REQUIRE(max_abs_diff(xt.grad(), gx_dense) < 1e-5);
}

TEST_CASE("RowParallelLinear: TP=3 backward weight+bias+input grads",
          "[distributed][tp]") {
  const int64_t in = 36, out = 12;  // 36 % 3 == 0
  nn::Linear dense(in, out, /*use_bias=*/true);

  Tensor xd = random_input(5, in, 0x62626262);
  Tensor xt = random_input(5, in, 0x62626262);
  xd.set_requires_grad(true);
  xt.set_requires_grad(true);

  Tensor y_dense = dense.forward(xd);
  Engine::backward(ops::sum(y_dense));
  Tensor gw_dense = dense.weight().grad().contiguous();  // [out, in]
  Tensor gb_dense = dense.bias().grad().contiguous();    // [out]
  Tensor gx_dense = xd.grad().contiguous();              // [5, in]

  auto comm = std::make_shared<SimCommBackend>(3);
  auto row = RowParallelLinear::from_dense(dense, comm);
  Tensor y_tp = row->forward(xt);
  Engine::backward(ops::sum(y_tp));

  std::vector<Tensor> w_sh, b_sh;
  collect_shard_grads(*row, w_sh, b_sh);
  REQUIRE(static_cast<int>(w_sh.size()) == 3);
  // Row-parallel splits input features: weights concat along dim 1.
  Tensor gw_tp = ops::cat(w_sh, 1).contiguous();
  REQUIRE(gw_tp.shape() == gw_dense.shape());
  REQUIRE(max_abs_diff(gw_tp, gw_dense) < 1e-5);
  // Bias is a single replicated tensor (added once after all-reduce); its grad
  // must equal the dense bias grad exactly.
  REQUIRE(static_cast<int>(b_sh.size()) == 1);
  REQUIRE(b_sh[0].shape() == gb_dense.shape());
  REQUIRE(max_abs_diff(b_sh[0], gb_dense) < 1e-5);
  // Input grad reassembles to the full [rows, in] dense grad.
  REQUIRE(xt.grad().shape() == gx_dense.shape());
  REQUIRE(max_abs_diff(xt.grad(), gx_dense) < 1e-5);
}

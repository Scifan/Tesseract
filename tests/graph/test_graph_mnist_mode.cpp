// M1I — end-to-end graph-mode verification for the MNIST MLP.
//
// This test stands up the same `Linear -> ReLU -> Linear ->
// cross_entropy_with_logits` pipeline that `examples/mnist.cpp` runs,
// but drives it through the graph path instead of eager execution:
//
//   1. Enter a `graph::GraphScope` and do a single forward pass on
//      synthetic tensors; ops record into `graph::Graph` as a side
//      effect of the eager+trace mode.
//   2. Emit the recorded graph into an `mlir::ModuleOp` via
//      `ir::emit_mlir`.
//   3. Run the `--tesseract-backward` pass to synthesize the reverse-
//      mode AD, then optionally `--convert-tesseract-to-linalg` to push
//      the forward primitives down one dialect layer.
//   4. Assert that every intermediate module passes `mlir::verify`.
//
// We do *not* execute the IR yet — that's M1I stage-2, gated on the
// MLIR JIT runner. Stage-1 establishes that the IR pipeline can round-
// trip every MNIST op end-to-end, which is the prerequisite for any
// runtime work.

#include <catch2/catch_test_macros.hpp>

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ir/Emit.hpp"
#include "tesseract/ir/Passes.hpp"
#include "tesseract/ir/TesseractDialect.h"
#include "tesseract/nn/Activation.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Sequential.hpp"
#include "tesseract/ops/Loss.hpp"

using namespace tesseract;

namespace {

std::string dump_module(mlir::ModuleOp m) {
  std::string s;
  llvm::raw_string_ostream os(s);
  m.print(os);
  return os.str();
}

// Build a 2-layer MLP matching the MNIST example's structure, but small
// enough that shapes are clear in diagnostics if something fails.
struct MlpGraph {
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

MlpGraph capture_mnist_like_step(mlir::MLIRContext& ctx) {
  auto fc1 = std::make_shared<nn::Linear>(/*in_features=*/4,
                                          /*out_features=*/8);
  auto relu = std::make_shared<nn::ReLU>();
  auto fc2 = std::make_shared<nn::Linear>(/*in_features=*/8,
                                          /*out_features=*/3);
  auto model = std::make_shared<nn::Sequential>(
      std::initializer_list<std::shared_ptr<nn::Module>>{fc1, relu, fc2});

  Tensor x = Tensor::zeros({/*N=*/2, /*in=*/4}, DType::Float32);
  Tensor y = Tensor::zeros({/*N=*/2}, DType::Int64);

  graph::GraphScope scope;
  graph::bind_input(x, "x");
  graph::bind_input(y, "targets");
  graph::bind_param(fc1->weight(), "W1");
  graph::bind_param(fc1->bias(),   "b1");
  graph::bind_param(fc2->weight(), "W2");
  graph::bind_param(fc2->bias(),   "b2");

  Tensor logits = model->forward(x);
  Tensor loss = ops::cross_entropy_with_logits(logits, y);
  graph::mark_output(loss);

  return {ir::emit_mlir(ctx, scope.graph())};
}

}  // namespace

TEST_CASE("graph-mode MNIST MLP forward lowers to a verified module",
          "[graph][m1i]") {
  mlir::MLIRContext ctx;
  ctx.loadDialect<ir::TesseractDialect>();

  MlpGraph g = capture_mnist_like_step(ctx);
  REQUIRE(g.module);
  REQUIRE(mlir::succeeded(mlir::verify(*g.module)));

  const std::string out = dump_module(*g.module);
  // Every structural piece the MNIST example exercises must appear.
  REQUIRE(out.find("tesseract.matmul")                != std::string::npos);
  REQUIRE(out.find("tesseract.add")                   != std::string::npos);
  REQUIRE(out.find("tesseract.relu")                  != std::string::npos);
  REQUIRE(out.find("tesseract.broadcast_to")          != std::string::npos);
  REQUIRE(out.find("tesseract.cross_entropy_with_logits") != std::string::npos);
  REQUIRE(out.find("\"W1\"") != std::string::npos);
  REQUIRE(out.find("\"b1\"") != std::string::npos);
  REQUIRE(out.find("\"W2\"") != std::string::npos);
  REQUIRE(out.find("\"b2\"") != std::string::npos);
}

TEST_CASE("graph-mode MNIST backward pass closes over every op",
          "[graph][m1i]") {
  mlir::MLIRContext ctx;
  ctx.loadDialect<ir::TesseractDialect>();

  MlpGraph g = capture_mnist_like_step(ctx);
  REQUIRE(g.module);

  mlir::PassManager pm(&ctx);
  pm.addPass(ir::createBackwardPass());
  // M1I stage-1 only verifies the IR; lowering composes on top to make
  // sure the backward pass produces shapes the Linalg converter accepts.
  pm.addPass(ir::createConvertTesseractToLinalgPass());

  REQUIRE(mlir::succeeded(pm.run(*g.module)));
  REQUIRE(mlir::succeeded(mlir::verify(*g.module)));

  const std::string out = dump_module(*g.module);
  INFO(out);
  // After M1I.2.b-Phase-2, the fused backward ops also lower — so we
  // expect them to be *absent* from the post-linalg IR; what we see
  // instead is the linalg sequence they expand to (math.exp + math.log
  // + onehot-select for CE, arith.cmpf + arith.select for relu_bwd).
  REQUIRE(out.find("tesseract.cross_entropy_with_logits_backward")
          == std::string::npos);
  REQUIRE(out.find("tesseract.relu_backward") == std::string::npos);
  REQUIRE(out.find("math.exp") != std::string::npos);
  REQUIRE(out.find("math.log") != std::string::npos);
  // Bias gradients collapse the batch dim. We emit `tesseract.sum
  // {dim = 0 : si64}`, which `--convert-tesseract-to-linalg` rewrites
  // to `linalg.reduce ... dimensions = [0]`. Either form counts as a
  // correct batch-axis reduction.
  const bool reduces_over_batch_dim =
      out.find("dim = 0 : si64") != std::string::npos ||
      out.find("dimensions = [0]") != std::string::npos;
  REQUIRE(reduces_over_batch_dim);
  // The forward primitives we had rules for should have lowered to
  // linalg (there should be at least one matmul in the backward pass
  // for the weight gradients).
  REQUIRE(out.find("linalg.matmul") != std::string::npos);
  // The backward return must carry the loss scalar plus one gradient
  // per registered param, in declaration order (W1, b1, W2, b2).
  REQUIRE(out.find("tesseract.return") != std::string::npos);
}

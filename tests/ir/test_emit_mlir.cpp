// Verifies the Graph → MLIR emitter (src/ir/Emit.cpp). Only compiled when
// TESSERACT_ENABLE_MLIR=ON.
//
// We build a few small graphs via `graph::GraphScope`, emit them with
// `ir::emit_mlir`, and check that (a) verify() succeeds (the emitter calls
// it too but we double-check), and (b) the printed textual form contains
// the mnemonics we expect.

#include <catch2/catch_test_macros.hpp>

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ir/Emit.hpp"
#include "tesseract/ir/TesseractDialect.h"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Reduction.hpp"

using namespace tesseract;

namespace {

std::string dump(mlir::ModuleOp m) {
  std::string s;
  llvm::raw_string_ostream os(s);
  m.print(os);
  return os.str();
}

}  // namespace

TEST_CASE("emit_mlir: elementwise chain", "[ir][emit]") {
  mlir::MLIRContext ctx;
  ctx.loadDialect<ir::TesseractDialect>();

  Tensor a = Tensor::zeros({2, 3}, DType::Float32);
  Tensor b = Tensor::zeros({2, 3}, DType::Float32);

  graph::GraphScope scope;
  graph::bind_input(a, "a");
  graph::bind_input(b, "b");

  Tensor c = ops::add(a, b);
  Tensor d = ops::mul(c, a);
  graph::mark_output(d);

  auto module = ir::emit_mlir(ctx, scope.graph());
  REQUIRE(module);
  REQUIRE(mlir::succeeded(mlir::verify(*module)));

  const std::string out = dump(*module);
  REQUIRE(out.find("tesseract.graph") != std::string::npos);
  REQUIRE(out.find("tesseract.function") != std::string::npos);
  REQUIRE(out.find("tesseract.add") != std::string::npos);
  REQUIRE(out.find("tesseract.mul") != std::string::npos);
  REQUIRE(out.find("tesseract.return") != std::string::npos);
  REQUIRE(out.find("tensor<2x3xf32>") != std::string::npos);
}

TEST_CASE("emit_mlir: matmul + relu chain with param", "[ir][emit]") {
  mlir::MLIRContext ctx;
  ctx.loadDialect<ir::TesseractDialect>();

  Tensor x = Tensor::zeros({4, 8}, DType::Float32);
  Tensor w = Tensor::zeros({8, 3}, DType::Float32);

  graph::GraphScope scope;
  graph::bind_input(x, "x");
  graph::bind_param(w, "weight");

  Tensor y = ops::matmul(x, w);
  Tensor z = ops::relu(y);
  graph::mark_output(z);

  auto module = ir::emit_mlir(ctx, scope.graph());
  REQUIRE(module);
  REQUIRE(mlir::succeeded(mlir::verify(*module)));

  const std::string out = dump(*module);
  REQUIRE(out.find("tesseract.matmul") != std::string::npos);
  REQUIRE(out.find("tesseract.relu") != std::string::npos);
  REQUIRE(out.find("tesseract.param") != std::string::npos);
  REQUIRE(out.find("\"weight\"") != std::string::npos);
  REQUIRE(out.find("tensor<4x8xf32>") != std::string::npos);
  REQUIRE(out.find("tensor<8x3xf32>") != std::string::npos);
  REQUIRE(out.find("tensor<4x3xf32>") != std::string::npos);
}

TEST_CASE("emit_mlir: reduction carries dim + keepdim attrs", "[ir][emit]") {
  mlir::MLIRContext ctx;
  ctx.loadDialect<ir::TesseractDialect>();

  Tensor x = Tensor::zeros({3, 4}, DType::Float32);

  graph::GraphScope scope;
  graph::bind_input(x, "x");

  Tensor s = ops::sum(x, /*dim=*/1, /*keepdim=*/true);
  graph::mark_output(s);

  auto module = ir::emit_mlir(ctx, scope.graph());
  REQUIRE(module);
  REQUIRE(mlir::succeeded(mlir::verify(*module)));

  const std::string out = dump(*module);
  REQUIRE(out.find("tesseract.sum") != std::string::npos);
  REQUIRE(out.find("dim = 1") != std::string::npos);
  REQUIRE(out.find("keepdim = true") != std::string::npos);
  REQUIRE(out.find("tensor<3x1xf32>") != std::string::npos);
}

TEST_CASE("emit_mlir: nn::Linear forward pass round-trips", "[ir][emit]") {
  mlir::MLIRContext ctx;
  ctx.loadDialect<ir::TesseractDialect>();

  nn::Linear layer(6, 2);
  Tensor x = Tensor::zeros({4, 6}, DType::Float32);

  graph::GraphScope scope;
  graph::bind_input(x, "x");
  graph::bind_param(layer.weight(), "W");
  graph::bind_param(layer.bias(), "b");

  Tensor y = layer.forward(x);
  graph::mark_output(y);

  auto module = ir::emit_mlir(ctx, scope.graph());
  REQUIRE(module);
  REQUIRE(mlir::succeeded(mlir::verify(*module)));

  const std::string out = dump(*module);
  REQUIRE(out.find("tesseract.matmul") != std::string::npos);
  REQUIRE(out.find("tesseract.add") != std::string::npos);
  REQUIRE(out.find("\"W\"") != std::string::npos);
  REQUIRE(out.find("\"b\"") != std::string::npos);
}

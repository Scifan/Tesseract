// Stage-1 graph IR smoke tests.
//
// These verify that `graph::GraphScope` captures ops executed through the
// public `ops::` API, that the captured graph has the expected structure,
// and that eager numerics are unchanged under recording (the default mode is
// "eager + trace").

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Sequential.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/ops/Softmax.hpp"
#include "tesseract/ops/View.hpp"

using namespace tesseract;

TEST_CASE("GraphScope captures binary elementwise ops", "[graph][smoke]") {
  Tensor a = Tensor::from_vector<float>({1, 2, 3, 4}, {2, 2});
  Tensor b = Tensor::from_vector<float>({5, 6, 7, 8}, {2, 2});

  {
    graph::GraphScope scope;
    Tensor c = ops::add(a, b);
    Tensor d = ops::mul(c, a);

    const graph::Graph& g = scope.graph();
    REQUIRE(g.num_ops() == 2);
    REQUIRE(g.ops()[0].kind == "add");
    REQUIRE(g.ops()[1].kind == "mul");
    REQUIRE(g.num_values() == 4);          // a, b, c, d
    REQUIRE(g.ops()[0].inputs.size() == 2);
    REQUIRE(g.ops()[0].outputs.size() == 1);
    // c is the output of add and an input to mul
    REQUIRE(g.ops()[1].inputs[0] == g.ops()[0].outputs[0]);

    // Eager numerics preserved: c[0] = 1 + 5 = 6, d[0] = 6 * 1 = 6.
    REQUIRE(c.data_ptr<float>()[0] == 6.0f);
    REQUIRE(d.data_ptr<float>()[0] == 6.0f);
  }

  // Outside the scope, ops no longer record; no active graph.
  REQUIRE_FALSE(graph::is_recording());
}

TEST_CASE("GraphScope records matmul + relu chain", "[graph][smoke]") {
  Tensor a = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  Tensor b = Tensor::from_vector<float>({1, 0, 0, 1, 1, 0}, {3, 2});

  graph::GraphScope scope;
  Tensor y = ops::matmul(a, b);
  Tensor z = ops::relu(y);
  graph::mark_output(z);

  const graph::Graph& g = scope.graph();
  REQUIRE(g.num_ops() == 2);
  REQUIRE(g.ops()[0].kind == "matmul");
  REQUIRE(g.ops()[1].kind == "relu");
  REQUIRE(g.outputs().size() == 1);
  REQUIRE(g.outputs()[0] == g.ops()[1].outputs[0]);
}

TEST_CASE("GraphScope records view-family ops with shape attrs", "[graph][smoke]") {
  Tensor x = Tensor::from_vector<float>({1, 2, 3, 4, 5, 6}, {2, 3});

  graph::GraphScope scope;
  Tensor y = ops::reshape(x, {3, 2});
  Tensor z = ops::transpose(y, 0, 1);

  const graph::Graph& g = scope.graph();
  REQUIRE(g.num_ops() == 2);
  REQUIRE(g.ops()[0].kind == "reshape");
  REQUIRE(g.ops()[1].kind == "transpose");

  auto it = g.ops()[0].attrs.find("shape");
  REQUIRE(it != g.ops()[0].attrs.end());
  const auto* shape_attr = std::get_if<std::vector<int64_t>>(&it->second);
  REQUIRE(shape_attr != nullptr);
  REQUIRE(*shape_attr == std::vector<int64_t>{3, 2});

  auto it_a = g.ops()[1].attrs.find("dim_a");
  auto it_b = g.ops()[1].attrs.find("dim_b");
  REQUIRE(it_a != g.ops()[1].attrs.end());
  REQUIRE(it_b != g.ops()[1].attrs.end());
  REQUIRE(std::get<int64_t>(it_a->second) == 0);
  REQUIRE(std::get<int64_t>(it_b->second) == 1);
}

TEST_CASE("GraphScope captures nn::Linear forward pass", "[graph][smoke]") {
  nn::Linear fc(4, 3);
  Tensor x = Tensor::zeros({2, 4}, DType::Float32);

  graph::GraphScope scope;
  Tensor y = fc.forward(x);
  graph::mark_output(y);

  const graph::Graph& g = scope.graph();
  // nn::Linear today runs transpose(weight) -> matmul(x, w^T) -> add(bias).
  // The exact op list is an implementation detail, but we expect at least
  // matmul and add to appear, and there should be at least one output.
  bool saw_matmul = false;
  bool saw_add = false;
  for (const auto& op : g.ops()) {
    if (op.kind == "matmul") saw_matmul = true;
    if (op.kind == "add")    saw_add    = true;
  }
  REQUIRE(saw_matmul);
  REQUIRE(saw_add);
  REQUIRE(g.outputs().size() == 1);
}

TEST_CASE("GraphScope nesting is rejected", "[graph][smoke]") {
  graph::GraphScope outer;
  REQUIRE(graph::is_recording());
  REQUIRE_THROWS([]() { graph::GraphScope inner; }());
}

TEST_CASE("Graph::to_string is non-empty and stable", "[graph][smoke]") {
  Tensor a = Tensor::ones({2, 2}, DType::Float32);
  Tensor b = Tensor::ones({2, 2}, DType::Float32);

  graph::GraphScope scope;
  Tensor c = ops::add(a, b);
  graph::mark_output(c);

  std::string dump = scope.graph().to_string();
  REQUIRE(!dump.empty());
  REQUIRE(dump.find("tesseract.add") != std::string::npos);
  REQUIRE(dump.find("return") != std::string::npos);
}

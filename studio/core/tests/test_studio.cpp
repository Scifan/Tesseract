// Tesseract Studio — headless core tests (M5 / B-047).
//
// Exercises the GUI-independent stack end-to-end without any display: JSON +
// .tsb round-trip, catalog, validation/shape-inference, codegen round-trip,
// and the executor actually training an MLP + generating from a Llama through
// the embedded C++ engine.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

#include "tesseract/studio/Analysis.hpp"
#include "tesseract/studio/BlockCatalog.hpp"
#include "tesseract/studio/BlockGraph.hpp"
#include "tesseract/studio/Codegen.hpp"
#include "tesseract/studio/Executor.hpp"
#include "tesseract/studio/Json.hpp"

using namespace tesseract::studio;

namespace {

// Builds the canonical MLP-training graph: Input -> Linear -> ReLU -> Linear
// -> SequentialModel, plus SyntheticClassification + Adam + CE + TrainLoop.
BlockGraph make_mlp_graph(int64_t epochs = 40) {
  BlockGraph g;
  g.name = "mlp";
  const int64_t feats = 8, classes = 3, hidden = 32;

  Json inp = Json::object();
  inp.set("batch", Json(32)); inp.set("features", Json(feats));
  int64_t in = g.add_node("Input", inp);

  Json l1p = Json::object();
  l1p.set("in_features", Json(feats)); l1p.set("out_features", Json(hidden));
  l1p.set("bias", Json(true));
  int64_t l1 = g.add_node("Linear", l1p);

  int64_t relu = g.add_node("ReLU");

  Json l2p = Json::object();
  l2p.set("in_features", Json(hidden)); l2p.set("out_features", Json(classes));
  l2p.set("bias", Json(true));
  int64_t l2 = g.add_node("Linear", l2p);

  int64_t sm = g.add_node("SequentialModel");

  Json dsp = Json::object();
  dsp.set("samples", Json(256)); dsp.set("features", Json(feats));
  dsp.set("classes", Json(classes)); dsp.set("seed", Json(0));
  int64_t ds = g.add_node("SyntheticClassification", dsp);

  Json op = Json::object();
  op.set("lr", Json(0.05));
  int64_t adam = g.add_node("Adam", op);

  int64_t loss = g.add_node("CrossEntropyLoss");

  Json lp = Json::object();
  lp.set("epochs", Json(epochs)); lp.set("batch_size", Json(32));
  int64_t loop = g.add_node("TrainLoop", lp);

  g.connect(in, "out", l1, "in");
  g.connect(l1, "out", relu, "in");
  g.connect(relu, "out", l2, "in");
  g.connect(l2, "out", sm, "in");
  g.connect(sm, "model", adam, "model");
  g.connect(sm, "model", loop, "model");
  g.connect(ds, "data", loop, "data");
  g.connect(adam, "opt", loop, "optimizer");
  g.connect(loss, "loss", loop, "loss");
  return g;
}

BlockGraph make_llama_graph() {
  BlockGraph g;
  g.name = "llama";
  Json mp = Json::object();
  mp.set("vocab_size", Json(64)); mp.set("hidden_size", Json(32));
  mp.set("num_hidden_layers", Json(2)); mp.set("num_attention_heads", Json(4));
  mp.set("num_key_value_heads", Json(4)); mp.set("intermediate_size", Json(64));
  int64_t m = g.add_node("LoadLlama", mp);

  Json gp = Json::object();
  gp.set("prompt_ids", Json(std::string("1, 5, 9"))); gp.set("max_new_tokens", Json(6));
  int64_t gen = g.add_node("Generate", gp);
  g.connect(m, "model", gen, "model");
  return g;
}

// Tensor playground: A[3x4] · B[4x2] -> ReLU -> Inspect(backward).
BlockGraph make_tensor_graph() {
  BlockGraph g;
  g.name = "tensor";
  Json ap = Json::object();
  ap.set("rows", Json(3)); ap.set("cols", Json(4));
  ap.set("init", Json(std::string("randn"))); ap.set("seed", Json(1));
  int64_t a = g.add_node("TensorConst", ap);

  Json bp = Json::object();
  bp.set("rows", Json(4)); bp.set("cols", Json(2));
  bp.set("init", Json(std::string("randn"))); bp.set("seed", Json(2));
  int64_t b = g.add_node("TensorConst", bp);

  int64_t mm = g.add_node("TMatMul");
  int64_t relu = g.add_node("TReLU");

  Json ip = Json::object();
  ip.set("backward", Json(true));
  int64_t insp = g.add_node("TensorInspect", ip);

  g.connect(a, "out", mm, "a");
  g.connect(b, "out", mm, "b");
  g.connect(mm, "out", relu, "in");
  g.connect(relu, "out", insp, "in");
  return g;
}

std::vector<RunEvent> run_collect(const BlockGraph& g) {
  std::vector<RunEvent> events;
  RunOptions opt;  // cpu
  Executor::run(g, [&](const RunEvent& e) { events.push_back(e); }, opt);
  return events;
}

}  // namespace

TEST_CASE("JSON round-trips objects, arrays, escapes", "[studio][json]") {
  Json o = Json::object();
  o.set("name", Json(std::string("a\"b\nc")));
  o.set("n", Json(42));
  o.set("f", Json(1.5));
  o.set("b", Json(true));
  Json arr = Json::array();
  arr.push_back(Json(1)); arr.push_back(Json(2));
  o.set("xs", arr);

  Json back = Json::parse(o.dump());
  REQUIRE(back.value("name").as_string() == "a\"b\nc");
  REQUIRE(back.value("n").as_int() == 42);
  REQUIRE(back.value("f").as_number() == 1.5);
  REQUIRE(back.value("b").as_bool() == true);
  REQUIRE(back.value("xs").items().size() == 2);
}

TEST_CASE("BlockGraph .tsb is fully bidirectional", "[studio][graph]") {
  BlockGraph g = make_mlp_graph();
  std::string tsb = g.to_tsb();
  BlockGraph g2 = BlockGraph::from_tsb(tsb);
  REQUIRE(g2.nodes.size() == g.nodes.size());
  REQUIRE(g2.edges.size() == g.edges.size());
  REQUIRE(g2.to_tsb() == tsb);  // stable
}

TEST_CASE("Catalog exposes the core block kinds", "[studio][catalog]") {
  const auto& cat = BlockCatalog::instance();
  REQUIRE(cat.find("Linear") != nullptr);
  REQUIRE(cat.find("TrainLoop") != nullptr);
  REQUIRE(cat.find("LoadLlama") != nullptr);
  REQUIRE(cat.find("Generate") != nullptr);
  // The palette JSON is well-formed and non-empty.
  REQUIRE(cat.to_json().items().size() >= 15);
}

TEST_CASE("Analysis validates a good graph and flags a bad one",
          "[studio][analysis]") {
  BlockGraph g = make_mlp_graph();
  AnalysisResult a = analyze(g);
  REQUIRE(a.ok);
  // Shape inference reached the last Linear: [32, 3].
  // (find the second Linear node id = it has out_features 3)
  bool saw_out3 = false;
  for (const auto& [id, shape] : a.out_shapes)
    if (shape.size() == 2 && shape[1] == 3) saw_out3 = true;
  REQUIRE(saw_out3);

  SECTION("disconnected required input is an error") {
    BlockGraph bad = g;
    // Drop the dataset edge into TrainLoop.
    bad.edges.erase(
        std::remove_if(bad.edges.begin(), bad.edges.end(),
                       [](const Edge& e) { return e.to.port == "data"; }),
        bad.edges.end());
    AnalysisResult b = analyze(bad);
    REQUIRE_FALSE(b.ok);
  }

  SECTION("type-mismatched wire is an error") {
    BlockGraph bad = g;
    // Wire a Dataset output into a Tensor input.
    bad.connect(/*ds is some node feeding loop.data*/ bad.nodes[5].id, "data",
                bad.nodes[1].id, "in");
    AnalysisResult b = analyze(bad);
    REQUIRE_FALSE(b.ok);
  }
}

TEST_CASE("Codegen emits round-trippable C++ and Python", "[studio][codegen]") {
  BlockGraph g = make_mlp_graph();
  std::string cpp = generate_cpp(g);
  std::string py = generate_python(g);
  REQUIRE(cpp.find("nn::Sequential") != std::string::npos);
  REQUIRE(cpp.find("cross_entropy_with_logits") != std::string::npos);
  REQUIRE(py.find("ts.nn.Sequential") != std::string::npos);

  auto rg_cpp = extract_tsb(cpp);
  REQUIRE(rg_cpp.has_value());
  REQUIRE(rg_cpp->nodes.size() == g.nodes.size());
  auto rg_py = extract_tsb(py);
  REQUIRE(rg_py.has_value());
  REQUIRE(rg_py->nodes.size() == g.nodes.size());
}

TEST_CASE("Executor trains an MLP (loss drops, accuracy rises)",
          "[studio][executor]") {
  std::vector<RunEvent> ev = run_collect(make_mlp_graph(60));
  // Collect loss points + the done summary.
  std::vector<double> losses;
  bool done_ok = false;
  double final_acc = 0.0;
  for (const auto& e : ev) {
    if (e.type == "loss") losses.push_back(e.data.value("loss").as_number());
    if (e.type == "done") {
      done_ok = e.data.value("ok").as_bool();
      final_acc = e.data.value("final_acc").as_number();
    }
  }
  REQUIRE(done_ok);
  REQUIRE(losses.size() > 10);
  REQUIRE(losses.back() < losses.front());  // learning happened
  REQUIRE(final_acc > 0.8);                 // separable mixture is learnable
}

TEST_CASE("Tensor catalog + shape inference for the playground",
          "[studio][tensor]") {
  const auto& cat = BlockCatalog::instance();
  REQUIRE(cat.find("TensorConst") != nullptr);
  REQUIRE(cat.find("TMatMul") != nullptr);
  REQUIRE(cat.find("TensorInspect") != nullptr);

  BlockGraph g = make_tensor_graph();
  AnalysisResult a = analyze(g);
  REQUIRE(a.ok);
  // The matmul output must infer to [3, 2].
  bool saw_3x2 = false;
  for (const auto& [id, shape] : a.out_shapes)
    if (shape.size() == 2 && shape[0] == 3 && shape[1] == 2) saw_3x2 = true;
  REQUIRE(saw_3x2);
}

TEST_CASE("Executor evaluates a tensor graph and backprops to leaves",
          "[studio][executor][tensor]") {
  std::vector<RunEvent> ev = run_collect(make_tensor_graph());
  bool done_ok = false, did_backward = false;
  int results = 0, grads = 0;
  std::vector<int64_t> result_shape;
  for (const auto& e : ev) {
    if (e.type == "tensor") {
      std::string role = e.data.value("role").as_string();
      if (role == "result") {
        ++results;
        for (const auto& d : e.data.value("shape").items())
          result_shape.push_back(d.as_int());
      } else if (role == "grad") {
        ++grads;
      }
    }
    if (e.type == "done") {
      done_ok = e.data.value("ok").as_bool();
      did_backward = e.data.value("backward").as_bool();
    }
  }
  REQUIRE(done_ok);
  REQUIRE(did_backward);
  REQUIRE(results == 1);
  REQUIRE(result_shape == std::vector<int64_t>({3, 2}));
  REQUIRE(grads == 2);  // one gradient per leaf TensorConst
}

TEST_CASE("IR emission lowers blocks to the tesseract dialect",
          "[studio][ir]") {
  std::string mlp_ir = generate_ir(make_mlp_graph());
  REQUIRE(mlp_ir.find("tesseract.matmul") != std::string::npos);
  REQUIRE(mlp_ir.find("tesseract.relu") != std::string::npos);
  REQUIRE(mlp_ir.find("func.func @forward") != std::string::npos);

  std::string tensor_ir = generate_ir(make_tensor_graph());
  REQUIRE(tensor_ir.find("tesseract.matmul") != std::string::npos);
  REQUIRE(tensor_ir.find("tensor<3x2xf32>") != std::string::npos);
}

TEST_CASE("Executor generates from a Llama block", "[studio][executor]") {
  std::vector<RunEvent> ev = run_collect(make_llama_graph());
  bool done_ok = false;
  int64_t generated = 0;
  int tokens = 0;
  for (const auto& e : ev) {
    if (e.type == "token") ++tokens;
    if (e.type == "done") {
      done_ok = e.data.value("ok").as_bool();
      generated = e.data.value("generated").as_int();
    }
  }
  REQUIRE(done_ok);
  REQUIRE(generated == 6);
  REQUIRE(tokens == 6);
}

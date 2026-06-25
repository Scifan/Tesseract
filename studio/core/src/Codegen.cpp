#include "tesseract/studio/Codegen.hpp"

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tesseract/studio/Analysis.hpp"
#include "tesseract/studio/BlockCatalog.hpp"

namespace tesseract::studio {

namespace {

// Layer chain feeding a SequentialModel's `in`, in forward order.
std::vector<const Node*> layer_chain(const BlockGraph& g, int64_t seq_node) {
  const BlockCatalog& cat = BlockCatalog::instance();
  std::vector<const Node*> chain;
  const Edge* e = g.incoming(seq_node, "in");
  while (e) {
    const Node* cur = g.find_node(e->from.node);
    if (!cur) break;
    const BlockSpec* spec = cat.find(cur->kind);
    if (!spec || spec->category != "Layers") break;
    chain.push_back(cur);
    e = g.incoming(cur->id, "in");
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

const Node* find_kind(const BlockGraph& g, const std::string& kind) {
  for (const auto& n : g.nodes)
    if (n.kind == kind) return &n;
  return nullptr;
}

std::string bstr(bool b) { return b ? "true" : "false"; }

// ---- C++ emission ------------------------------------------------------- //

void emit_cpp_layer(std::ostream& o, const Node& n) {
  if (n.kind == "Linear")
    o << "  model->add(std::make_shared<nn::Linear>("
      << n.param_int("in_features", 8) << ", " << n.param_int("out_features", 32)
      << ", " << bstr(n.param_bool("bias", true)) << "));\n";
  else if (n.kind == "ReLU")
    o << "  model->add(std::make_shared<nn::ReLU>());\n";
  else if (n.kind == "Sigmoid")
    o << "  model->add(std::make_shared<nn::Sigmoid>());\n";
  else if (n.kind == "Tanh")
    o << "  model->add(std::make_shared<nn::Tanh>());\n";
  else if (n.kind == "RMSNorm")
    o << "  model->add(std::make_shared<nn::RMSNorm>(" << n.param_int("dim", 32)
      << ", " << n.param_num("eps", 1e-5) << "));\n";
  else if (n.kind == "LayerNorm")
    o << "  model->add(std::make_shared<nn::LayerNorm>(" << n.param_int("dim", 32)
      << ", " << n.param_num("eps", 1e-5) << ", "
      << bstr(n.param_bool("bias", true)) << "));\n";
  else if (n.kind == "FeedForward")
    o << "  model->add(std::make_shared<nn::FeedForward>("
      << n.param_int("d_model", 32) << ", " << n.param_int("d_ff", 64) << ", "
      << bstr(n.param_bool("bias", false)) << "));\n";
  else
    o << "  // (" << n.kind << " — add via the C++ nn API)\n";
}

std::string cpp_train(const BlockGraph& g, const Node& seq) {
  std::ostringstream o;
  const Node* opt = nullptr;
  for (const auto& n : g.nodes)
    if (n.kind == "SGD" || n.kind == "Adam") opt = &n;
  const Node* loop = find_kind(g, "TrainLoop");
  const Node* data = find_kind(g, "SyntheticClassification");

  o << "  auto model = std::make_shared<nn::Sequential>();\n";
  for (const Node* n : layer_chain(g, seq.id)) emit_cpp_layer(o, *n);
  o << "  model->to(Device{DeviceType::CPU, 0});\n\n";

  if (data)
    o << "  // Dataset: " << data->param_int("samples", 256) << " samples, "
      << data->param_int("features", 8) << " features, "
      << data->param_int("classes", 3) << " classes (build X/Y as you like).\n";

  if (opt && opt->kind == "Adam")
    o << "  optim::Adam opt(model->parameters(), " << opt->param_num("lr", 1e-3)
      << ");\n";
  else if (opt)
    o << "  optim::SGD opt(model->parameters(), " << opt->param_num("lr", 0.1)
      << ", " << opt->param_num("momentum", 0.0) << ");\n";

  int64_t epochs = loop ? loop->param_int("epochs", 30) : 30;
  o << "  for (int e = 0; e < " << epochs << "; ++e) {\n"
    << "    Tensor logits = model->forward(x);\n"
    << "    Tensor loss = ops::cross_entropy_with_logits(logits, y);\n"
    << "    opt.zero_grad();\n"
    << "    Engine::backward(loss);\n"
    << "    opt.step();\n"
    << "  }\n";
  return o.str();
}

std::string cpp_generate(const BlockGraph& g, const Node& model_node) {
  std::ostringstream o;
  const Node* gen = find_kind(g, "Generate");
  if (model_node.kind == "LoadLlama") {
    o << "  models::LlamaConfig cfg;\n"
      << "  cfg.vocab_size = " << model_node.param_int("vocab_size", 256) << ";\n"
      << "  cfg.hidden_size = " << model_node.param_int("hidden_size", 64) << ";\n"
      << "  cfg.num_hidden_layers = "
      << model_node.param_int("num_hidden_layers", 2) << ";\n"
      << "  cfg.num_attention_heads = "
      << model_node.param_int("num_attention_heads", 4) << ";\n"
      << "  cfg.num_key_value_heads = "
      << model_node.param_int("num_key_value_heads", 4) << ";\n"
      << "  cfg.intermediate_size = "
      << model_node.param_int("intermediate_size", 128) << ";\n";
    std::string dir = model_node.param_str("model_dir", "");
    if (!dir.empty())
      o << "  auto model = models::LlamaModel::from_pretrained(\"" << dir
        << "\", cfg);\n";
    else
      o << "  auto model = std::make_shared<models::LlamaModel>(cfg);\n";
    o << "  models::LlamaModel::GenerateConfig gc;\n"
      << "  gc.max_new_tokens = " << (gen ? gen->param_int("max_new_tokens", 16) : 16)
      << ";\n"
      << "  std::vector<int32_t> prompt = {1, 2, 3};\n"
      << "  auto out = model->generate(prompt, gc);\n";
  } else if (model_node.kind == "LoadMamba") {
    o << "  models::MambaConfig cfg;\n"
      << "  cfg.vocab_size = " << model_node.param_int("vocab_size", 256) << ";\n"
      << "  cfg.hidden_size = " << model_node.param_int("hidden_size", 64) << ";\n"
      << "  cfg.num_hidden_layers = "
      << model_node.param_int("num_hidden_layers", 2) << ";\n"
      << "  auto model = std::make_shared<models::MambaModel>(cfg);\n"
      << "  models::MambaModel::GenerateConfig gc;\n"
      << "  gc.max_new_tokens = " << (gen ? gen->param_int("max_new_tokens", 16) : 16)
      << ";\n"
      << "  std::vector<int32_t> prompt = {1, 2, 3};\n"
      << "  auto out = model->generate(prompt, gc);\n";
  }
  return o.str();
}

// ---- Python emission ---------------------------------------------------- //

void emit_py_layer(std::ostream& o, const Node& n, bool& first) {
  auto comma = [&] { if (!first) o << ",\n"; first = false; };
  if (n.kind == "Linear") {
    comma();
    o << "    ts.nn.Linear(" << n.param_int("in_features", 8) << ", "
      << n.param_int("out_features", 32) << ", bias="
      << (n.param_bool("bias", true) ? "True" : "False") << ")";
  } else if (n.kind == "ReLU") {
    comma(); o << "    ts.nn.ReLU()";
  } else if (n.kind == "Sigmoid") {
    comma(); o << "    ts.nn.Sigmoid()";
  } else if (n.kind == "Tanh") {
    comma(); o << "    ts.nn.Tanh()";
  } else if (n.kind == "Embedding") {
    comma();
    o << "    ts.nn.Embedding(" << n.param_int("num_embeddings", 1000) << ", "
      << n.param_int("embedding_dim", 32) << ")";
  }
  // (RMSNorm/LayerNorm/FeedForward/MHA are C++-only in the current bindings.)
}

std::string py_train(const BlockGraph& g, const Node& seq) {
  std::ostringstream o;
  const Node* opt = nullptr;
  for (const auto& n : g.nodes)
    if (n.kind == "SGD" || n.kind == "Adam") opt = &n;
  const Node* loop = find_kind(g, "TrainLoop");

  o << "model = ts.nn.Sequential([\n";
  bool first = true;
  for (const Node* n : layer_chain(g, seq.id)) emit_py_layer(o, *n, first);
  o << "\n])\n\n";
  if (opt && opt->kind == "Adam")
    o << "opt = ts.optim.Adam(model.parameters(), lr=" << opt->param_num("lr", 1e-3)
      << ")\n";
  else if (opt)
    o << "opt = ts.optim.SGD(model.parameters(), lr=" << opt->param_num("lr", 0.1)
      << ", momentum=" << opt->param_num("momentum", 0.0) << ")\n";
  int64_t epochs = loop ? loop->param_int("epochs", 30) : 30;
  o << "\n# X: float32 [N, features], y: int64 [N]\n"
    << "for _ in range(" << epochs << "):\n"
    << "    logits = model(X)\n"
    << "    loss = ts.ops.cross_entropy(logits, y)\n"
    << "    opt.zero_grad()\n"
    << "    loss.backward()\n"
    << "    opt.step()\n";
  return o.str();
}

std::string py_generate(const BlockGraph& g, const Node& model_node) {
  std::ostringstream o;
  const Node* gen = find_kind(g, "Generate");
  if (model_node.kind == "LoadLlama") {
    o << "cfg = ts.models.LlamaConfig()\n"
      << "cfg.vocab_size = " << model_node.param_int("vocab_size", 256) << "\n"
      << "cfg.hidden_size = " << model_node.param_int("hidden_size", 64) << "\n"
      << "cfg.num_hidden_layers = "
      << model_node.param_int("num_hidden_layers", 2) << "\n"
      << "cfg.num_attention_heads = "
      << model_node.param_int("num_attention_heads", 4) << "\n"
      << "cfg.num_key_value_heads = "
      << model_node.param_int("num_key_value_heads", 4) << "\n"
      << "cfg.intermediate_size = "
      << model_node.param_int("intermediate_size", 128) << "\n"
      << "model = ts.models.LlamaModel(cfg)\n"
      << "gc = ts.models.LlamaGenerateConfig()\n"
      << "gc.max_new_tokens = " << (gen ? gen->param_int("max_new_tokens", 16) : 16)
      << "\n"
      << "out = model.generate([1, 2, 3], gc)\n"
      << "print(out)\n";
  } else if (model_node.kind == "LoadMamba") {
    o << "cfg = ts.models.MambaConfig()\n"
      << "cfg.vocab_size = " << model_node.param_int("vocab_size", 256) << "\n"
      << "cfg.hidden_size = " << model_node.param_int("hidden_size", 64) << "\n"
      << "cfg.num_hidden_layers = "
      << model_node.param_int("num_hidden_layers", 2) << "\n"
      << "model = ts.models.MambaModel(cfg)\n"
      << "gc = ts.models.MambaGenerateConfig()\n"
      << "gc.max_new_tokens = " << (gen ? gen->param_int("max_new_tokens", 16) : 16)
      << "\n"
      << "out = model.generate([1, 2, 3], gc)\n"
      << "print(out)\n";
  }
  return o.str();
}

}  // namespace

std::string generate_cpp(const BlockGraph& g) {
  std::ostringstream o;
  o << "// Tesseract Studio — generated C++ (graph: " << g.name << ").\n"
    << "// Re-openable in Studio via the embedded graph header below; edits\n"
    << "// outside it are not parsed back.\n"
    << "// @tsb-graph " << g.to_json().dump() << "\n"
    << "#include <memory>\n#include <vector>\n\n"
    << "#include \"tesseract/autograd/Engine.hpp\"\n"
    << "#include \"tesseract/core/Tensor.hpp\"\n"
    << "#include \"tesseract/nn/Sequential.hpp\"\n"
    << "#include \"tesseract/nn/Linear.hpp\"\n"
    << "#include \"tesseract/nn/Activation.hpp\"\n"
    << "#include \"tesseract/optim/Adam.hpp\"\n"
    << "#include \"tesseract/optim/SGD.hpp\"\n"
    << "#include \"tesseract/ops/Loss.hpp\"\n"
    << "#include \"tesseract/models/Llama.hpp\"\n"
    << "#include \"tesseract/models/MambaModel.hpp\"\n\n"
    << "using namespace tesseract;\n\n"
    << "int main() {\n";
  const Node* seq = find_kind(g, "SequentialModel");
  const Node* llama = find_kind(g, "LoadLlama");
  const Node* mamba = find_kind(g, "LoadMamba");
  if (seq) o << cpp_train(g, *seq);
  else if (llama) o << cpp_generate(g, *llama);
  else if (mamba) o << cpp_generate(g, *mamba);
  else o << "  // (empty graph)\n";
  o << "  return 0;\n}\n";
  return o.str();
}

std::string generate_python(const BlockGraph& g) {
  std::ostringstream o;
  o << "# Tesseract Studio — generated Python (graph: " << g.name << ").\n"
    << "# Re-openable in Studio via the embedded graph header below.\n"
    << "# @tsb-graph " << g.to_json().dump() << "\n"
    << "import numpy as np\n"
    << "import tesseract as ts\n\n";
  const Node* seq = find_kind(g, "SequentialModel");
  const Node* llama = find_kind(g, "LoadLlama");
  const Node* mamba = find_kind(g, "LoadMamba");
  if (seq) o << py_train(g, *seq);
  else if (llama) o << py_generate(g, *llama);
  else if (mamba) o << py_generate(g, *mamba);
  else o << "# (empty graph)\n";
  return o.str();
}

// --------------------------------------------------------------------------- //
// IR emission — block graph -> tesseract-dialect MLIR (display only).
// --------------------------------------------------------------------------- //
namespace {

std::string ttype(const std::vector<int64_t>& s) {
  if (s.empty()) return "tensor<f32>";
  std::string o = "tensor<";
  for (size_t i = 0; i < s.size(); ++i) o += std::to_string(s[i]) + "x";
  o += "f32>";
  return o;
}

const std::vector<int64_t>* shape_of(const AnalysisResult& a, int64_t id) {
  auto it = a.out_shapes.find(id);
  return it == a.out_shapes.end() ? nullptr : &it->second;
}

std::string ir_sequential(const BlockGraph& g, const Node& seq,
                          const AnalysisResult& a) {
  std::ostringstream args, body;
  int ssa = 0, argn = 0;
  auto chain = layer_chain(g, seq.id);

  std::vector<int64_t> cur_shape = {32, 8};
  if (!chain.empty()) {
    if (const Edge* in = g.incoming(chain.front()->id, "in"))
      if (const std::vector<int64_t>* s = shape_of(a, in->from.node))
        cur_shape = *s;
  }
  std::string cur = "%arg0";
  args << "%arg0: " << ttype(cur_shape);
  ++argn;

  auto unary = [&](const char* op) {
    std::string y = "%" + std::to_string(ssa++);
    body << "    " << y << " = \"tesseract." << op << "\"(" << cur << ") : ("
         << ttype(cur_shape) << ") -> " << ttype(cur_shape) << "\n";
    cur = y;
  };

  for (const Node* n : chain) {
    const std::vector<int64_t>* os = shape_of(a, n->id);
    std::vector<int64_t> out_shape = os ? *os : cur_shape;
    if (n->kind == "Linear") {
      int64_t in_f = n->param_int("in_features", 8);
      int64_t out_f = n->param_int("out_features", 32);
      std::string w = "%arg" + std::to_string(argn++);
      args << ", " << w << ": " << ttype({in_f, out_f});
      std::string mm = "%" + std::to_string(ssa++);
      body << "    " << mm << " = \"tesseract.matmul\"(" << cur << ", " << w
           << ") : (" << ttype(cur_shape) << ", " << ttype({in_f, out_f})
           << ") -> " << ttype(out_shape) << "\n";
      cur = mm;
      if (n->param_bool("bias", true)) {
        std::string b = "%arg" + std::to_string(argn++);
        args << ", " << b << ": " << ttype({out_f});
        std::string ad = "%" + std::to_string(ssa++);
        body << "    " << ad << " = \"tesseract.add\"(" << mm << ", " << b
             << ") : (" << ttype(out_shape) << ", " << ttype({out_f}) << ") -> "
             << ttype(out_shape) << "\n";
        cur = ad;
      }
      cur_shape = out_shape;
    } else if (n->kind == "ReLU") {
      unary("relu");
    } else if (n->kind == "Sigmoid") {
      unary("sigmoid");
    } else if (n->kind == "Tanh") {
      unary("tanh");
    } else if (n->kind == "RMSNorm" || n->kind == "LayerNorm") {
      int64_t dim = n->param_int("dim", 32);
      std::string w = "%arg" + std::to_string(argn++);
      args << ", " << w << ": " << ttype({dim});
      std::string y = "%" + std::to_string(ssa++);
      const char* op = n->kind == "RMSNorm" ? "rms_norm" : "layer_norm";
      body << "    " << y << " = \"tesseract." << op << "\"(" << cur << ", " << w
           << ") : (" << ttype(cur_shape) << ", " << ttype({dim}) << ") -> "
           << ttype(cur_shape) << "\n";
      cur = y;
    } else {
      body << "    // " << n->kind
           << " lowers to a sub-graph of tesseract ops (omitted)\n";
      cur_shape = out_shape;
    }
  }

  std::ostringstream o;
  o << "  func.func @forward(" << args.str() << ") -> " << ttype(cur_shape)
    << " {\n"
    << body.str() << "    \"func.return\"(" << cur << ") : (" << ttype(cur_shape)
    << ") -> ()\n"
    << "  }\n";
  return o.str();
}

std::string ir_tensor(const BlockGraph& g, const Node& sink,
                      const AnalysisResult& a) {
  std::ostringstream args, body;
  int ssa = 0, argn = 0;
  std::unordered_map<int64_t, std::string> name;  // node -> ssa value
  bool first_arg = true;

  auto sh = [&](int64_t id) -> std::vector<int64_t> {
    const std::vector<int64_t>* s = shape_of(a, id);
    return s ? *s : std::vector<int64_t>{};
  };
  auto in_name = [&](int64_t node, const std::string& port) -> std::string {
    const Edge* e = g.incoming(node, port);
    if (!e) return "%?";
    auto it = name.find(e->from.node);
    return it == name.end() ? "%?" : it->second;
  };
  auto in_shape = [&](int64_t node, const std::string& port) {
    const Edge* e = g.incoming(node, port);
    return e ? sh(e->from.node) : std::vector<int64_t>{};
  };

  for (int64_t id : g.topo_order()) {
    const Node* n = g.find_node(id);
    if (!n) continue;
    const std::string& k = n->kind;
    if (k == "TensorConst") {
      std::string an = "%arg" + std::to_string(argn++);
      if (!first_arg) args << ", ";
      first_arg = false;
      args << an << ": " << ttype(sh(id));
      name[id] = an;
    } else if (k == "TReLU" || k == "TSigmoid" || k == "TTanh" || k == "TExp") {
      const char* op = k == "TReLU"      ? "relu"
                       : k == "TSigmoid" ? "sigmoid"
                       : k == "TTanh"    ? "tanh"
                                         : "exp";
      std::string y = "%" + std::to_string(ssa++);
      body << "    " << y << " = \"tesseract." << op << "\"("
           << in_name(id, "in") << ") : (" << ttype(in_shape(id, "in"))
           << ") -> " << ttype(sh(id)) << "\n";
      name[id] = y;
    } else if (k == "TAdd" || k == "TSub" || k == "TMul" || k == "TMatMul") {
      const char* op = k == "TAdd"   ? "add"
                       : k == "TSub" ? "sub"
                       : k == "TMul" ? "mul"
                                     : "matmul";
      std::string y = "%" + std::to_string(ssa++);
      body << "    " << y << " = \"tesseract." << op << "\"("
           << in_name(id, "a") << ", " << in_name(id, "b") << ") : ("
           << ttype(in_shape(id, "a")) << ", " << ttype(in_shape(id, "b"))
           << ") -> " << ttype(sh(id)) << "\n";
      name[id] = y;
    }
  }

  const Edge* ie = g.incoming(sink.id, "in");
  std::string ret = ie ? in_name(sink.id, "in") : "%?";
  std::vector<int64_t> ret_shape = ie ? sh(ie->from.node) : std::vector<int64_t>{};
  if (sink.param_bool("backward", true))
    body << "    %loss = \"tesseract.sum\"(" << ret
         << ") {dim = -1 : si64, keepdim = false} : (" << ttype(ret_shape)
         << ") -> tensor<f32>\n"
         << "    // autograd: tesseract --backward materializes ∂%loss/∂argK\n";

  std::ostringstream o;
  o << "  func.func @tensor_graph(" << args.str() << ") -> " << ttype(ret_shape)
    << " {\n"
    << body.str() << "    \"func.return\"(" << ret << ") : ("
    << ttype(ret_shape) << ") -> ()\n"
    << "  }\n";
  return o.str();
}

std::string ir_model(const Node& m) {
  std::ostringstream o;
  int64_t h = m.param_int("hidden_size", 64);
  int64_t L = m.param_int("num_hidden_layers", 2);
  o << "  // " << m.kind << ": " << L << " decoder layers, hidden=" << h
    << " — one representative layer shown.\n"
    << "  func.func @decoder_layer(%x: " << ttype({1, h}) << ", %wn: "
    << ttype({h}) << ") -> " << ttype({1, h}) << " {\n"
    << "    %0 = \"tesseract.rms_norm\"(%x, %wn) : (" << ttype({1, h}) << ", "
    << ttype({h}) << ") -> " << ttype({1, h}) << "\n"
    << "    // ... attention + SwiGLU lower to tesseract.matmul / softmax / "
       "add ...\n"
    << "    %1 = \"tesseract.add\"(%x, %0) : (" << ttype({1, h}) << ", "
    << ttype({1, h}) << ") -> " << ttype({1, h}) << "\n"
    << "    \"func.return\"(%1) : (" << ttype({1, h}) << ") -> ()\n"
    << "  }\n";
  return o.str();
}

}  // namespace

std::string generate_ir(const BlockGraph& g) {
  AnalysisResult a = analyze(g);
  std::ostringstream o;
  o << "// Tesseract Studio — tesseract-dialect IR for graph: " << g.name
    << "\n"
    << "// One IR, many front-ends: these blocks lower to the same dialect the\n"
    << "// C++/Python APIs and .mlir files use (see tests/ir/*.mlir).\n"
    << "module {\n";
  const Node* seq = find_kind(g, "SequentialModel");
  const Node* inspect = find_kind(g, "TensorInspect");
  const Node* llama = find_kind(g, "LoadLlama");
  const Node* mamba = find_kind(g, "LoadMamba");
  if (seq) o << ir_sequential(g, *seq, a);
  else if (inspect) o << ir_tensor(g, *inspect, a);
  else if (llama) o << ir_model(*llama);
  else if (mamba) o << ir_model(*mamba);
  else o << "  // (empty graph — add blocks to see their IR)\n";
  o << "}\n";
  return o.str();
}

std::optional<BlockGraph> extract_tsb(const std::string& source) {
  std::istringstream in(source);
  std::string line;
  const std::string marker = "@tsb-graph ";
  while (std::getline(in, line)) {
    auto pos = line.find(marker);
    if (pos != std::string::npos) {
      std::string json = line.substr(pos + marker.size());
      try {
        return BlockGraph::from_tsb(json);
      } catch (...) {
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

}  // namespace tesseract::studio

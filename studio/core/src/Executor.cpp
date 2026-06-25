#include "tesseract/studio/Executor.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/io/BpeTokenizer.hpp"
#include "tesseract/io/Tokenizer.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/models/MambaModel.hpp"
#include "tesseract/nn/Activation.hpp"
#include "tesseract/nn/Embedding.hpp"
#include "tesseract/nn/FeedForward.hpp"
#include "tesseract/nn/LayerNorm.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Module.hpp"
#include "tesseract/nn/MultiHeadAttention.hpp"
#include "tesseract/nn/RMSNorm.hpp"
#include "tesseract/nn/Sequential.hpp"
#include "tesseract/nn/TransformerBlock.hpp"
#include "tesseract/optim/Adam.hpp"
#include "tesseract/optim/SGD.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Loss.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Reduction.hpp"
#include "tesseract/studio/Analysis.hpp"
#include "tesseract/studio/BlockCatalog.hpp"

namespace tesseract::studio {

namespace {

Device parse_device(const std::string& s) {
  if (s.rfind("cuda", 0) == 0) {
    int idx = 0;
    auto pos = s.find(':');
    if (pos != std::string::npos) idx = std::stoi(s.substr(pos + 1));
    return Device{DeviceType::CUDA, idx};
  }
  return cpu_device();
}

bool cancelled(const RunOptions& opt) {
  return opt.cancel && opt.cancel->load();
}

void log(const EventSink& sink, const std::string& msg) {
  Json d = Json::object();
  d.set("message", Json(msg));
  sink({"log", d});
}

void emit_error(const EventSink& sink, const std::string& msg) {
  Json d = Json::object();
  d.set("message", Json(msg));
  sink({"error", d});
}

// ---- model assembly ----------------------------------------------------- //

struct BuiltModel {
  std::shared_ptr<nn::Module> module;             // Sequential / generic
  std::shared_ptr<models::LlamaModel> llama;      // optional
  std::shared_ptr<models::MambaModel> mamba;      // optional
};

std::shared_ptr<nn::Module> instantiate_layer(const Node& n) {
  const DType dt = DType::Float32;
  if (n.kind == "Linear")
    return std::make_shared<nn::Linear>(n.param_int("in_features", 8),
                                        n.param_int("out_features", 32),
                                        n.param_bool("bias", true), dt);
  if (n.kind == "ReLU") return std::make_shared<nn::ReLU>();
  if (n.kind == "Sigmoid") return std::make_shared<nn::Sigmoid>();
  if (n.kind == "Tanh") return std::make_shared<nn::Tanh>();
  if (n.kind == "RMSNorm")
    return std::make_shared<nn::RMSNorm>(n.param_int("dim", 32),
                                         n.param_num("eps", 1e-5), dt);
  if (n.kind == "LayerNorm")
    return std::make_shared<nn::LayerNorm>(n.param_int("dim", 32),
                                           n.param_num("eps", 1e-5),
                                           n.param_bool("bias", true), dt);
  if (n.kind == "Embedding")
    return std::make_shared<nn::Embedding>(n.param_int("num_embeddings", 1000),
                                           n.param_int("embedding_dim", 32), dt);
  if (n.kind == "FeedForward")
    return std::make_shared<nn::FeedForward>(n.param_int("d_model", 32),
                                             n.param_int("d_ff", 64),
                                             n.param_bool("bias", false), dt);
  if (n.kind == "MultiHeadAttention")
    return std::make_shared<nn::MultiHeadAttention>(
        n.param_int("d_model", 32), n.param_int("num_heads", 4),
        /*use_bias=*/true, n.param_bool("causal", true), dt,
        /*rope_base=*/0.0, /*rope_max_seq=*/0,
        n.param_int("num_kv_heads", 0));
  if (n.kind == "TransformerBlock")
    return std::make_shared<nn::TransformerBlock>(
        n.param_int("d_model", 32), n.param_int("num_heads", 4),
        n.param_int("d_ff", 64), /*norm_eps=*/1e-5, n.param_bool("causal", true),
        /*use_bias=*/false, dt, /*rope_base=*/0.0, /*rope_max_seq=*/0,
        n.param_int("num_kv_heads", 0));
  return nullptr;
}

// Walk back from a SequentialModel node through the layer chain (port "in")
// and assemble an nn::Sequential in forward order.
std::shared_ptr<nn::Sequential> assemble_sequential(const BlockGraph& g,
                                                    int64_t seq_node) {
  const BlockCatalog& cat = BlockCatalog::instance();
  std::vector<std::shared_ptr<nn::Module>> layers;
  const Edge* e = g.incoming(seq_node, "in");
  while (e) {
    const Node* cur = g.find_node(e->from.node);
    if (!cur) break;
    const BlockSpec* spec = cat.find(cur->kind);
    if (!spec || spec->category != "Layers") break;  // hit the Input seed
    auto mod = instantiate_layer(*cur);
    if (!mod) break;
    layers.push_back(mod);
    e = g.incoming(cur->id, "in");
  }
  auto seq = std::make_shared<nn::Sequential>();
  for (auto it = layers.rbegin(); it != layers.rend(); ++it) seq->add(*it);
  return seq;
}

models::LlamaConfig llama_config_from(const Node& n) {
  models::LlamaConfig c;
  c.vocab_size = n.param_int("vocab_size", 256);
  c.hidden_size = n.param_int("hidden_size", 64);
  c.num_hidden_layers = n.param_int("num_hidden_layers", 2);
  c.num_attention_heads = n.param_int("num_attention_heads", 4);
  c.num_key_value_heads = n.param_int("num_key_value_heads", 4);
  c.intermediate_size = n.param_int("intermediate_size", 128);
  c.max_position_embeddings = n.param_int("max_position_embeddings", 512);
  c.num_experts = n.param_int("num_experts", 0);
  c.num_experts_per_tok = n.param_int("num_experts_per_tok", 0);
  c.dtype = n.param_bool("fp16", false) ? DType::Float16 : DType::Float32;
  return c;
}

models::MambaConfig mamba_config_from(const Node& n) {
  models::MambaConfig c;
  c.vocab_size = n.param_int("vocab_size", 256);
  c.hidden_size = n.param_int("hidden_size", 64);
  c.num_hidden_layers = n.param_int("num_hidden_layers", 2);
  c.d_state = n.param_int("d_state", 16);
  c.d_conv = n.param_int("d_conv", 4);
  c.expand = n.param_int("expand", 2);
  return c;
}

// ---- datasets ----------------------------------------------------------- //

struct HostDataset {
  int64_t n = 0;
  int64_t features = 0;
  int64_t classes = 0;
  std::vector<float> X;     // [n, features]
  std::vector<int64_t> Y;   // [n]
};

HostDataset make_synthetic(const Node& n) {
  HostDataset ds;
  ds.n = n.param_int("samples", 256);
  ds.features = n.param_int("features", 8);
  ds.classes = n.param_int("classes", 3);
  std::mt19937 rng(static_cast<uint32_t>(n.param_int("seed", 0)));
  std::normal_distribution<float> noise(0.0f, 0.6f);
  std::uniform_real_distribution<float> center(-2.5f, 2.5f);
  // Per-class centroids.
  std::vector<std::vector<float>> centroids(ds.classes,
                                            std::vector<float>(ds.features));
  for (auto& c : centroids)
    for (auto& v : c) v = center(rng);
  ds.X.resize(static_cast<size_t>(ds.n) * ds.features);
  ds.Y.resize(ds.n);
  for (int64_t i = 0; i < ds.n; ++i) {
    int64_t cls = i % ds.classes;
    ds.Y[i] = cls;
    for (int64_t f = 0; f < ds.features; ++f)
      ds.X[i * ds.features + f] = centroids[cls][f] + noise(rng);
  }
  return ds;
}

HostDataset load_csv(const Node& n) {
  HostDataset ds;
  std::ifstream in(n.param_str("path", "data.csv"));
  if (!in) throw std::runtime_error("CSV not found: " + n.param_str("path"));
  int64_t label_col = n.param_int("label_col", -1);
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;
    bool numeric = true;
    while (std::getline(ss, cell, ',')) {
      try {
        row.push_back(std::stod(cell));
      } catch (...) {
        numeric = false;
        break;
      }
    }
    if (numeric && !row.empty()) rows.push_back(std::move(row));
  }
  if (rows.empty()) throw std::runtime_error("CSV had no numeric rows");
  int64_t ncol = static_cast<int64_t>(rows[0].size());
  if (label_col < 0) label_col = ncol - 1;
  ds.n = static_cast<int64_t>(rows.size());
  ds.features = ncol - 1;
  int64_t max_label = 0;
  for (const auto& r : rows) {
    auto lbl = static_cast<int64_t>(r[label_col]);
    max_label = std::max(max_label, lbl);
    ds.Y.push_back(lbl);
    for (int64_t c = 0; c < ncol; ++c)
      if (c != label_col) ds.X.push_back(static_cast<float>(r[c]));
  }
  ds.classes = max_label + 1;
  return ds;
}

HostDataset build_dataset(const BlockGraph& g, int64_t node) {
  const Node* n = g.find_node(node);
  if (!n) throw std::runtime_error("dataset node missing");
  if (n->kind == "SyntheticClassification") return make_synthetic(*n);
  if (n->kind == "CsvDataset") return load_csv(*n);
  throw std::runtime_error("unsupported dataset block: " + n->kind);
}

Tensor batch_x(const HostDataset& ds, const std::vector<int64_t>& idx,
               Device dev) {
  int64_t b = static_cast<int64_t>(idx.size());
  Tensor x = Tensor::empty(Shape({b, ds.features}), DType::Float32, cpu_device());
  float* p = x.data_ptr<float>();
  for (int64_t i = 0; i < b; ++i)
    std::memcpy(p + i * ds.features, &ds.X[idx[i] * ds.features],
                static_cast<size_t>(ds.features) * sizeof(float));
  return dev.is_cpu() ? x : x.to(dev);
}

Tensor batch_y(const HostDataset& ds, const std::vector<int64_t>& idx,
               Device dev) {
  int64_t b = static_cast<int64_t>(idx.size());
  Tensor y = Tensor::empty(Shape({b}), DType::Int64, cpu_device());
  int64_t* p = y.data_ptr<int64_t>();
  for (int64_t i = 0; i < b; ++i) p[i] = ds.Y[idx[i]];
  return dev.is_cpu() ? y : y.to(dev);
}

float scalar_of(const Tensor& t) {
  Tensor h = t.device().is_cpu() ? t : t.to(cpu_device());
  h = h.is_contiguous() ? h : h.contiguous();
  return h.data_ptr<float>()[0];
}

double batch_accuracy(const Tensor& logits, const HostDataset& ds,
                      const std::vector<int64_t>& idx) {
  Tensor h = logits.device().is_cpu() ? logits : logits.to(cpu_device());
  h = h.is_contiguous() ? h : h.contiguous();
  const float* p = h.data_ptr<float>();
  int64_t b = static_cast<int64_t>(idx.size());
  int64_t c = ds.classes;
  int64_t correct = 0;
  for (int64_t i = 0; i < b; ++i) {
    int64_t best = 0;
    float bv = p[i * c];
    for (int64_t j = 1; j < c; ++j)
      if (p[i * c + j] > bv) { bv = p[i * c + j]; best = j; }
    if (best == ds.Y[idx[i]]) ++correct;
  }
  return b ? static_cast<double>(correct) / static_cast<double>(b) : 0.0;
}

// ---- training ----------------------------------------------------------- //

Json run_train(const BlockGraph& g, const Node& loop,
               std::unordered_map<int64_t, BuiltModel>& models,
               const EventSink& sink, const RunOptions& opt) {
  Device dev = parse_device(opt.device);

  const Edge* me = g.incoming(loop.id, "model");
  const Edge* de = g.incoming(loop.id, "data");
  const Edge* oe = g.incoming(loop.id, "optimizer");
  if (!me || !de || !oe)
    throw std::runtime_error("TrainLoop needs model, data, optimizer wired");

  auto mit = models.find(me->from.node);
  if (mit == models.end() || !mit->second.module)
    throw std::runtime_error("TrainLoop.model must be a SequentialModel");
  auto model = mit->second.module;

  HostDataset ds = build_dataset(g, de->from.node);
  log(sink, "dataset: " + std::to_string(ds.n) + " samples, " +
                std::to_string(ds.features) + " features, " +
                std::to_string(ds.classes) + " classes");

  model->to(dev);
  model->train(true);
  auto params = model->parameters();
  if (params.empty())
    throw std::runtime_error("model has no trainable parameters (empty chain?)");

  // Build the optimizer from the optimizer block.
  const Node* on = g.find_node(oe->from.node);
  std::unique_ptr<optim::SGD> sgd;
  std::unique_ptr<optim::Adam> adam;
  if (on->kind == "SGD")
    sgd = std::make_unique<optim::SGD>(params, on->param_num("lr", 0.1),
                                       on->param_num("momentum", 0.0));
  else
    adam = std::make_unique<optim::Adam>(params, on->param_num("lr", 1e-3),
                                         on->param_num("beta1", 0.9),
                                         on->param_num("beta2", 0.999),
                                         on->param_num("eps", 1e-8));
  auto zero_grad = [&] { if (sgd) sgd->zero_grad(); else adam->zero_grad(); };
  auto step = [&] { if (sgd) sgd->step(); else adam->step(); };

  int64_t epochs = std::max<int64_t>(1, loop.param_int("epochs", 30));
  int64_t bs = std::max<int64_t>(1, loop.param_int("batch_size", 32));

  std::vector<int64_t> order(ds.n);
  for (int64_t i = 0; i < ds.n; ++i) order[i] = i;
  std::mt19937 shuffle_rng(1234);

  int64_t global_step = 0;
  float last_loss = 0.0f;
  double last_acc = 0.0;
  for (int64_t ep = 0; ep < epochs; ++ep) {
    if (cancelled(opt)) { log(sink, "cancelled"); break; }
    std::shuffle(order.begin(), order.end(), shuffle_rng);
    double ep_loss = 0.0, ep_acc = 0.0;
    int64_t batches = 0;
    for (int64_t s = 0; s < ds.n; s += bs) {
      if (cancelled(opt)) break;
      std::vector<int64_t> idx(order.begin() + s,
                               order.begin() + std::min(ds.n, s + bs));
      Tensor x = batch_x(ds, idx, dev);
      Tensor y = batch_y(ds, idx, dev);
      Tensor logits = model->forward(x);
      Tensor loss = ops::cross_entropy_with_logits(logits, y);
      zero_grad();
      tesseract::Engine::backward(loss);
      step();
      last_loss = scalar_of(loss);
      last_acc = batch_accuracy(logits, ds, idx);
      ep_loss += last_loss;
      ep_acc += last_acc;
      ++batches;
      Json ld = Json::object();
      ld.set("step", Json(global_step));
      ld.set("epoch", Json(ep));
      ld.set("loss", Json(static_cast<double>(last_loss)));
      ld.set("acc", Json(last_acc));
      sink({"loss", ld});
      ++global_step;
    }
    if (batches) {
      Json md = Json::object();
      md.set("epoch", Json(ep));
      md.set("loss", Json(ep_loss / batches));
      md.set("acc", Json(ep_acc / batches));
      sink({"metric", md});
    }
  }

  Json summary = Json::object();
  summary.set("ok", Json(true));
  summary.set("mode", Json(std::string("train")));
  summary.set("final_loss", Json(static_cast<double>(last_loss)));
  summary.set("final_acc", Json(last_acc));
  summary.set("steps", Json(global_step));
  sink({"done", summary});
  return summary;
}

// ---- generation --------------------------------------------------------- //

std::vector<int32_t> parse_ids(const std::string& csv) {
  std::vector<int32_t> ids;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    // trim
    size_t a = tok.find_first_not_of(" \t");
    if (a == std::string::npos) continue;
    size_t b = tok.find_last_not_of(" \t");
    try {
      ids.push_back(static_cast<int32_t>(std::stol(tok.substr(a, b - a + 1))));
    } catch (...) {
    }
  }
  return ids;
}

Json run_generate(const BlockGraph& g, const Node& gen,
                  std::unordered_map<int64_t, BuiltModel>& models,
                  const EventSink& sink, const RunOptions& opt) {
  const Edge* me = g.incoming(gen.id, "model");
  if (!me) throw std::runtime_error("Generate needs a model wired");
  auto mit = models.find(me->from.node);
  if (mit == models.end())
    throw std::runtime_error("Generate.model is not a model block");

  // Optional tokenizer.
  std::shared_ptr<io::Tokenizer> tok;
  if (const Edge* te = g.incoming(gen.id, "tokenizer")) {
    const Node* tn = g.find_node(te->from.node);
    if (tn && tn->kind == "BpeTokenizer") {
      tok = std::make_shared<io::BpeTokenizer>(
          io::BpeTokenizer::from_file(tn->param_str("path", "tokenizer.json")));
    }
  }

  // Resolve prompt ids.
  std::vector<int32_t> prompt;
  std::string prompt_text = gen.param_str("prompt", "");
  if (!prompt_text.empty() && tok) {
    prompt = tok->encode(prompt_text, true);
  } else {
    prompt = parse_ids(gen.param_str("prompt_ids", "1, 2, 3"));
  }
  if (prompt.empty()) prompt = {1};

  int64_t max_new = std::max<int64_t>(1, gen.param_int("max_new_tokens", 16));

  std::vector<int32_t> out;
  if (mit->second.llama) {
    models::LlamaModel::GenerateConfig gc;
    gc.max_new_tokens = max_new;
    gc.do_sample = gen.param_bool("do_sample", false);
    gc.sampling.temperature = gen.param_num("temperature", 1.0);
    gc.sampling.top_k = static_cast<int>(gen.param_int("top_k", 0));
    gc.sampling.top_p = gen.param_num("top_p", 1.0);
    gc.seed = static_cast<uint64_t>(gen.param_int("seed", 0));
    out = mit->second.llama->generate(prompt, gc);
  } else if (mit->second.mamba) {
    models::MambaModel::GenerateConfig gc;
    gc.max_new_tokens = max_new;
    out = mit->second.mamba->generate(prompt, gc);
  } else {
    throw std::runtime_error("Generate.model must be a Llama or Mamba block");
  }

  // Stream the generated tail token-by-token; decode incrementally if we can.
  std::string text;
  for (size_t i = prompt.size(); i < out.size(); ++i) {
    if (cancelled(opt)) break;
    Json td = Json::object();
    td.set("id", Json(static_cast<int64_t>(out[i])));
    if (tok) {
      std::vector<int32_t> upto(out.begin() + static_cast<long>(prompt.size()),
                                out.begin() + static_cast<long>(i) + 1);
      std::string decoded = tok->decode(upto, true);
      td.set("text", Json(decoded));
      text = decoded;
    }
    sink({"token", td});
  }
  if (tok) {
    Json xd = Json::object();
    xd.set("text", Json(text));
    sink({"text", xd});
  }

  Json summary = Json::object();
  summary.set("ok", Json(true));
  summary.set("mode", Json(std::string("generate")));
  summary.set("prompt_len", Json(static_cast<int64_t>(prompt.size())));
  summary.set("generated", Json(static_cast<int64_t>(out.size() - prompt.size())));
  Json ids = Json::array();
  for (int32_t id : out) ids.push_back(Json(static_cast<int64_t>(id)));
  summary.set("ids", std::move(ids));
  sink({"done", summary});
  return summary;
}

// ---- tensor / autograd playground --------------------------------------- //

Tensor make_leaf(const Node& n, Device dev) {
  int64_t rows = std::max<int64_t>(1, n.param_int("rows", 2));
  int64_t cols = std::max<int64_t>(1, n.param_int("cols", 3));
  std::string init = n.param_str("init", "randn");
  double scale = n.param_num("scale", 1.0);
  Tensor t = Tensor::empty(Shape({rows, cols}), DType::Float32, cpu_device());
  float* p = t.data_ptr<float>();
  std::mt19937 rng(static_cast<uint32_t>(n.param_int("seed", 0)));
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (int64_t i = 0; i < rows * cols; ++i) {
    float v;
    if (init == "zeros") v = 0.0f;
    else if (init == "ones") v = 1.0f;
    else if (init == "arange") v = static_cast<float>(i);
    else if (init == "eye") v = ((i / cols) == (i % cols)) ? 1.0f : 0.0f;
    else v = nd(rng);
    p[i] = v * static_cast<float>(scale);
  }
  Tensor out = dev.is_cpu() ? t : t.to(dev);
  out.set_requires_grad(true);
  return out;
}

void emit_tensor(const EventSink& sink, const std::string& role, int64_t node,
                 const std::string& label, const Tensor& t) {
  Json d = Json::object();
  d.set("role", Json(role));  // "result" | "grad" | "value"
  d.set("node", Json(node));
  d.set("label", Json(label));
  if (!t.defined()) {
    d.set("shape", Json::array());
    d.set("values", Json::array());
    d.set("numel", Json(static_cast<int64_t>(0)));
    sink({"tensor", d});
    return;
  }
  Tensor h = t.device().is_cpu() ? t : t.to(cpu_device());
  h = h.is_contiguous() ? h : h.contiguous();
  Json shp = Json::array();
  for (std::size_t i = 0; i < h.shape().rank(); ++i)
    shp.push_back(Json(h.shape()[i]));
  d.set("shape", std::move(shp));
  int64_t numel = h.numel();
  int64_t lim = std::min<int64_t>(numel, 4096);
  Json vals = Json::array();
  const float* p = h.data_ptr<float>();
  for (int64_t i = 0; i < lim; ++i)
    vals.push_back(Json(static_cast<double>(p[i])));
  d.set("values", std::move(vals));
  d.set("numel", Json(numel));
  sink({"tensor", d});
}

Json run_tensor(const BlockGraph& g, const Node& sink_node, const EventSink& sink,
                const RunOptions& opt) {
  Device dev = parse_device(opt.device);
  std::unordered_map<int64_t, Tensor> vals;
  std::vector<std::pair<int64_t, std::string>> leaves;  // (node, label)

  auto val_of = [&](const Edge* e) -> Tensor {
    if (!e) return Tensor{};
    auto it = vals.find(e->from.node);
    return it == vals.end() ? Tensor{} : it->second;
  };

  for (int64_t id : g.topo_order()) {
    if (cancelled(opt)) break;
    const Node* n = g.find_node(id);
    if (!n) continue;
    const std::string& k = n->kind;
    if (k == "TensorConst") {
      Tensor leaf = make_leaf(*n, dev);
      vals[id] = leaf;
      leaves.emplace_back(id, "t" + std::to_string(id));
      emit_tensor(sink, "value", id, "t" + std::to_string(id), leaf);
    } else if (k == "TReLU" || k == "TSigmoid" || k == "TTanh" || k == "TExp") {
      Tensor x = val_of(g.incoming(id, "in"));
      if (!x.defined()) continue;
      Tensor y = k == "TReLU"      ? ops::relu(x)
                 : k == "TSigmoid" ? ops::sigmoid(x)
                 : k == "TTanh"    ? ops::tanh(x)
                                   : ops::exp(x);
      vals[id] = y;
      emit_tensor(sink, "value", id, k, y);
    } else if (k == "TAdd" || k == "TSub" || k == "TMul" || k == "TMatMul") {
      Tensor a = val_of(g.incoming(id, "a"));
      Tensor b = val_of(g.incoming(id, "b"));
      if (!a.defined() || !b.defined()) continue;
      Tensor y = k == "TAdd"   ? ops::add(a, b)
                 : k == "TSub" ? ops::sub(a, b)
                 : k == "TMul" ? ops::mul(a, b)
                               : ops::matmul(a, b);
      vals[id] = y;
      emit_tensor(sink, "value", id, k, y);
    }
  }

  Tensor out = val_of(g.incoming(sink_node.id, "in"));
  if (!out.defined())
    throw std::runtime_error("Inspect: could not evaluate the input tensor");
  const Edge* ie = g.incoming(sink_node.id, "in");
  emit_tensor(sink, "result", ie->from.node, "output", out);

  bool did_backward = false;
  if (sink_node.param_bool("backward", true)) {
    Tensor loss = ops::sum(out);
    tesseract::Engine::backward(loss);
    did_backward = true;
    for (const auto& [lid, label] : leaves) {
      const Tensor& gr = vals[lid].grad();
      emit_tensor(sink, "grad", lid, "∂Σ/∂" + label, gr);
    }
  }

  Json summary = Json::object();
  summary.set("ok", Json(true));
  summary.set("mode", Json(std::string("tensor")));
  summary.set("backward", Json(did_backward));
  summary.set("leaves", Json(static_cast<int64_t>(leaves.size())));
  sink({"done", summary});
  return summary;
}

}  // namespace

Json Executor::run(const BlockGraph& g, const EventSink& sink,
                   const RunOptions& opt) {
  try {
    AnalysisResult a = analyze(g);
    if (!a.ok) {
      for (const auto& d : a.diagnostics)
        if (d.severity == "error") emit_error(sink, d.message);
      Json fail = Json::object();
      fail.set("ok", Json(false));
      fail.set("reason", Json(std::string("validation failed")));
      return fail;
    }

    Device dev = parse_device(opt.device);

    // Instantiate every Model-producing node up front so optimizer + sink
    // share the same instance.
    std::unordered_map<int64_t, BuiltModel> models;
    for (const auto& n : g.nodes) {
      if (n.kind == "SequentialModel") {
        BuiltModel bm;
        bm.module = assemble_sequential(g, n.id);
        models[n.id] = bm;
      } else if (n.kind == "LoadLlama") {
        BuiltModel bm;
        auto cfg = llama_config_from(n);
        std::string dir = n.param_str("model_dir", "");
        if (!dir.empty()) {
          log(sink, "loading Llama from " + dir);
          bm.llama = models::LlamaModel::from_pretrained(dir, cfg);
        } else {
          bm.llama = std::make_shared<models::LlamaModel>(cfg);
        }
        bm.llama->to(dev);
        bm.module = bm.llama;
        models[n.id] = bm;
      } else if (n.kind == "LoadMamba") {
        BuiltModel bm;
        bm.mamba = std::make_shared<models::MambaModel>(mamba_config_from(n));
        bm.mamba->to(dev);
        bm.module = bm.mamba;
        models[n.id] = bm;
      }
    }

    // Dispatch on the sink block.
    for (const auto& n : g.nodes) {
      if (n.kind == "TrainLoop") return run_train(g, n, models, sink, opt);
    }
    for (const auto& n : g.nodes) {
      if (n.kind == "Generate") return run_generate(g, n, models, sink, opt);
    }
    for (const auto& n : g.nodes) {
      if (n.kind == "TensorInspect") return run_tensor(g, n, sink, opt);
    }

    emit_error(sink,
               "graph has no runnable sink (add a TrainLoop, Generate, or "
               "Tensor Inspect)");
    Json fail = Json::object();
    fail.set("ok", Json(false));
    fail.set("reason", Json(std::string("no sink")));
    return fail;
  } catch (const std::exception& ex) {
    emit_error(sink, std::string("execution error: ") + ex.what());
    Json fail = Json::object();
    fail.set("ok", Json(false));
    fail.set("reason", Json(std::string(ex.what())));
    sink({"done", fail});
    return fail;
  }
}

}  // namespace tesseract::studio

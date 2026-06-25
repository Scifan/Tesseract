#include "tesseract/studio/BlockCatalog.hpp"

namespace tesseract::studio {

const PortSpec* BlockSpec::input(const std::string& name) const {
  for (const auto& p : inputs)
    if (p.name == name) return &p;
  return nullptr;
}

const PortSpec* BlockSpec::output(const std::string& name) const {
  for (const auto& p : outputs)
    if (p.name == name) return &p;
  return nullptr;
}

namespace {

ParamSpec pi(std::string name, int64_t def, std::string label) {
  return {std::move(name), "int", Json(def), std::move(label), {}};
}
ParamSpec pf(std::string name, double def, std::string label) {
  return {std::move(name), "float", Json(def), std::move(label), {}};
}
ParamSpec pb(std::string name, bool def, std::string label) {
  return {std::move(name), "bool", Json(def), std::move(label), {}};
}
ParamSpec ps(std::string name, std::string def, std::string label) {
  return {std::move(name), "string", Json(std::move(def)), std::move(label), {}};
}
ParamSpec pe(std::string name, std::string def, std::string label,
             std::vector<std::string> options) {
  return {std::move(name), "enum", Json(std::move(def)), std::move(label),
          std::move(options)};
}

PortSpec port(std::string name, PortType t) { return {std::move(name), t}; }

}  // namespace

BlockCatalog::BlockCatalog() {
  using PT = PortType;

  // ---- Data ------------------------------------------------------------- //
  specs_.push_back({"Input", "Data", "Input", "Input tensor [batch, features].",
                    {}, {port("out", PT::Tensor)},
                    {pi("batch", 32, "Batch size"),
                     pi("features", 8, "Features")}});

  specs_.push_back(
      {"SyntheticClassification", "Data", "Synthetic classification",
       "Gaussian-mixture classification set (in-memory).", {},
       {port("data", PT::Dataset)},
       {pi("samples", 256, "Samples"), pi("features", 8, "Features"),
        pi("classes", 3, "Classes"), pi("seed", 0, "Seed")}});

  specs_.push_back({"CsvDataset", "Data", "CSV dataset",
                    "Numeric CSV; one column is the integer class label.", {},
                    {port("data", PT::Dataset)},
                    {ps("path", "data.csv", "CSV path"),
                     pi("label_col", -1, "Label column (-1 = last)")}});

  // ---- Layers ----------------------------------------------------------- //
  specs_.push_back({"Linear", "Layers", "Linear", "Affine: y = x·Wᵀ + b.",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)},
                    {pi("in_features", 8, "In features"),
                     pi("out_features", 32, "Out features"),
                     pb("bias", true, "Bias")}});
  specs_.push_back({"ReLU", "Layers", "ReLU", "max(0, x).",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)}, {}});
  specs_.push_back({"Sigmoid", "Layers", "Sigmoid", "1 / (1 + e^-x).",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)}, {}});
  specs_.push_back({"Tanh", "Layers", "Tanh", "Hyperbolic tangent.",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)}, {}});
  specs_.push_back({"RMSNorm", "Layers", "RMSNorm", "Llama RMS normalization.",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)},
                    {pi("dim", 32, "Normalized dim"), pf("eps", 1e-5, "Eps")}});
  specs_.push_back({"LayerNorm", "Layers", "LayerNorm", "Standard LayerNorm.",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)},
                    {pi("dim", 32, "Normalized dim"), pf("eps", 1e-5, "Eps"),
                     pb("bias", true, "Bias")}});
  specs_.push_back({"Embedding", "Layers", "Embedding",
                    "Lookup table [num_embeddings, dim].",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)},
                    {pi("num_embeddings", 1000, "Vocab"),
                     pi("embedding_dim", 32, "Embedding dim")}});
  specs_.push_back({"FeedForward", "Layers", "FeedForward (SwiGLU)",
                    "Llama SwiGLU MLP.", {port("in", PT::Tensor)},
                    {port("out", PT::Tensor)},
                    {pi("d_model", 32, "d_model"), pi("d_ff", 64, "d_ff"),
                     pb("bias", false, "Bias")}});
  specs_.push_back({"MultiHeadAttention", "Layers", "Multi-head attention",
                    "Causal MHA / GQA over [B, S, D].",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)},
                    {pi("d_model", 32, "d_model"), pi("num_heads", 4, "Heads"),
                     pi("num_kv_heads", 0, "KV heads (0=MHA)"),
                     pb("causal", true, "Causal")}});
  specs_.push_back({"TransformerBlock", "Layers", "Transformer block",
                    "Pre-norm attention + SwiGLU + residuals.",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)},
                    {pi("d_model", 32, "d_model"), pi("num_heads", 4, "Heads"),
                     pi("d_ff", 64, "d_ff"), pi("num_kv_heads", 0, "KV heads"),
                     pb("causal", true, "Causal")}});

  // ---- Tensor (eager tensor & autograd playground) ---------------------- //
  specs_.push_back(
      {"TensorConst", "Tensor", "Tensor", "A leaf tensor (requires_grad).", {},
       {port("out", PT::Tensor)},
       {pi("rows", 2, "Rows"), pi("cols", 3, "Cols"),
        pe("init", "randn", "Init", {"randn", "zeros", "ones", "arange", "eye"}),
        pf("scale", 1.0, "Scale"), pi("seed", 0, "Seed")}});
  specs_.push_back({"TAdd", "Tensor", "Add (a + b)", "Element-wise add.",
                    {port("a", PT::Tensor), port("b", PT::Tensor)},
                    {port("out", PT::Tensor)}, {}});
  specs_.push_back({"TSub", "Tensor", "Sub (a − b)", "Element-wise subtract.",
                    {port("a", PT::Tensor), port("b", PT::Tensor)},
                    {port("out", PT::Tensor)}, {}});
  specs_.push_back({"TMul", "Tensor", "Mul (a ⊙ b)", "Element-wise multiply.",
                    {port("a", PT::Tensor), port("b", PT::Tensor)},
                    {port("out", PT::Tensor)}, {}});
  specs_.push_back({"TMatMul", "Tensor", "MatMul (a · b)", "Matrix product.",
                    {port("a", PT::Tensor), port("b", PT::Tensor)},
                    {port("out", PT::Tensor)}, {}});
  specs_.push_back({"TReLU", "Tensor", "ReLU(x)", "max(0, x).",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)}, {}});
  specs_.push_back({"TSigmoid", "Tensor", "Sigmoid(x)", "1/(1+e^-x).",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)}, {}});
  specs_.push_back({"TTanh", "Tensor", "Tanh(x)", "Hyperbolic tangent.",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)}, {}});
  specs_.push_back({"TExp", "Tensor", "Exp(x)", "e^x.",
                    {port("in", PT::Tensor)}, {port("out", PT::Tensor)}, {}});
  specs_.push_back(
      {"TensorInspect", "Tensor", "Inspect / backward",
       "Evaluate `in`; show shape + values, optionally run autograd backward "
       "and visualize leaf gradients.",
       {port("in", PT::Tensor)}, {},
       {pb("backward", true, "Run backward (∂Σout/∂leaf)")}});

  // ---- Model ------------------------------------------------------------ //
  specs_.push_back({"SequentialModel", "Model", "Sequential model",
                    "Assemble the layer chain feeding `in` into a model.",
                    {port("in", PT::Tensor)}, {port("model", PT::Model)}, {}});
  specs_.push_back(
      {"LoadLlama", "Model", "Llama model",
       "Build/load a Llama (set model_dir to load HF weights).", {},
       {port("model", PT::Model)},
       {ps("model_dir", "", "HF model dir (blank = random init)"),
        pi("vocab_size", 256, "Vocab"), pi("hidden_size", 64, "Hidden"),
        pi("num_hidden_layers", 2, "Layers"),
        pi("num_attention_heads", 4, "Heads"),
        pi("num_key_value_heads", 4, "KV heads"),
        pi("intermediate_size", 128, "Intermediate"),
        pi("max_position_embeddings", 512, "Max pos"),
        pi("num_experts", 0, "Experts (0=dense)"),
        pi("num_experts_per_tok", 0, "Experts/token"),
        pb("fp16", false, "FP16")}});
  specs_.push_back({"LoadMamba", "Model", "Mamba model",
                    "Build a Mamba (SSM) model.", {},
                    {port("model", PT::Model)},
                    {pi("vocab_size", 256, "Vocab"),
                     pi("hidden_size", 64, "Hidden"),
                     pi("num_hidden_layers", 2, "Layers"),
                     pi("d_state", 16, "d_state"), pi("d_conv", 4, "d_conv"),
                     pi("expand", 2, "Expand")}});

  // ---- Loss ------------------------------------------------------------- //
  specs_.push_back({"CrossEntropyLoss", "Loss", "Cross-entropy loss",
                    "Softmax cross-entropy with integer targets.", {},
                    {port("loss", PT::Loss)}, {}});

  // ---- Optimizer -------------------------------------------------------- //
  specs_.push_back({"SGD", "Optimizer", "SGD",
                    "Stochastic gradient descent (+ momentum).",
                    {port("model", PT::Model)}, {port("opt", PT::Optimizer)},
                    {pf("lr", 0.1, "Learning rate"),
                     pf("momentum", 0.0, "Momentum")}});
  specs_.push_back({"Adam", "Optimizer", "Adam", "Adam optimizer.",
                    {port("model", PT::Model)}, {port("opt", PT::Optimizer)},
                    {pf("lr", 1e-3, "Learning rate"), pf("beta1", 0.9, "Beta1"),
                     pf("beta2", 0.999, "Beta2"), pf("eps", 1e-8, "Eps")}});

  // ---- Tokenizer -------------------------------------------------------- //
  specs_.push_back({"BpeTokenizer", "Tokenizer", "BPE tokenizer",
                    "Load a tokenizer.json (byte-level BPE).", {},
                    {port("tokenizer", PT::Tokenizer)},
                    {ps("path", "tokenizer.json", "tokenizer.json path")}});

  // ---- Train (sink) ----------------------------------------------------- //
  specs_.push_back(
      {"TrainLoop", "Train", "Train loop",
       "Train `model` on `data` with `optimizer` minimizing `loss`.",
       {port("model", PT::Model), port("data", PT::Dataset),
        port("optimizer", PT::Optimizer), port("loss", PT::Loss)},
       {},
       {pi("epochs", 30, "Epochs"), pi("batch_size", 32, "Batch size")}});

  // ---- Inference (sink) ------------------------------------------------- //
  specs_.push_back(
      {"Generate", "Inference", "Generate",
       "Autoregressively generate tokens from `model`.",
       {port("model", PT::Model), port("tokenizer", PT::Tokenizer)},
       {},
       {ps("prompt", "", "Prompt text (needs tokenizer)"),
        ps("prompt_ids", "1, 2, 3", "or prompt ids (comma-sep)"),
        pi("max_new_tokens", 16, "Max new tokens"),
        pb("do_sample", false, "Sample"), pf("temperature", 1.0, "Temperature"),
        pi("top_k", 0, "Top-k"), pf("top_p", 1.0, "Top-p"),
        pi("seed", 0, "Seed")}});
}

const BlockCatalog& BlockCatalog::instance() {
  static const BlockCatalog kCatalog;
  return kCatalog;
}

const BlockSpec* BlockCatalog::find(const std::string& kind) const {
  for (const auto& s : specs_)
    if (s.kind == kind) return &s;
  return nullptr;
}

Json BlockCatalog::default_params(const std::string& kind) const {
  Json out = Json::object();
  const BlockSpec* s = find(kind);
  if (!s) return out;
  for (const auto& p : s->params) out.set(p.name, p.def);
  return out;
}

Json BlockCatalog::to_json() const {
  Json arr = Json::array();
  for (const auto& s : specs_) {
    Json o = Json::object();
    o.set("kind", Json(s.kind));
    o.set("category", Json(s.category));
    o.set("label", Json(s.label));
    o.set("summary", Json(s.summary));
    Json ins = Json::array();
    for (const auto& p : s.inputs) {
      Json po = Json::object();
      po.set("name", Json(p.name));
      po.set("type", Json(std::string(to_string(p.type))));
      ins.push_back(std::move(po));
    }
    o.set("inputs", std::move(ins));
    Json outs = Json::array();
    for (const auto& p : s.outputs) {
      Json po = Json::object();
      po.set("name", Json(p.name));
      po.set("type", Json(std::string(to_string(p.type))));
      outs.push_back(std::move(po));
    }
    o.set("outputs", std::move(outs));
    Json params = Json::array();
    for (const auto& p : s.params) {
      Json po = Json::object();
      po.set("name", Json(p.name));
      po.set("type", Json(p.type));
      po.set("label", Json(p.label));
      po.set("default", p.def);
      if (!p.options.empty()) {
        Json opts = Json::array();
        for (const auto& opt : p.options) opts.push_back(Json(opt));
        po.set("options", std::move(opts));
      }
      params.push_back(std::move(po));
    }
    o.set("params", std::move(params));
    arr.push_back(std::move(o));
  }
  return arr;
}

}  // namespace tesseract::studio

#include "tesseract/studio/Analysis.hpp"

#include <algorithm>

#include "tesseract/studio/BlockCatalog.hpp"

namespace tesseract::studio {

Json AnalysisResult::to_json() const {
  Json o = Json::object();
  o.set("ok", Json(ok));
  Json diags = Json::array();
  for (const auto& d : diagnostics) {
    Json jd = Json::object();
    jd.set("severity", Json(d.severity));
    jd.set("node", Json(d.node));
    jd.set("message", Json(d.message));
    diags.push_back(std::move(jd));
  }
  o.set("diagnostics", std::move(diags));
  Json shapes = Json::object();
  for (const auto& [id, shape] : out_shapes) {
    Json arr = Json::array();
    for (int64_t d : shape) arr.push_back(Json(d));
    shapes.set(std::to_string(id), std::move(arr));
  }
  o.set("shapes", std::move(shapes));
  return o;
}

namespace {

std::string shape_str(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    out += std::to_string(s[i]);
    if (i + 1 < s.size()) out += ", ";
  }
  out += "]";
  return out;
}

// Propagate a Tensor shape through one layer node. Records warnings on
// dimension mismatch. Returns the output shape (may equal input).
std::vector<int64_t> propagate(const Node& n, const std::vector<int64_t>& in,
                               std::vector<Diagnostic>& diags) {
  auto warn_last = [&](int64_t expect, const char* what) {
    if (!in.empty() && in.back() != expect) {
      diags.push_back({"warning", n.id,
                       n.kind + ": " + what + " expects last dim " +
                           std::to_string(expect) + " but got " +
                           shape_str(in)});
    }
  };
  if (n.kind == "Linear") {
    warn_last(n.param_int("in_features"), "in_features");
    std::vector<int64_t> out = in;
    if (!out.empty()) out.back() = n.param_int("out_features");
    return out;
  }
  if (n.kind == "RMSNorm" || n.kind == "LayerNorm") {
    warn_last(n.param_int("dim"), "dim");
    return in;
  }
  if (n.kind == "FeedForward" || n.kind == "MultiHeadAttention" ||
      n.kind == "TransformerBlock") {
    warn_last(n.param_int("d_model"), "d_model");
    return in;
  }
  if (n.kind == "Embedding") {
    std::vector<int64_t> out = in;
    out.push_back(n.param_int("embedding_dim"));
    return out;
  }
  // ReLU / Sigmoid / Tanh and anything else: identity.
  return in;
}

}  // namespace

AnalysisResult analyze(const BlockGraph& g) {
  AnalysisResult r;
  const BlockCatalog& cat = BlockCatalog::instance();

  // 1. Unknown kinds.
  for (const auto& n : g.nodes) {
    if (!cat.find(n.kind))
      r.diagnostics.push_back(
          {"error", n.id, "unknown block kind '" + n.kind + "'"});
  }

  // 2. Edge endpoints + type compatibility.
  for (const auto& e : g.edges) {
    const Node* fn = g.find_node(e.from.node);
    const Node* tn = g.find_node(e.to.node);
    if (!fn || !tn) {
      r.diagnostics.push_back(
          {"error", -1, "edge references a missing node"});
      continue;
    }
    const BlockSpec* fs = cat.find(fn->kind);
    const BlockSpec* ts = cat.find(tn->kind);
    if (!fs || !ts) continue;
    const PortSpec* fp = fs->output(e.from.port);
    const PortSpec* tp = ts->input(e.to.port);
    if (!fp) {
      r.diagnostics.push_back({"error", fn->id,
                               "no output port '" + e.from.port + "'"});
      continue;
    }
    if (!tp) {
      r.diagnostics.push_back({"error", tn->id,
                               "no input port '" + e.to.port + "'"});
      continue;
    }
    if (fp->type != tp->type && fp->type != PortType::Any &&
        tp->type != PortType::Any) {
      r.diagnostics.push_back(
          {"error", tn->id,
           std::string("type mismatch: ") + fn->kind + "." + fp->name + " (" +
               to_string(fp->type) + ") -> " + tn->kind + "." + tp->name +
               " (" + to_string(tp->type) + ")"});
    }
  }

  // 3. Required inputs connected. Generate.tokenizer is optional.
  for (const auto& n : g.nodes) {
    const BlockSpec* s = cat.find(n.kind);
    if (!s) continue;
    for (const auto& in : s->inputs) {
      const bool optional = (n.kind == "Generate" && in.name == "tokenizer");
      if (!optional && !g.incoming(n.id, in.name)) {
        r.diagnostics.push_back(
            {"error", n.id,
             n.kind + ": input '" + in.name + "' is not connected"});
      }
    }
  }

  // 4. DAG-ness.
  std::vector<int64_t> order;
  try {
    order = g.topo_order();
  } catch (const std::exception& ex) {
    r.diagnostics.push_back({"error", -1, ex.what()});
  }

  // 5. Tensor shape propagation along the chain (best-effort).
  if (!order.empty()) {
    auto shape_of = [&](int64_t node) -> const std::vector<int64_t>* {
      auto it = r.out_shapes.find(node);
      return it == r.out_shapes.end() ? nullptr : &it->second;
    };
    for (int64_t id : order) {
      const Node* n = g.find_node(id);
      if (!n) continue;
      if (n->kind == "Input") {
        r.out_shapes[id] = {n->param_int("batch", 32),
                            n->param_int("features", 8)};
        continue;
      }
      if (n->kind == "TensorConst") {
        r.out_shapes[id] = {n->param_int("rows", 2), n->param_int("cols", 3)};
        continue;
      }
      // Binary tensor ops carry ports a/b.
      if (n->kind == "TAdd" || n->kind == "TSub" || n->kind == "TMul" ||
          n->kind == "TMatMul") {
        const Edge* ea = g.incoming(id, "a");
        const Edge* eb = g.incoming(id, "b");
        const std::vector<int64_t>* sa = ea ? shape_of(ea->from.node) : nullptr;
        const std::vector<int64_t>* sb = eb ? shape_of(eb->from.node) : nullptr;
        if (sa && sb) {
          if (n->kind == "TMatMul") {
            if (sa->size() == 2 && sb->size() == 2) {
              if (sa->back() != sb->front())
                r.diagnostics.push_back(
                    {"warning", id,
                     "MatMul inner dims disagree: " + shape_str(*sa) + " · " +
                         shape_str(*sb)});
              r.out_shapes[id] = {(*sa)[0], (*sb)[1]};
            }
          } else {
            if (*sa != *sb)
              r.diagnostics.push_back(
                  {"warning", id,
                   n->kind + ": shapes differ " + shape_str(*sa) + " vs " +
                       shape_str(*sb)});
            r.out_shapes[id] = *sa;
          }
        }
        continue;
      }
      // Find the Tensor-typed input edge (port "in").
      const Edge* in_edge = g.incoming(id, "in");
      if (in_edge) {
        auto it = r.out_shapes.find(in_edge->from.node);
        if (it != r.out_shapes.end())
          r.out_shapes[id] = propagate(*n, it->second, r.diagnostics);
      }
    }
  }

  r.ok = std::none_of(r.diagnostics.begin(), r.diagnostics.end(),
                      [](const Diagnostic& d) { return d.severity == "error"; });
  return r;
}

}  // namespace tesseract::studio

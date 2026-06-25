#include "tesseract/studio/BlockGraph.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace tesseract::studio {

const char* to_string(PortType t) {
  switch (t) {
    case PortType::Tensor:    return "Tensor";
    case PortType::Model:     return "Model";
    case PortType::Params:    return "Params";
    case PortType::Optimizer: return "Optimizer";
    case PortType::Loss:      return "Loss";
    case PortType::Tokenizer: return "Tokenizer";
    case PortType::Dataset:   return "Dataset";
    case PortType::Scalar:    return "Scalar";
    case PortType::Any:       return "Any";
  }
  return "Any";
}

PortType port_type_from_string(const std::string& s) {
  if (s == "Tensor")    return PortType::Tensor;
  if (s == "Model")     return PortType::Model;
  if (s == "Params")    return PortType::Params;
  if (s == "Optimizer") return PortType::Optimizer;
  if (s == "Loss")      return PortType::Loss;
  if (s == "Tokenizer") return PortType::Tokenizer;
  if (s == "Dataset")   return PortType::Dataset;
  if (s == "Scalar")    return PortType::Scalar;
  return PortType::Any;
}

int64_t BlockGraph::add_node(const std::string& kind, Json params, double x,
                             double y) {
  Node n;
  n.id = next_id_++;
  n.kind = kind;
  n.params = params.is_object() ? std::move(params) : Json::object();
  n.x = x;
  n.y = y;
  nodes.push_back(std::move(n));
  return nodes.back().id;
}

Node* BlockGraph::find_node(int64_t id) {
  for (auto& n : nodes)
    if (n.id == id) return &n;
  return nullptr;
}

const Node* BlockGraph::find_node(int64_t id) const {
  for (const auto& n : nodes)
    if (n.id == id) return &n;
  return nullptr;
}

void BlockGraph::connect(int64_t from_node, const std::string& from_port,
                         int64_t to_node, const std::string& to_port) {
  Edge e;
  e.from = {from_node, from_port};
  e.to = {to_node, to_port};
  edges.push_back(e);
}

const Edge* BlockGraph::incoming(int64_t node, const std::string& port) const {
  for (const auto& e : edges)
    if (e.to.node == node && e.to.port == port) return &e;
  return nullptr;
}

std::vector<const Edge*> BlockGraph::outgoing(int64_t node,
                                              const std::string& port) const {
  std::vector<const Edge*> out;
  for (const auto& e : edges)
    if (e.from.node == node && e.from.port == port) out.push_back(&e);
  return out;
}

std::vector<int64_t> BlockGraph::topo_order() const {
  std::unordered_map<int64_t, int> indeg;
  std::unordered_map<int64_t, std::vector<int64_t>> succ;
  for (const auto& n : nodes) indeg[n.id] = 0;
  for (const auto& e : edges) {
    if (e.from.node == e.to.node) continue;  // self-loop guard
    succ[e.from.node].push_back(e.to.node);
    indeg[e.to.node] += 1;
  }
  // Seed with indegree-0 nodes in declaration order for stable output.
  std::vector<int64_t> ready;
  for (const auto& n : nodes)
    if (indeg[n.id] == 0) ready.push_back(n.id);

  std::vector<int64_t> order;
  while (!ready.empty()) {
    int64_t id = ready.front();
    ready.erase(ready.begin());
    order.push_back(id);
    for (int64_t s : succ[id]) {
      if (--indeg[s] == 0) ready.push_back(s);
    }
  }
  if (order.size() != nodes.size())
    throw std::runtime_error("BlockGraph: cycle detected (graph must be a DAG)");
  return order;
}

Json BlockGraph::to_json() const {
  Json j = Json::object();
  j.set("tsb_version", Json(1));
  j.set("name", Json(name));
  j.set("device", Json(device));
  Json jn = Json::array();
  for (const auto& n : nodes) {
    Json o = Json::object();
    o.set("id", Json(n.id));
    o.set("kind", Json(n.kind));
    o.set("params", n.params);
    o.set("x", Json(n.x));
    o.set("y", Json(n.y));
    jn.push_back(std::move(o));
  }
  j.set("nodes", std::move(jn));
  Json je = Json::array();
  for (const auto& e : edges) {
    Json o = Json::object();
    Json f = Json::object();
    f.set("node", Json(e.from.node));
    f.set("port", Json(e.from.port));
    Json t = Json::object();
    t.set("node", Json(e.to.node));
    t.set("port", Json(e.to.port));
    o.set("from", std::move(f));
    o.set("to", std::move(t));
    je.push_back(std::move(o));
  }
  j.set("edges", std::move(je));
  return j;
}

BlockGraph BlockGraph::from_json(const Json& j) {
  BlockGraph g;
  g.name = j.value("name").as_string("untitled");
  g.device = j.value("device").as_string("cpu");
  int64_t max_id = 0;
  for (const auto& jn : j.value("nodes").items()) {
    Node n;
    n.id = jn.value("id").as_int(-1);
    n.kind = jn.value("kind").as_string();
    n.params = jn.contains("params") ? jn.at("params") : Json::object();
    if (!n.params.is_object()) n.params = Json::object();
    n.x = jn.value("x").as_number(0.0);
    n.y = jn.value("y").as_number(0.0);
    max_id = std::max(max_id, n.id);
    g.nodes.push_back(std::move(n));
  }
  for (const auto& je : j.value("edges").items()) {
    Edge e;
    const Json& f = je.value("from");
    const Json& t = je.value("to");
    e.from = {f.value("node").as_int(-1), f.value("port").as_string()};
    e.to = {t.value("node").as_int(-1), t.value("port").as_string()};
    g.edges.push_back(e);
  }
  g.next_id_ = max_id + 1;
  return g;
}

}  // namespace tesseract::studio

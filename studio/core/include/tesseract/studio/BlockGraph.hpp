// Tesseract Studio — block-graph data model (M5 / B-047).
//
// A BlockGraph is the in-memory representation of a visual program: a set of
// typed nodes (blocks) wired together through named ports. It is the single
// source of truth that the UI edits, the validator/shape-inference reads, the
// code generators emit from, and the executor runs. It serializes to/from the
// `.tsb` JSON format (fully bidirectional).

#ifndef TESSERACT_STUDIO_BLOCKGRAPH_HPP
#define TESSERACT_STUDIO_BLOCKGRAPH_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tesseract/studio/Json.hpp"

namespace tesseract::studio {

// The wire/port value categories. Connections are only legal between matching
// types (Any matches anything — used for generic pass-through ports).
enum class PortType {
  Tensor,
  Model,
  Params,      // a parameter list feeding an optimizer
  Optimizer,
  Loss,
  Tokenizer,
  Dataset,
  Scalar,      // a numeric/string config value
  Any,
};

const char* to_string(PortType t);
PortType port_type_from_string(const std::string& s);

struct Endpoint {
  int64_t node = -1;
  std::string port;
};

struct Edge {
  Endpoint from;  // an output port
  Endpoint to;    // an input port
};

struct Node {
  int64_t id = -1;
  std::string kind;     // catalog key, e.g. "Linear", "LlamaModel", "Adam"
  Json params;          // object: free-form, schema defined by the catalog
  double x = 0.0;       // canvas position (UI only)
  double y = 0.0;

  Node() : params(Json::object()) {}

  int64_t param_int(const std::string& key, int64_t d = 0) const {
    return params.value(key).as_int(d);
  }
  double param_num(const std::string& key, double d = 0.0) const {
    return params.value(key).as_number(d);
  }
  bool param_bool(const std::string& key, bool d = false) const {
    return params.value(key).as_bool(d);
  }
  std::string param_str(const std::string& key, const std::string& d = "") const {
    return params.value(key).as_string(d);
  }
};

class BlockGraph {
 public:
  std::string name = "untitled";
  std::string device = "cpu";  // "cpu" | "cuda"
  std::vector<Node> nodes;
  std::vector<Edge> edges;

  // Node management. add_node returns a freshly-allocated unique id.
  int64_t add_node(const std::string& kind, Json params = Json::object(),
                   double x = 0.0, double y = 0.0);
  Node* find_node(int64_t id);
  const Node* find_node(int64_t id) const;
  void connect(int64_t from_node, const std::string& from_port,
               int64_t to_node, const std::string& to_port);

  // Returns the edge feeding the given input port, if any.
  const Edge* incoming(int64_t node, const std::string& port) const;
  // All edges leaving the given output port.
  std::vector<const Edge*> outgoing(int64_t node, const std::string& port) const;

  // Topological order of node ids (Kahn). Throws on a cycle.
  std::vector<int64_t> topo_order() const;

  // .tsb serialization (bidirectional).
  Json to_json() const;
  static BlockGraph from_json(const Json& j);
  std::string to_tsb(int indent = 2) const { return to_json().dump(indent); }
  static BlockGraph from_tsb(const std::string& text) {
    return from_json(Json::parse(text));
  }

 private:
  int64_t next_id_ = 1;
};

}  // namespace tesseract::studio

#endif  // TESSERACT_STUDIO_BLOCKGRAPH_HPP

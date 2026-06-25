// Tesseract Studio — block catalog (M5 / B-047).
//
// The catalog is the single declarative source of every block kind: its
// category, ports (typed in/out), and editable parameters with defaults. The
// UI builds its palette + node editors from it, the validator checks ports +
// required params against it, and codegen/executor switch on the same `kind`
// strings. Adding a block = adding one BlockSpec here (plus a codegen/executor
// case where it does real work).

#ifndef TESSERACT_STUDIO_BLOCKCATALOG_HPP
#define TESSERACT_STUDIO_BLOCKCATALOG_HPP

#include <string>
#include <vector>

#include "tesseract/studio/BlockGraph.hpp"
#include "tesseract/studio/Json.hpp"

namespace tesseract::studio {

struct ParamSpec {
  std::string name;
  std::string type;     // "int" | "float" | "bool" | "string" | "enum"
  Json def;             // default value
  std::string label;
  std::vector<std::string> options;  // for "enum"
};

struct PortSpec {
  std::string name;
  PortType type;
};

struct BlockSpec {
  std::string kind;
  std::string category;
  std::string label;
  std::string summary;
  std::vector<PortSpec> inputs;
  std::vector<PortSpec> outputs;
  std::vector<ParamSpec> params;

  const PortSpec* input(const std::string& name) const;
  const PortSpec* output(const std::string& name) const;
};

class BlockCatalog {
 public:
  // The process-wide catalog (built once).
  static const BlockCatalog& instance();

  const BlockSpec* find(const std::string& kind) const;
  const std::vector<BlockSpec>& specs() const { return specs_; }

  // Serialize the whole catalog for the web UI palette.
  Json to_json() const;

  // Convenience: a node's params merged over its spec defaults.
  Json default_params(const std::string& kind) const;

 private:
  BlockCatalog();
  std::vector<BlockSpec> specs_;
};

}  // namespace tesseract::studio

#endif  // TESSERACT_STUDIO_BLOCKCATALOG_HPP

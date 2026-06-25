// Tesseract Studio — validation + shape inference (M5 / B-047).
//
// Runs on every graph edit (and before execution). It checks the graph against
// the catalog (known kinds, real endpoints, type-compatible wires, required
// inputs connected, DAG-ness) and propagates tensor shapes along the layer
// chain so the UI can surface shape mismatches before anything runs.

#ifndef TESSERACT_STUDIO_ANALYSIS_HPP
#define TESSERACT_STUDIO_ANALYSIS_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "tesseract/studio/BlockGraph.hpp"
#include "tesseract/studio/Json.hpp"

namespace tesseract::studio {

struct Diagnostic {
  std::string severity;  // "error" | "warning"
  int64_t node = -1;     // -1 = graph-level
  std::string message;
};

struct AnalysisResult {
  bool ok = true;  // false if any "error" diagnostic exists
  std::vector<Diagnostic> diagnostics;
  // Inferred shape of each node's primary Tensor output (empty if unknown).
  std::map<int64_t, std::vector<int64_t>> out_shapes;

  Json to_json() const;
};

AnalysisResult analyze(const BlockGraph& g);

}  // namespace tesseract::studio

#endif  // TESSERACT_STUDIO_ANALYSIS_HPP

// Tesseract Studio — code generation (M5 / B-047).
//
// Emits a runnable program from a BlockGraph in either C++ (against the
// tesseract C++ API) or Python (against the `tesseract` package). Every
// generated file carries a compact `@tsb {json}` header so Studio can re-open
// (round-trip) a file it produced back into blocks — the pragmatic reading of
// "bidirectional code <-> blocks": .tsb JSON is fully bidirectional, and
// generated source round-trips via its embedded header.

#ifndef TESSERACT_STUDIO_CODEGEN_HPP
#define TESSERACT_STUDIO_CODEGEN_HPP

#include <optional>
#include <string>

#include "tesseract/studio/BlockGraph.hpp"

namespace tesseract::studio {

std::string generate_cpp(const BlockGraph& g);
std::string generate_python(const BlockGraph& g);

// Emit a textual MLIR module in the `tesseract` dialect that the block graph
// lowers to — the visual realization of the "one IR" philosophy. Display-only
// (it mirrors the dialect's real op set/assembly so it reads like the genuine
// article, but is not fed back into the editor).
std::string generate_ir(const BlockGraph& g);

// Recover a BlockGraph from the `@tsb {json}` header of a generated file
// (C++ `//` or Python `#`). Returns nullopt if no header is present.
std::optional<BlockGraph> extract_tsb(const std::string& source);

}  // namespace tesseract::studio

#endif  // TESSERACT_STUDIO_CODEGEN_HPP

#pragma once

// SSA value and attribute descriptors used by the C++-level graph IR.
//
// Values carry (shape, dtype, device) metadata and a stable integer id that
// uniquely identifies them inside a single Graph. They do NOT carry storage:
// recorded Tensors remain the owner of the underlying buffer.
//
// Attributes are a small value-typed variant used to annotate ops (axis
// indices, keepdim flags, permutation orderings, etc.). Keeping them
// lightweight lets the graph stay cheap to construct during eager training.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"

namespace tesseract::graph {

using ValueId = int64_t;

inline constexpr ValueId kInvalidValueId = -1;

// Provenance of a value inside the graph. `kParam` and `kInput` are both
// graph-level inputs; the distinction matters to the autograd pass (params
// accumulate gradients, inputs do not).
enum class ValueKind : std::uint8_t {
  kIntermediate = 0,
  kInput = 1,
  kParam = 2,
  kConstant = 3,
};

struct Value {
  ValueId id{kInvalidValueId};
  Shape shape;
  DType dtype{DType::Float32};
  Device device{};
  ValueKind kind{ValueKind::kIntermediate};
  // Human-readable label; optional, used only by the pretty-printer.
  std::string name;
};

// Minimal attribute variant. We intentionally keep the set small; exotic
// attributes (e.g. distributed-sharding specs) come in later milestones.
using Attr = std::variant<std::monostate,
                          int64_t,
                          double,
                          bool,
                          std::string,
                          std::vector<int64_t>>;

using AttrMap = std::unordered_map<std::string, Attr>;

}  // namespace tesseract::graph

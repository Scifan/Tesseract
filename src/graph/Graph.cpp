#include "tesseract/graph/Graph.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>

#include "tesseract/utils/Logging.hpp"

namespace tesseract::graph {

namespace {

std::string attr_to_string(const Attr& a) {
  struct Visitor {
    std::string operator()(std::monostate) const { return "()"; }
    std::string operator()(int64_t v) const { return std::to_string(v); }
    std::string operator()(double v) const {
      std::ostringstream os; os << v; return os.str();
    }
    std::string operator()(bool v) const { return v ? "true" : "false"; }
    std::string operator()(const std::string& s) const { return "\"" + s + "\""; }
    std::string operator()(const std::vector<int64_t>& v) const {
      std::ostringstream os; os << "[";
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) os << ", ";
        os << v[i];
      }
      os << "]";
      return os.str();
    }
  };
  return std::visit(Visitor{}, a);
}

}  // namespace

ValueId Graph::new_value(Shape shape, DType dtype, Device device,
                         ValueKind kind, std::string name) {
  Value v{};
  v.id = static_cast<ValueId>(values_.size());
  v.shape = std::move(shape);
  v.dtype = dtype;
  v.device = device;
  v.kind = kind;
  v.name = std::move(name);
  values_.push_back(std::move(v));
  return values_.back().id;
}

ValueId Graph::add_input(Shape shape, DType dtype, Device device, std::string name) {
  ValueId id = new_value(std::move(shape), dtype, device, ValueKind::kInput, std::move(name));
  inputs_.push_back(id);
  return id;
}

ValueId Graph::add_param(Shape shape, DType dtype, Device device, std::string name) {
  ValueId id = new_value(std::move(shape), dtype, device, ValueKind::kParam, std::move(name));
  params_.push_back(id);
  return id;
}

ValueId Graph::add_constant(Shape shape, DType dtype, Device device, std::string name) {
  ValueId id = new_value(std::move(shape), dtype, device, ValueKind::kConstant, std::move(name));
  constants_.push_back(id);
  return id;
}

void Graph::mark_output(ValueId id) {
  TESSERACT_CHECK(id >= 0 && id < static_cast<ValueId>(values_.size()),
                  "mark_output: id {} out of range (num_values={})", id, values_.size());
  if (std::find(outputs_.begin(), outputs_.end(), id) == outputs_.end()) {
    outputs_.push_back(id);
  }
}

std::size_t Graph::add_op(std::string kind,
                          std::vector<ValueId> inputs,
                          std::vector<ValueId> outputs,
                          AttrMap attrs) {
  Op op;
  op.kind = std::move(kind);
  op.inputs = std::move(inputs);
  op.outputs = std::move(outputs);
  op.attrs = std::move(attrs);
  const std::size_t idx = ops_.size();
  ops_.push_back(std::move(op));
  return idx;
}

const Value& Graph::value(ValueId id) const {
  TESSERACT_CHECK(id >= 0 && id < static_cast<ValueId>(values_.size()),
                  "Graph::value({}) out of range (num_values={})", id, values_.size());
  return values_[static_cast<std::size_t>(id)];
}

std::string Graph::to_string() const {
  std::ostringstream os;
  os << "graph {\n";

  auto print_val = [&](ValueId id) {
    const Value& v = this->value(id);
    os << "  %" << v.id;
    if (!v.name.empty()) os << " (\"" << v.name << "\")";
    os << " : " << v.shape.to_string() << ":" << dtype_name(v.dtype);
    if (v.device.is_cuda()) os << "[cuda:" << v.device.index << "]";
    os << "\n";
  };

  if (!inputs_.empty()) {
    os << "  # inputs:\n";
    for (ValueId id : inputs_) print_val(id);
  }
  if (!params_.empty()) {
    os << "  # params:\n";
    for (ValueId id : params_) print_val(id);
  }
  if (!constants_.empty()) {
    os << "  # constants:\n";
    for (ValueId id : constants_) print_val(id);
  }

  os << "  # ops:\n";
  for (const Op& op : ops_) {
    os << "  ";
    for (std::size_t i = 0; i < op.outputs.size(); ++i) {
      if (i) os << ", ";
      os << "%" << op.outputs[i];
    }
    if (!op.outputs.empty()) os << " = ";
    os << "tesseract." << op.kind << "(";
    for (std::size_t i = 0; i < op.inputs.size(); ++i) {
      if (i) os << ", ";
      os << "%" << op.inputs[i];
    }
    os << ")";
    if (!op.attrs.empty()) {
      os << " {";
      bool first = true;
      for (const auto& [k, v] : op.attrs) {
        if (!first) os << ", ";
        first = false;
        os << k << "=" << attr_to_string(v);
      }
      os << "}";
    }
    os << "\n";
  }

  if (!outputs_.empty()) {
    os << "  return";
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      os << (i == 0 ? " " : ", ") << "%" << outputs_[i];
    }
    os << "\n";
  }

  os << "}\n";
  return os.str();
}

}  // namespace tesseract::graph

#include "tesseract/ir/Emit.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"

#include "tesseract/ir/TesseractDialect.h"
#include "tesseract/ir/TesseractOps.h"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::ir {

namespace {

mlir::Type dtype_to_mlir(mlir::MLIRContext& ctx, DType dt) {
  switch (dt) {
    case DType::Float32: return mlir::Float32Type::get(&ctx);
    case DType::Float64: return mlir::Float64Type::get(&ctx);
    case DType::Int32:   return mlir::IntegerType::get(&ctx, 32);
    case DType::Int64:   return mlir::IntegerType::get(&ctx, 64);
    case DType::Bool:    return mlir::IntegerType::get(&ctx, 1);
    default: break;
  }
  TESSERACT_CHECK(false, "emit_mlir: unsupported dtype {}", static_cast<int>(dt));
  return {};
}

mlir::RankedTensorType value_to_type(mlir::MLIRContext& ctx,
                                     const graph::Value& v) {
  llvm::SmallVector<int64_t, 8> dims;
  dims.reserve(v.shape.rank());
  for (std::size_t i = 0; i < v.shape.rank(); ++i) dims.push_back(v.shape[i]);
  return mlir::RankedTensorType::get(dims, dtype_to_mlir(ctx, v.dtype));
}

mlir::Attribute attr_to_mlir(mlir::OpBuilder& b, const std::string& key,
                             const graph::Attr& a) {
  return std::visit(
      [&](auto&& v) -> mlir::Attribute {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return b.getUnitAttr();
        } else if constexpr (std::is_same_v<T, int64_t>) {
          // Default to signed i64 — matches the SI64Attr constraint on
          // reduction / transpose ops in TesseractOps.td.
          auto ty = b.getIntegerType(64, /*isSigned=*/true);
          return mlir::IntegerAttr::get(ty, llvm::APInt(64, v, /*isSigned=*/true));
        } else if constexpr (std::is_same_v<T, double>) {
          return b.getF64FloatAttr(v);
        } else if constexpr (std::is_same_v<T, bool>) {
          return b.getBoolAttr(v);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return b.getStringAttr(v);
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          return b.getI64ArrayAttr(v);
        } else {
          TESSERACT_CHECK(false, "emit_mlir: unhandled attr variant for key '{}'", key);
          return {};
        }
      },
      a);
}

// Return a fully-qualified op name (e.g. "tesseract.add") from a bare
// mnemonic as stored in graph::Op::kind.
std::string qualified_op_name(const std::string& mnemonic) {
  return "tesseract." + mnemonic;
}

struct EmitContext {
  mlir::MLIRContext& ctx;
  mlir::OpBuilder builder;
  const graph::Graph& g;
  // ValueId → mlir::Value inside the function body.
  std::unordered_map<graph::ValueId, mlir::Value> value_map;

  EmitContext(mlir::MLIRContext& c, const graph::Graph& graph)
      : ctx(c), builder(&c), g(graph) {}

  mlir::Location loc() { return builder.getUnknownLoc(); }

  void set(graph::ValueId id, mlir::Value v) {
    TESSERACT_CHECK(!value_map.count(id),
                    "emit_mlir: duplicate ValueId {} in value_map", id);
    value_map.emplace(id, v);
  }

  mlir::Value get(graph::ValueId id) const {
    auto it = value_map.find(id);
    TESSERACT_CHECK(it != value_map.end(),
                    "emit_mlir: missing ValueId {} in value_map (op uses an "
                    "SSA value that was never produced)",
                    id);
    return it->second;
  }

  // Binary elementwise ops in the `tesseract` dialect carry
  // `SameOperandsAndResultType` so any implicit broadcast that the eager op
  // would have handled silently must be materialized as a
  // `tesseract.broadcast_to` before the op. We do this in the emitter rather
  // than in the C++ recorder so the graph IR stays a faithful transcript of
  // the user program, with the "pre-broadcast" contract a property of the
  // MLIR lowering (ADR-0004).
  mlir::Value maybe_broadcast(mlir::Value v, mlir::RankedTensorType target) {
    auto t = mlir::cast<mlir::RankedTensorType>(v.getType());
    if (t == target) return v;
    TESSERACT_CHECK(t.getElementType() == target.getElementType(),
                    "emit_mlir: broadcast across dtypes is not allowed");

    llvm::SmallVector<int64_t, 8> target_dims(target.getShape().begin(),
                                              target.getShape().end());
    mlir::OperationState state(loc(), qualified_op_name("broadcast_to"));
    state.addOperands({v});
    state.addTypes({target});
    state.addAttribute("shape", builder.getI64ArrayAttr(target_dims));
    mlir::Operation* op = builder.create(state);
    return op->getResult(0);
  }

  static bool is_binary_elementwise(const std::string& kind) {
    return kind == "add" || kind == "sub" || kind == "mul" || kind == "div";
  }

  void emit_one_op(const graph::Op& op) {
    llvm::SmallVector<mlir::Value, 4> operands;
    operands.reserve(op.inputs.size());
    for (graph::ValueId id : op.inputs) operands.push_back(get(id));

    llvm::SmallVector<mlir::Type, 4> result_types;
    result_types.reserve(op.outputs.size());
    for (graph::ValueId id : op.outputs) {
      result_types.push_back(value_to_type(ctx, g.value(id)));
    }

    if (is_binary_elementwise(op.kind) && result_types.size() == 1 &&
        operands.size() == 2) {
      auto target = mlir::cast<mlir::RankedTensorType>(result_types[0]);
      operands[0] = maybe_broadcast(operands[0], target);
      operands[1] = maybe_broadcast(operands[1], target);
    }

    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    attrs.reserve(op.attrs.size());
    for (const auto& [k, v] : op.attrs) {
      attrs.emplace_back(builder.getStringAttr(k),
                         attr_to_mlir(builder, k, v));
    }

    mlir::OperationState state(loc(), qualified_op_name(op.kind));
    state.addOperands(operands);
    state.addTypes(result_types);
    state.addAttributes(attrs);

    mlir::Operation* created = builder.create(state);
    TESSERACT_CHECK(created != nullptr,
                    "emit_mlir: failed to create op '{}'", op.kind);

    for (std::size_t i = 0; i < op.outputs.size(); ++i) {
      set(op.outputs[i], created->getResult(i));
    }
  }
};

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> emit_mlir(mlir::MLIRContext& ctx,
                                            const graph::Graph& graph,
                                            const EmitOptions& opts) {
  ctx.getOrLoadDialect<TesseractDialect>();

  mlir::OpBuilder top_builder(&ctx);
  auto unknown = top_builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(unknown);

  top_builder.setInsertionPointToEnd(module->getBody());

  // tesseract.graph { ... }
  auto graph_op = top_builder.create<GraphOp>(unknown);
  mlir::Block* graph_block = &graph_op.getBody().emplaceBlock();

  // Build function signature: block arguments are inputs followed by
  // params. Outputs become the function's result types.
  llvm::SmallVector<mlir::Type, 8> arg_types;
  arg_types.reserve(graph.inputs().size() + graph.params().size());
  for (graph::ValueId id : graph.inputs()) {
    arg_types.push_back(value_to_type(ctx, graph.value(id)));
  }
  for (graph::ValueId id : graph.params()) {
    arg_types.push_back(value_to_type(ctx, graph.value(id)));
  }

  llvm::SmallVector<mlir::Type, 4> result_types;
  result_types.reserve(graph.outputs().size());
  for (graph::ValueId id : graph.outputs()) {
    result_types.push_back(value_to_type(ctx, graph.value(id)));
  }

  auto func_type = mlir::FunctionType::get(&ctx, arg_types, result_types);

  mlir::OpBuilder graph_builder(&ctx);
  graph_builder.setInsertionPointToEnd(graph_block);
  auto func_op = graph_builder.create<FunctionOp>(
      unknown,
      /*sym_name=*/graph_builder.getStringAttr(opts.function_name),
      /*function_type=*/mlir::TypeAttr::get(func_type));

  mlir::Block* body = &func_op.getBody().emplaceBlock();
  llvm::SmallVector<mlir::Location, 8> arg_locs(arg_types.size(), unknown);
  body->addArguments(arg_types, arg_locs);

  EmitContext ec(ctx, graph);
  ec.builder.setInsertionPointToEnd(body);

  // Map inputs → block args (straight pass-through).
  std::size_t arg_idx = 0;
  for (graph::ValueId id : graph.inputs()) {
    ec.set(id, body->getArgument(arg_idx++));
  }
  // Map params → tesseract.param(block-arg) so passes can find trainable
  // leaves without inspecting the parent function signature.
  for (graph::ValueId id : graph.params()) {
    mlir::Value block_arg = body->getArgument(arg_idx++);
    const auto& v = graph.value(id);
    auto tagged = ec.builder.create<ParamOp>(
        unknown,
        /*result=*/value_to_type(ctx, v),
        /*input=*/block_arg,
        /*name=*/ec.builder.getStringAttr(v.name.empty()
                                              ? ("param_" + std::to_string(id))
                                              : v.name));
    ec.set(id, tagged.getResult());
  }

  for (const graph::Op& op : graph.ops()) {
    ec.emit_one_op(op);
  }

  llvm::SmallVector<mlir::Value, 4> return_values;
  return_values.reserve(graph.outputs().size());
  for (graph::ValueId id : graph.outputs()) {
    return_values.push_back(ec.get(id));
  }
  ec.builder.create<ReturnOp>(unknown, return_values);

  // Final verification: if the emitter built something the dialect can't
  // stomach, fail loudly here rather than down the lowering pipeline.
  if (mlir::failed(mlir::verify(*module))) {
    std::string dump;
    llvm::raw_string_ostream os(dump);
    module->print(os);
    throw std::runtime_error(
        "tesseract::ir::emit_mlir produced a module that failed verify():\n" +
        dump);
  }
  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> emit_func_mlir(mlir::MLIRContext& ctx,
                                                 const graph::Graph& graph,
                                                 const EmitOptions& opts) {
  ctx.getOrLoadDialect<TesseractDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();

  mlir::OpBuilder top_builder(&ctx);
  auto unknown = top_builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(unknown);
  top_builder.setInsertionPointToEnd(module->getBody());

  // Build func type: block args = inputs ++ params; results = outputs.
  llvm::SmallVector<mlir::Type, 8> arg_types;
  arg_types.reserve(graph.inputs().size() + graph.params().size());
  for (graph::ValueId id : graph.inputs())
    arg_types.push_back(value_to_type(ctx, graph.value(id)));
  for (graph::ValueId id : graph.params())
    arg_types.push_back(value_to_type(ctx, graph.value(id)));

  llvm::SmallVector<mlir::Type, 4> result_types;
  result_types.reserve(graph.outputs().size());
  for (graph::ValueId id : graph.outputs())
    result_types.push_back(value_to_type(ctx, graph.value(id)));

  auto func_type = mlir::FunctionType::get(&ctx, arg_types, result_types);
  auto func_op =
      top_builder.create<mlir::func::FuncOp>(unknown, opts.function_name,
                                             func_type);
  // Tell the ExecutionEngine to materialize a C-callable wrapper
  // (`_mlir_ciface_<name>`) that takes StridedMemRefType* descriptors.
  func_op->setAttr("llvm.emit_c_interface", top_builder.getUnitAttr());

  mlir::Block* body = func_op.addEntryBlock();
  EmitContext ec(ctx, graph);
  ec.builder.setInsertionPointToEnd(body);

  // Bind inputs and params directly to block arguments (no tesseract.param
  // wrapper — that op has no lowering target).
  std::size_t arg_idx = 0;
  for (graph::ValueId id : graph.inputs())
    ec.set(id, body->getArgument(arg_idx++));
  for (graph::ValueId id : graph.params())
    ec.set(id, body->getArgument(arg_idx++));

  for (const graph::Op& op : graph.ops()) ec.emit_one_op(op);

  llvm::SmallVector<mlir::Value, 4> rets;
  rets.reserve(graph.outputs().size());
  for (graph::ValueId id : graph.outputs()) rets.push_back(ec.get(id));
  ec.builder.create<mlir::func::ReturnOp>(unknown, rets);

  if (mlir::failed(mlir::verify(*module))) {
    std::string dump;
    llvm::raw_string_ostream os(dump);
    module->print(os);
    throw std::runtime_error(
        "tesseract::ir::emit_func_mlir produced a module that failed "
        "verify():\n" +
        dump);
  }
  return module;
}

}  // namespace tesseract::ir

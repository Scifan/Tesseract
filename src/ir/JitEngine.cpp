// M1I.2.b — MLIR ExecutionEngine wrapper. See JitEngine.hpp.

#include "tesseract/ir/JitEngine.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/ir/Emit.hpp"
#include "tesseract/ir/Passes.hpp"
#include "tesseract/ir/TesseractDialect.h"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::ir {

namespace {

// ---------------------------------------------------------------------------
// Global one-time initialization for the native JIT target. Safe to call
// repeatedly from any thread; guarded by `std::call_once`.
void initialize_llvm_once() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });
}

// ---------------------------------------------------------------------------
// Build a `llvm::TargetMachine` pinned to the host CPU with the host feature
// set — "native" in the usual `-mcpu=native -mattr=+...` sense.
//
// Why this matters: `mlir::ExecutionEngine::create(..., /*tm=*/nullptr)` falls
// back to a TargetMachine built from the LLVM default triple + a generic
// subtarget. On x86_64 the default subtarget is `x86-64` (SSE2 baseline
// only), so the JIT never emits AVX2 / AVX-512 even at `optLevel=3`.
// That turned into a real perf cliff on `bench_graph_vs_eager`'s `wide`
// case (2.23× slower than eager on a 512×512 matmul), because eager's
// hand-written kernel compiled with `-O3 -march=native` sees the full
// SIMD instruction set while the JIT's identical linalg-to-scf loop did
// not. Pinning to the host features closes that gap without any change
// to the lowering pipeline itself.
//
// Returns `nullptr` if the host triple is somehow unsupported (non-x86 /
// ARM / exotic targets where the LLVM build has not registered a
// backend); the caller then falls back to the default TM.
std::unique_ptr<llvm::TargetMachine> make_host_target_machine(int opt_level) {
  const std::string triple = llvm::sys::getProcessTriple();
  std::string err;
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, err);
  if (target == nullptr) return nullptr;

  const std::string cpu = std::string(llvm::sys::getHostCPUName());

  // Turn the host feature map into a comma-separated `+f1,+f2,-f3,...`
  // string in the format `createTargetMachine` expects.
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> host_features;
  if (llvm::sys::getHostCPUFeatures(host_features)) {
    for (const auto& kv : host_features) {
      features.AddFeature(kv.first(), kv.second);
    }
  }

  llvm::TargetOptions topts;
  const auto cg_opt = static_cast<llvm::CodeGenOptLevel>(opt_level);
  return std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine(triple, cpu, features.getString(), topts,
                                  /*RelocModel=*/std::nullopt,
                                  /*CodeModel=*/std::nullopt, cg_opt,
                                  /*JIT=*/true));
}

// ---------------------------------------------------------------------------
// Dynamic-rank StridedMemRef descriptor.
//
// MLIR's C ABI for `memref<...>` is a `StridedMemRefType<T, Rank>` struct:
//   { T* basePtr; T* data; int64_t offset; int64_t sizes[Rank];
//     int64_t strides[Rank]; }
// `Rank` is statically known to the JIT but, on our side, varies per value.
// We keep the layout identical by computing the total struct size at runtime
// and writing the fields into a plain byte buffer. This matches what the
// `_mlir_ciface_*` wrapper emitted by `llvm.emit_c_interface` expects.
struct MemRefDescriptor {
  std::vector<std::byte> bytes;   // The descriptor struct itself.
  void* self_ptr = nullptr;       // &bytes[0] — indirection slot for
                                  // invokePacked (see comment below).
};

std::size_t descriptor_size(std::size_t rank) {
  // 2 pointers + 1 offset + rank sizes + rank strides. We don't bother
  // aligning here: StridedMemRefType uses 8-byte aligned fields and the
  // std::vector allocator gives us >= 8-byte alignment on every supported
  // target.
  return 2 * sizeof(void*) + sizeof(int64_t) + 2 * rank * sizeof(int64_t);
}

// Populate a descriptor for a contiguous, row-major Tensor view onto `data`.
// The shape is read from `shape_dims` (which is usually the recorded graph
// value's shape). For the graph values that participate here, every tensor
// is required to be contiguous (the precondition is enforced by the caller).
MemRefDescriptor make_descriptor(const std::vector<int64_t>& shape_dims,
                                 void* data) {
  const std::size_t rank = shape_dims.size();
  MemRefDescriptor d;
  d.bytes.resize(descriptor_size(rank));

  auto* p = d.bytes.data();

  // basePtr, data: both point at the (0,...,0) element. We do not support
  // non-zero `offset`s here.
  std::memcpy(p + 0, &data, sizeof(void*));
  std::memcpy(p + sizeof(void*), &data, sizeof(void*));
  int64_t zero = 0;
  std::memcpy(p + 2 * sizeof(void*), &zero, sizeof(int64_t));

  // Contiguous row-major strides: stride[i] = prod(shape[i+1:]).
  std::vector<int64_t> strides(rank, 1);
  for (std::size_t i = rank; i-- > 0;) {
    if (i + 1 < rank) strides[i] = strides[i + 1] * shape_dims[i + 1];
  }

  auto* p_sizes = p + 2 * sizeof(void*) + sizeof(int64_t);
  auto* p_strides = p_sizes + rank * sizeof(int64_t);
  for (std::size_t i = 0; i < rank; ++i) {
    std::memcpy(p_sizes + i * sizeof(int64_t), &shape_dims[i], sizeof(int64_t));
    std::memcpy(p_strides + i * sizeof(int64_t), &strides[i], sizeof(int64_t));
  }

  d.self_ptr = d.bytes.data();
  return d;
}

// Decide the MLIR LayoutMapOption and pass pipeline we use.
void build_lowering_pipeline(mlir::PassManager& pm) {
  pm.addPass(createConvertTesseractToLinalgPass());

  // Before bufferization we still have `linalg.generic` / `linalg.reduce`
  // on `tensor<...>` values, which is the right level to fuse together
  // chains of elementwise ops (e.g. the many `linalg.generic` blocks
  // produced by the cross-entropy forward lowering). Fusion cuts
  // intermediate `tensor.empty` allocations and gives the later scf loops
  // more to chew on at once. Canonicalize + CSE tidies up
  // `linalg.generic` iteration maps + arithmetic so downstream passes
  // see a normal form.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgElementwiseOpFusionPass());

  // Permute `linalg.matmul` from (m, n, k) to (m, k, n) so the later
  // `convert-linalg-to-loops` produces an N-innermost loop nest. With
  // N innermost all three memref accesses in the body are contiguous,
  // which is the shape LLVM's LoopVectorize knows how to turn into
  // AVX / AVX-512 code. See `passes/InterchangeMatmul.cpp` + B-007
  // for the full rationale.
  pm.addNestedPass<mlir::func::FuncOp>(createInterchangeMatmulPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // One-shot bufferize: turn `tensor<...>` values into `memref<...>` values.
  // IdentityLayoutMap keeps the result types as `memref<MxNxf32>` (no
  // layout map), so the JIT's C ABI exactly matches our static strides.
  mlir::bufferization::OneShotBufferizationOptions bufferize_opts;
  bufferize_opts.bufferizeFunctionBoundaries = true;
  bufferize_opts.setFunctionBoundaryTypeConversion(
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferize_opts));

  // Move returned memrefs to leading out-params so the caller can pre-
  // allocate outputs and avoid the "who frees the alloc?" problem.
  pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());

  // Structured linalg → scf loops over memrefs.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());

  // Lower affine (used by linalg.reduce's indexing maps) to std arith + scf.
  pm.addPass(mlir::createLowerAffinePass());

  // scf → cf (structured CFG to unstructured).
  pm.addPass(mlir::createConvertSCFToCFPass());

  // memref.* → llvm.*. `expand-strided-metadata` breaks memref descriptor
  // construction down into primitive `memref.extract_*` ops before the
  // llvm lowering sees them.
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

  // func / arith / cf → llvm. Math first: `math.exp` / `math.log` (and
  // friends, used by cross-entropy's log-sum-exp) lower to llvm intrinsics
  // where possible, falling back to libm calls (libc's expf / logf) for
  // the remainder. Must run before `convert-arith-to-llvm` because the
  // libm path uses `func.call` + declarations that the func dialect
  // conversion then simplifies.
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createConvertMathToLibmPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());

  // Clean up any leftover unrealized_conversion_cast ops.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl: owns the MLIR context, the lowered module, and the ExecutionEngine.
// Also caches per-value shape info used by `invoke`.

struct JitEngine::Impl {
  Options opts;
  mlir::DialectRegistry registry;
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::unique_ptr<mlir::ExecutionEngine> engine;

  // Cached per-argument shape/dtype info, computed once at construction:
  // outputs come first (post `buffer-results-to-out-params`), then inputs
  // followed by params.
  struct ArgInfo {
    std::vector<int64_t> shape;
    DType dtype;
  };
  std::vector<ArgInfo> output_infos;
  std::vector<ArgInfo> input_infos;  // inputs + params, in bind order.
  std::vector<graph::ValueId> input_ids;  // matches input_infos.

  Impl(const graph::Graph& g, Options options) : opts(std::move(options)) {
    initialize_llvm_once();

    // Register every dialect the lowering pipeline will introduce before
    // creating the MLIRContext. Anything missing here shows up as a
    // verifier failure deep in the passes.
    registry.insert<TesseractDialect, mlir::func::FuncDialect,
                    mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::math::MathDialect, mlir::tensor::TensorDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::cf::ControlFlowDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::LLVM::LLVMDialect>();
    // Bufferization external-model interface impls. Without these, the
    // one-shot pass reports "op was not bufferized" on every linalg /
    // tensor / arith op it encounters.
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    // Translation hooks so `mlir::translateModuleToLLVMIR` (used by the
    // ExecutionEngine internally) understands the `llvm.*` ops we produce.
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);

    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();

    // Record per-arg metadata BEFORE any lowering mutates the module —
    // we need the original Tensor-level shapes/dtypes to marshal inputs
    // and pre-allocate outputs.
    for (graph::ValueId id : g.outputs()) {
      const auto& v = g.value(id);
      ArgInfo ai;
      ai.shape.reserve(v.shape.rank());
      for (std::size_t i = 0; i < v.shape.rank(); ++i)
        ai.shape.push_back(v.shape[i]);
      ai.dtype = v.dtype;
      output_infos.push_back(std::move(ai));
    }
    for (graph::ValueId id : g.inputs()) input_ids.push_back(id);
    for (graph::ValueId id : g.params()) input_ids.push_back(id);
    for (graph::ValueId id : input_ids) {
      const auto& v = g.value(id);
      ArgInfo ai;
      ai.shape.reserve(v.shape.rank());
      for (std::size_t i = 0; i < v.shape.rank(); ++i)
        ai.shape.push_back(v.shape[i]);
      ai.dtype = v.dtype;
      input_infos.push_back(std::move(ai));
    }

    // Emit `func.func @<name>` and run the CPU lowering pipeline.
    EmitOptions emit_opts;
    emit_opts.function_name = opts.function_name;
    module = emit_func_mlir(ctx, g, emit_opts);

    if (opts.dump_ir) {
      llvm::errs() << "[tesseract-jit] pre-lowering module:\n";
      module->print(llvm::errs());
      llvm::errs() << "\n";
    }

    mlir::PassManager pm(&ctx);
    build_lowering_pipeline(pm);
    if (mlir::failed(pm.run(*module))) {
      std::string dump;
      llvm::raw_string_ostream os(dump);
      module->print(os);
      throw std::runtime_error(
          "tesseract::ir::JitEngine: CPU lowering pipeline failed:\n" + dump);
    }

    if (opts.dump_ir) {
      llvm::errs() << "[tesseract-jit] post-lowering (LLVM dialect) "
                      "module:\n";
      module->print(llvm::errs());
      llvm::errs() << "\n";
    }

    // Build a host-native TargetMachine so the LLVM backend actually
    // sees AVX2 / AVX-512 / etc. (see `make_host_target_machine`
    // rationale). We pass the raw pointer into the optimizing
    // transformer (which stashes a `TargetMachine*` captured by ref),
    // then move ownership into `ExecutionEngine::create` so the TM
    // outlives every JIT-compiled function. On platforms where we
    // fail to obtain a host TM we fall back to the upstream default
    // (generic subtarget + no `-mcpu`), which still works but loses
    // the SIMD codegen.
    auto host_tm = make_host_target_machine(opts.opt_level);
    llvm::TargetMachine* tm_raw = host_tm.get();

    mlir::ExecutionEngineOptions ee_opts;
    auto opt_level = static_cast<llvm::CodeGenOptLevel>(opts.opt_level);
    ee_opts.jitCodeGenOptLevel = opt_level;
    ee_opts.transformer = mlir::makeOptimizingTransformer(
        /*optLevel=*/opts.opt_level,
        /*sizeLevel=*/0, /*targetMachine=*/tm_raw);

    auto maybe_engine =
        mlir::ExecutionEngine::create(*module, ee_opts, std::move(host_tm));
    if (!maybe_engine) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << maybe_engine.takeError();
      throw std::runtime_error(
          "tesseract::ir::JitEngine: ExecutionEngine::create failed: " + msg);
    }
    engine = std::move(*maybe_engine);
  }
};

JitEngine::JitEngine(const graph::Graph& g, Options opts)
    : impl_(std::make_unique<Impl>(g, std::move(opts))) {}
JitEngine::~JitEngine() = default;
JitEngine::JitEngine(JitEngine&&) noexcept = default;
JitEngine& JitEngine::operator=(JitEngine&&) noexcept = default;

std::vector<Tensor> JitEngine::invoke(
    const std::unordered_map<graph::ValueId, Tensor>& bindings) const {
  TESSERACT_CHECK(impl_ != nullptr, "JitEngine::invoke: moved-from instance");

  // 1) Allocate output tensors. Each gets its own storage; we hand its
  //    contiguous pointer to the JIT via a memref descriptor.
  std::vector<Tensor> outputs;
  outputs.reserve(impl_->output_infos.size());
  for (const auto& ai : impl_->output_infos) {
    outputs.push_back(Tensor::empty(Shape(ai.shape), ai.dtype));
  }

  // 2) Build memref descriptors. `buffer-results-to-out-params`
  //    *appends* the out-param memrefs at the end of the argument list,
  //    so the C-iface signature is (inputs..., params..., outputs...).
  std::vector<MemRefDescriptor> descriptors;
  descriptors.reserve(impl_->input_infos.size() + impl_->output_infos.size());

  for (std::size_t i = 0; i < impl_->input_ids.size(); ++i) {
    graph::ValueId id = impl_->input_ids[i];
    auto it = bindings.find(id);
    TESSERACT_CHECK(it != bindings.end(),
                    "JitEngine::invoke: missing binding for ValueId {}", id);
    const Tensor& t = it->second;
    TESSERACT_CHECK(t.is_contiguous(),
                    "JitEngine::invoke: input/param tensor for ValueId {} "
                    "must be contiguous",
                    id);
    TESSERACT_CHECK(t.dtype() == impl_->input_infos[i].dtype,
                    "JitEngine::invoke: dtype mismatch for ValueId {}", id);
    // Shape check (a mismatch here would be a silent memory corruption if
    // we let it through).
    const auto& expected = impl_->input_infos[i].shape;
    TESSERACT_CHECK(static_cast<std::size_t>(t.rank()) == expected.size(),
                    "JitEngine::invoke: rank mismatch for ValueId {}: "
                    "bound={}, expected={}",
                    id, t.rank(), expected.size());
    for (std::size_t d = 0; d < expected.size(); ++d) {
      TESSERACT_CHECK(t.shape()[d] == expected[d],
                      "JitEngine::invoke: shape mismatch for ValueId {} at "
                      "dim {}: bound={}, expected={}",
                      id, d, t.shape()[d], expected[d]);
    }
    // NOTE: raw_data is non-const on Tensor, so const_cast is needed here
    // because bindings map stores Tensor by value (shared TensorImpl
    // underneath). The JIT does not write into input buffers.
    descriptors.push_back(make_descriptor(
        expected, const_cast<void*>(t.raw_data())));
  }

  for (std::size_t i = 0; i < impl_->output_infos.size(); ++i) {
    TESSERACT_CHECK(outputs[i].is_contiguous(),
                    "JitEngine::invoke: output tensor {} must be contiguous",
                    i);
    descriptors.push_back(make_descriptor(impl_->output_infos[i].shape,
                                          outputs[i].raw_data()));
  }

  // 3) Build the packed argument array that invokePacked expects. Each
  //    slot must hold the *address* of a variable whose value is the
  //    `StridedMemRefType*` pointing to the descriptor. We keep the
  //    pointer variables in a side-vector so their addresses stay
  //    stable until invokePacked returns.
  std::vector<void*> desc_ptrs(descriptors.size());
  for (std::size_t i = 0; i < descriptors.size(); ++i)
    desc_ptrs[i] = descriptors[i].self_ptr;

  std::vector<void*> packed_args(descriptors.size());
  for (std::size_t i = 0; i < desc_ptrs.size(); ++i)
    packed_args[i] = &desc_ptrs[i];

  const std::string c_iface_name =
      std::string("_mlir_ciface_") + impl_->opts.function_name;
  if (auto err = impl_->engine->invokePacked(c_iface_name, packed_args)) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << err;
    throw std::runtime_error(
        "tesseract::ir::JitEngine::invoke: invokePacked('" + c_iface_name +
        "') failed: " + msg);
  }

  return outputs;
}

}  // namespace tesseract::ir

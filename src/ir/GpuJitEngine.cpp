// M4 Phase 8 (B-009 tail) — GPU JIT execution path. See GpuJitEngine.hpp.

#include "tesseract/ir/GpuJitEngine.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mutex>

#include <cuda.h>

#include "llvm/Support/TargetSelect.h"

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

#include "tesseract/core/DType.hpp"
#include "tesseract/ir/Emit.hpp"
#include "tesseract/ir/Passes.hpp"
#include "tesseract/ir/TesseractDialect.h"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::ir {

namespace {

void cu_check(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* msg = nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &msg);
    throw std::runtime_error(std::string("GpuJitEngine CUDA driver error in ") +
                             what + ": " + (name ? name : "?") + " — " +
                             (msg ? msg : "?"));
  }
}

// One-time driver init. cuInit is idempotent; guard with call_once anyway.
bool ensure_cuda_init() {
  static bool ok = []() {
    return cuInit(0) == CUDA_SUCCESS;
  }();
  return ok;
}

// Where each kernel operand's data comes from at launch time. A fused
// elementwise kernel can take memref operands (inputs / the output alloc) plus
// scalar operands (e.g. relu's 0.0 threshold materialized as arith.constant).
struct OperandDesc {
  enum class Kind { Input, Output, ScalarF32 };
  Kind kind = Kind::Input;
  std::size_t index = 0;            // index into inputs (or outputs).
  std::vector<int64_t> shape;       // row-major shape of the memref.
  float scalar = 0.0f;             // value for ScalarF32 operands.
};

}  // namespace

struct GpuJitEngine::Impl {
  Options opts;

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;  // kept alive for provenance.

  // CUDA driver state.
  CUdevice device = 0;
  CUcontext cuctx = nullptr;
  CUmodule cumod = nullptr;
  CUfunction cufun = nullptr;
  unsigned grid[3] = {1, 1, 1};
  unsigned block[3] = {1, 1, 1};

  // Per-arg metadata, matching JitEngine's marshaling contract.
  struct ArgInfo {
    std::vector<int64_t> shape;
    DType dtype;
  };
  std::vector<ArgInfo> output_infos;
  std::vector<ArgInfo> input_infos;       // inputs + params, in bind order.
  std::vector<graph::ValueId> input_ids;  // matches input_infos.

  // Launch operand order (== cuLaunchKernel memref order).
  std::vector<OperandDesc> operands;

  Impl(const graph::Graph& g, Options options) : opts(std::move(options)) {
    // 1. Record per-value shape/dtype before lowering mutates the IR.
    auto record = [&](graph::ValueId id, std::vector<ArgInfo>& dst) {
      const auto& v = g.value(id);
      ArgInfo ai;
      ai.shape.reserve(v.shape.rank());
      for (std::size_t i = 0; i < v.shape.rank(); ++i)
        ai.shape.push_back(v.shape[i]);
      ai.dtype = v.dtype;
      dst.push_back(std::move(ai));
    };
    for (graph::ValueId id : g.outputs()) record(id, output_infos);
    for (graph::ValueId id : g.inputs()) input_ids.push_back(id);
    for (graph::ValueId id : g.params()) input_ids.push_back(id);
    for (graph::ValueId id : input_ids) record(id, input_infos);

    TESSERACT_CHECK(output_infos.size() == 1,
                    "GpuJitEngine: exactly one graph output is supported "
                    "(got {})",
                    output_infos.size());
    for (const auto& ai : output_infos)
      TESSERACT_CHECK(ai.dtype == DType::Float32,
                      "GpuJitEngine: only Float32 graphs are supported");
    for (const auto& ai : input_infos)
      TESSERACT_CHECK(ai.dtype == DType::Float32,
                      "GpuJitEngine: only Float32 graphs are supported");

    // 2. MLIR context + dialects + the translation interfaces the
    //    gpu-module-to-binary serializer needs.
    registry.insert<TesseractDialect, mlir::func::FuncDialect,
                    mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                    mlir::NVVM::NVVMDialect>();
    // Bufferization external models — convert-tesseract-to-gpu runs
    // one-shot-bufferize, which needs these or every linalg/tensor/arith/scf
    // op reports "op was not bufferized".
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
    mlir::registerGPUDialectTranslation(registry);
    mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);

    ctx = std::make_unique<mlir::MLIRContext>();
    ctx->appendDialectRegistry(registry);
    ctx->loadAllAvailableDialects();

    // 3. Emit func.func and run the device lowering pipeline.
    EmitOptions emit_opts;
    emit_opts.function_name = opts.function_name;
    module = emit_func_mlir(*ctx, g, emit_opts);

    if (opts.dump_ir) {
      llvm::errs() << "[tesseract-gpujit] pre-lowering:\n";
      module->print(llvm::errs());
    }

    run_device_pipeline();

    if (opts.dump_ir) {
      llvm::errs() << "[tesseract-gpujit] post-serialization:\n";
      module->print(llvm::errs());
    }

    // 4. Extract launch config + cubin from the lowered module.
    std::string cubin;
    extract(cubin);

    // 5. Load the cubin and resolve the kernel on the target device.
    TESSERACT_CHECK(ensure_cuda_init(), "GpuJitEngine: cuInit failed");
    cu_check(cuDeviceGet(&device, opts.device_index), "cuDeviceGet");
    cu_check(cuDevicePrimaryCtxRetain(&cuctx, device),
             "cuDevicePrimaryCtxRetain");
    cu_check(cuCtxSetCurrent(cuctx), "cuCtxSetCurrent");
    cu_check(cuModuleLoadData(&cumod, cubin.data()), "cuModuleLoadData");
    cu_check(cuModuleGetFunction(&cufun, cumod, kernel_name_.c_str()),
             "cuModuleGetFunction");
  }

  ~Impl() {
    if (cumod) cuModuleUnload(cumod);
    if (cuctx) cuDevicePrimaryCtxRelease(device);
  }

  std::string kernel_name_;

  void run_device_pipeline() {
    // The NVPTX serializer (gpu-module-to-binary) looks the nvptx64 backend up
    // in the LLVM TargetRegistry and runs its asm printer to emit PTX. Make
    // sure every target the LLVM build registered (NVPTX + host) is live.
    static std::once_flag tgt_flag;
    std::call_once(tgt_flag, []() {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmPrinters();
    });

    mlir::PassManager pm(ctx.get());
    buildConvertTesseractToGpuPipeline(pm);

    mlir::GpuNVVMAttachTargetOptions attach;
    attach.chip = opts.chip;
    attach.optLevel = static_cast<unsigned>(opts.opt_level);
    pm.addPass(mlir::createGpuNVVMAttachTarget(attach));

    pm.addNestedPass<mlir::gpu::GPUModuleOp>(
        mlir::createConvertGpuOpsToNVVMOps());
    pm.addPass(mlir::createConvertNVVMToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    // Serialize to a raw cubin ELF (not a fatbin wrapper) — cuModuleLoadData
    // consumes a cubin for the exact target arch directly.
    mlir::GpuModuleToBinaryPassOptions bin_opts;
    bin_opts.compilationTarget = "bin";
    pm.addPass(mlir::createGpuModuleToBinaryPass(bin_opts));

    if (mlir::failed(pm.run(*module))) {
      std::string dump;
      llvm::raw_string_ostream os(dump);
      module->print(os);
      throw std::runtime_error(
          "GpuJitEngine: device lowering pipeline failed:\n" + dump);
    }
  }

  void extract(std::string& cubin_out) {
    mlir::ModuleOp mod = *module;

    // Find the single host func + its launch_func.
    mlir::func::FuncOp entry;
    mlir::gpu::LaunchFuncOp launch;
    mod.walk([&](mlir::func::FuncOp f) {
      if (f.getName() == opts.function_name) entry = f;
    });
    TESSERACT_CHECK(entry, "GpuJitEngine: entry func '{}' not found",
                    opts.function_name);

    int n_launch = 0;
    entry.walk([&](mlir::gpu::LaunchFuncOp op) {
      launch = op;
      ++n_launch;
    });
    TESSERACT_CHECK(n_launch == 1,
                    "GpuJitEngine: graph must lower to exactly one GPU kernel "
                    "(got {} launches; multi-kernel graphs are unsupported)",
                    n_launch);

    kernel_name_ = launch.getKernelName().str();

    // Grid / block sizes: constant-folded operands.
    auto dim = [&](const mlir::gpu::KernelDim3& d, unsigned out[3]) {
      auto cx = mlir::getConstantIntValue(d.x);
      auto cy = mlir::getConstantIntValue(d.y);
      auto cz = mlir::getConstantIntValue(d.z);
      TESSERACT_CHECK(cx && cy && cz,
                      "GpuJitEngine: dynamic grid/block sizes are unsupported "
                      "(static shapes required)");
      out[0] = static_cast<unsigned>(*cx);
      out[1] = static_cast<unsigned>(*cy);
      out[2] = static_cast<unsigned>(*cz);
    };
    dim(launch.getGridSizeOperandValues(), grid);
    dim(launch.getBlockSizeOperandValues(), block);

    // Map each kernel operand. Memref operands are inputs (entry block args)
    // or the output alloc; scalar operands are constants folded into the
    // kernel-arg list (e.g. relu's threshold).
    mlir::Block& entry_block = entry.getBody().front();
    for (mlir::Value operand : launch.getKernelOperands()) {
      OperandDesc d;
      mlir::Type ty = operand.getType();
      if (mlir::isa<mlir::MemRefType>(ty)) {
        if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
          TESSERACT_CHECK(
              ba.getOwner() == &entry_block,
              "GpuJitEngine: kernel operand is a non-entry block arg");
          d.kind = OperandDesc::Kind::Input;
          d.index = ba.getArgNumber();
          TESSERACT_CHECK(d.index < input_infos.size(),
                          "GpuJitEngine: kernel arg {} out of input range",
                          d.index);
          d.shape = input_infos[d.index].shape;
        } else {
          // A non-argument memref is the internally-allocated output.
          d.kind = OperandDesc::Kind::Output;
          d.index = 0;
          d.shape = output_infos[0].shape;
        }
      } else if (mlir::isa<mlir::Float32Type>(ty)) {
        auto cst = operand.getDefiningOp<mlir::arith::ConstantOp>();
        TESSERACT_CHECK(cst,
                        "GpuJitEngine: scalar kernel operand must be a "
                        "constant (got a non-constant f32)");
        auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue());
        TESSERACT_CHECK(fattr,
                        "GpuJitEngine: scalar f32 operand is not a FloatAttr");
        d.kind = OperandDesc::Kind::ScalarF32;
        d.scalar = static_cast<float>(fattr.getValueAsDouble());
      } else {
        std::string tystr;
        llvm::raw_string_ostream os(tystr);
        ty.print(os);
        TESSERACT_THROW(
            "GpuJitEngine: unsupported kernel operand type '{}' (only "
            "memref<...xf32> and f32 constants are supported)",
            tystr);
      }
      operands.push_back(std::move(d));
    }

    // Pull the serialized cubin out of the gpu.binary the
    // --gpu-module-to-binary pass produced (replacing the gpu.module).
    mlir::gpu::BinaryOp binary;
    int n_bin = 0;
    mod.walk([&](mlir::gpu::BinaryOp op) {
      binary = op;
      ++n_bin;
    });
    TESSERACT_CHECK(n_bin == 1,
                    "GpuJitEngine: expected exactly one gpu.binary (got {})",
                    n_bin);
    mlir::ArrayAttr objs = binary.getObjects();
    TESSERACT_CHECK(!objs.empty(),
                    "GpuJitEngine: gpu.binary has no serialized objects");
    auto obj = mlir::cast<mlir::gpu::ObjectAttr>(objs[0]);
    llvm::StringRef bytes = obj.getObject().getValue();
    cubin_out.assign(bytes.data(), bytes.size());
  }
};

GpuJitEngine::GpuJitEngine(const graph::Graph& g, Options opts)
    : impl_(std::make_unique<Impl>(g, std::move(opts))) {}
GpuJitEngine::~GpuJitEngine() = default;
GpuJitEngine::GpuJitEngine(GpuJitEngine&&) noexcept = default;
GpuJitEngine& GpuJitEngine::operator=(GpuJitEngine&&) noexcept = default;

bool GpuJitEngine::available() {
  if (!ensure_cuda_init()) return false;
  int count = 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS) return false;
  return count > 0;
}

std::vector<Tensor> GpuJitEngine::invoke(
    const std::unordered_map<graph::ValueId, Tensor>& bindings) const {
  TESSERACT_CHECK(impl_ != nullptr, "GpuJitEngine::invoke: moved-from");
  cu_check(cuCtxSetCurrent(impl_->cuctx), "cuCtxSetCurrent");

  const auto numel = [](const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (int64_t d : s) n *= d;
    return n;
  };

  // Device buffers, one per kernel operand (inputs copied H2D, output fresh).
  std::vector<CUdeviceptr> dbufs(impl_->operands.size(), 0);
  std::vector<Tensor> outputs;
  outputs.reserve(impl_->output_infos.size());
  for (const auto& ai : impl_->output_infos)
    outputs.push_back(Tensor::empty(Shape(ai.shape), ai.dtype));

  for (std::size_t i = 0; i < impl_->operands.size(); ++i) {
    const auto& od = impl_->operands[i];
    if (od.kind == OperandDesc::Kind::ScalarF32) continue;  // no device buffer.
    const int64_t bytes = numel(od.shape) * static_cast<int64_t>(sizeof(float));
    cu_check(cuMemAlloc(&dbufs[i], static_cast<size_t>(bytes)), "cuMemAlloc");
    if (od.kind == OperandDesc::Kind::Input) {
      graph::ValueId id = impl_->input_ids[od.index];
      auto it = bindings.find(id);
      TESSERACT_CHECK(it != bindings.end(),
                      "GpuJitEngine::invoke: missing binding for ValueId {}",
                      id);
      const Tensor& t = it->second;
      TESSERACT_CHECK(t.is_contiguous(),
                      "GpuJitEngine::invoke: input ValueId {} must be "
                      "contiguous",
                      id);
      cu_check(cuMemcpyHtoD(dbufs[i], t.raw_data(), static_cast<size_t>(bytes)),
               "cuMemcpyHtoD");
    }
  }

  // Build the flattened kernel-argument list. Each memref<rank R> contributes
  // { allocPtr, alignedPtr, offset, sizes[R], strides[R] } — the MLIR memref
  // C ABI. We keep the backing storage stable (reserved exactly) so the
  // address-of-element pointers handed to cuLaunchKernel stay valid.
  std::size_t n_ptr_slots = 0, n_int_slots = 0, n_scalar_slots = 0;
  for (const auto& od : impl_->operands) {
    if (od.kind == OperandDesc::Kind::ScalarF32) {
      ++n_scalar_slots;
    } else {
      n_ptr_slots += 2;
      n_int_slots += 1 + 2 * od.shape.size();
    }
  }
  std::vector<CUdeviceptr> ptr_store;
  std::vector<int64_t> int_store;
  std::vector<float> scalar_store;
  ptr_store.reserve(n_ptr_slots);
  int_store.reserve(n_int_slots);
  scalar_store.reserve(n_scalar_slots);
  std::vector<void*> kparams;
  kparams.reserve(n_ptr_slots + n_int_slots + n_scalar_slots);

  for (std::size_t i = 0; i < impl_->operands.size(); ++i) {
    const auto& od = impl_->operands[i];
    if (od.kind == OperandDesc::Kind::ScalarF32) {
      scalar_store.push_back(od.scalar);
      kparams.push_back(&scalar_store.back());
      continue;
    }
    const std::size_t rank = od.shape.size();
    // allocPtr, alignedPtr (both = device base).
    ptr_store.push_back(dbufs[i]);
    kparams.push_back(&ptr_store.back());
    ptr_store.push_back(dbufs[i]);
    kparams.push_back(&ptr_store.back());
    // offset = 0.
    int_store.push_back(0);
    kparams.push_back(&int_store.back());
    // sizes.
    for (std::size_t d = 0; d < rank; ++d) {
      int_store.push_back(od.shape[d]);
      kparams.push_back(&int_store.back());
    }
    // strides (row-major).
    std::vector<int64_t> strides(rank, 1);
    for (std::size_t d = rank; d-- > 0;)
      if (d + 1 < rank) strides[d] = strides[d + 1] * od.shape[d + 1];
    for (std::size_t d = 0; d < rank; ++d) {
      int_store.push_back(strides[d]);
      kparams.push_back(&int_store.back());
    }
  }

  cu_check(cuLaunchKernel(impl_->cufun, impl_->grid[0], impl_->grid[1],
                          impl_->grid[2], impl_->block[0], impl_->block[1],
                          impl_->block[2], /*sharedMemBytes=*/0,
                          /*stream=*/nullptr, kparams.data(),
                          /*extra=*/nullptr),
           "cuLaunchKernel");
  cu_check(cuCtxSynchronize(), "cuCtxSynchronize");

  // D2H copy the output(s).
  for (std::size_t i = 0; i < impl_->operands.size(); ++i) {
    const auto& od = impl_->operands[i];
    if (od.kind != OperandDesc::Kind::Output) continue;
    const int64_t bytes = numel(od.shape) * static_cast<int64_t>(sizeof(float));
    cu_check(cuMemcpyDtoH(outputs[od.index].raw_data(), dbufs[i],
                          static_cast<size_t>(bytes)),
             "cuMemcpyDtoH");
  }

  for (CUdeviceptr p : dbufs)
    if (p) cuMemFree(p);

  return outputs;
}

}  // namespace tesseract::ir

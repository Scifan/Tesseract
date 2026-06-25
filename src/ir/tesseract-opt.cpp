// tesseract-opt: a minimal MLIR opt-like driver that registers the Tesseract
// dialect alongside a useful set of upstream dialects for round-trip and IR
// exploration. Build with -DTESSERACT_ENABLE_MLIR=ON.

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "tesseract/ir/Passes.hpp"
#include "tesseract/ir/TesseractDialect.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;

  // Core upstream dialects useful for lowering experiments in M1.
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  // Translation interfaces so `--gpu-module-to-binary` (and any LLVM-IR
  // export) can lower gpu.module / NVVM / LLVM ops to an llvm::Module on the
  // way to PTX/cubin. registerAllDialects alone does not pull these in.
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::registerBuiltinDialectTranslation(registry);

  registry.insert<tesseract::ir::TesseractDialect>();

  // Tesseract-owned conversion / analysis passes.
  tesseract::ir::registerTesseractPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tesseract optimizer driver\n", registry));
}

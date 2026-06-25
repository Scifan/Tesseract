//===- TesseractDialect.td - Tesseract dialect -------------*- tablegen -*-===//
//
// Defines the Tesseract MLIR dialect. At M0 this is a skeleton that registers
// only a handful of high-level tensor ops. Lowering passes (to Linalg / Func /
// LLVM) land in M1.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERACT_IR_DIALECT
#define TESSERACT_IR_DIALECT

include "mlir/IR/OpBase.td"

def Tesseract_Dialect : Dialect {
  let name = "tesseract";
  let summary = "Tesseract high-level tensor dialect (placeholder)";
  let description = [{
    The Tesseract dialect exposes high-level tensor operations that mirror the
    eager-mode C++ runtime. It is intentionally minimal at M0: three core ops
    serve as a correctness sanity-check for the dialect registration pipeline
    and as a lowering target for upcoming passes.
  }];
  let cppNamespace = "::tesseract::ir";
}

#endif  // TESSERACT_IR_DIALECT

#include "tesseract/ir/TesseractDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "tesseract/ir/TesseractOps.h"

using namespace mlir;
using namespace tesseract::ir;

#include "tesseract/ir/TesseractDialect.cpp.inc"

void TesseractDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tesseract/ir/TesseractOps.cpp.inc"
      >();
}

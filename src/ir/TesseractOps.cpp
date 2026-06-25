#include "tesseract/ir/TesseractOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace tesseract::ir;

#define GET_OP_CLASSES
#include "tesseract/ir/TesseractOps.cpp.inc"

//===----------------------------------------------------------------------===//
// tesseract.constant
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  auto attrTy = llvm::dyn_cast<ShapedType>(getValue().getType());
  if (!resultTy || !attrTy) {
    return emitOpError("expects ranked tensor types on both value attr and result");
  }
  if (resultTy != attrTy) {
    return emitOpError("result type must match the value attribute type (got ")
           << resultTy << " vs " << attrTy << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// tesseract.matmul
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsTy = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  auto resTy = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhsTy || !rhsTy || !resTy) {
    return emitOpError("all operands and the result must be ranked tensors");
  }
  const int64_t rank = lhsTy.getRank();
  if ((rank != 2 && rank != 3) || rhsTy.getRank() != rank ||
      resTy.getRank() != rank) {
    return emitOpError("only rank-2 or rank-3 (batched) operands of equal "
                       "rank are supported");
  }
  // For rank-3, dim 0 is the batch axis: it must agree across all operands.
  // The trailing [M,K]·[K,N]→[M,N] checks below apply to the last two dims.
  const int64_t mAx = rank - 2;
  const int64_t kAxL = rank - 1;
  const int64_t kAxR = rank - 2;
  const int64_t nAx = rank - 1;
  const auto lhsShape = lhsTy.getShape();
  const auto rhsShape = rhsTy.getShape();
  const auto resShape = resTy.getShape();
  if (rank == 3) {
    if (!lhsTy.isDynamicDim(0) && !rhsTy.isDynamicDim(0) &&
        lhsShape[0] != rhsShape[0]) {
      return emitOpError("batch dims disagree: lhs[0]=")
             << lhsShape[0] << " vs rhs[0]=" << rhsShape[0];
    }
    if (!lhsTy.isDynamicDim(0) && !resTy.isDynamicDim(0) &&
        lhsShape[0] != resShape[0]) {
      return emitOpError("result batch dim must match lhs[0]");
    }
  }
  if (!lhsTy.isDynamicDim(kAxL) && !rhsTy.isDynamicDim(kAxR) &&
      lhsShape[kAxL] != rhsShape[kAxR]) {
    return emitOpError("inner dims disagree: lhs[")
           << kAxL << "]=" << lhsShape[kAxL] << " vs rhs[" << kAxR
           << "]=" << rhsShape[kAxR];
  }
  if (!lhsTy.isDynamicDim(mAx) && !resTy.isDynamicDim(mAx) &&
      lhsShape[mAx] != resShape[mAx]) {
    return emitOpError("result row count must match lhs row count");
  }
  if (!rhsTy.isDynamicDim(nAx) && !resTy.isDynamicDim(nAx) &&
      rhsShape[nAx] != resShape[nAx]) {
    return emitOpError("result col count must match rhs col count");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// tesseract.dequant_matmul
//===----------------------------------------------------------------------===//

LogicalResult DequantMatMulOp::verify() {
  const llvm::StringRef scheme = getScheme();
  const bool grouped = (scheme == "int4_group" || scheme == "fp4");
  const bool per_tensor =
      (scheme == "int8" || scheme == "fp8_e4m3" || scheme == "fp8_e5m2");
  if (!grouped && !per_tensor) {
    return emitOpError("unknown quantization scheme '")
           << scheme
           << "' (expected one of int8, int4_group, fp8_e4m3, fp8_e5m2, fp4)";
  }
  const int64_t gs = getGroupSize();
  if (grouped && gs <= 0) {
    return emitOpError("scheme '") << scheme
           << "' is grouped and requires group_size > 0 (got " << gs << ")";
  }
  if (per_tensor && gs != -1) {
    return emitOpError("scheme '") << scheme
           << "' is per-tensor and requires group_size == -1 (got " << gs << ")";
  }
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto resTy = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (lhsTy && resTy && lhsTy.getRank() != 2) {
    return emitOpError("lhs must be rank-2 [M, K]");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// tesseract.paged_kv_alloc  (M4 Track C2 / B-045)
//===----------------------------------------------------------------------===//

LogicalResult PagedKVAllocOp::verify() {
  auto poolTy = llvm::dyn_cast<RankedTensorType>(getPool().getType());
  if (!poolTy || poolTy.getRank() != 4) {
    return emitOpError(
        "pool must be a rank-4 tensor [num_blocks, block_size, num_kv_heads, "
        "head_dim]");
  }
  // Static dims must agree with the attributes; dynamic dims (`?`) are left
  // unchecked so a dynamically-sized pool is representable.
  const int64_t want[4] = {getNumBlocks(), getBlockSize(), getNumKvHeads(),
                           getHeadDim()};
  const char *names[4] = {"num_blocks", "block_size", "num_kv_heads",
                          "head_dim"};
  for (int i = 0; i < 4; ++i) {
    if (!poolTy.isDynamicDim(i) && poolTy.getShape()[i] != want[i]) {
      return emitOpError("pool dim ")
             << i << " (" << names[i] << ") = " << poolTy.getShape()[i]
             << " disagrees with attribute " << want[i];
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// tesseract.paged_kv_append
//===----------------------------------------------------------------------===//

LogicalResult PagedKVAppendOp::verify() {
  if (getPool().getType() != getUpdated().getType()) {
    return emitOpError("updated pool type must match the input pool type");
  }
  auto slotTy = llvm::dyn_cast<RankedTensorType>(getSlotMapping().getType());
  if (slotTy && !slotTy.getElementType().isInteger(32) &&
      !slotTy.getElementType().isInteger(64)) {
    return emitOpError("slot_mapping must be an integer tensor (Int32/Int64)");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// tesseract.paged_attention
//===----------------------------------------------------------------------===//

LogicalResult PagedAttentionOp::verify() {
  auto qTy = llvm::dyn_cast<RankedTensorType>(getQuery().getType());
  auto outTy = llvm::dyn_cast<RankedTensorType>(getOut().getType());
  if (!qTy || qTy.getRank() != 3) {
    return emitOpError("query must be rank-3 [num_tokens, num_heads, head_dim]");
  }
  if (outTy && outTy != qTy) {
    return emitOpError("result type must match the query type");
  }
  auto kTy = llvm::dyn_cast<RankedTensorType>(getKPool().getType());
  auto vTy = llvm::dyn_cast<RankedTensorType>(getVPool().getType());
  if (kTy && kTy.getRank() != 4) {
    return emitOpError("k_pool must be a rank-4 paged block buffer");
  }
  if (vTy && vTy.getRank() != 4) {
    return emitOpError("v_pool must be a rank-4 paged block buffer");
  }
  auto btTy = llvm::dyn_cast<RankedTensorType>(getBlockTable().getType());
  if (btTy && btTy.getRank() != 2) {
    return emitOpError("block_table must be rank-2 [num_seqs, max_blocks]");
  }
  return success();
}

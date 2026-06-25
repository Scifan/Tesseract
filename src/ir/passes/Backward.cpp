// Reverse-mode AD as a graph-level pass on the tesseract dialect (M1H).
//
// Semantics
// ---------
// For every `tesseract.function` inside a `tesseract.graph`, the pass
// appends one block-argument per existing result (the cotangents / grad of
// outputs) and one result per `tesseract.param` (the gradient of that
// param). The forward ops are kept intact so that backward use-sites can
// refer to their results directly — this is the vjp lowering, and it lets
// the existing MLIR CSE/canonicalization passes deduplicate forward values
// that get read both by the forward return and by the backward accumulator.
//
// Op coverage
// -----------
// * add, sub, mul, neg             — trivial.
// * div                            — d/dlhs = dy/rhs, d/drhs = −dy·lhs/rhs².
// * matmul                         — dA = dOut · Bᵀ, dB = Aᵀ · dOut
//                                     (rank-2 and rank-3 batched).
// * sum (any dim, keepdim)         — dx = broadcast_to(dOut, x.shape), with a
//                                     size-1 reshape at the reduced axis when
//                                     keepdim=false. dim=-1 reduces all.
// * softmax (any dim)              — dx = y ⊙ (dy − Σ_dim(y ⊙ dy)).
// * sigmoid                        — dx = dy ⊙ y ⊙ (1 − y).
// * relu                           — dx = tesseract.relu_backward(x, dOut).
// * rotary_embedding (RoPE)        — dx = rope(dOut, cos, −sin) (adjoint).
// * permute / view / reshape       — inverse permutation / reshape of dOut.
// * cross_entropy_with_logits      — d_logits =
//                                     tesseract.cross_entropy_with_logits_backward
//                                     (logits, targets, grad_loss).
// * broadcast_to                   — dx = sum(dOut) over every axis that
//                                     was introduced or size-1-tiled
//                                     (enough for the bias case that the
//                                     emitter generates).
// * tesseract.param                — gradient propagates to the
//                                     corresponding block-argument.
//
// Ops not in this list fail the pass with a clear diagnostic. Registering
// another rule is a ~10 LOC change.
//
// Design notes
// ------------
// * The pass is **in-place**: we modify the function's entry block and
//   function_type attribute instead of cloning. This keeps SSA stable for
//   the forward outputs that also appear in the new return.
// * Gradient accumulation (when a value is used by multiple downstream
//   ops) is materialized as `tesseract.add` chains. CSE can fold them but
//   we don't rely on it for correctness.
// * The cotangent block-arguments are appended in the same order as the
//   forward outputs, so the caller wires `grad_output_i` ↔ `output_i`
//   positionally. See ADR-0004 for the training-loop contract.

#include "tesseract/ir/Passes.hpp"
#include "tesseract/ir/TesseractDialect.h"
#include "tesseract/ir/TesseractOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace tesseract::ir {

namespace {

// ---------------- IR construction helpers ----------------

mlir::Value emitBinary(mlir::OpBuilder &b, mlir::Location loc,
                       const std::string &mnemonic, mlir::Value lhs,
                       mlir::Value rhs) {
  mlir::OperationState state(loc, std::string("tesseract.") + mnemonic);
  state.addOperands({lhs, rhs});
  state.addTypes({lhs.getType()});
  return b.create(state)->getResult(0);
}

mlir::Value emitUnary(mlir::OpBuilder &b, mlir::Location loc,
                      const std::string &mnemonic, mlir::Value input) {
  mlir::OperationState state(loc, std::string("tesseract.") + mnemonic);
  state.addOperands({input});
  state.addTypes({input.getType()});
  return b.create(state)->getResult(0);
}

mlir::Value emitTranspose(mlir::OpBuilder &b, mlir::Location loc,
                          mlir::Value input, int64_t dimA, int64_t dimB) {
  auto inTy = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t, 4> shape(inTy.getShape().begin(),
                                      inTy.getShape().end());
  std::swap(shape[dimA], shape[dimB]);
  auto outTy = mlir::RankedTensorType::get(shape, inTy.getElementType());

  mlir::OperationState state(loc, "tesseract.transpose");
  state.addOperands({input});
  state.addTypes({outTy});
  auto si64 = b.getIntegerType(64, /*isSigned=*/true);
  state.addAttribute(
      "dim_a", mlir::IntegerAttr::get(si64, llvm::APInt(64, dimA, true)));
  state.addAttribute(
      "dim_b", mlir::IntegerAttr::get(si64, llvm::APInt(64, dimB, true)));
  return b.create(state)->getResult(0);
}

mlir::Value emitMatMul(mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
                       mlir::Value rhs) {
  auto lhsTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsTy = mlir::cast<mlir::RankedTensorType>(rhs.getType());
  const int64_t rank = lhsTy.getRank();
  // Result = [<batch dims from lhs>, lhs[rank-2], rhs[rank-1]]. Handles both
  // rank-2 (plain) and rank-3 (batched, multi-head attention) matmul.
  llvm::SmallVector<int64_t, 4> shape(lhsTy.getShape().begin(),
                                      lhsTy.getShape().end());
  shape[rank - 1] = rhsTy.getShape()[rank - 1];
  auto resultTy = mlir::RankedTensorType::get(shape, lhsTy.getElementType());

  mlir::OperationState state(loc, "tesseract.matmul");
  state.addOperands({lhs, rhs});
  state.addTypes({resultTy});
  return b.create(state)->getResult(0);
}

// Emit `tesseract.rotary_embedding` with the given input / cos / sin.
mlir::Value emitRope(mlir::OpBuilder &b, mlir::Location loc, mlir::Value input,
                     mlir::Value cos, mlir::Value sin) {
  mlir::OperationState state(loc, "tesseract.rotary_embedding");
  state.addOperands({input, cos, sin});
  state.addTypes({input.getType()});
  return b.create(state)->getResult(0);
}

// Emit `tesseract.permute` with the given axis list, computing the result type.
mlir::Value emitPermute(mlir::OpBuilder &b, mlir::Location loc,
                        mlir::Value input, llvm::ArrayRef<int64_t> axes) {
  auto inTy = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t, 4> shape;
  shape.reserve(axes.size());
  for (int64_t a : axes) shape.push_back(inTy.getShape()[a]);
  auto outTy = mlir::RankedTensorType::get(shape, inTy.getElementType());
  mlir::OperationState state(loc, "tesseract.permute");
  state.addOperands({input});
  state.addTypes({outTy});
  state.addAttribute("axes", b.getI64ArrayAttr(
                                 llvm::SmallVector<int64_t, 4>(axes.begin(),
                                                               axes.end())));
  return b.create(state)->getResult(0);
}

// Emit `tesseract.reshape` of `input` to `targetTy`.
mlir::Value emitReshape(mlir::OpBuilder &b, mlir::Location loc,
                        mlir::Value input, mlir::RankedTensorType targetTy) {
  mlir::OperationState state(loc, "tesseract.reshape");
  state.addOperands({input});
  state.addTypes({targetTy});
  llvm::SmallVector<int64_t, 4> dims(targetTy.getShape().begin(),
                                     targetTy.getShape().end());
  state.addAttribute("shape", b.getI64ArrayAttr(dims));
  return b.create(state)->getResult(0);
}

mlir::Value emitBroadcastTo(mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value input,
                            mlir::RankedTensorType targetTy) {
  mlir::OperationState state(loc, "tesseract.broadcast_to");
  state.addOperands({input});
  state.addTypes({targetTy});
  llvm::SmallVector<int64_t, 8> dims(targetTy.getShape().begin(),
                                     targetTy.getShape().end());
  state.addAttribute("shape", b.getI64ArrayAttr(dims));
  return b.create(state)->getResult(0);
}

// Emit `tesseract.sum` reducing `axis` from `input`. `keepdim` controls
// whether the reduced axis survives as size 1. We compute the result
// type statically so lowering passes can pattern-match on shape.
mlir::Value emitSumAlong(mlir::OpBuilder &b, mlir::Location loc,
                         mlir::Value input, int64_t axis, bool keepdim) {
  auto inTy = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t, 8> outShape;
  outShape.reserve(inTy.getRank());
  for (int64_t i = 0; i < inTy.getRank(); ++i) {
    if (i == axis) {
      if (keepdim) outShape.push_back(1);
    } else {
      outShape.push_back(inTy.getShape()[i]);
    }
  }
  auto outTy = mlir::RankedTensorType::get(outShape, inTy.getElementType());

  mlir::OperationState state(loc, "tesseract.sum");
  state.addOperands({input});
  state.addTypes({outTy});
  auto si64 = b.getIntegerType(64, /*isSigned=*/true);
  state.addAttribute(
      "dim", mlir::IntegerAttr::get(si64, llvm::APInt(64, axis, true)));
  state.addAttribute("keepdim", b.getBoolAttr(keepdim));
  return b.create(state)->getResult(0);
}

// Reduce `grad` (shape = outTy) back to `input`'s shape. Handles the two
// broadcast patterns our emitter produces:
//   * leading-rank expansion (input had fewer dims than output — drop
//     leading axes with keepdim=false), and
//   * size-1 tiling (input had size 1 along an axis where output is >1 —
//     reduce that axis with keepdim=true so the rank stays in sync).
// Returns a new Value with input.type(). Bails out on axes whose output
// extent is dynamic (shape = ShapedType::kDynamic) since we don't emit
// dynamic shapes from the C++ graph yet.
mlir::Value reduceToShape(mlir::OpBuilder &b, mlir::Location loc,
                          mlir::Value grad, mlir::RankedTensorType inTy) {
  auto gradTy = mlir::cast<mlir::RankedTensorType>(grad.getType());
  const int64_t outRank = gradTy.getRank();
  const int64_t inRank = inTy.getRank();
  // Collapse leading axes introduced by the broadcast.
  mlir::Value v = grad;
  for (int64_t i = 0; i < outRank - inRank; ++i) {
    v = emitSumAlong(b, loc, v, /*axis=*/0, /*keepdim=*/false);
  }
  // Now v and inTy have the same rank. Reduce each axis where input is
  // size-1 but output is >1, preserving the axis (keepdim=true) so the
  // resulting shape matches the input exactly.
  auto curTy = mlir::cast<mlir::RankedTensorType>(v.getType());
  for (int64_t i = 0; i < inRank; ++i) {
    const int64_t inDim = inTy.getShape()[i];
    const int64_t outDim = curTy.getShape()[i];
    if (inDim == 1 && outDim != 1) {
      v = emitSumAlong(b, loc, v, /*axis=*/i, /*keepdim=*/true);
      curTy = mlir::cast<mlir::RankedTensorType>(v.getType());
    }
  }
  return v;
}

// ---------------- Backward context ----------------

struct BackwardCtx {
  mlir::OpBuilder &builder;
  mlir::Location loc;
  llvm::DenseMap<mlir::Value, mlir::Value> grad;

  // Accumulate a new gradient contribution for `v`. If `v` already has a
  // grad entry, emit a tesseract.add to combine them.
  void addContribution(mlir::Value v, mlir::Value contribution) {
    auto it = grad.find(v);
    if (it == grad.end()) {
      grad[v] = contribution;
      return;
    }
    grad[v] = emitBinary(builder, loc, "add", it->second, contribution);
  }

  mlir::Value getOrZero(mlir::Value v) const {
    auto it = grad.find(v);
    if (it != grad.end()) return it->second;
    // No contribution observed: in a well-formed graph every used value
    // should receive at least one upstream grad. We bail out loudly
    // instead of silently emitting zero so missed rules surface early.
    v.getDefiningOp()->emitOpError(
        "tesseract-backward: value has no gradient contribution; an op along "
        "its use chain likely lacks a backward rule");
    return nullptr;
  }
};

// ---------------- Per-op backward rules ----------------

mlir::LogicalResult backwardOp(mlir::Operation *op, BackwardCtx &ctx) {
  mlir::OpBuilder &b = ctx.builder;
  mlir::Location loc = op->getLoc();

  if (auto add = mlir::dyn_cast<AddOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(add.getResult());
    if (!dout) return mlir::success();
    ctx.addContribution(add.getLhs(), dout);
    ctx.addContribution(add.getRhs(), dout);
    return mlir::success();
  }
  if (auto sub = mlir::dyn_cast<SubOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(sub.getResult());
    if (!dout) return mlir::success();
    mlir::Value negDout = emitUnary(b, loc, "neg", dout);
    ctx.addContribution(sub.getLhs(), dout);
    ctx.addContribution(sub.getRhs(), negDout);
    return mlir::success();
  }
  if (auto mul = mlir::dyn_cast<MulOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(mul.getResult());
    if (!dout) return mlir::success();
    mlir::Value dLhs = emitBinary(b, loc, "mul", dout, mul.getRhs());
    mlir::Value dRhs = emitBinary(b, loc, "mul", dout, mul.getLhs());
    ctx.addContribution(mul.getLhs(), dLhs);
    ctx.addContribution(mul.getRhs(), dRhs);
    return mlir::success();
  }
  if (auto neg = mlir::dyn_cast<NegOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(neg.getResult());
    if (!dout) return mlir::success();
    ctx.addContribution(neg.getInput(), emitUnary(b, loc, "neg", dout));
    return mlir::success();
  }
  if (auto mm = mlir::dyn_cast<MatMulOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(mm.getResult());
    if (!dout) return mlir::success();
    // dA = dOut @ Bᵀ, dB = Aᵀ @ dOut, transposing the last two dims so the
    // rule works for both rank-2 and rank-3 (batched, multi-head) matmul.
    const int64_t rank =
        mlir::cast<mlir::RankedTensorType>(mm.getLhs().getType()).getRank();
    const int64_t a = rank - 2, c = rank - 1;
    mlir::Value bT = emitTranspose(b, loc, mm.getRhs(), a, c);
    mlir::Value aT = emitTranspose(b, loc, mm.getLhs(), a, c);
    mlir::Value dA = emitMatMul(b, loc, dout, bT);
    mlir::Value dB = emitMatMul(b, loc, aT, dout);
    ctx.addContribution(mm.getLhs(), dA);
    ctx.addContribution(mm.getRhs(), dB);
    return mlir::success();
  }
  if (auto sm = mlir::dyn_cast<SoftmaxOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(sm.getResult());
    if (!dout) return mlir::success();
    // y = softmax(x, dim); dx = y ⊙ (dy − Σ_dim(y ⊙ dy)). Honor the op's
    // `dim` attribute (negative => from the end) rather than assuming the
    // last axis, so softmax over any axis differentiates correctly.
    mlir::Value y = sm.getResult();
    auto yTy = mlir::cast<mlir::RankedTensorType>(y.getType());
    int64_t ax = sm.getDim();
    if (ax < 0) ax += yTy.getRank();
    mlir::Value ydy = emitBinary(b, loc, "mul", y, dout);
    mlir::Value s = emitSumAlong(b, loc, ydy, ax, /*keepdim=*/true);
    mlir::Value sB = emitBroadcastTo(b, loc, s, yTy);
    mlir::Value diff = emitBinary(b, loc, "sub", dout, sB);
    mlir::Value dx = emitBinary(b, loc, "mul", y, diff);
    ctx.addContribution(sm.getInput(), dx);
    return mlir::success();
  }
  if (auto sig = mlir::dyn_cast<SigmoidOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(sig.getResult());
    if (!dout) return mlir::success();
    // y = sigmoid(x); dx = dy ⊙ (y − y²) = dy ⊙ y ⊙ (1 − y), expressed
    // without a ones tensor.
    mlir::Value y = sig.getResult();
    mlir::Value yy = emitBinary(b, loc, "mul", y, y);
    mlir::Value yMinusYY = emitBinary(b, loc, "sub", y, yy);
    ctx.addContribution(sig.getInput(), emitBinary(b, loc, "mul", dout, yMinusYY));
    return mlir::success();
  }
  if (auto dv = mlir::dyn_cast<DivOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(dv.getResult());
    if (!dout) return mlir::success();
    // d/dlhs = dy / rhs ; d/drhs = −dy ⊙ lhs / rhs².
    mlir::Value dLhs = emitBinary(b, loc, "div", dout, dv.getRhs());
    mlir::Value lhsOverRhs = emitBinary(b, loc, "div", dv.getLhs(), dv.getRhs());
    mlir::Value tmp = emitBinary(b, loc, "mul", dLhs, lhsOverRhs);
    mlir::Value dRhs = emitUnary(b, loc, "neg", tmp);
    ctx.addContribution(dv.getLhs(), dLhs);
    ctx.addContribution(dv.getRhs(), dRhs);
    return mlir::success();
  }
  if (auto rope = mlir::dyn_cast<RotaryEmbeddingOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(rope.getResult());
    if (!dout) return mlir::success();
    // RoPE applies an orthogonal per-pair rotation R(θ); its adjoint is
    // R(−θ), i.e. rotary_embedding with sin negated. cos/sin are inputs
    // (no gradient flows to the position tables).
    mlir::Value negSin = emitUnary(b, loc, "neg", rope.getSin());
    mlir::Value dx = emitRope(b, loc, dout, rope.getCos(), negSin);
    ctx.addContribution(rope.getInput(), dx);
    return mlir::success();
  }
  if (auto pm = mlir::dyn_cast<PermuteOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(pm.getResult());
    if (!dout) return mlir::success();
    // Inverse permutation: if y = permute(x, axes) then dx = permute(dy, inv).
    llvm::SmallVector<int64_t, 4> axes;
    for (mlir::Attribute a : pm.getAxes())
      axes.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());
    llvm::SmallVector<int64_t, 4> inv(axes.size());
    for (std::size_t i = 0; i < axes.size(); ++i)
      inv[static_cast<std::size_t>(axes[i])] = static_cast<int64_t>(i);
    ctx.addContribution(pm.getInput(), emitPermute(b, loc, dout, inv));
    return mlir::success();
  }
  if (auto vw = mlir::dyn_cast<ViewOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(vw.getResult());
    if (!dout) return mlir::success();
    // Reshape the cotangent back to the input shape (view is metadata-only).
    auto inTy = mlir::cast<mlir::RankedTensorType>(vw.getInput().getType());
    ctx.addContribution(vw.getInput(), emitReshape(b, loc, dout, inTy));
    return mlir::success();
  }
  if (auto rs = mlir::dyn_cast<ReshapeOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(rs.getResult());
    if (!dout) return mlir::success();
    auto inTy = mlir::cast<mlir::RankedTensorType>(rs.getInput().getType());
    ctx.addContribution(rs.getInput(), emitReshape(b, loc, dout, inTy));
    return mlir::success();
  }
  if (auto sumOp = mlir::dyn_cast<SumOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(sumOp.getResult());
    if (!dout) return mlir::success();
    auto xTy = mlir::cast<mlir::RankedTensorType>(sumOp.getInput().getType());
    const int64_t inRank = xTy.getRank();
    int64_t dim = sumOp.getDim();
    // dim == -1 (or reduce-all): cotangent is a scalar; broadcast back.
    if (dim < 0 || inRank <= 1) {
      ctx.addContribution(sumOp.getInput(), emitBroadcastTo(b, loc, dout, xTy));
      return mlir::success();
    }
    // Per-axis reduction: dx = broadcast(dOut) along the reduced axis. When
    // keepdim=false the cotangent dropped that axis, so first reshape it back
    // to a size-1 slot at `dim`; with keepdim=true it is already size-1 there.
    if (!sumOp.getKeepdim()) {
      llvm::SmallVector<int64_t, 4> keepShape(xTy.getShape().begin(),
                                              xTy.getShape().end());
      keepShape[static_cast<std::size_t>(dim)] = 1;
      auto keepTy =
          mlir::RankedTensorType::get(keepShape, xTy.getElementType());
      dout = emitReshape(b, loc, dout, keepTy);
    }
    ctx.addContribution(sumOp.getInput(), emitBroadcastTo(b, loc, dout, xTy));
    return mlir::success();
  }
  if (auto reluOp = mlir::dyn_cast<ReluOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(reluOp.getResult());
    if (!dout) return mlir::success();
    // dx = relu_backward(x, dout). Kept as a fused op at M1I; lowering to
    // a compare + select lands with M1J's pattern rewrites.
    mlir::OperationState state(loc, "tesseract.relu_backward");
    state.addOperands({reluOp.getInput(), dout});
    state.addTypes({reluOp.getInput().getType()});
    mlir::Value dx = b.create(state)->getResult(0);
    ctx.addContribution(reluOp.getInput(), dx);
    return mlir::success();
  }
  if (auto ceOp = mlir::dyn_cast<CrossEntropyWithLogitsOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(ceOp.getResult());
    if (!dout) return mlir::success();
    // d_logits = cross_entropy_with_logits_backward(logits, targets, dout)
    // Produces a tensor with the same type as logits; targets gets no
    // gradient (Int64 index tensor).
    mlir::OperationState state(loc,
                               "tesseract.cross_entropy_with_logits_backward");
    state.addOperands({ceOp.getLogits(), ceOp.getTargets(), dout});
    state.addTypes({ceOp.getLogits().getType()});
    mlir::Value dLogits = b.create(state)->getResult(0);
    ctx.addContribution(ceOp.getLogits(), dLogits);
    // No contribution for `targets` — it's an index tensor.
    return mlir::success();
  }
  if (auto tp = mlir::dyn_cast<TransposeOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(tp.getResult());
    if (!dout) return mlir::success();
    // transpose is self-inverse along the same pair of axes.
    mlir::Value dx = emitTranspose(b, loc, dout, tp.getDimA(), tp.getDimB());
    ctx.addContribution(tp.getInput(), dx);
    return mlir::success();
  }
  if (auto bcast = mlir::dyn_cast<BroadcastToOp>(op)) {
    mlir::Value dout = ctx.grad.lookup(bcast.getResult());
    if (!dout) return mlir::success();
    auto inTy = mlir::cast<mlir::RankedTensorType>(bcast.getInput().getType());
    ctx.addContribution(bcast.getInput(), reduceToShape(b, loc, dout, inTy));
    return mlir::success();
  }
  if (auto paramOp = mlir::dyn_cast<ParamOp>(op)) {
    // Pass the gradient straight through: grad of the param's result is
    // also the grad of its block-argument input.
    mlir::Value dout = ctx.grad.lookup(paramOp.getResult());
    if (!dout) return mlir::success();
    ctx.addContribution(paramOp.getInput(), dout);
    return mlir::success();
  }

  return op->emitOpError(
      "tesseract-backward: no backward rule registered for this op");
}

// ---------------- Function-level driver ----------------

mlir::LogicalResult runOnFunction(FunctionOp func) {
  mlir::Block *body = &func.getBody().front();
  auto *terminator = body->getTerminator();
  auto ret = mlir::dyn_cast<ReturnOp>(terminator);
  if (!ret) return func.emitOpError("expected tesseract.return terminator");

  // --- Collect forward structure. --- //
  llvm::SmallVector<mlir::Value, 4> forwardOutputs(ret.getOperands().begin(),
                                                   ret.getOperands().end());
  llvm::SmallVector<ParamOp, 4> paramOps;
  for (auto pOp : body->getOps<ParamOp>()) paramOps.push_back(pOp);

  // --- Extend block with grad-output arguments. --- //
  llvm::SmallVector<mlir::Type, 4> gradOutTypes;
  gradOutTypes.reserve(forwardOutputs.size());
  for (mlir::Value out : forwardOutputs) gradOutTypes.push_back(out.getType());

  llvm::SmallVector<mlir::Location, 4> gradOutLocs(gradOutTypes.size(),
                                                   func.getLoc());
  const unsigned firstGradArgIdx = body->getNumArguments();
  for (auto [ty, l] : llvm::zip(gradOutTypes, gradOutLocs)) {
    body->addArgument(ty, l);
  }

  // --- Seed grad_map: forward output → new block-arg. --- //
  mlir::OpBuilder builder(func.getContext());
  builder.setInsertionPoint(terminator);
  BackwardCtx ctx{builder, func.getLoc(), {}};
  for (auto [i, out] : llvm::enumerate(forwardOutputs)) {
    ctx.grad[out] = body->getArgument(firstGradArgIdx + i);
  }

  // --- Walk ops in reverse, computing input gradients. --- //
  llvm::SmallVector<mlir::Operation *, 16> ops;
  for (auto &op : body->without_terminator()) ops.push_back(&op);
  for (auto *op : llvm::reverse(ops)) {
    if (mlir::failed(backwardOp(op, ctx))) return mlir::failure();
  }

  // --- Collect param gradients in declaration order. --- //
  llvm::SmallVector<mlir::Value, 4> paramGrads;
  paramGrads.reserve(paramOps.size());
  for (ParamOp pOp : paramOps) {
    mlir::Value g = ctx.grad.lookup(pOp.getInput());
    if (!g) {
      // Param not used in any op that feeds an output: emit a
      // broadcast-of-zero. Build a `tesseract.constant` of zeros matching
      // the param type.
      auto ty = mlir::cast<mlir::RankedTensorType>(pOp.getInput().getType());
      mlir::Attribute zeroElems = mlir::DenseElementsAttr::get(
          ty, builder.getFloatAttr(ty.getElementType(), 0.0));
      mlir::OperationState cstState(func.getLoc(), "tesseract.constant");
      cstState.addTypes({ty});
      cstState.addAttribute("value", zeroElems);
      g = builder.create(cstState)->getResult(0);
    }
    paramGrads.push_back(g);
  }

  // --- Replace terminator with new return (forward outs ++ paramGrads). --- //
  llvm::SmallVector<mlir::Value, 8> newRetValues(forwardOutputs.begin(),
                                                 forwardOutputs.end());
  newRetValues.append(paramGrads.begin(), paramGrads.end());

  builder.setInsertionPoint(terminator);
  mlir::OperationState retState(terminator->getLoc(), "tesseract.return");
  retState.addOperands(newRetValues);
  builder.create(retState);
  terminator->erase();

  // --- Update the function_type attribute. --- //
  llvm::SmallVector<mlir::Type, 8> newResultTypes;
  newResultTypes.reserve(newRetValues.size());
  for (mlir::Value v : newRetValues) newResultTypes.push_back(v.getType());

  llvm::SmallVector<mlir::Type, 8> newInputTypes(
      body->getArgumentTypes().begin(), body->getArgumentTypes().end());

  auto newFuncTy =
      mlir::FunctionType::get(func.getContext(), newInputTypes, newResultTypes);
  func->setAttr(func.getFunctionTypeAttrName(),
                mlir::TypeAttr::get(newFuncTy));
  return mlir::success();
}

// ---------------- Pass ----------------

struct BackwardPass
    : public mlir::PassWrapper<BackwardPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BackwardPass)

  llvm::StringRef getArgument() const final { return "tesseract-backward"; }
  llvm::StringRef getDescription() const final {
    return "Append reverse-mode AD to every tesseract.function: extend the "
           "signature with cotangents as inputs and parameter gradients as "
           "outputs (vjp-style).";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::LogicalResult status = mlir::success();
    module.walk([&](FunctionOp f) {
      if (mlir::failed(runOnFunction(f))) {
        status = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (mlir::failed(status)) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createBackwardPass() {
  return std::make_unique<BackwardPass>();
}

void registerBackwardPass() {
  mlir::PassRegistration<BackwardPass>();
}

}  // namespace tesseract::ir

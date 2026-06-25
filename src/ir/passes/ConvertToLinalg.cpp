// tesseract → linalg lowering (M1G).
//
// Rewrites the core `tesseract` dialect ops that the Graph emitter produces
// into the upstream `linalg` / `tensor` / `arith` dialects. The intent is to
// produce Linalg on tensors that the standard CPU pipeline
// (`--one-shot-bufferize` → `--convert-linalg-to-loops` →
// `--convert-scf-to-cf` → `--convert-func-to-llvm` → ...) already knows how
// to process.
//
// At M1G we cover the canonical compute-bound ops:
//
//   tesseract.add   → linalg.add       (ins/outs on tensor.empty)
//   tesseract.sub   → linalg.sub
//   tesseract.mul   → linalg.mul
//   tesseract.div   → linalg.div
//   tesseract.matmul→ linalg.fill(0) → linalg.matmul
//   tesseract.sum   → linalg.fill(0) → linalg.reduce { arith.addf }
//
// The remaining ops (activations, softmax family, shape manipulation,
// composite loss) are deferred to follow-up passes (M1.γ+). Every unhandled
// op is left untouched; `applyPartialConversion` is not used here so the
// pass composes cleanly with other dialects.

#include "tesseract/ir/Passes.hpp"
#include "tesseract/ir/TesseractDialect.h"
#include "tesseract/ir/TesseractOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tesseract::ir {

namespace {

// Materialize a `tensor.empty` of the same shape / element type as `ty`. The
// structured linalg ops require an explicit destination passed via `outs`.
mlir::Value createEmptyLike(mlir::OpBuilder &b, mlir::Location loc,
                            mlir::RankedTensorType ty) {
  return b.create<mlir::tensor::EmptyOp>(loc, ty.getShape(),
                                         ty.getElementType());
}

// Materialize a zero-initialized tensor via `linalg.fill`. Used as the init
// operand for reductions / matmul where the body accumulates into outs.
mlir::Value createZeroFilled(mlir::OpBuilder &b, mlir::Location loc,
                             mlir::RankedTensorType ty) {
  mlir::Value empty = createEmptyLike(b, loc, ty);
  mlir::Attribute zero;
  mlir::Type elemTy = ty.getElementType();
  if (mlir::isa<mlir::FloatType>(elemTy)) {
    zero = b.getFloatAttr(elemTy, 0.0);
  } else if (auto itype = mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
    zero = b.getIntegerAttr(itype, 0);
  } else {
    // Fallback: let the verifier complain; this branch is not hit by any
    // M1 lowering today.
    zero = b.getZeroAttr(elemTy);
  }
  mlir::Value zeroCst = b.create<mlir::arith::ConstantOp>(
      loc, mlir::cast<mlir::TypedAttr>(zero));
  return b.create<mlir::linalg::FillOp>(
              loc, mlir::ValueRange{zeroCst}, mlir::ValueRange{empty})
      .getResult(0);
}

// ---------------- Binary elementwise ----------------
//
// All four ops share the same rewrite shape: allocate an `empty`, hand it to
// the matching named linalg op, replace.

template <typename TesseractOp, typename LinalgOp>
struct BinaryElementwiseLowering : public mlir::OpRewritePattern<TesseractOp> {
  using mlir::OpRewritePattern<TesseractOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TesseractOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    mlir::Value init = createEmptyLike(rewriter, op.getLoc(), resultTy);
    auto newOp = rewriter.create<LinalgOp>(
        op.getLoc(), mlir::ValueRange{op.getLhs(), op.getRhs()},
        mlir::ValueRange{init});
    rewriter.replaceOp(op, newOp.getResults());
    return mlir::success();
  }
};

// ---------------- MatMul ----------------

struct MatMulLowering : public mlir::OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MatMulOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto lhsTy = mlir::cast<mlir::RankedTensorType>(op.getLhs().getType());
    auto rhsTy = mlir::cast<mlir::RankedTensorType>(op.getRhs().getType());
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());

    // Legality: only rank-2 (M,K)x(K,N) and rank-3 batched (B,M,K)x(B,K,N)
    // map to linalg.matmul / linalg.batch_matmul. lhs/rhs/result ranks must
    // agree, and the contracting (and batch) dims must be statically
    // compatible — otherwise we'd emit a verifier-invalid linalg op. Reject
    // cleanly so an unsupported matmul surfaces as an unlowered tesseract op
    // rather than malformed IR.
    const int64_t rank = resultTy.getRank();
    if (rank != lhsTy.getRank() || rank != rhsTy.getRank())
      return rewriter.notifyMatchFailure(op, "matmul lhs/rhs/result rank mismatch");
    if (rank != 2 && rank != 3)
      return rewriter.notifyMatchFailure(
          op, "matmul lowering supports only rank-2 or rank-3 (batched)");

    auto ls = lhsTy.getShape();
    auto rs = rhsTy.getShape();
    auto os = resultTy.getShape();
    const int64_t kd = rank - 1;       // contracting axis index
    const int64_t md = rank - 2;       // M / output-row axis
    auto dynamicOrEq = [](int64_t a, int64_t b) {
      return a == mlir::ShapedType::kDynamic ||
             b == mlir::ShapedType::kDynamic || a == b;
    };
    // K dims: lhs[..,K] vs rhs[K,..].
    if (!dynamicOrEq(ls[kd], rs[md]))
      return rewriter.notifyMatchFailure(op, "matmul contracting dim mismatch");
    // Result rows/cols.
    if (!dynamicOrEq(os[md], ls[md]) || !dynamicOrEq(os[kd], rs[kd]))
      return rewriter.notifyMatchFailure(op, "matmul result shape mismatch");
    // Batch dim for rank-3.
    if (rank == 3 &&
        (!dynamicOrEq(ls[0], rs[0]) || !dynamicOrEq(os[0], ls[0])))
      return rewriter.notifyMatchFailure(op, "batched matmul batch-dim mismatch");

    // linalg matmul accumulates, so the init must be zero.
    mlir::Value init = createZeroFilled(rewriter, op.getLoc(), resultTy);
    if (rank == 3) {
      auto newOp = rewriter.create<mlir::linalg::BatchMatmulOp>(
          op.getLoc(), mlir::TypeRange{resultTy},
          mlir::ValueRange{op.getLhs(), op.getRhs()}, mlir::ValueRange{init});
      rewriter.replaceOp(op, newOp.getResults());
      return mlir::success();
    }
    auto newOp = rewriter.create<mlir::linalg::MatmulOp>(
        op.getLoc(), mlir::ValueRange{op.getLhs(), op.getRhs()},
        mlir::ValueRange{init});
    rewriter.replaceOp(op, newOp.getResults());
    return mlir::success();
  }
};

// ---------------- Sum reduction ----------------
//
// Semantics we handle here:
//   * `dim == -1, keepdim == false`   → scalar tensor<f32>, reduce all dims.
//   * `dim >= 0,  keepdim == false`   → reduce a single axis.
//   * `dim >= 0,  keepdim == true`    → reduce + tensor.expand_shape back.
//
// For M1G we only emit the reduction for floating-point element types; the
// body yields `arith.addf`. Other element types fall through and the op is
// left in place.

struct SumLowering : public mlir::OpRewritePattern<SumOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SumOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputTy =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!mlir::isa<mlir::FloatType>(inputTy.getElementType())) {
      return rewriter.notifyMatchFailure(op, "only floating-point sum is "
                                             "lowered at M1");
    }

    const int64_t inRank = inputTy.getRank();
    const int64_t dim = op.getDim();
    const bool keepdim = op.getKeepdim();

    llvm::SmallVector<int64_t, 4> reduceDims;
    if (dim < 0) {
      for (int64_t i = 0; i < inRank; ++i) reduceDims.push_back(i);
    } else {
      reduceDims.push_back(dim);
    }

    // When keepdim is true, linalg.reduce's result has the reduced axes
    // collapsed — we re-expand after the reduction to match the expected
    // output shape.
    llvm::SmallVector<int64_t, 4> reducedShape;
    for (int64_t i = 0; i < inRank; ++i) {
      if (llvm::is_contained(reduceDims, i)) continue;
      reducedShape.push_back(inputTy.getShape()[i]);
    }

    auto reducedTy =
        mlir::RankedTensorType::get(reducedShape, inputTy.getElementType());
    mlir::Value init = createZeroFilled(rewriter, op.getLoc(), reducedTy);

    auto reduceOp = rewriter.create<mlir::linalg::ReduceOp>(
        op.getLoc(), mlir::ValueRange{op.getInput()}, mlir::ValueRange{init},
        reduceDims,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
          // args[0] is the input element, args[1] is the accumulator.
          mlir::Value added =
              b.create<mlir::arith::AddFOp>(loc, args[0], args[1]);
          b.create<mlir::linalg::YieldOp>(loc, added);
        });

    mlir::Value reduced = reduceOp.getResult(0);

    if (!keepdim || reducedTy == resultTy) {
      rewriter.replaceOp(op, reduced);
      return mlir::success();
    }

    // keepdim=true: the recorded result shape keeps the reduced axes at
    // size 1. Reassociate by inserting a unit dim for each reduced axis.
    llvm::SmallVector<mlir::ReassociationIndices, 4> reassoc;
    int64_t srcIdx = 0;
    reassoc.resize(reducedShape.size());
    for (int64_t i = 0; i < static_cast<int64_t>(resultTy.getShape().size());
         ++i) {
      if (llvm::is_contained(reduceDims, i)) continue;
      reassoc[srcIdx].push_back(i);
      ++srcIdx;
    }
    // Attach every reduced axis to the nearest surviving source dim (prefer
    // the one to its left; fall back to srcIdx 0 if reducing a prefix).
    for (int64_t i = 0; i < static_cast<int64_t>(resultTy.getShape().size());
         ++i) {
      if (!llvm::is_contained(reduceDims, i)) continue;
      int64_t target = 0;
      for (int64_t j = static_cast<int64_t>(reassoc.size()) - 1; j >= 0; --j) {
        if (!reassoc[j].empty() && reassoc[j].front() < i) {
          target = j;
          break;
        }
      }
      reassoc[target].push_back(i);
    }

    auto expanded = rewriter.create<mlir::tensor::ExpandShapeOp>(
        op.getLoc(), resultTy, reduced, reassoc);
    rewriter.replaceOp(op, expanded.getResult());
    return mlir::success();
  }
};

// ---------------- Unary elementwise via linalg.generic ----------------
//
// `linalg` ships named ops for add/sub/mul/div but not for neg or max(0, x).
// Both are trivially expressed as `linalg.generic` with all-parallel
// iterators and an identity indexing map. We share a single helper that
// the relu / neg / relu_backward rewrites call.

// Build a linalg.generic whose body is `bodyBuilder(b, loc, args)` and
// yields a single value of the result's element type. Takes N inputs
// (same rank/shape as the result) and writes into `init`.
mlir::Value createElementwiseGeneric(
    mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs,
    mlir::RankedTensorType resultTy,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                    mlir::ValueRange)>
        bodyBuilder) {
  mlir::Value init = createEmptyLike(b, loc, resultTy);
  const int64_t rank = resultTy.getRank();
  llvm::SmallVector<mlir::AffineMap, 4> maps(
      inputs.size() + 1,
      mlir::AffineMap::getMultiDimIdentityMap(rank, b.getContext()));
  llvm::SmallVector<mlir::utils::IteratorType, 4> iters(
      rank, mlir::utils::IteratorType::parallel);
  auto gen = b.create<mlir::linalg::GenericOp>(
      loc, mlir::TypeRange{resultTy}, inputs, mlir::ValueRange{init}, maps,
      iters,
      [&](mlir::OpBuilder &nested, mlir::Location nloc, mlir::ValueRange args) {
        // args = {input0_elem, input1_elem, ..., init_elem}. Drop the init.
        mlir::Value v = bodyBuilder(nested, nloc, args.drop_back(1));
        nested.create<mlir::linalg::YieldOp>(nloc, v);
      });
  return gen.getResult(0);
}

struct NegLowering : public mlir::OpRewritePattern<NegOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(NegOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!mlir::isa<mlir::FloatType>(resultTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "only float neg is lowered");
    mlir::Value out = createElementwiseGeneric(
        rewriter, op.getLoc(), mlir::ValueRange{op.getInput()}, resultTy,
        [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
          return b.create<mlir::arith::NegFOp>(loc, args[0]).getResult();
        });
    rewriter.replaceOp(op, out);
    return mlir::success();
  }
};

struct ReluLowering : public mlir::OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ReluOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto elemTy = resultTy.getElementType();
    if (!mlir::isa<mlir::FloatType>(elemTy))
      return rewriter.notifyMatchFailure(op, "only float relu is lowered");
    mlir::Value out = createElementwiseGeneric(
        rewriter, op.getLoc(), mlir::ValueRange{op.getInput()}, resultTy,
        [elemTy](mlir::OpBuilder &b, mlir::Location loc,
                 mlir::ValueRange args) {
          mlir::Value zero = b.create<mlir::arith::ConstantOp>(
              loc, b.getFloatAttr(elemTy, 0.0));
          return b.create<mlir::arith::MaximumFOp>(loc, args[0], zero)
              .getResult();
        });
    rewriter.replaceOp(op, out);
    return mlir::success();
  }
};

// dx = dout * (x > 0 ? 1 : 0). Fold the mask + multiply into a single
// arith.select on the grad operand.
struct ReluBackwardLowering : public mlir::OpRewritePattern<ReluBackwardOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ReluBackwardOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto elemTy = resultTy.getElementType();
    if (!mlir::isa<mlir::FloatType>(elemTy))
      return rewriter.notifyMatchFailure(
          op, "only float relu_backward is lowered");
    mlir::Value out = createElementwiseGeneric(
        rewriter, op.getLoc(),
        mlir::ValueRange{op.getInput(), op.getGradOutput()}, resultTy,
        [elemTy](mlir::OpBuilder &b, mlir::Location loc,
                 mlir::ValueRange args) {
          mlir::Value zero = b.create<mlir::arith::ConstantOp>(
              loc, b.getFloatAttr(elemTy, 0.0));
          mlir::Value mask = b.create<mlir::arith::CmpFOp>(
              loc, mlir::arith::CmpFPredicate::OGT, args[0], zero);
          return b.create<mlir::arith::SelectOp>(loc, mask, args[1], zero)
              .getResult();
        });
    rewriter.replaceOp(op, out);
    return mlir::success();
  }
};

// ---------------- Unary math activations ----------------
//
// The transformer FFN + normalization stack decomposes into these scalar
// activations (M4 Track C1 / B-044). Each is a single `linalg.generic` with
// an all-parallel body that calls the matching `math`/`arith` scalar op:
//
//   tesseract.exp/log/sqrt/tanh → math.exp/log/sqrt/tanh
//   tesseract.sigmoid           → 1 / (1 + exp(-x))   (math has no sigmoid)
//
// These are exactly the primitives `ops::rms_norm` (mul/mean/add/sqrt/div/mul)
// and `ops::swiglu_silu_gate` (sigmoid/mul/mul) record into the graph, so once
// they lower a captured RMSNorm / SwiGLU-FFN runs through the CPU JitEngine.

template <typename TesseractOp, typename MathOp>
struct UnaryMathLowering : public mlir::OpRewritePattern<TesseractOp> {
  using mlir::OpRewritePattern<TesseractOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(TesseractOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!mlir::isa<mlir::FloatType>(resultTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "only float unary math lowered");
    mlir::Value out = createElementwiseGeneric(
        rewriter, op.getLoc(), mlir::ValueRange{op.getInput()}, resultTy,
        [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
          return b.create<MathOp>(loc, args[0]).getResult();
        });
    rewriter.replaceOp(op, out);
    return mlir::success();
  }
};

struct SigmoidLowering : public mlir::OpRewritePattern<SigmoidOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(SigmoidOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto elemTy = resultTy.getElementType();
    if (!mlir::isa<mlir::FloatType>(elemTy))
      return rewriter.notifyMatchFailure(op, "only float sigmoid is lowered");
    mlir::Value out = createElementwiseGeneric(
        rewriter, op.getLoc(), mlir::ValueRange{op.getInput()}, resultTy,
        [elemTy](mlir::OpBuilder &b, mlir::Location loc,
                 mlir::ValueRange args) {
          mlir::Value one = b.create<mlir::arith::ConstantOp>(
              loc, b.getFloatAttr(elemTy, 1.0));
          mlir::Value negx = b.create<mlir::arith::NegFOp>(loc, args[0]);
          mlir::Value e = b.create<mlir::math::ExpOp>(loc, negx);
          mlir::Value denom = b.create<mlir::arith::AddFOp>(loc, one, e);
          return b.create<mlir::arith::DivFOp>(loc, one, denom).getResult();
        });
    rewriter.replaceOp(op, out);
    return mlir::success();
  }
};

// ---------------- Mean reduction ----------------
//
// Mirrors `SumLowering` (linalg.reduce { addf } + optional expand_shape for
// keepdim) then scales the result by 1/N where N is the product of the
// reduced extents. `ops::rms_norm` uses `mean(dim=-1, keepdim=true)`.
struct MeanLowering : public mlir::OpRewritePattern<MeanOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(MeanOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputTy =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto elemTy = inputTy.getElementType();
    if (!mlir::isa<mlir::FloatType>(elemTy))
      return rewriter.notifyMatchFailure(op, "only float mean is lowered");

    const int64_t inRank = inputTy.getRank();
    const int64_t dim = op.getDim();
    const bool keepdim = op.getKeepdim();

    llvm::SmallVector<int64_t, 4> reduceDims;
    if (dim < 0) {
      for (int64_t i = 0; i < inRank; ++i) reduceDims.push_back(i);
    } else {
      reduceDims.push_back(dim);
    }

    int64_t N = 1;
    llvm::SmallVector<int64_t, 4> reducedShape;
    for (int64_t i = 0; i < inRank; ++i) {
      if (llvm::is_contained(reduceDims, i)) {
        N *= inputTy.getShape()[i];
        continue;
      }
      reducedShape.push_back(inputTy.getShape()[i]);
    }
    if (N <= 0)
      return rewriter.notifyMatchFailure(op, "dynamic reduced extent");

    auto reducedTy = mlir::RankedTensorType::get(reducedShape, elemTy);
    mlir::Value init = createZeroFilled(rewriter, op.getLoc(), reducedTy);
    auto reduceOp = rewriter.create<mlir::linalg::ReduceOp>(
        op.getLoc(), mlir::ValueRange{op.getInput()}, mlir::ValueRange{init},
        reduceDims,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
          mlir::Value added =
              b.create<mlir::arith::AddFOp>(loc, args[0], args[1]);
          b.create<mlir::linalg::YieldOp>(loc, added);
        });

    // Scale the summed result by 1/N in place (all-parallel generic).
    const double invN = 1.0 / static_cast<double>(N);
    mlir::Value meaned = createElementwiseGeneric(
        rewriter, op.getLoc(), mlir::ValueRange{reduceOp.getResult(0)},
        reducedTy,
        [elemTy, invN](mlir::OpBuilder &b, mlir::Location loc,
                       mlir::ValueRange args) {
          mlir::Value s = b.create<mlir::arith::ConstantOp>(
              loc, b.getFloatAttr(elemTy, invN));
          return b.create<mlir::arith::MulFOp>(loc, args[0], s).getResult();
        });

    if (!keepdim || reducedTy == resultTy) {
      rewriter.replaceOp(op, meaned);
      return mlir::success();
    }

    llvm::SmallVector<mlir::ReassociationIndices, 4> reassoc;
    reassoc.resize(reducedShape.empty() ? 1 : reducedShape.size());
    int64_t srcIdx = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(resultTy.getShape().size());
         ++i) {
      if (llvm::is_contained(reduceDims, i)) continue;
      reassoc[srcIdx].push_back(i);
      ++srcIdx;
    }
    for (int64_t i = 0; i < static_cast<int64_t>(resultTy.getShape().size());
         ++i) {
      if (!llvm::is_contained(reduceDims, i)) continue;
      int64_t target = 0;
      for (int64_t j = static_cast<int64_t>(reassoc.size()) - 1; j >= 0; --j) {
        if (!reassoc[j].empty() && reassoc[j].front() < i) {
          target = j;
          break;
        }
      }
      reassoc[target].push_back(i);
    }
    auto expanded = rewriter.create<mlir::tensor::ExpandShapeOp>(
        op.getLoc(), resultTy, meaned, reassoc);
    rewriter.replaceOp(op, expanded.getResult());
    return mlir::success();
  }
};

// ---------------- Transpose ----------------
//
// `tesseract.transpose` swaps two named dims of a ranked tensor. Upstream
// ships a dedicated `linalg.transpose` op that takes a permutation
// attribute — we build that permutation from `dim_a` / `dim_b` and let
// the downstream pipeline handle it.
struct TransposeLowering : public mlir::OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputTy =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    const int64_t rank = inputTy.getRank();
    int64_t a = op.getDimA();
    int64_t b = op.getDimB();
    if (a < 0) a += rank;
    if (b < 0) b += rank;
    if (a < 0 || a >= rank || b < 0 || b >= rank)
      return rewriter.notifyMatchFailure(op, "dim out of range");

    llvm::SmallVector<int64_t, 4> perm(rank);
    for (int64_t i = 0; i < rank; ++i) perm[i] = i;
    std::swap(perm[a], perm[b]);

    mlir::Value init = createEmptyLike(rewriter, op.getLoc(), resultTy);
    auto tr = rewriter.create<mlir::linalg::TransposeOp>(
        op.getLoc(), op.getInput(), init, perm);
    rewriter.replaceOp(op, tr.getResults());
    return mlir::success();
  }
};

// ---------------- View / Reshape (contiguous) ----------------
//
// Both ops are pure metadata reshapes of a contiguous row-major tensor. We
// lower them by collapsing the input to a flat 1-D tensor and re-expanding to
// the (static) result shape. `tensor.collapse_shape` / `tensor.expand_shape`
// bufferize to `memref.collapse_shape` / `memref.expand_shape`, which are
// metadata-only — no `memrefCopy` runtime symbol is pulled in (the trap RoPE
// hit earlier with `insert_slice`). This unlocks the multi-head reshape path
// ([S, H*Dh] ⇄ [S, H, Dh]) needed for a full Llama block through the JIT.
template <typename ShapeOp>
struct ReshapeViewLowering : public mlir::OpRewritePattern<ShapeOp> {
  using mlir::OpRewritePattern<ShapeOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ShapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inTy = mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto outTy = mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!inTy.hasStaticShape() || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic reshape unsupported");

    const int64_t inRank = inTy.getRank();
    const int64_t outRank = outTy.getRank();
    int64_t total = 1;
    for (int64_t d : inTy.getShape()) total *= d;
    // Element-count legality: a reshape/view must preserve numel. A mismatch
    // means the collapse→expand below would produce a verifier-invalid
    // expand_shape; reject so the malformed reshape surfaces as an unlowered
    // op rather than broken IR.
    int64_t outTotal = 1;
    for (int64_t d : outTy.getShape()) outTotal *= d;
    if (total != outTotal)
      return rewriter.notifyMatchFailure(
          op, "reshape changes element count (in != out numel)");

    // Flatten to 1-D (skip when already rank-1).
    mlir::Value flat = op.getInput();
    if (inRank != 1) {
      mlir::ReassociationIndices all;
      for (int64_t i = 0; i < inRank; ++i) all.push_back(i);
      auto flatTy =
          mlir::RankedTensorType::get({total}, inTy.getElementType());
      flat = rewriter.create<mlir::tensor::CollapseShapeOp>(
          loc, flatTy, op.getInput(),
          llvm::SmallVector<mlir::ReassociationIndices, 1>{all});
    }

    // Expand the flat tensor to the result shape (skip when result is rank-1).
    mlir::Value res = flat;
    if (outRank != 1) {
      mlir::ReassociationIndices all;
      for (int64_t i = 0; i < outRank; ++i) all.push_back(i);
      res = rewriter.create<mlir::tensor::ExpandShapeOp>(
          loc, outTy, flat,
          llvm::SmallVector<mlir::ReassociationIndices, 1>{all});
    }
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

// ---------------- Permute ----------------
//
// General axis permutation → `linalg.transpose` with the explicit axis list.
// `axes[i]` is the input dim that lands at output position `i`, which matches
// linalg.transpose's `permutation` convention directly.
struct PermuteLowering : public mlir::OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(PermuteOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputTy =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    const int64_t rank = inputTy.getRank();
    llvm::SmallVector<int64_t, 4> perm;
    perm.reserve(static_cast<std::size_t>(rank));
    llvm::SmallVector<bool, 4> seen(static_cast<std::size_t>(rank), false);
    for (mlir::Attribute a : op.getAxes()) {
      int64_t v = mlir::cast<mlir::IntegerAttr>(a).getInt();
      if (v < 0) v += rank;
      if (v < 0 || v >= rank)
        return rewriter.notifyMatchFailure(op, "axis out of range");
      // Bijection check: each input axis must appear exactly once. A repeated
      // axis (and, with the count check below, a missing one) would make
      // linalg.transpose's permutation invalid.
      if (seen[static_cast<std::size_t>(v)])
        return rewriter.notifyMatchFailure(op, "permute axes not a bijection "
                                               "(duplicate axis)");
      seen[static_cast<std::size_t>(v)] = true;
      perm.push_back(v);
    }
    if (static_cast<int64_t>(perm.size()) != rank)
      return rewriter.notifyMatchFailure(op, "axes count != rank");

    // Result shape must equal the input shape with `perm` applied
    // (result[i] == input[perm[i]]).
    auto inShape = inputTy.getShape();
    auto outShape = resultTy.getShape();
    for (int64_t i = 0; i < rank; ++i) {
      const int64_t want = inShape[static_cast<std::size_t>(perm[i])];
      const int64_t got = outShape[static_cast<std::size_t>(i)];
      if (want != mlir::ShapedType::kDynamic &&
          got != mlir::ShapedType::kDynamic && want != got)
        return rewriter.notifyMatchFailure(op, "permute result shape != "
                                               "permuted input shape");
    }

    mlir::Value init = createEmptyLike(rewriter, op.getLoc(), resultTy);
    auto tr = rewriter.create<mlir::linalg::TransposeOp>(
        op.getLoc(), op.getInput(), init, perm);
    rewriter.replaceOp(op, tr.getResults());
    return mlir::success();
  }
};

// ---------------- BroadcastTo ----------------
//
// Supports arbitrary numpy-style broadcast: align shapes to the right,
// missing leading dims in the input are treated as size-1 (they get
// collapsed into the result's corresponding output index via a constant
// 0 in the affine map), and size-1 dims expand to the output size.
// Implemented as a single `linalg.generic` with:
//   * iterator types: all-parallel
//   * input map: (d0, …, dR-1) -> (mapped dims…) where each mapped dim
//     is either `d_{i + rank_diff}` if the input dim is non-unit, or the
//     constant 0 otherwise (size-1 broadcast).
//   * output map: identity.
struct BroadcastToLowering : public mlir::OpRewritePattern<BroadcastToOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(BroadcastToOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputTy =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    const int64_t inRank = inputTy.getRank();
    const int64_t outRank = resultTy.getRank();
    if (outRank < inRank)
      return rewriter.notifyMatchFailure(op, "output rank < input rank");

    // Build the input AffineMap: one expression per input dim.
    const int64_t rankDiff = outRank - inRank;
    llvm::SmallVector<mlir::AffineExpr, 4> inExprs;
    inExprs.reserve(inRank);
    for (int64_t i = 0; i < inRank; ++i) {
      const int64_t outDim = i + rankDiff;
      const int64_t inSize = inputTy.getShape()[i];
      const int64_t outSize = resultTy.getShape()[outDim];
      if (inSize == outSize) {
        inExprs.push_back(rewriter.getAffineDimExpr(outDim));
      } else if (inSize == 1) {
        inExprs.push_back(rewriter.getAffineConstantExpr(0));
      } else {
        return rewriter.notifyMatchFailure(
            op, "non-broadcastable dim: input dim " + std::to_string(inSize) +
                    " vs output dim " + std::to_string(outSize));
      }
    }

    mlir::AffineMap inputMap = inRank == 0
        ? mlir::AffineMap::get(outRank, /*symbolCount=*/0, rewriter.getContext())
        : mlir::AffineMap::get(outRank, /*symbolCount=*/0, inExprs,
                               rewriter.getContext());
    mlir::AffineMap outputMap =
        mlir::AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());

    llvm::SmallVector<mlir::utils::IteratorType, 4> iters(
        outRank, mlir::utils::IteratorType::parallel);

    mlir::Value init = createEmptyLike(rewriter, op.getLoc(), resultTy);
    auto gen = rewriter.create<mlir::linalg::GenericOp>(
        op.getLoc(), mlir::TypeRange{resultTy},
        mlir::ValueRange{op.getInput()}, mlir::ValueRange{init},
        llvm::ArrayRef<mlir::AffineMap>{inputMap, outputMap}, iters,
        [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
          b.create<mlir::linalg::YieldOp>(loc, args[0]);
        });
    rewriter.replaceOp(op, gen.getResults());
    return mlir::success();
  }
};

// ---------------- Contiguous / Clone (layout ops) ----------------
//
// Both ops are layout-only in the eager API: `contiguous` promises a
// row-major copy, `clone` promises fresh storage. We materialize both as a
// real `linalg.copy` into a fresh `tensor.empty` rather than folding to the
// input. This honors `clone`'s fresh-storage contract (a no-op alias would
// let a later in-place write corrupt the original) and gives bufferization a
// well-defined materialization point even when producer == consumer
// (`x = contiguous(x)`). The copy is trivially DCE'd if the result is unused
// and elided by later fusion when provably safe — correctness first.
namespace {
mlir::Value materializeCopy(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value input,
                            mlir::RankedTensorType resultTy) {
  mlir::Value init = createEmptyLike(rewriter, loc, resultTy);
  auto cp = rewriter.create<mlir::linalg::CopyOp>(
      loc, mlir::ValueRange{input}, mlir::ValueRange{init});
  return cp.getResult(0);
}
}  // namespace

struct ContiguousLowering : public mlir::OpRewritePattern<ContiguousOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ContiguousOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    rewriter.replaceOp(
        op, materializeCopy(rewriter, op.getLoc(), op.getInput(), resultTy));
    return mlir::success();
  }
};

struct CloneLowering : public mlir::OpRewritePattern<CloneOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(CloneOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    rewriter.replaceOp(
        op, materializeCopy(rewriter, op.getLoc(), op.getInput(), resultTy));
    return mlir::success();
  }
};

// ---------------- Cross-entropy with logits (forward + backward) ----------------
//
// Both ops share a two-stage numeric skeleton (row-wise log-sum-exp +
// optional softmax materialization). We emit the skeleton as a sequence
// of `linalg.generic` ops — one per shape-affine stage — and let the
// downstream fusion passes collapse adjacent loops.
//
// The forward op decomposes as:
//   row_max [N]     : row-wise max reduction of logits [N, C]
//   sum_exp [N]     : sum_c exp(logits[n, c] - row_max[n])
//   lse     [N]     : log(sum_exp) + row_max          (= log sum exp)
//   selected[N]     : sum_c onehot(targets)[n, c] * logits[n, c]
//                     (equivalent to logits[n, targets[n]], but without
//                      runtime indirection — onehot fuses into the
//                      reduction body via `linalg.index` + `arith.cmpi`.)
//   per_row [N]     : lse - selected
//   sum_all []      : sum_n per_row[n]
//   loss    []      : sum_all / N
//
// The backward op decomposes as:
//   row_max, sum_exp, inv_sum_exp (= 1 / sum_exp)
//   probs  [N, C]   : exp(logits[n, c] - row_max[n]) * inv_sum_exp[n]
//   d_logits[N, C]  : (probs - onehot(targets)) * (grad_scalar / N)
//
// Both forms assume `logits : [N, C] (float)`, `targets : [N] (i64)`,
// `grad : [] (float)` (scalar). Anything else bails out — upstream
// passes will surface the unlowered op.

namespace {

// Create a `linalg.fill`-initialized tensor of `ty` filled with `attr`.
mlir::Value createFloatFilled(mlir::OpBuilder &b, mlir::Location loc,
                              mlir::RankedTensorType ty, mlir::Attribute attr) {
  mlir::Value empty = createEmptyLike(b, loc, ty);
  mlir::Value cst = b.create<mlir::arith::ConstantOp>(
      loc, mlir::cast<mlir::TypedAttr>(attr));
  return b.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{cst},
                                         mlir::ValueRange{empty})
      .getResult(0);
}

// Row-wise reduction along the last (C) axis:
//   output[n] = reduce_op(logits[n, 0], logits[n, 1], ..., init[n])
// `init` must be a pre-filled [N] tensor. `reduceBody(b, loc, elem, acc)`
// produces the new accumulator.
mlir::Value createRowReduce(
    mlir::OpBuilder &b, mlir::Location loc, mlir::Value input, mlir::Value init,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                    mlir::Value, mlir::Value)>
        reduceBody) {
  auto *ctx = b.getContext();
  auto inputMap = mlir::AffineMap::getMultiDimIdentityMap(2, ctx);
  auto outputMap = mlir::AffineMap::get(
      2, 0, {mlir::getAffineDimExpr(0, ctx)}, ctx);
  llvm::SmallVector<mlir::utils::IteratorType, 2> iters{
      mlir::utils::IteratorType::parallel,
      mlir::utils::IteratorType::reduction};
  auto gen = b.create<mlir::linalg::GenericOp>(
      loc, mlir::TypeRange{init.getType()}, mlir::ValueRange{input},
      mlir::ValueRange{init},
      llvm::ArrayRef<mlir::AffineMap>{inputMap, outputMap}, iters,
      [&](mlir::OpBuilder &nested, mlir::Location nloc, mlir::ValueRange args) {
        mlir::Value v = reduceBody(nested, nloc, args[0], args[1]);
        nested.create<mlir::linalg::YieldOp>(nloc, v);
      });
  return gen.getResult(0);
}

// Compute `logits[n, c] - row_val[n]` via a linalg.generic that
// broadcasts `row_val` across the C axis.
mlir::Value createShiftedLogits(mlir::OpBuilder &b, mlir::Location loc,
                                mlir::Value logits, mlir::Value rowVal,
                                mlir::RankedTensorType logitsTy) {
  auto *ctx = b.getContext();
  auto logitsMap = mlir::AffineMap::getMultiDimIdentityMap(2, ctx);
  auto rowMap = mlir::AffineMap::get(
      2, 0, {mlir::getAffineDimExpr(0, ctx)}, ctx);
  mlir::Value init = createEmptyLike(b, loc, logitsTy);
  llvm::SmallVector<mlir::utils::IteratorType, 2> iters(
      2, mlir::utils::IteratorType::parallel);
  auto gen = b.create<mlir::linalg::GenericOp>(
      loc, mlir::TypeRange{logitsTy},
      mlir::ValueRange{logits, rowVal}, mlir::ValueRange{init},
      llvm::ArrayRef<mlir::AffineMap>{logitsMap, rowMap, logitsMap}, iters,
      [](mlir::OpBuilder &nested, mlir::Location nloc, mlir::ValueRange args) {
        mlir::Value d = nested.create<mlir::arith::SubFOp>(nloc, args[0], args[1]);
        nested.create<mlir::linalg::YieldOp>(nloc, d);
      });
  return gen.getResult(0);
}

}  // namespace

struct CrossEntropyWithLogitsLowering
    : public mlir::OpRewritePattern<CrossEntropyWithLogitsOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(CrossEntropyWithLogitsOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value logits = op.getLogits();
    mlir::Value targets = op.getTargets();
    auto logitsTy =
        mlir::cast<mlir::RankedTensorType>(logits.getType());
    auto targetsTy =
        mlir::cast<mlir::RankedTensorType>(targets.getType());
    auto resultTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    auto elemTy =
        mlir::dyn_cast<mlir::FloatType>(logitsTy.getElementType());
    if (!elemTy || logitsTy.getRank() != 2 || targetsTy.getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "only rank-2 float logits / rank-1 int targets are lowered");

    const int64_t N = logitsTy.getShape()[0];
    auto *ctx = rewriter.getContext();
    auto rowTy = mlir::RankedTensorType::get({N}, elemTy);

    // row_max[n] = max_c logits[n, c], initialized to -inf.
    llvm::APFloat negInf =
        llvm::APFloat::getInf(elemTy.getFloatSemantics(), /*Negative=*/true);
    mlir::Value maxInit =
        createFloatFilled(rewriter, loc, rowTy,
                          rewriter.getFloatAttr(elemTy, negInf));
    mlir::Value rowMax = createRowReduce(
        rewriter, loc, logits, maxInit,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::Value v, mlir::Value acc) {
          return b.create<mlir::arith::MaximumFOp>(l, v, acc).getResult();
        });

    // sum_exp[n] = sum_c exp(logits[n, c] - row_max[n]).
    mlir::Value sumInit =
        createFloatFilled(rewriter, loc, rowTy,
                          rewriter.getFloatAttr(elemTy, 0.0));
    auto logitsMap = mlir::AffineMap::getMultiDimIdentityMap(2, ctx);
    auto rowMap = mlir::AffineMap::get(
        2, 0, {mlir::getAffineDimExpr(0, ctx)}, ctx);
    llvm::SmallVector<mlir::utils::IteratorType, 2> sumIters{
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::reduction};
    auto sumExpGen = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{rowTy},
        mlir::ValueRange{logits, rowMax}, mlir::ValueRange{sumInit},
        llvm::ArrayRef<mlir::AffineMap>{logitsMap, rowMap, rowMap}, sumIters,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          // args = {logits, row_max, sum_acc}
          mlir::Value d = b.create<mlir::arith::SubFOp>(l, args[0], args[1]);
          mlir::Value e = b.create<mlir::math::ExpOp>(l, d);
          mlir::Value a = b.create<mlir::arith::AddFOp>(l, e, args[2]);
          b.create<mlir::linalg::YieldOp>(l, a);
        });
    mlir::Value sumExp = sumExpGen.getResult(0);

    // lse[n] = log(sum_exp[n]) + row_max[n]
    mlir::Value lse = createElementwiseGeneric(
        rewriter, loc, mlir::ValueRange{sumExp, rowMax}, rowTy,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::Value logv = b.create<mlir::math::LogOp>(l, args[0]);
          return b.create<mlir::arith::AddFOp>(l, logv, args[1]).getResult();
        });

    // selected[n] = sum_c onehot(targets[n])[c] * logits[n, c]
    //             = logits[n, targets[n]]
    mlir::Value selInit =
        createFloatFilled(rewriter, loc, rowTy,
                          rewriter.getFloatAttr(elemTy, 0.0));
    auto targetsMap = mlir::AffineMap::get(
        2, 0, {mlir::getAffineDimExpr(0, ctx)}, ctx);
    auto selGen = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{rowTy},
        mlir::ValueRange{logits, targets}, mlir::ValueRange{selInit},
        llvm::ArrayRef<mlir::AffineMap>{logitsMap, targetsMap, rowMap}, sumIters,
        [elemTy](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          // args = {logits_elem (f32), target_elem (i64), acc (f32)}
          mlir::Value cIdx = b.create<mlir::linalg::IndexOp>(l, /*dim=*/1);
          mlir::Value cI64 = b.create<mlir::arith::IndexCastOp>(
              l, b.getI64Type(), cIdx);
          mlir::Value mask = b.create<mlir::arith::CmpIOp>(
              l, mlir::arith::CmpIPredicate::eq, args[1], cI64);
          mlir::Value zero = b.create<mlir::arith::ConstantOp>(
              l, b.getFloatAttr(elemTy, 0.0));
          mlir::Value contrib =
              b.create<mlir::arith::SelectOp>(l, mask, args[0], zero);
          mlir::Value acc =
              b.create<mlir::arith::AddFOp>(l, contrib, args[2]);
          b.create<mlir::linalg::YieldOp>(l, acc);
        });
    mlir::Value selected = selGen.getResult(0);

    // per_row[n] = lse[n] - selected[n]
    mlir::Value perRow = createElementwiseGeneric(
        rewriter, loc, mlir::ValueRange{lse, selected}, rowTy,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          return b.create<mlir::arith::SubFOp>(l, args[0], args[1]).getResult();
        });

    // total[] = sum_n per_row[n]
    auto scalarTy = mlir::RankedTensorType::get({}, elemTy);
    mlir::Value totalInit =
        createFloatFilled(rewriter, loc, scalarTy,
                          rewriter.getFloatAttr(elemTy, 0.0));
    auto totalReduce = rewriter.create<mlir::linalg::ReduceOp>(
        loc, mlir::ValueRange{perRow}, mlir::ValueRange{totalInit},
        llvm::ArrayRef<int64_t>{0},
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::Value s = b.create<mlir::arith::AddFOp>(l, args[0], args[1]);
          b.create<mlir::linalg::YieldOp>(l, s);
        });
    mlir::Value total = totalReduce.getResult(0);

    // loss[] = total / N
    mlir::Value loss = createElementwiseGeneric(
        rewriter, loc, mlir::ValueRange{total}, scalarTy,
        [elemTy, N](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::Value n = b.create<mlir::arith::ConstantOp>(
              l, b.getFloatAttr(elemTy, static_cast<double>(N)));
          return b.create<mlir::arith::DivFOp>(l, args[0], n).getResult();
        });

    // `resultTy` is always the 0-D float tensor; no reshape needed.
    (void)resultTy;
    rewriter.replaceOp(op, loss);
    return mlir::success();
  }
};

struct CrossEntropyWithLogitsBackwardLowering
    : public mlir::OpRewritePattern<CrossEntropyWithLogitsBackwardOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(CrossEntropyWithLogitsBackwardOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value logits = op.getLogits();
    mlir::Value targets = op.getTargets();
    mlir::Value grad = op.getGrad();
    auto logitsTy =
        mlir::cast<mlir::RankedTensorType>(logits.getType());
    auto targetsTy =
        mlir::cast<mlir::RankedTensorType>(targets.getType());
    auto gradTy =
        mlir::cast<mlir::RankedTensorType>(grad.getType());
    auto elemTy =
        mlir::dyn_cast<mlir::FloatType>(logitsTy.getElementType());
    if (!elemTy || logitsTy.getRank() != 2 || targetsTy.getRank() != 1 ||
        gradTy.getRank() != 0)
      return rewriter.notifyMatchFailure(
          op, "only rank-2 float logits / rank-1 targets / scalar grad is "
              "lowered");

    const int64_t N = logitsTy.getShape()[0];
    auto *ctx = rewriter.getContext();
    auto rowTy = mlir::RankedTensorType::get({N}, elemTy);

    // row_max[n]
    llvm::APFloat negInf =
        llvm::APFloat::getInf(elemTy.getFloatSemantics(), /*Negative=*/true);
    mlir::Value maxInit =
        createFloatFilled(rewriter, loc, rowTy,
                          rewriter.getFloatAttr(elemTy, negInf));
    mlir::Value rowMax = createRowReduce(
        rewriter, loc, logits, maxInit,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::Value v, mlir::Value acc) {
          return b.create<mlir::arith::MaximumFOp>(l, v, acc).getResult();
        });

    // sum_exp[n] = sum_c exp(logits[n, c] - row_max[n])
    mlir::Value sumInit =
        createFloatFilled(rewriter, loc, rowTy,
                          rewriter.getFloatAttr(elemTy, 0.0));
    auto logitsMap = mlir::AffineMap::getMultiDimIdentityMap(2, ctx);
    auto rowMap = mlir::AffineMap::get(
        2, 0, {mlir::getAffineDimExpr(0, ctx)}, ctx);
    llvm::SmallVector<mlir::utils::IteratorType, 2> sumIters{
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::reduction};
    auto sumExpGen = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{rowTy},
        mlir::ValueRange{logits, rowMax}, mlir::ValueRange{sumInit},
        llvm::ArrayRef<mlir::AffineMap>{logitsMap, rowMap, rowMap}, sumIters,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::Value d = b.create<mlir::arith::SubFOp>(l, args[0], args[1]);
          mlir::Value e = b.create<mlir::math::ExpOp>(l, d);
          mlir::Value a = b.create<mlir::arith::AddFOp>(l, e, args[2]);
          b.create<mlir::linalg::YieldOp>(l, a);
        });
    mlir::Value sumExp = sumExpGen.getResult(0);

    // d_logits[n, c] = (exp(logits - row_max) / sum_exp - onehot) * (grad / N)
    // Fused into a single linalg.generic over [N, C] that reads the
    // scalar `grad` + the per-row row_max / sum_exp / targets and the
    // full logits tensor. Targets are accessed via (d0, d1) -> (d0).
    auto scalarMap = mlir::AffineMap::get(2, 0, {}, ctx);
    auto targetsMap = mlir::AffineMap::get(
        2, 0, {mlir::getAffineDimExpr(0, ctx)}, ctx);
    mlir::Value dLogitsInit = createEmptyLike(rewriter, loc, logitsTy);
    llvm::SmallVector<mlir::utils::IteratorType, 2> parIters(
        2, mlir::utils::IteratorType::parallel);
    auto dLogitsGen = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{logitsTy},
        mlir::ValueRange{logits, rowMax, sumExp, targets, grad},
        mlir::ValueRange{dLogitsInit},
        llvm::ArrayRef<mlir::AffineMap>{logitsMap, rowMap, rowMap, targetsMap,
                                         scalarMap, logitsMap},
        parIters,
        [elemTy, N](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          // args = {logit, row_max, sum_exp, target_i64, grad_scalar, init}
          mlir::Value shifted =
              b.create<mlir::arith::SubFOp>(l, args[0], args[1]);
          mlir::Value e = b.create<mlir::math::ExpOp>(l, shifted);
          mlir::Value softmax =
              b.create<mlir::arith::DivFOp>(l, e, args[2]);

          mlir::Value cIdx = b.create<mlir::linalg::IndexOp>(l, /*dim=*/1);
          mlir::Value cI64 = b.create<mlir::arith::IndexCastOp>(
              l, b.getI64Type(), cIdx);
          mlir::Value mask = b.create<mlir::arith::CmpIOp>(
              l, mlir::arith::CmpIPredicate::eq, args[3], cI64);
          mlir::Value one = b.create<mlir::arith::ConstantOp>(
              l, b.getFloatAttr(elemTy, 1.0));
          mlir::Value zero = b.create<mlir::arith::ConstantOp>(
              l, b.getFloatAttr(elemTy, 0.0));
          mlir::Value onehot =
              b.create<mlir::arith::SelectOp>(l, mask, one, zero);
          mlir::Value diff =
              b.create<mlir::arith::SubFOp>(l, softmax, onehot);

          mlir::Value invN = b.create<mlir::arith::ConstantOp>(
              l, b.getFloatAttr(elemTy, 1.0 / static_cast<double>(N)));
          mlir::Value scale =
              b.create<mlir::arith::MulFOp>(l, args[4], invN);
          mlir::Value scaled =
              b.create<mlir::arith::MulFOp>(l, diff, scale);
          b.create<mlir::linalg::YieldOp>(l, scaled);
        });

    rewriter.replaceOp(op, dLogitsGen.getResults());
    return mlir::success();
  }
};

// ---------------- Softmax ----------------
//
// softmax(x, dim) = exp(x - max_dim x) / sum_dim exp(x - max_dim x), the
// numerically stable form. Decomposes into reduce-max over `dim` → subtract
// (broadcasting the reduced row stat back over `dim`) → exp → reduce-sum over
// `dim` → divide. This is the one primitive a captured single-head attention
// (matmul → scale → mask → softmax → matmul) still needed to run through the
// CPU JIT (M4 Track C1 tail / B-044): every other op already lowers.
struct SoftmaxLowering : public mlir::OpRewritePattern<SoftmaxOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(SoftmaxOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ty = mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto elemTy = mlir::dyn_cast<mlir::FloatType>(ty.getElementType());
    if (!elemTy)
      return rewriter.notifyMatchFailure(op, "only float softmax is lowered");
    const int64_t R = ty.getRank();
    if (R == 0)
      return rewriter.notifyMatchFailure(op, "scalar softmax has no axis");
    int64_t d = op.getDim();
    if (d < 0) d += R;
    if (d < 0 || d >= R)
      return rewriter.notifyMatchFailure(op, "softmax dim out of range");
    for (int64_t i = 0; i < R; ++i)
      if (ty.isDynamicDim(i))
        return rewriter.notifyMatchFailure(op, "dynamic softmax not lowered");

    mlir::Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    mlir::Value input = op.getInput();

    // Reduced (rank R-1) tensor: input shape with axis `d` removed.
    llvm::SmallVector<int64_t, 4> redShape;
    for (int64_t i = 0; i < R; ++i)
      if (i != d) redShape.push_back(ty.getShape()[i]);
    auto redTy = mlir::RankedTensorType::get(redShape, elemTy);

    // Affine maps for the two broadcast generics: the full operand is
    // identity over R dims; the reduced operand drops dim `d`.
    auto idMap = mlir::AffineMap::getMultiDimIdentityMap(R, ctx);
    llvm::SmallVector<mlir::AffineExpr, 4> redExprs;
    for (int64_t i = 0; i < R; ++i)
      if (i != d) redExprs.push_back(rewriter.getAffineDimExpr(i));
    auto redMap = mlir::AffineMap::get(R, 0, redExprs, ctx);
    llvm::SmallVector<mlir::utils::IteratorType, 4> parIters(
        R, mlir::utils::IteratorType::parallel);

    // rowMax = max over d (init -inf).
    llvm::APFloat negInf =
        llvm::APFloat::getInf(elemTy.getFloatSemantics(), /*Negative=*/true);
    mlir::Value maxInit = createFloatFilled(
        rewriter, loc, redTy, rewriter.getFloatAttr(elemTy, negInf));
    auto maxRed = rewriter.create<mlir::linalg::ReduceOp>(
        loc, mlir::ValueRange{input}, mlir::ValueRange{maxInit},
        llvm::ArrayRef<int64_t>{d},
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange a) {
          b.create<mlir::linalg::YieldOp>(
              l, b.create<mlir::arith::MaximumFOp>(l, a[0], a[1]).getResult());
        });
    mlir::Value rowMax = maxRed.getResult(0);

    // expv = exp(input - broadcast(rowMax)).
    mlir::Value expInit = createEmptyLike(rewriter, loc, ty);
    auto expGen = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{ty}, mlir::ValueRange{input, rowMax},
        mlir::ValueRange{expInit},
        llvm::ArrayRef<mlir::AffineMap>{idMap, redMap, idMap}, parIters,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange a) {
          mlir::Value s = b.create<mlir::arith::SubFOp>(l, a[0], a[1]);
          b.create<mlir::linalg::YieldOp>(
              l, b.create<mlir::math::ExpOp>(l, s).getResult());
        });
    mlir::Value expv = expGen.getResult(0);

    // sumExp = sum over d.
    mlir::Value sumInit = createFloatFilled(
        rewriter, loc, redTy, rewriter.getFloatAttr(elemTy, 0.0));
    auto sumRed = rewriter.create<mlir::linalg::ReduceOp>(
        loc, mlir::ValueRange{expv}, mlir::ValueRange{sumInit},
        llvm::ArrayRef<int64_t>{d},
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange a) {
          b.create<mlir::linalg::YieldOp>(
              l, b.create<mlir::arith::AddFOp>(l, a[0], a[1]).getResult());
        });
    mlir::Value sumExp = sumRed.getResult(0);

    // out = expv / broadcast(sumExp).
    mlir::Value outInit = createEmptyLike(rewriter, loc, ty);
    auto outGen = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{ty}, mlir::ValueRange{expv, sumExp},
        mlir::ValueRange{outInit},
        llvm::ArrayRef<mlir::AffineMap>{idMap, redMap, idMap}, parIters,
        [](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange a) {
          b.create<mlir::linalg::YieldOp>(
              l, b.create<mlir::arith::DivFOp>(l, a[0], a[1]).getResult());
        });
    rewriter.replaceOp(op, outGen.getResult(0));
    return mlir::success();
  }
};

// ---------------- Rotary position embedding ----------------
//
// RoPE rotates each adjacent pair (x[..,2j], x[..,2j+1]) of the last dim by the
// position angle stored in cos/sin. Per element c this is
//   out[c] = x[c]·cos[p,c] + rot[c]·sin[p,c]
// where rot[2j] = -x[2j+1] and rot[2j+1] = x[2j] ("rotate_half" on interleaved
// pairs). We emit a single all-parallel `linalg.generic` over the output: the
// diagonal x[c] and the cos/sin tables come in through affine maps (cos/sin's
// [S,D] rows broadcast over the leading batch dims), while the cross-lane
// partner x[c^1] is fetched in-body via `tensor.extract` at the index computed
// from `linalg.index`. This stays copy-free (no strided insert_slice → no
// `memrefCopy` runtime dependency) and is numerically identical to the eager
// kernel.
struct RotaryEmbeddingLowering
    : public mlir::OpRewritePattern<RotaryEmbeddingOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(RotaryEmbeddingOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto xTy = mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto cosTy = mlir::cast<mlir::RankedTensorType>(op.getCos().getType());
    auto sinTy = mlir::cast<mlir::RankedTensorType>(op.getSin().getType());
    auto elemTy = mlir::dyn_cast<mlir::FloatType>(xTy.getElementType());
    if (!elemTy)
      return rewriter.notifyMatchFailure(op, "only float RoPE is lowered");
    const int64_t R = xTy.getRank();
    if (R < 2)
      return rewriter.notifyMatchFailure(op, "RoPE needs rank >= 2");
    if (!xTy.hasStaticShape() || !cosTy.hasStaticShape() ||
        !sinTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic RoPE not lowered");
    if (cosTy.getRank() != 2 || sinTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "cos/sin must be rank-2 tables");
    const int64_t D = xTy.getShape()[R - 1];
    const int64_t S = xTy.getShape()[R - 2];
    if (D % 2 != 0)
      return rewriter.notifyMatchFailure(op, "RoPE last dim must be even");

    mlir::Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    mlir::Value x = op.getInput();

    auto idxA = [&](int64_t v) { return rewriter.getIndexAttr(v); };

    // First S rows of the cos/sin tables: [S, D] (subview, folds away when the
    // table already has exactly S rows).
    auto rowTy = mlir::RankedTensorType::get({S, D}, elemTy);
    llvm::SmallVector<mlir::OpFoldResult, 2> tblOff{idxA(0), idxA(0)};
    llvm::SmallVector<mlir::OpFoldResult, 2> tblSize{idxA(S), idxA(D)};
    llvm::SmallVector<mlir::OpFoldResult, 2> tblStride{idxA(1), idxA(1)};
    mlir::Value cosS = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, rowTy, op.getCos(), tblOff, tblSize, tblStride);
    mlir::Value sinS = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, rowTy, op.getSin(), tblOff, tblSize, tblStride);

    // out[idx] = x[idx]*cos[p,c] + rot*sin[p,c], rot = ±x[..,c^1].
    auto idMap = mlir::AffineMap::getMultiDimIdentityMap(R, ctx);
    llvm::SmallVector<mlir::AffineExpr, 2> last2{
        rewriter.getAffineDimExpr(R - 2), rewriter.getAffineDimExpr(R - 1)};
    auto tblMap = mlir::AffineMap::get(R, 0, last2, ctx);
    llvm::SmallVector<mlir::utils::IteratorType, 4> parR(
        R, mlir::utils::IteratorType::parallel);
    mlir::Value outInit = createEmptyLike(rewriter, loc, xTy);
    auto comb = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{xTy}, mlir::ValueRange{x, cosS, sinS},
        mlir::ValueRange{outInit},
        llvm::ArrayRef<mlir::AffineMap>{idMap, tblMap, tblMap, idMap}, parR,
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange a) {
          // Current multi-index, then the partner index along the last dim.
          llvm::SmallVector<mlir::Value, 4> idxs;
          idxs.reserve(static_cast<std::size_t>(R));
          for (int64_t d = 0; d < R; ++d)
            idxs.push_back(b.create<mlir::linalg::IndexOp>(l, d));
          mlir::Value last = idxs[static_cast<std::size_t>(R - 1)];
          mlir::Value one = b.create<mlir::arith::ConstantIndexOp>(l, 1);
          mlir::Value two = b.create<mlir::arith::ConstantIndexOp>(l, 2);
          mlir::Value zero = b.create<mlir::arith::ConstantIndexOp>(l, 0);
          mlir::Value rem = b.create<mlir::arith::RemUIOp>(l, last, two);
          mlir::Value isOdd = b.create<mlir::arith::CmpIOp>(
              l, mlir::arith::CmpIPredicate::ne, rem, zero);
          mlir::Value partner = b.create<mlir::arith::SelectOp>(
              l, isOdd, b.create<mlir::arith::SubIOp>(l, last, one),
              b.create<mlir::arith::AddIOp>(l, last, one));
          llvm::SmallVector<mlir::Value, 4> pidx(idxs.begin(), idxs.end());
          pidx[static_cast<std::size_t>(R - 1)] = partner;
          mlir::Value xp = b.create<mlir::tensor::ExtractOp>(l, x, pidx);
          // rot = isOdd ? +x[partner] : -x[partner].
          mlir::Value rot = b.create<mlir::arith::SelectOp>(
              l, isOdd, xp, b.create<mlir::arith::NegFOp>(l, xp).getResult());
          mlir::Value xc = b.create<mlir::arith::MulFOp>(l, a[0], a[1]);
          mlir::Value rs = b.create<mlir::arith::MulFOp>(l, rot, a[2]);
          b.create<mlir::linalg::YieldOp>(
              l, b.create<mlir::arith::AddFOp>(l, xc, rs).getResult());
        });
    rewriter.replaceOp(op, comb.getResult(0));
    return mlir::success();
  }
};

// ---------------- Pass ----------------

struct ConvertTesseractToLinalgPass
    : public mlir::PassWrapper<ConvertTesseractToLinalgPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTesseractToLinalgPass)

  llvm::StringRef getArgument() const final {
    return "convert-tesseract-to-linalg";
  }
  llvm::StringRef getDescription() const final {
    return "Lower a subset of the tesseract dialect (add/sub/mul/div, neg, "
           "relu, relu_backward, sigmoid, exp, log, sqrt, tanh, softmax, "
           "rotary_embedding, matmul, sum, mean, transpose, broadcast_to, "
           "cross_entropy_with_logits + its fused backward) to linalg + "
           "tensor + arith + math.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
        BinaryElementwiseLowering<AddOp, mlir::linalg::AddOp>,
        BinaryElementwiseLowering<SubOp, mlir::linalg::SubOp>,
        BinaryElementwiseLowering<MulOp, mlir::linalg::MulOp>,
        BinaryElementwiseLowering<DivOp, mlir::linalg::DivOp>,
        MatMulLowering, SumLowering, MeanLowering, NegLowering, ReluLowering,
        ReluBackwardLowering, TransposeLowering, PermuteLowering,
        ReshapeViewLowering<ViewOp>, ReshapeViewLowering<ReshapeOp>,
        BroadcastToLowering,
        ContiguousLowering, CloneLowering,
        SigmoidLowering,
        UnaryMathLowering<ExpOp, mlir::math::ExpOp>,
        UnaryMathLowering<LogOp, mlir::math::LogOp>,
        UnaryMathLowering<SqrtOp, mlir::math::SqrtOp>,
        UnaryMathLowering<TanhOp, mlir::math::TanhOp>,
        SoftmaxLowering,
        RotaryEmbeddingLowering,
        CrossEntropyWithLogitsLowering,
        CrossEntropyWithLogitsBackwardLowering>(&getContext());

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createConvertTesseractToLinalgPass() {
  return std::make_unique<ConvertTesseractToLinalgPass>();
}

void registerConvertTesseractToLinalgPass() {
  mlir::PassRegistration<ConvertTesseractToLinalgPass>();
}

void registerTesseractPasses() {
  registerConvertTesseractToLinalgPass();
  registerBackwardPass();
  registerInterchangeMatmulPass();
  registerConvertTesseractToGpuPipeline();
}

}  // namespace tesseract::ir

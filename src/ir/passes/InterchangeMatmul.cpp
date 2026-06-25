// M1I.2.c / B-007 Phase-2 — linalg.matmul loop-order rewrite.
//
// `--convert-linalg-to-loops` emits one `scf.for` per dimension of the
// linalg op's iteration space, in the order those dimensions appear
// in `iterator_types`. `linalg.matmul`'s default iteration space is
// `(m, n, k)` with types `[parallel, parallel, reduction]`, so the
// generated loop nest is
//
//     for m { for n { for k { C[m,n] += A[m,k] * B[k,n] } } }  // M-N-K
//
// with the reduction dim innermost. For a row-major `B[k,n]`, the
// innermost iteration strides by `N` elements through `B`, which
// blows every L1 cache line after the first iteration and is why
// `bench_graph_vs_eager --engine mlir` was ~2× slower than eager on
// the 512×512 `wide` config (see docs/backlog.md #B-007).
//
// Permuting the iteration space to `(m, k, n)` — i.e. moving the
// reduction to the middle and the second parallel dim innermost —
// produces
//
//     for m { for k { for n { C[m,n] += A[m,k] * B[k,n] } } }  // M-K-N
//
// where the innermost loop walks `C[m, *]`, `B[k, *]` *and* a loop-
// invariant `A[m, k]`, all contiguous / constant in `n`. That layout
// is what every hand-written row-major matmul uses, it trivially
// auto-vectorizes via LLVM's LoopVectorize, and it composes with the
// AVX/AVX-512 SIMD emission enabled by the host `TargetMachine`
// rewire landed alongside this pass.
//
// Implementation strategy (kept deliberately minimal):
//   1. Walk every `linalg::MatmulOp` in the function.
//   2. `generalizeNamedOp(matmul)` → named op becomes a `linalg.generic`
//      whose `indexing_maps` / `iterator_types` we can freely permute.
//   3. `interchangeGenericOp(generic, [0, 2, 1])` → swap the `n` and `k`
//      axes in the iteration domain, giving us the M-K-N order above.
//   4. Leave everything else untouched; the later
//      `--convert-linalg-to-loops` pass expands this generic into the
//      desired scf nest as usual.
//
// Why not also tile/vectorize here? The host-TM + loop-order fix
// already recovers most of the matmul gap on MLP-class shapes. Full
// `linalg::vectorize` + `vector.contract` lowering is the real
// follow-up (second half of B-007) but needs a vector-dialect
// lowering pipeline and is a much larger change. This pass is the
// cheap intermediate that unblocks LoopVectorize.

#include "tesseract/ir/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace tesseract::ir {

namespace {

class InterchangeMatmulPass
    : public mlir::PassWrapper<InterchangeMatmulPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InterchangeMatmulPass)

  llvm::StringRef getArgument() const final {
    return "tesseract-interchange-matmul";
  }

  llvm::StringRef getDescription() const final {
    return "Permute linalg.matmul iteration order from (m, n, k) to "
           "(m, k, n) so the innermost loop is contiguous in both "
           "operands and output (unblocks LLVM LoopVectorize).";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    mlir::func::FuncOp fn = getOperation();
    mlir::IRRewriter rewriter(fn.getContext());

    // Collect up front — generalize/interchange replaces ops and we
    // don't want to mutate the walker's traversal underneath it.
    llvm::SmallVector<mlir::linalg::MatmulOp, 4> matmuls;
    fn.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

    for (mlir::linalg::MatmulOp m : matmuls) {
      rewriter.setInsertionPoint(m);
      auto generic = mlir::linalg::generalizeNamedOp(
          rewriter, mlir::cast<mlir::linalg::LinalgOp>(m.getOperation()));
      if (mlir::failed(generic)) {
        // `generalizeNamedOp` only fails on ops that lack a region
        // builder, which doesn't apply to `linalg.matmul`. Tolerate
        // the failure anyway so one odd op can't sink the whole pass.
        continue;
      }

      // Iteration domain is (m, n, k) = (0, 1, 2); target (m, k, n)
      // means the result's dim i should read the original's dim
      // `perm[i]`, so perm = [0, 2, 1].
      const llvm::SmallVector<unsigned, 3> perm = {0, 2, 1};
      auto interchanged =
          mlir::linalg::interchangeGenericOp(rewriter, *generic, perm);
      if (mlir::failed(interchanged)) {
        // Unexpected: generalize succeeded, so the domain size is 3.
        // Bail gracefully to keep the rest of the function alive; the
        // original (now-generalized) op stays and lowers via the
        // default M-N-K path.
        continue;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createInterchangeMatmulPass() {
  return std::make_unique<InterchangeMatmulPass>();
}

void registerInterchangeMatmulPass() {
  mlir::PassRegistration<InterchangeMatmulPass>();
}

}  // namespace tesseract::ir

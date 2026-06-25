// RUN: tesseract-opt %s --tesseract-interchange-matmul --verify-each | FileCheck %s
//
// Pins the behaviour of the M1I.2.c / B-007 pass
// `--tesseract-interchange-matmul`. The pass generalizes every
// `linalg.matmul` to a `linalg.generic` and then interchanges its
// iteration space from the default `(m, n, k)` (K innermost, stride-N
// access on row-major RHS) to `(m, k, n)` (N innermost, contiguous
// access on RHS + output). The resulting loop nest is what LLVM
// LoopVectorize knows how to turn into AVX / AVX-512 code.
//
// Three things we want to hold across refactors:
//   1. No `linalg.matmul` survives the pass.
//   2. The replacement is a `linalg.generic` whose `iterator_types`
//      list the reduction axis in the MIDDLE position — i.e. the
//      interchanged `[parallel, reduction, parallel]` rather than the
//      original `[parallel, parallel, reduction]`.
//   3. The body still implements the standard multiply-accumulate
//      (`arith.mulf` + `arith.addf`) so numeric semantics are
//      preserved — the pass is a pure re-ordering.
//
// CHECK-LABEL: func @matmul_gets_interchanged
func.func @matmul_gets_interchanged(%a: tensor<4x8xf32>,
                                     %b: tensor<8x3xf32>) -> tensor<4x3xf32> {
  // Standard pre-lowering shape: an empty output, zero-filled, then
  // `linalg.matmul` accumulating into it — exactly what
  // `--convert-tesseract-to-linalg` produces for `tesseract.matmul`.
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4x3xf32>
  %zero = linalg.fill ins(%cst : f32) outs(%init : tensor<4x3xf32>)
      -> tensor<4x3xf32>
  // CHECK-NOT: linalg.matmul
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: linalg.yield
  %0 = linalg.matmul
      ins(%a, %b : tensor<4x8xf32>, tensor<8x3xf32>)
      outs(%zero : tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

// CHECK-LABEL: func @non_matmul_is_untouched
func.func @non_matmul_is_untouched(%a: tensor<4x3xf32>,
                                    %b: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // The pass is scoped to `linalg.matmul` specifically — other
  // `linalg.*` ops are left alone so the pipeline stays additive.
  // Any future widening should update this test accordingly.
  %init = tensor.empty() : tensor<4x3xf32>
  // CHECK: linalg.add
  // CHECK-NOT: iterator_types = ["parallel", "reduction", "parallel"]
  %0 = linalg.add ins(%a, %b : tensor<4x3xf32>, tensor<4x3xf32>)
                  outs(%init : tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

// CHECK-LABEL: func @two_matmuls_in_a_row
func.func @two_matmuls_in_a_row(%a: tensor<4x8xf32>,
                                 %b: tensor<8x6xf32>,
                                 %c: tensor<6x3xf32>) -> tensor<4x3xf32> {
  // Multiple matmuls in one function should all interchange — pins
  // the walk in `InterchangeMatmulPass::runOnOperation`.
  %cst = arith.constant 0.0 : f32
  %init_ab = tensor.empty() : tensor<4x6xf32>
  %zero_ab = linalg.fill ins(%cst : f32) outs(%init_ab : tensor<4x6xf32>)
      -> tensor<4x6xf32>
  %ab = linalg.matmul
      ins(%a, %b : tensor<4x8xf32>, tensor<8x6xf32>)
      outs(%zero_ab : tensor<4x6xf32>) -> tensor<4x6xf32>
  %init = tensor.empty() : tensor<4x3xf32>
  %zero = linalg.fill ins(%cst : f32) outs(%init : tensor<4x3xf32>)
      -> tensor<4x3xf32>
  %abc = linalg.matmul
      ins(%ab, %c : tensor<4x6xf32>, tensor<6x3xf32>)
      outs(%zero : tensor<4x3xf32>) -> tensor<4x3xf32>
  // CHECK-NOT: linalg.matmul
  // CHECK-COUNT-2: iterator_types = ["parallel", "reduction", "parallel"]
  return %abc : tensor<4x3xf32>
}

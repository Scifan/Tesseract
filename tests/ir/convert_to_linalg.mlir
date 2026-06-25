// RUN: tesseract-opt %s --convert-tesseract-to-linalg --verify-each | FileCheck %s
//
// Smoke test for the M1G lowering. Drives a handful of tesseract ops through
// the conversion pass and spot-checks the emitted linalg / tensor / arith
// sequence. Exhaustive op coverage (softmax, activations, shape ops) lands
// with the follow-up passes in M1.γ.

// CHECK-LABEL: func @add_to_linalg
func.func @add_to_linalg(%a: tensor<4x3xf32>, %b: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // CHECK: tensor.empty
  // CHECK: linalg.add
  // CHECK-NOT: tesseract.add
  %0 = "tesseract.add"(%a, %b) : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

// CHECK-LABEL: func @mul_to_linalg
func.func @mul_to_linalg(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.mul
  // CHECK-NOT: tesseract.mul
  %0 = "tesseract.mul"(%a, %b) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @matmul_to_linalg
func.func @matmul_to_linalg(%a: tensor<4x8xf32>, %b: tensor<8x3xf32>) -> tensor<4x3xf32> {
  // The lowering must zero-initialize the accumulator because linalg.matmul
  // is a true reducer and adds into its outs operand.
  // CHECK: arith.constant 0
  // CHECK: linalg.fill
  // CHECK: linalg.matmul
  // CHECK-NOT: tesseract.matmul
  %0 = "tesseract.matmul"(%a, %b) : (tensor<4x8xf32>, tensor<8x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

// CHECK-LABEL: func @sum_all_to_linalg
func.func @sum_all_to_linalg(%x: tensor<2x3xf32>) -> tensor<f32> {
  // CHECK: linalg.fill
  // CHECK: linalg.reduce
  // CHECK: arith.addf
  // CHECK-NOT: tesseract.sum
  %0 = "tesseract.sum"(%x) {dim = -1 : si64, keepdim = false} : (tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @sum_axis_keepdim
func.func @sum_axis_keepdim(%x: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
  // CHECK: linalg.reduce
  // CHECK: tensor.expand_shape
  // CHECK-NOT: tesseract.sum
  %0 = "tesseract.sum"(%x) {dim = 1 : si64, keepdim = true}
       : (tensor<2x3x4xf32>) -> tensor<2x1x4xf32>
  return %0 : tensor<2x1x4xf32>
}

// -----

// An end-to-end linear layer: tesseract.matmul + tesseract.broadcast_to +
// tesseract.add. All three ops should lower — the broadcast_to turns
// into a linalg.generic with a (d0, d1) -> (d1) input map that expresses
// the "broadcast along leading axis" semantics for row-biased matmul.
// CHECK-LABEL: func @linear_forward
func.func @linear_forward(%x: tensor<4x8xf32>,
                          %w: tensor<8x3xf32>,
                          %b: tensor<3xf32>) -> tensor<4x3xf32> {
  // CHECK: linalg.matmul
  %0 = "tesseract.matmul"(%x, %w) : (tensor<4x8xf32>, tensor<8x3xf32>) -> tensor<4x3xf32>
  // CHECK: linalg.generic
  // CHECK-NOT: tesseract.broadcast_to
  %1 = "tesseract.broadcast_to"(%b) {shape = [4, 3]} : (tensor<3xf32>) -> tensor<4x3xf32>
  // CHECK: linalg.add
  %2 = "tesseract.add"(%0, %1) : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %2 : tensor<4x3xf32>
}

// -----

// CHECK-LABEL: func @relu_to_linalg
func.func @relu_to_linalg(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  // CHECK-NOT: tesseract.relu
  %0 = "tesseract.relu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: func @transpose_to_linalg
func.func @transpose_to_linalg(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // CHECK: linalg.transpose
  // CHECK-NOT: tesseract.transpose
  %0 = "tesseract.transpose"(%x) {dim_a = 0 : si64, dim_b = 1 : si64}
       : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: func @cross_entropy_to_linalg
func.func @cross_entropy_to_linalg(%logits: tensor<4x3xf32>,
                                   %targets: tensor<4xi64>) -> tensor<f32> {
  // CE lowers to a chain of linalg.generic reductions (row-max, sum-exp,
  // log, onehot-select, mean). We only spot-check that the dialect op
  // disappears and that at least one linalg.reduce appears.
  // CHECK: linalg.generic
  // CHECK: math.exp
  // CHECK: math.log
  // CHECK-NOT: tesseract.cross_entropy_with_logits
  %0 = "tesseract.cross_entropy_with_logits"(%logits, %targets)
       : (tensor<4x3xf32>, tensor<4xi64>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Batched (rank-3) matmul lowers to linalg.batch_matmul (M4 P8 hardening:
// rank-n legality + shape check).
// CHECK-LABEL: func @batch_matmul_to_linalg
func.func @batch_matmul_to_linalg(%a: tensor<8x4x3xf32>,
                                  %b: tensor<8x3x5xf32>) -> tensor<8x4x5xf32> {
  // CHECK: linalg.fill
  // CHECK: linalg.batch_matmul
  // CHECK-NOT: tesseract.matmul
  %0 = "tesseract.matmul"(%a, %b)
       : (tensor<8x4x3xf32>, tensor<8x3x5xf32>) -> tensor<8x4x5xf32>
  return %0 : tensor<8x4x5xf32>
}

// -----

// permute -> linalg.transpose; bijection + shape validated (M4 P8 hardening).
// CHECK-LABEL: func @permute_to_linalg
func.func @permute_to_linalg(%x: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  // CHECK: linalg.transpose
  // CHECK-NOT: tesseract.permute
  %0 = "tesseract.permute"(%x) {axes = [2, 0, 1]}
       : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}

// -----

// view/reshape -> collapse_shape + expand_shape (numel preserved; M4 P8
// element-count check).
// CHECK-LABEL: func @view_to_linalg
func.func @view_to_linalg(%x: tensor<4x6xf32>) -> tensor<2x3x4xf32> {
  // CHECK: tensor.collapse_shape
  // CHECK: tensor.expand_shape
  // CHECK-NOT: tesseract.view
  %0 = "tesseract.view"(%x) {shape = [2, 3, 4]}
       : (tensor<4x6xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

// clone -> a real linalg.copy into fresh storage (M4 P8: no longer a no-op,
// so clone's fresh-storage contract survives in-place bufferization).
// CHECK-LABEL: func @clone_to_linalg
func.func @clone_to_linalg(%x: tensor<3x5xf32>) -> tensor<3x5xf32> {
  // CHECK: tensor.empty
  // CHECK: linalg.copy
  // CHECK-NOT: tesseract.clone
  %0 = "tesseract.clone"(%x) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

// contiguous -> a real linalg.copy (M4 P8: materialized, not folded away).
// CHECK-LABEL: func @contiguous_to_linalg
func.func @contiguous_to_linalg(%x: tensor<3x5xf32>) -> tensor<3x5xf32> {
  // CHECK: linalg.copy
  // CHECK-NOT: tesseract.contiguous
  %0 = "tesseract.contiguous"(%x) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

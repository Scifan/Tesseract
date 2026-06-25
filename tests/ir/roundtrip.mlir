// RUN: tesseract-opt %s --verify-diagnostics --verify-each | FileCheck %s
//
// Round-trip + verification smoke test for every op in the tesseract dialect.
// Parsed once, verified, and printed back. Any op whose custom assembly
// format breaks will fail the first run; any op whose verifier rejects the
// canonical input will fail the second.

// CHECK-LABEL: func @constant_roundtrip
func.func @constant_roundtrip() -> tensor<2x2xf32> {
  // CHECK: tesseract.constant
  %0 = "tesseract.constant"() {value = dense<1.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @binary_elementwise_roundtrip
func.func @binary_elementwise_roundtrip(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tesseract.add
  %0 = "tesseract.add"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK: tesseract.sub
  %1 = "tesseract.sub"(%0, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK: tesseract.mul
  %2 = "tesseract.mul"(%1, %a) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK: tesseract.div
  %3 = "tesseract.div"(%2, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @unary_elementwise_roundtrip
func.func @unary_elementwise_roundtrip(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: tesseract.neg
  %0 = "tesseract.neg"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: tesseract.relu
  %1 = "tesseract.relu"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: tesseract.sigmoid
  %2 = "tesseract.sigmoid"(%1) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: tesseract.tanh
  %3 = "tesseract.tanh"(%2) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: tesseract.exp
  %4 = "tesseract.exp"(%3) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: tesseract.log
  %5 = "tesseract.log"(%4) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %5 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: func @matmul_roundtrip
func.func @matmul_roundtrip(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // CHECK: tesseract.matmul
  %0 = "tesseract.matmul"(%a, %b) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func @reduction_roundtrip
func.func @reduction_roundtrip(%x: tensor<2x3xf32>) -> tensor<f32> {
  // CHECK: tesseract.sum
  %0 = "tesseract.sum"(%x) {dim = -1 : si64, keepdim = false} : (tensor<2x3xf32>) -> tensor<f32>
  // CHECK: tesseract.mean
  %1 = "tesseract.mean"(%x) {dim = -1 : si64, keepdim = false} : (tensor<2x3xf32>) -> tensor<f32>
  // CHECK: tesseract.max
  %2 = "tesseract.max"(%x) {dim = -1 : si64, keepdim = false} : (tensor<2x3xf32>) -> tensor<f32>
  %3 = "tesseract.add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %4 = "tesseract.add"(%3, %2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %4 : tensor<f32>
}

// -----

// CHECK-LABEL: func @dequant_matmul_roundtrip
func.func @dequant_matmul_roundtrip(%a: tensor<2x8xf32>, %wq: tensor<8x4xi8>, %s: tensor<4xf32>) -> tensor<2x4xf32> {
  // CHECK: tesseract.dequant_matmul
  // CHECK-SAME: scheme = "int8"
  %0 = "tesseract.dequant_matmul"(%a, %wq, %s) {scheme = "int8", group_size = -1 : si64}
       : (tensor<2x8xf32>, tensor<8x4xi8>, tensor<4xf32>) -> tensor<2x4xf32>
  // CHECK: tesseract.dequant_matmul
  // CHECK-SAME: scheme = "int4_group"
  %1 = "tesseract.dequant_matmul"(%a, %wq, %s) {scheme = "int4_group", group_size = 4 : si64}
       : (tensor<2x8xf32>, tensor<8x4xi8>, tensor<4xf32>) -> tensor<2x4xf32>
  // CHECK: tesseract.dequant_matmul
  // CHECK-SAME: scheme = "fp8_e4m3"
  %2 = "tesseract.dequant_matmul"(%a, %wq, %s) {scheme = "fp8_e4m3", group_size = -1 : si64}
       : (tensor<2x8xf32>, tensor<8x4xi8>, tensor<4xf32>) -> tensor<2x4xf32>
  // CHECK: tesseract.dequant_matmul
  // CHECK-SAME: scheme = "fp4"
  %3 = "tesseract.dequant_matmul"(%a, %wq, %s) {scheme = "fp4", group_size = 8 : si64}
       : (tensor<2x8xf32>, tensor<8x4xi8>, tensor<4xf32>) -> tensor<2x4xf32>
  %4 = "tesseract.add"(%0, %1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %5 = "tesseract.add"(%4, %2) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %6 = "tesseract.add"(%5, %3) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %6 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func @softmax_roundtrip
func.func @softmax_roundtrip(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: tesseract.softmax
  %0 = "tesseract.softmax"(%x) {dim = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: tesseract.log_softmax
  %1 = "tesseract.log_softmax"(%0) {dim = 1 : si64} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func @cross_entropy_roundtrip
func.func @cross_entropy_roundtrip(%logits: tensor<8x10xf32>, %targets: tensor<8xi64>) -> tensor<f32> {
  // CHECK: tesseract.cross_entropy_with_logits
  %0 = "tesseract.cross_entropy_with_logits"(%logits, %targets)
       : (tensor<8x10xf32>, tensor<8xi64>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @shape_ops_roundtrip
func.func @shape_ops_roundtrip(%x: tensor<6xf32>) -> tensor<3x2xf32> {
  // CHECK: tesseract.view
  %0 = "tesseract.view"(%x) {shape = [2, 3]} : (tensor<6xf32>) -> tensor<2x3xf32>
  // CHECK: tesseract.reshape
  %1 = "tesseract.reshape"(%0) {shape = [3, 2]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK: tesseract.transpose
  %2 = "tesseract.transpose"(%1) {dim_a = 0 : si64, dim_b = 1 : si64} : (tensor<3x2xf32>) -> tensor<2x3xf32>
  // CHECK: tesseract.permute
  %3 = "tesseract.permute"(%2) {axes = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK: tesseract.contiguous
  %4 = "tesseract.contiguous"(%3) : (tensor<3x2xf32>) -> tensor<3x2xf32>
  // CHECK: tesseract.clone
  %5 = "tesseract.clone"(%4) : (tensor<3x2xf32>) -> tensor<3x2xf32>
  // CHECK: tesseract.broadcast_to
  %6 = "tesseract.broadcast_to"(%5) {shape = [4, 3, 2]} : (tensor<3x2xf32>) -> tensor<4x3x2xf32>
  // Reduce back to [3,2] for the return type.
  %7 = "tesseract.sum"(%6) {dim = 0 : si64, keepdim = false} : (tensor<4x3x2xf32>) -> tensor<3x2xf32>
  return %7 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: tesseract.graph
"tesseract.graph"() ({
  // CHECK: tesseract.function
  "tesseract.function"() ({
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
    // CHECK: tesseract.param
    %w = "tesseract.param"(%arg1) {name = "weight"} : (tensor<4xf32>) -> tensor<4xf32>
    %y = "tesseract.mul"(%arg0, %w) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    // CHECK: tesseract.return
    "tesseract.return"(%y) : (tensor<4xf32>) -> ()
  }) {sym_name = "fwd", function_type = (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>} : () -> ()
}) : () -> ()

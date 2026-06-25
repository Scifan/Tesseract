// RUN: tesseract-opt %s --convert-tesseract-to-linalg --verify-each | FileCheck %s
//
// M4 Track C1 (B-044): transformer FFN + normalization primitives lower to
// linalg/arith/math. These are the ops `ops::rms_norm` (mul/mean/add/sqrt/
// div/mul) and `ops::swiglu_silu_gate` (sigmoid/mul/mul) record into the
// graph, so covering them lets a captured RMSNorm / SwiGLU-FFN execute
// through the CPU JitEngine.

// CHECK-LABEL: func @sigmoid_to_linalg
func.func @sigmoid_to_linalg(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // sigmoid(x) = 1 / (1 + exp(-x))
  // CHECK: linalg.generic
  // CHECK: arith.negf
  // CHECK: math.exp
  // CHECK: arith.addf
  // CHECK: arith.divf
  // CHECK-NOT: tesseract.sigmoid
  %0 = "tesseract.sigmoid"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func @sqrt_to_linalg
func.func @sqrt_to_linalg(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: linalg.generic
  // CHECK: math.sqrt
  // CHECK-NOT: tesseract.sqrt
  %0 = "tesseract.sqrt"(%x) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @exp_to_linalg
func.func @exp_to_linalg(%x: tensor<3xf32>) -> tensor<3xf32> {
  // CHECK: math.exp
  // CHECK-NOT: tesseract.exp
  %0 = "tesseract.exp"(%x) : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func @tanh_to_linalg
func.func @tanh_to_linalg(%x: tensor<3xf32>) -> tensor<3xf32> {
  // CHECK: math.tanh
  // CHECK-NOT: tesseract.tanh
  %0 = "tesseract.tanh"(%x) : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func @mean_keepdim_to_linalg
func.func @mean_keepdim_to_linalg(%x: tensor<2x3xf32>) -> tensor<2x1xf32> {
  // mean over the last axis: reduce(addf) then scale by 1/3, re-expand dim.
  // CHECK: linalg.reduce
  // CHECK: arith.addf
  // CHECK: arith.mulf
  // CHECK: tensor.expand_shape
  // CHECK-NOT: tesseract.mean
  %0 = "tesseract.mean"(%x) {dim = 1 : si64, keepdim = true} : (tensor<2x3xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// CHECK-LABEL: func @swiglu_decomposition_to_linalg
func.func @swiglu_decomposition_to_linalg(%gate: tensor<2x4xf32>, %up: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // silu(gate) * up = (gate * sigmoid(gate)) * up — the SwiGLU FFN activation.
  // CHECK: math.exp
  // CHECK: linalg.mul
  // CHECK-NOT: tesseract.sigmoid
  // CHECK-NOT: tesseract.mul
  %s = "tesseract.sigmoid"(%gate) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %silu = "tesseract.mul"(%gate, %s) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %out = "tesseract.mul"(%silu, %up) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %out : tensor<2x4xf32>
}

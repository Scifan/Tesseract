// RUN: tesseract-opt %s --convert-tesseract-to-linalg --verify-each | FileCheck %s
//
// M4 Track C1 tail (B-044): single-head scaled-dot-product attention lowers to
// linalg/arith/math. These are the ops a captured SDPA records — transpose,
// matmul, mul (scale), add (mask), softmax, matmul — so covering softmax (the
// one primitive C1 had not lowered) lets a full attention sub-block execute
// through the CPU JitEngine alongside RMSNorm + SwiGLU.

// CHECK-LABEL: func @softmax_to_linalg
func.func @softmax_to_linalg(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // softmax(x) = exp(x - rowmax) / rowsum: reduce-max, exp, reduce-sum, div.
  // CHECK: linalg.reduce
  // CHECK: arith.maximumf
  // CHECK: math.exp
  // CHECK: linalg.reduce
  // CHECK: arith.addf
  // CHECK: arith.divf
  // CHECK-NOT: tesseract.softmax
  %0 = "tesseract.softmax"(%x) {dim = -1 : si64} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @sdpa_to_linalg
func.func @sdpa_to_linalg(%q: tensor<4x8xf32>, %k: tensor<4x8xf32>,
                          %v: tensor<4x8xf32>, %scale: tensor<4x4xf32>,
                          %mask: tensor<4x4xf32>) -> tensor<4x8xf32> {
  // out = softmax(scale * (Q · Kᵀ) + mask) · V — the whole attention block.
  // CHECK: linalg.transpose
  // CHECK: linalg.matmul
  // CHECK: linalg.mul
  // CHECK: linalg.add
  // CHECK: math.exp
  // CHECK: arith.divf
  // CHECK: linalg.matmul
  // CHECK-NOT: tesseract.softmax
  // CHECK-NOT: tesseract.matmul
  // CHECK-NOT: tesseract.transpose
  %kt = "tesseract.transpose"(%k) {dim_a = 0 : si64, dim_b = 1 : si64} : (tensor<4x8xf32>) -> tensor<8x4xf32>
  %scores = "tesseract.matmul"(%q, %kt) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  %scaled = "tesseract.mul"(%scores, %scale) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %masked = "tesseract.add"(%scaled, %mask) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %probs = "tesseract.softmax"(%masked) {dim = -1 : si64} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %out = "tesseract.matmul"(%probs, %v) : (tensor<4x4xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}

// CHECK-LABEL: func @rope_to_linalg
func.func @rope_to_linalg(%x: tensor<2x4x8xf32>, %cos: tensor<4x8xf32>,
                          %sin: tensor<4x8xf32>) -> tensor<2x4x8xf32> {
  // out = x*cos + rotate_half(x)*sin via one broadcast multiply-add generic;
  // the cross-lane partner is read with linalg.index + tensor.extract.
  // CHECK: linalg.generic
  // CHECK: linalg.index
  // CHECK: tensor.extract
  // CHECK: arith.negf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK-NOT: tesseract.rotary_embedding
  %0 = "tesseract.rotary_embedding"(%x, %cos, %sin) : (tensor<2x4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<2x4x8xf32>
  return %0 : tensor<2x4x8xf32>
}

// CHECK-LABEL: func @multihead_to_linalg
func.func @multihead_to_linalg(%q: tensor<4x8xf32>) -> tensor<2x4x4xf32> {
  // The multi-head reshape/permute/batched-matmul path: view [S,H*Dh] ->
  // [S,H,Dh] (expand_shape), permute -> [H,S,Dh] (transpose), batched matmul
  // -> [H,S,S]. Exercises the lowerings that let a full Llama block run
  // through the CPU JIT.
  // CHECK: tensor.expand_shape
  // CHECK: linalg.transpose
  // CHECK: linalg.batch_matmul
  // CHECK-NOT: tesseract.view
  // CHECK-NOT: tesseract.permute
  // CHECK-NOT: tesseract.matmul
  %v = "tesseract.view"(%q) {shape = [4, 2, 4]} : (tensor<4x8xf32>) -> tensor<4x2x4xf32>
  %p = "tesseract.permute"(%v) {axes = [1, 0, 2]} : (tensor<4x2x4xf32>) -> tensor<2x4x4xf32>
  %pt = "tesseract.transpose"(%p) {dim_a = 1 : si64, dim_b = 2 : si64} : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  %s = "tesseract.matmul"(%p, %pt) : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %s : tensor<2x4x4xf32>
}

// RUN: tesseract-opt %s --tesseract-backward --verify-each | FileCheck %s
//
// Smoke tests for M1H — the graph-level reverse-mode AD pass. For each
// function we check the post-pass signature (new cotangent input + new
// param-gradient outputs) and a handful of structural ops that the rule
// table must emit. Checks target the dialect's pretty-printed assembly
// format (see TesseractOps.td).

// Forward: loss = sum(mul(x, w)).
// Params: [w]. After backward: signature gains one f32 cotangent and one
// w-shaped gradient.
// CHECK-LABEL: tesseract.function @elt_loss
// CHECK-SAME: (tensor<4xf32>, tensor<4xf32>, tensor<f32>) -> (tensor<f32>, tensor<4xf32>)
"tesseract.graph"() ({
  "tesseract.function"() ({
  ^bb0(%x: tensor<4xf32>, %w_in: tensor<4xf32>):
    %w = "tesseract.param"(%w_in) {name = "w"} : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: tesseract.mul
    %y   = "tesseract.mul"(%x, %w) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %sum = "tesseract.sum"(%y) {dim = -1 : si64, keepdim = false}
         : (tensor<4xf32>) -> tensor<f32>
    // Backward must broadcast the scalar cotangent back to tensor<4xf32>.
    // CHECK: tesseract.broadcast_to
    // Then multiply by the forward operands to get dL/dx and dL/dw.
    // CHECK: tesseract.mul
    // CHECK: tesseract.mul
    // The new return carries (loss, d_w).
    // CHECK: tesseract.return {{.*}} tensor<f32>, tensor<4xf32>
    "tesseract.return"(%sum) : (tensor<f32>) -> ()
  }) {sym_name = "elt_loss",
      function_type = (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>} : () -> ()
}) : () -> ()

// -----

// Forward: y = (x · W) + b, loss = sum(y). Params: [W, b].
// After backward: signature gains a scalar cotangent and two gradients
// (dW and db) in declaration order.
// CHECK-LABEL: tesseract.function @mlp_head
// CHECK-SAME: (tensor<3x4xf32>, tensor<4x2xf32>, tensor<3x2xf32>, tensor<f32>) -> (tensor<f32>, tensor<4x2xf32>, tensor<3x2xf32>)
"tesseract.graph"() ({
  "tesseract.function"() ({
  ^bb0(%x: tensor<3x4xf32>, %W_in: tensor<4x2xf32>, %b_in: tensor<3x2xf32>):
    %W = "tesseract.param"(%W_in) {name = "W"} : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %b = "tesseract.param"(%b_in) {name = "b"} : (tensor<3x2xf32>) -> tensor<3x2xf32>
    // CHECK: tesseract.matmul
    %z   = "tesseract.matmul"(%x, %W)
         : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
    // CHECK: tesseract.add
    %y   = "tesseract.add"(%z, %b)
         : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
    %loss = "tesseract.sum"(%y) {dim = -1 : si64, keepdim = false}
          : (tensor<3x2xf32>) -> tensor<f32>
    // Matmul backward must emit two transposes (one per operand) plus two
    // follow-up matmuls for dA and dB.
    // CHECK: tesseract.transpose
    // CHECK: tesseract.transpose
    // CHECK: tesseract.matmul
    // CHECK: tesseract.matmul
    // CHECK: tesseract.return {{.*}} tensor<f32>, tensor<4x2xf32>, tensor<3x2xf32>
    "tesseract.return"(%loss) : (tensor<f32>) -> ()
  }) {sym_name = "mlp_head",
      function_type = (tensor<3x4xf32>, tensor<4x2xf32>, tensor<3x2xf32>) -> tensor<f32>}
      : () -> ()
}) : () -> ()

// -----

// Forward covers the full MNIST-head shape: matmul + broadcast bias + relu
// + matmul + broadcast bias + cross-entropy. Exercises every new backward
// rule that M1I added on top of M1H.
// CHECK-LABEL: tesseract.function @mnist_head
// CHECK-SAME: (tensor<2x4xf32>, tensor<2xi64>, tensor<4x8xf32>, tensor<8xf32>, tensor<8x3xf32>, tensor<3xf32>, tensor<f32>) -> (tensor<f32>, tensor<4x8xf32>, tensor<8xf32>, tensor<8x3xf32>, tensor<3xf32>)
"tesseract.graph"() ({
  "tesseract.function"() ({
  ^bb0(%x: tensor<2x4xf32>, %y: tensor<2xi64>,
       %W1_in: tensor<4x8xf32>, %b1_in: tensor<8xf32>,
       %W2_in: tensor<8x3xf32>, %b2_in: tensor<3xf32>):
    %W1 = "tesseract.param"(%W1_in) {name = "W1"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %b1 = "tesseract.param"(%b1_in) {name = "b1"} : (tensor<8xf32>) -> tensor<8xf32>
    %W2 = "tesseract.param"(%W2_in) {name = "W2"} : (tensor<8x3xf32>) -> tensor<8x3xf32>
    %b2 = "tesseract.param"(%b2_in) {name = "b2"} : (tensor<3xf32>) -> tensor<3xf32>
    %h  = "tesseract.matmul"(%x, %W1)
        : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %bb1 = "tesseract.broadcast_to"(%b1) {shape = [2, 8]}
         : (tensor<8xf32>) -> tensor<2x8xf32>
    %h1 = "tesseract.add"(%h, %bb1)
        : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %a  = "tesseract.relu"(%h1) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %o  = "tesseract.matmul"(%a, %W2)
        : (tensor<2x8xf32>, tensor<8x3xf32>) -> tensor<2x3xf32>
    %bb2 = "tesseract.broadcast_to"(%b2) {shape = [2, 3]}
         : (tensor<3xf32>) -> tensor<2x3xf32>
    %logits = "tesseract.add"(%o, %bb2)
        : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %loss = "tesseract.cross_entropy_with_logits"(%logits, %y)
        : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<f32>
    // Cross-entropy backward becomes a single fused op.
    // CHECK: tesseract.cross_entropy_with_logits_backward
    // Walking the ops in reverse, the second layer's bias gradient is
    // materialized first (sum of d_logits along batch dim).
    // CHECK: tesseract.sum {{.*}} {dim = 0 : si64
    // Second-layer matmul backward emits two transposes + two matmuls.
    // ReLU backward then contracts the activation gradient.
    // CHECK: tesseract.relu_backward
    // After relu_backward we get the first-layer bias gradient.
    // CHECK: tesseract.sum {{.*}} {dim = 0 : si64
    // Final return: (loss, dW1, db1, dW2, db2).
    // CHECK: tesseract.return {{.*}} tensor<f32>, tensor<4x8xf32>, tensor<8xf32>, tensor<8x3xf32>, tensor<3xf32>
    "tesseract.return"(%loss) : (tensor<f32>) -> ()
  }) {sym_name = "mnist_head",
      function_type = (tensor<2x4xf32>, tensor<2xi64>, tensor<4x8xf32>,
                       tensor<8xf32>, tensor<8x3xf32>, tensor<3xf32>)
                      -> tensor<f32>}
      : () -> ()
}) : () -> ()

// -----

// Softmax backward: forward p = softmax(x ⊙ w), loss = sum(p). The backward
// rule must emit y⊙dy, a keepdim sum over the last axis, a broadcast, a sub,
// and a final multiply by the softmax output.
// CHECK-LABEL: tesseract.function @softmax_loss
"tesseract.graph"() ({
  "tesseract.function"() ({
  ^bb0(%x: tensor<3x5xf32>, %w_in: tensor<3x5xf32>):
    %w = "tesseract.param"(%w_in) {name = "w"} : (tensor<3x5xf32>) -> tensor<3x5xf32>
    %s = "tesseract.mul"(%x, %w) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
    %p = "tesseract.softmax"(%s) {dim = -1 : si64} : (tensor<3x5xf32>) -> tensor<3x5xf32>
    %loss = "tesseract.sum"(%p) {dim = -1 : si64, keepdim = false}
          : (tensor<3x5xf32>) -> tensor<f32>
    // Softmax backward structure: Σ_last(y⊙dy) with keepdim, broadcast, sub.
    // CHECK: tesseract.sum {{.*}} {dim = 1 : si64, keepdim = true
    // CHECK: tesseract.broadcast_to
    // CHECK: tesseract.sub
    // CHECK: tesseract.return {{.*}} tensor<f32>, tensor<3x5xf32>
    "tesseract.return"(%loss) : (tensor<f32>) -> ()
  }) {sym_name = "softmax_loss",
      function_type = (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<f32>}
      : () -> ()
}) : () -> ()

// -----

// RoPE backward: forward y = rotary_embedding(x, cos, sin), loss = sum(y). The
// adjoint of the orthogonal rotation is rotary_embedding with sin negated, so
// the backward must emit a neg on the sin table and a rotary_embedding on the
// cotangent. cos/sin are plain inputs (no gradient).
// CHECK-LABEL: tesseract.function @rope_loss
"tesseract.graph"() ({
  "tesseract.function"() ({
  ^bb0(%x_in: tensor<2x4x8xf32>, %cos: tensor<4x8xf32>, %sin: tensor<4x8xf32>):
    %x = "tesseract.param"(%x_in) {name = "x"} : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %y = "tesseract.rotary_embedding"(%x, %cos, %sin)
       : (tensor<2x4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<2x4x8xf32>
    %loss = "tesseract.sum"(%y) {dim = -1 : si64, keepdim = false}
          : (tensor<2x4x8xf32>) -> tensor<f32>
    // CHECK: tesseract.neg
    // CHECK: tesseract.rotary_embedding
    // CHECK: tesseract.return {{.*}} tensor<f32>, tensor<2x4x8xf32>
    "tesseract.return"(%loss) : (tensor<f32>) -> ()
  }) {sym_name = "rope_loss",
      function_type = (tensor<2x4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<f32>}
      : () -> ()
}) : () -> ()

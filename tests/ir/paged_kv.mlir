// RUN: tesseract-opt %s | tesseract-opt | FileCheck %s
//
// M4 Track C2 (B-045): round-trip (parse → print → re-parse) for the paged
// KV-cache + dynamic-shape dialect ops. No lowering — this pins the IR
// representation only. The `?` token axis exercises dynamic shapes as a
// first-class IR concept.

// CHECK-LABEL: func @paged_kv_alloc
func.func @paged_kv_alloc() -> tensor<256x16x8x64xf32> {
  // CHECK: tesseract.paged_kv_alloc
  // CHECK-SAME: num_blocks = 256
  %pool = "tesseract.paged_kv_alloc"() {
    num_blocks = 256 : si64, block_size = 16 : si64,
    num_kv_heads = 8 : si64, head_dim = 64 : si64
  } : () -> tensor<256x16x8x64xf32>
  return %pool : tensor<256x16x8x64xf32>
}

// CHECK-LABEL: func @paged_kv_append
func.func @paged_kv_append(
    %pool: tensor<256x16x8x64xf32>,
    %kv: tensor<?x8x64xf32>,
    %slots: tensor<?xi32>) -> tensor<256x16x8x64xf32> {
  // CHECK: tesseract.paged_kv_append
  %updated = "tesseract.paged_kv_append"(%pool, %kv, %slots)
      : (tensor<256x16x8x64xf32>, tensor<?x8x64xf32>, tensor<?xi32>)
        -> tensor<256x16x8x64xf32>
  return %updated : tensor<256x16x8x64xf32>
}

// CHECK-LABEL: func @paged_attention_dynamic
func.func @paged_attention_dynamic(
    %q: tensor<?x8x64xf32>,
    %kpool: tensor<256x16x8x64xf32>,
    %vpool: tensor<256x16x8x64xf32>,
    %block_table: tensor<?x?xi32>,
    %seq_lens: tensor<?xi32>) -> tensor<?x8x64xf32> {
  // The token axis (%q dim 0) is dynamic: decode handles a variable number
  // of in-flight sequences. CHECK that the op + dynamic types round-trip.
  // `causal = true` is the attribute default and is elided on print; use
  // false here so the round-trip of the attribute is actually observable.
  // CHECK: tesseract.paged_attention
  // CHECK-SAME: causal = false
  %out = "tesseract.paged_attention"(%q, %kpool, %vpool, %block_table, %seq_lens) {
    scale = 0.125 : f64, causal = false
  } : (tensor<?x8x64xf32>, tensor<256x16x8x64xf32>, tensor<256x16x8x64xf32>,
       tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x8x64xf32>
  return %out : tensor<?x8x64xf32>
}

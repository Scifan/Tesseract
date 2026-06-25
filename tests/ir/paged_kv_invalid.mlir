// RUN: tesseract-opt %s --split-input-file --verify-diagnostics
//
// M4 Track C2 (B-045): verifier negatives for the paged KV-cache ops.

// Pool dim disagrees with the num_blocks attribute.
func.func @bad_pool() -> tensor<128x16x8x64xf32> {
  // expected-error @+1 {{pool dim 0 (num_blocks) = 128 disagrees with attribute 256}}
  %pool = "tesseract.paged_kv_alloc"() {
    num_blocks = 256 : si64, block_size = 16 : si64,
    num_kv_heads = 8 : si64, head_dim = 64 : si64
  } : () -> tensor<128x16x8x64xf32>
  return %pool : tensor<128x16x8x64xf32>
}

// -----

// Query must be rank-3 [num_tokens, num_heads, head_dim].
func.func @bad_query(
    %q: tensor<8x64xf32>,
    %kpool: tensor<256x16x8x64xf32>,
    %vpool: tensor<256x16x8x64xf32>,
    %block_table: tensor<?x?xi32>,
    %seq_lens: tensor<?xi32>) -> tensor<8x64xf32> {
  // expected-error @+1 {{query must be rank-3}}
  %out = "tesseract.paged_attention"(%q, %kpool, %vpool, %block_table, %seq_lens) {
    scale = 0.125 : f64, causal = true
  } : (tensor<8x64xf32>, tensor<256x16x8x64xf32>, tensor<256x16x8x64xf32>,
       tensor<?x?xi32>, tensor<?xi32>) -> tensor<8x64xf32>
  return %out : tensor<8x64xf32>
}

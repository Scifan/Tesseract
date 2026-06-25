// RUN: tesseract-opt %s --split-input-file --verify-diagnostics
//
// Negative tests for tesseract.dequant_matmul's verifier (Wave 20 / B-037):
// unknown schemes, and group_size that disagrees with the scheme family.

func.func @bad_scheme(%a: tensor<2x8xf32>, %wq: tensor<8x4xi8>, %s: tensor<4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{unknown quantization scheme 'int3'}}
  %0 = "tesseract.dequant_matmul"(%a, %wq, %s) {scheme = "int3", group_size = -1 : si64}
       : (tensor<2x8xf32>, tensor<8x4xi8>, tensor<4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @grouped_needs_group_size(%a: tensor<2x8xf32>, %wq: tensor<8x4xi8>, %s: tensor<4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{is grouped and requires group_size > 0}}
  %0 = "tesseract.dequant_matmul"(%a, %wq, %s) {scheme = "int4_group", group_size = -1 : si64}
       : (tensor<2x8xf32>, tensor<8x4xi8>, tensor<4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @per_tensor_rejects_group_size(%a: tensor<2x8xf32>, %wq: tensor<8x4xi8>, %s: tensor<4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{is per-tensor and requires group_size == -1}}
  %0 = "tesseract.dequant_matmul"(%a, %wq, %s) {scheme = "int8", group_size = 4 : si64}
       : (tensor<2x8xf32>, tensor<8x4xi8>, tensor<4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

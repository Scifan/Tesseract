// RUN: tesseract-opt %s --convert-tesseract-to-gpu | FileCheck %s
//
// Wave 15 (B-009 / M2L.2) — device-IR half of graph-mode GPU codegen.
//
// Drives data-parallel tesseract ops through the full lowering pipeline
//     tesseract → linalg → bufferize → parallel-loops → gpu → outlining
// and pins the structural result: the module is a gpu.container_module, each
// elementwise function dispatches via `gpu.launch_func`, and the outlined
// `gpu.func` kernel carries `gpu.block_id` indexing + the lowered arithmetic
// (with the original tesseract op gone).
//
// NOTE: this stops at verified `gpu.*` IR. JIT-compiling the gpu.module to
// PTX/cubin + GPU parity is the B-009 tail, gated on an LLVM build with the
// NVPTX target and a free device (see ConvertToGpu.cpp header).

// CHECK: module attributes {gpu.container_module}

// CHECK-LABEL: func @add_to_gpu
// CHECK: gpu.launch_func @add_to_gpu_kernel
// CHECK-NOT: tesseract.add
func.func @add_to_gpu(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = "tesseract.add"(%a, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK: gpu.module @add_to_gpu_kernel
// CHECK: gpu.func @add_to_gpu_kernel{{.*}} kernel
// CHECK-DAG: gpu.block_id
// CHECK-DAG: gpu.thread_id
// CHECK: arith.addf
// CHECK: gpu.return

// CHECK-LABEL: func @mul_relu_to_gpu
// CHECK: gpu.launch_func @mul_relu_to_gpu_kernel
// CHECK-NOT: tesseract.mul
// CHECK-NOT: tesseract.relu
func.func @mul_relu_to_gpu(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = "tesseract.mul"(%a, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  %1 = "tesseract.relu"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}

// CHECK: gpu.module @mul_relu_to_gpu_kernel
// CHECK: arith.mulf

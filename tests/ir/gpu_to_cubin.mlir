// RUN: tesseract-opt %s \
// RUN:   --convert-tesseract-to-gpu \
// RUN:   --nvvm-attach-target='chip=sm_89 O=3' \
// RUN:   --convert-gpu-to-nvvm \
// RUN:   --convert-nvvm-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   --gpu-module-to-binary \
// RUN: | FileCheck %s
//
// M4 Phase 8 (B-009 tail) — the device-compilation half of GPU codegen.
//
// This drives the elementwise op all the way through the modern MLIR GPU
// compilation chain: tesseract -> linalg -> bufferize -> parallel-loops ->
// gpu -> outlining (the --convert-tesseract-to-gpu pipeline), then attaches an
// `#nvvm.target<chip=sm_89>`, lowers the kernel body to NVVM/LLVM, and runs
// `--gpu-module-to-binary`, which invokes the in-tree NVPTX backend + `ptxas`
// to serialize the kernel to a real cubin/fatbin. The check pins:
//   * the host `gpu.launch_func` survives (so the runtime launcher can read
//     the kernel name + grid/block + operand order);
//   * the `gpu.module` has been replaced by a serialized `gpu.binary` carrying
//     a `#gpu.object` for the sm_89 target (the cubin bytes).
//
// `GpuJitEngine` runs exactly this pipeline in-process and then loads the
// `gpu.binary` via the CUDA driver (cuModuleLoadData + cuLaunchKernel). A GPU
// is NOT required to run this test — ptxas compiles offline.

// CHECK: module attributes {gpu.container_module}
// CHECK: func @add_to_gpu
// CHECK: gpu.launch_func @add_to_gpu_kernel
// CHECK-NOT: gpu.module
// CHECK: gpu.binary @add_to_gpu_kernel
// CHECK-SAME: #gpu.object
// CHECK-SAME: chip = "sm_89"
func.func @add_to_gpu(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = "tesseract.add"(%a, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

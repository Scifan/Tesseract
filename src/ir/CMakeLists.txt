# MLIR dialect placeholder. Only evaluated when TESSERACT_ENABLE_MLIR=ON.
#
# Expects a local LLVM/MLIR install under third_party/llvm-install/ produced by
# scripts/build_llvm.sh, or an explicit -DMLIR_DIR=<...>/lib/cmake/mlir on the
# cmake command line. Dependencies.cmake wires both paths.

# The TableGen generator puts outputs into an include subfolder that mirrors the
# public include layout so that consumers can write `#include
# "tesseract/ir/TesseractOps.h.inc"` without bespoke include dirs.
set(_tblgen_out_dir "${CMAKE_CURRENT_BINARY_DIR}/include/tesseract/ir")
file(MAKE_DIRECTORY "${_tblgen_out_dir}")

# mlir_tablegen consumes `include_directories()` (via LLVM_TABLEGEN_FLAGS).
# Export the installed MLIR/LLVM include roots so that `include
# "mlir/IR/OpBase.td"` resolves, and the local `.td` files can find their
# siblings.
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

set(LLVM_TARGET_DEFINITIONS TesseractOps.td)
mlir_tablegen(include/tesseract/ir/TesseractOps.h.inc   -gen-op-decls)
mlir_tablegen(include/tesseract/ir/TesseractOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS TesseractDialect.td)
mlir_tablegen(include/tesseract/ir/TesseractDialect.h.inc   -gen-dialect-decls -dialect=tesseract)
mlir_tablegen(include/tesseract/ir/TesseractDialect.cpp.inc -gen-dialect-defs  -dialect=tesseract)

add_public_tablegen_target(TesseractIRIncGen)

add_library(tesseract_ir STATIC
  TesseractDialect.cpp
  TesseractOps.cpp
  Emit.cpp
  JitEngine.cpp
  passes/ConvertToLinalg.cpp
  passes/Backward.cpp
  passes/InterchangeMatmul.cpp
  passes/ConvertToGpu.cpp
)
add_library(Tesseract::ir ALIAS tesseract_ir)
add_dependencies(tesseract_ir TesseractIRIncGen)

target_include_directories(tesseract_ir
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(tesseract_ir
  PUBLIC
    tesseract_graph
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRTransforms
    # Lowering targets for the M1G conversion pass.
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMathDialect
    # Traits / interfaces our dialect pulls in via TesseractOps.td (+ the
    # generated .inc files). Exposed PUBLIC so downstream targets that link
    # `tesseract_ir` do not also need to know the exact list.
    MLIRInferTypeOpInterface
    MLIRControlFlowInterfaces
    MLIRFunctionInterfaces
    MLIRSideEffectInterfaces
    # -------- M1I.2.b: MLIR ExecutionEngine JIT path --------
    # Pipeline dialects.
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRLLVMDialect
    MLIRBufferizationDialect
    # Passes on those dialects.
    MLIRLinalgTransforms
    MLIRMemRefTransforms
    MLIRBufferizationTransforms
    MLIRFuncTransforms
    # -------- Wave 15 (B-009): tesseract → GPU dialect lowering --------
    MLIRGPUDialect
    MLIRGPUTransforms
    MLIRSCFToGPU
    # Conversion passes.
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRMemRefToLLVM
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRMathToLLVM
    MLIRMathToLibm
    MLIRControlFlowToLLVM
    MLIRReconcileUnrealizedCasts
    # LLVM IR translation + ExecutionEngine.
    MLIRExecutionEngine
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
  PRIVATE
    Tesseract::compile_options
)

# -------- M4 Phase 8 (B-009 tail): GPU JIT (cubin) execution path --------
# GpuJitEngine drives gpu.module -> NVVM -> PTX -> cubin (via ptxas) and runs
# the kernel through the CUDA driver API. It only compiles when both MLIR and
# CUDA are enabled; the header stays available regardless (callers gate on
# GpuJitEngine::available()).
if(TESSERACT_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(tesseract_ir PRIVATE GpuJitEngine.cpp)
  target_compile_definitions(tesseract_ir PUBLIC TESSERACT_IR_HAS_GPU_JIT=1)
  target_link_libraries(tesseract_ir
    PUBLIC
      # Device lowering + serialization.
      MLIRGPUToNVVMTransforms
      MLIRNVVMToLLVM
      MLIRNVVMDialect
      MLIRNVVMTarget
      MLIRGPUToLLVMIRTranslation
      MLIRNVVMToLLVMIRTranslation
    PRIVATE
      CUDA::cuda_driver
  )
endif()

# tesseract-opt CLI: links in the common mlir-opt front-end plus every upstream
# dialect and pass so contributors can experiment with lowerings interactively.
# Following the upstream mlir-opt pattern: pull every registered dialect / pass /
# extension library out of the global properties and link them explicitly.
get_property(dialect_libs    GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs  GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_executable(tesseract-opt tesseract-opt.cpp)
target_link_libraries(tesseract-opt
  PRIVATE
    Tesseract::ir
    Tesseract::compile_options
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    ${translation_libs}
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRSupport
)

install(TARGETS tesseract-opt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

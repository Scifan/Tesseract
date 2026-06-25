# Tesseract build options.

include_guard(GLOBAL)

option(TESSERACT_BUILD_TESTS      "Build Catch2-based unit tests"                                 ON)
option(TESSERACT_BUILD_EXAMPLES   "Build example executables (e.g. MNIST)"                        OFF)
option(TESSERACT_BUILD_BENCHMARKS "Build micro-benchmarks"                                        OFF)
option(TESSERACT_ENABLE_MLIR      "Build the MLIR dialect and tesseract-opt (requires LLVM/MLIR)" OFF)
option(TESSERACT_ENABLE_CUDA      "Build the CUDA backend (requires CUDA Toolkit 12.x+ and nvcc)" OFF)
option(TESSERACT_USE_EIGEN        "Use Eigen as the reference linalg backend (fallback: naive)"   OFF)
option(TESSERACT_WERROR           "Treat all compiler warnings as errors"                         OFF)
option(TESSERACT_NATIVE_ARCH      "Pass -march=native on supported compilers"                     OFF)
option(TESSERACT_ENABLE_OPENMP    "Enable OpenMP-based parallel CPU kernels"                      ON)
option(TESSERACT_BUILD_PYTHON     "Build the pybind11 Python frontend (tesseract._core)"          OFF)
option(TESSERACT_BUILD_STUDIO     "Build Tesseract Studio (visual block builder, native C++)"     OFF)
option(TESSERACT_ENABLE_CUTLASS   "Fetch CUTLASS for custom GEMM / grouped-GEMM kernels (needs CUDA)" OFF)
option(TESSERACT_ENABLE_NCCL      "Build the NCCL multi-GPU collective backend (needs CUDA + NCCL)"  OFF)
option(TESSERACT_ENABLE_FP8       "Enable Ada/Hopper FP8 (E4M3/E5M2) GEMM paths (needs CUDA 12.x)"   OFF)

# TESSERACT_CUDA_ARCHITECTURES controls which GPU architectures nvcc compiles
# for. The default auto-picks `native` (CMake >= 3.24, matches the developer's
# host GPU and keeps compile times short) or falls back to a conservative
# Ada/Hopper/Blackwell superset on older CMake. CI typically overrides with
# an explicit release fatbin list. Only consulted when TESSERACT_ENABLE_CUDA
# is ON.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set(_tesseract_cuda_arch_default "native")
else()
  set(_tesseract_cuda_arch_default "80;86;89;90")
endif()
set(TESSERACT_CUDA_ARCHITECTURES "${_tesseract_cuda_arch_default}" CACHE STRING
  "CMAKE_CUDA_ARCHITECTURES value for the CUDA backend (e.g. 'native', '86', '89;90').")
unset(_tesseract_cuda_arch_default)

message(STATUS "Tesseract options:")
message(STATUS "  TESSERACT_BUILD_TESTS       = ${TESSERACT_BUILD_TESTS}")
message(STATUS "  TESSERACT_BUILD_EXAMPLES    = ${TESSERACT_BUILD_EXAMPLES}")
message(STATUS "  TESSERACT_BUILD_BENCHMARKS  = ${TESSERACT_BUILD_BENCHMARKS}")
message(STATUS "  TESSERACT_ENABLE_MLIR       = ${TESSERACT_ENABLE_MLIR}")
message(STATUS "  TESSERACT_ENABLE_CUDA       = ${TESSERACT_ENABLE_CUDA}")
message(STATUS "  TESSERACT_USE_EIGEN         = ${TESSERACT_USE_EIGEN}")
message(STATUS "  TESSERACT_WERROR            = ${TESSERACT_WERROR}")
message(STATUS "  TESSERACT_NATIVE_ARCH       = ${TESSERACT_NATIVE_ARCH}")
message(STATUS "  TESSERACT_ENABLE_OPENMP     = ${TESSERACT_ENABLE_OPENMP}")
message(STATUS "  TESSERACT_BUILD_PYTHON      = ${TESSERACT_BUILD_PYTHON}")
message(STATUS "  TESSERACT_BUILD_STUDIO      = ${TESSERACT_BUILD_STUDIO}")
message(STATUS "  TESSERACT_ENABLE_CUTLASS    = ${TESSERACT_ENABLE_CUTLASS}")
message(STATUS "  TESSERACT_ENABLE_NCCL       = ${TESSERACT_ENABLE_NCCL}")
message(STATUS "  TESSERACT_ENABLE_FP8        = ${TESSERACT_ENABLE_FP8}")

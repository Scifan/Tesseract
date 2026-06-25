cmake_minimum_required(VERSION 3.22)

project(tesseract
  VERSION 0.1.0
  DESCRIPTION "Tesseract: a C++20 deep learning framework"
  HOMEPAGE_URL "https://example.invalid/tesseract"
  LANGUAGES C CXX
)

# ---------------------------------------------------------------------------- #
# Project-wide settings                                                        #
# ---------------------------------------------------------------------------- #
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for clangd / clang-tidy by default.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)

# Install layout.
include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Options)
include(CompilerFlags)
include(Dependencies)

# ---------------------------------------------------------------------------- #
# Subdirectories                                                               #
# ---------------------------------------------------------------------------- #
add_subdirectory(src)

if(TESSERACT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TESSERACT_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
endif()

if(TESSERACT_BUILD_BENCHMARKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
  add_subdirectory(benchmarks)
endif()

if(TESSERACT_BUILD_PYTHON AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/CMakeLists.txt")
  add_subdirectory(python)
endif()

if(TESSERACT_BUILD_STUDIO AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/studio/CMakeLists.txt")
  add_subdirectory(studio)
endif()

# ---------------------------------------------------------------------------- #
# Summary                                                                      #
# ---------------------------------------------------------------------------- #
message(STATUS "")
message(STATUS "========== Tesseract configuration summary ==========")
message(STATUS "  Version          : ${PROJECT_VERSION}")
message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler         : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ standard     : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests            : ${TESSERACT_BUILD_TESTS}")
message(STATUS "  Examples         : ${TESSERACT_BUILD_EXAMPLES}")
message(STATUS "  Benchmarks       : ${TESSERACT_BUILD_BENCHMARKS}")
message(STATUS "  Studio           : ${TESSERACT_BUILD_STUDIO}")
message(STATUS "  MLIR dialect     : ${TESSERACT_ENABLE_MLIR}")
message(STATUS "  CUDA backend     : ${TESSERACT_ENABLE_CUDA}")
if(TESSERACT_ENABLE_CUDA)
  message(STATUS "  CUDA archs       : ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "  CUDA toolkit     : ${CUDAToolkit_VERSION} (${CMAKE_CUDA_COMPILER})")
endif()
message(STATUS "======================================================")
message(STATUS "")

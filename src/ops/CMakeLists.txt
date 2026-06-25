set(_tesseract_ops_sources
  cpu/Broadcast.cpp
  cpu/Arithmetic.cpp
  cpu/MatMul.cpp
  cpu/GemmAvx2.cpp
  cpu/GemvVnni.cpp
  cpu/Reduction.cpp
  cpu/Activation.cpp
  cpu/Softmax.cpp
  cpu/Loss.cpp
  cpu/View.cpp
  cpu/Indexing.cpp
  cpu/Attention.cpp
  cpu/RMSNorm.cpp
  cpu/LayerNorm.cpp
  cpu/BatchNorm.cpp
  cpu/RotaryEmbedding.cpp
  # Wave 3.1 (B-021): INT8 weight-only dequantize-matmul op. CPU
  # reference + CUDA dispatch live side-by-side in this TU (same
  # layering as `cpu/RMSNorm.cpp` — the CUDA launcher is wired via
  # `detail/DequantMatMul.hpp` and throws in CPU-only builds).
  cpu/DequantMatMul.cpp
  # Wave 4.1 (B-025): fused SwiGLU activation (silu(gate) * up).
  # CPU reference + CUDA dispatch + composite autograd fallback
  # live together here, mirroring `cpu/RMSNorm.cpp`. CUDA bridge
  # is `detail/SwiGLU.hpp`, routed to `SwiGLUStub.cpp` in CPU-only
  # builds.
  cpu/SwiGLU.cpp
  # M4 Track A2 (B-039): selective state-space scan (Mamba / S6). CPU
  # reference + CUDA dispatch live together here, mirroring `cpu/SwiGLU.cpp`.
  # CUDA bridge is `detail/SelectiveScan.hpp`, routed to
  # `SelectiveScanStub.cpp` in CPU-only builds.
  cpu/SelectiveScan.cpp
)

add_library(tesseract_ops STATIC ${_tesseract_ops_sources})
add_library(Tesseract::ops ALIAS tesseract_ops)

target_include_directories(tesseract_ops
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu
)

target_link_libraries(tesseract_ops
  PUBLIC  tesseract_core tesseract_graph
  PRIVATE Tesseract::compile_options fmt::fmt
)

if(TESSERACT_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(tesseract_ops PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(tesseract_ops PRIVATE TESSERACT_HAS_OPENMP=1)
endif()

if(TESSERACT_USE_EIGEN)
  target_link_libraries(tesseract_ops PRIVATE Eigen3::Eigen)
  target_compile_definitions(tesseract_ops PRIVATE TESSERACT_HAS_EIGEN=1)
endif()

set_target_properties(tesseract_ops PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

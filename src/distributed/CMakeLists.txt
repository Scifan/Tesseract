set(_tesseract_distributed_sources
  CommBackend.cpp
  TensorParallel.cpp
  # M4 Phase 7: real multi-GPU NCCL collectives. Always compiled — the body is
  # a throwing stub unless TESSERACT_HAS_NCCL (carried by tesseract_nccl).
  NcclCommBackend.cpp
)

add_library(tesseract_distributed STATIC ${_tesseract_distributed_sources})
add_library(Tesseract::distributed ALIAS tesseract_distributed)

target_include_directories(tesseract_distributed
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(tesseract_distributed
  PUBLIC  tesseract_core tesseract_ops tesseract_nn tesseract_autograd
  PRIVATE Tesseract::compile_options fmt::fmt
)

# NCCL collectives: link nccl (carries TESSERACT_HAS_NCCL + nccl.h) and the
# CUDA runtime headers/lib used by NcclCommBackend.cpp.
if(TESSERACT_ENABLE_NCCL AND TARGET tesseract_nccl)
  target_link_libraries(tesseract_distributed PRIVATE tesseract_nccl)
  if(TARGET CUDA::cudart)
    target_link_libraries(tesseract_distributed PRIVATE CUDA::cudart)
  endif()
endif()

set_target_properties(tesseract_distributed PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

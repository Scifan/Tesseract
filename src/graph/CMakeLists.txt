# Stage-1 C++ graph IR. Always built (no MLIR dependency). See ADR-0004.

set(_tesseract_graph_sources
  Graph.cpp
  GraphScope.cpp
  Autograd.cpp
)

add_library(tesseract_graph STATIC ${_tesseract_graph_sources})
add_library(Tesseract::graph ALIAS tesseract_graph)

target_include_directories(tesseract_graph
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_link_libraries(tesseract_graph
  PUBLIC
    tesseract_core
  PRIVATE
    Tesseract::compile_options
)

set_target_properties(tesseract_graph PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# Graph interpreter: dispatches graph ops to the eager `ops::` kernels.
# Kept in its own static lib so `tesseract_graph` itself stays free of a
# circular dependency with `tesseract_ops` (which itself links
# `tesseract_graph` for the recording side).
add_library(tesseract_graph_runtime STATIC Interpreter.cpp)
add_library(Tesseract::graph_runtime ALIAS tesseract_graph_runtime)

target_include_directories(tesseract_graph_runtime
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_link_libraries(tesseract_graph_runtime
  PUBLIC
    tesseract_graph
    tesseract_ops
  PRIVATE
    Tesseract::compile_options
)

set_target_properties(tesseract_graph_runtime PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

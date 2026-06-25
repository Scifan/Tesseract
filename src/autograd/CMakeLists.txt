set(_tesseract_autograd_sources
  Engine.cpp
)

add_library(tesseract_autograd STATIC ${_tesseract_autograd_sources})
add_library(Tesseract::autograd ALIAS tesseract_autograd)

target_include_directories(tesseract_autograd
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(tesseract_autograd
  PUBLIC  tesseract_core tesseract_ops
  PRIVATE Tesseract::compile_options fmt::fmt
)

set_target_properties(tesseract_autograd PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

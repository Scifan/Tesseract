set(_tesseract_optim_sources
  Optimizer.cpp
  SGD.cpp
  Adam.cpp
)

add_library(tesseract_optim STATIC ${_tesseract_optim_sources})
add_library(Tesseract::optim ALIAS tesseract_optim)

target_include_directories(tesseract_optim
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(tesseract_optim
  PUBLIC  tesseract_core tesseract_autograd
  PRIVATE Tesseract::compile_options fmt::fmt
)

set_target_properties(tesseract_optim PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

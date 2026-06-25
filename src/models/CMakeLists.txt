set(_tesseract_models_sources
  Disaggregated.cpp
  Llama.cpp
  MambaModel.cpp
  Sampler.cpp
  Scheduler.cpp
  Speculative.cpp
  StructuredDecoding.cpp
)

add_library(tesseract_models STATIC ${_tesseract_models_sources})
add_library(Tesseract::models ALIAS tesseract_models)

target_include_directories(tesseract_models
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(tesseract_models
  PUBLIC  tesseract_core tesseract_ops tesseract_nn tesseract_io
  PRIVATE Tesseract::compile_options fmt::fmt
)

set_target_properties(tesseract_models PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

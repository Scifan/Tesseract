set(_tesseract_io_sources
  SafeTensors.cpp
  Tokenizer.cpp
  BpeTokenizer.cpp
)

add_library(tesseract_io STATIC ${_tesseract_io_sources})
add_library(Tesseract::io ALIAS tesseract_io)

target_include_directories(tesseract_io
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(tesseract_io
  PUBLIC  tesseract_core
  PRIVATE Tesseract::compile_options fmt::fmt
)

set_target_properties(tesseract_io PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

set(TESSERACT_CORE_SOURCES
  DType.cpp
  Device.cpp
  Shape.cpp
  Allocator.cpp
  Storage.cpp
  Stream.cpp
  Tensor.cpp
  GradMode.cpp
)

add_library(tesseract_core STATIC ${TESSERACT_CORE_SOURCES})
add_library(Tesseract::core ALIAS tesseract_core)

target_include_directories(tesseract_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(tesseract_core
  PUBLIC
    fmt::fmt
  PRIVATE
    Tesseract::compile_options
    # tesseract_core references detail symbols defined in tesseract_cuda
    # (e.g. `cuda_default_allocator`, plus the whole stream/event bridge
    # consumed by src/core/Stream.cpp). Declaring a PRIVATE link dep
    # ensures CMake puts tesseract_cuda AFTER tesseract_core on every
    # consumer's link line — GNU ld resolves the back-reference in a
    # single left-to-right scan. tesseract_cuda does not link
    # tesseract_core back (it only needs the Allocator header), so the
    # graph stays acyclic.
    tesseract_cuda
)

set_target_properties(tesseract_core PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

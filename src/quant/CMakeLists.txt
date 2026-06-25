# tesseract_quant — Wave 3 (B-021) weight-only quantization utilities.
#
# Pure host-side packer right now (Pack.cpp). Future waves will land
# INT4 group packing + optional CUDA-accelerated packers here; the
# target already links against tesseract_core so those extensions
# don't need a new library.

set(_tesseract_quant_sources
  Pack.cpp
  # Wave 9 (B-031): device-resident KV-cache INT8 quantize/dequantize.
  # Unlike Pack.cpp these run on the decode hot path, so they dispatch
  # to the CUDA kernels in tesseract_cuda (or a CPU loop on host).
  QuantizeKV.cpp
)

add_library(tesseract_quant STATIC ${_tesseract_quant_sources})
add_library(Tesseract::quant ALIAS tesseract_quant)

target_include_directories(tesseract_quant
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(tesseract_quant
  PUBLIC  tesseract_core
  PRIVATE Tesseract::compile_options fmt::fmt Tesseract::cuda
)

set_target_properties(tesseract_quant PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

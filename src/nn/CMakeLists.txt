set(_tesseract_nn_sources
  Module.cpp
  Linear.cpp
  QuantizedLinear.cpp
  QuantizedLinearInt4G.cpp
  Scheme.cpp
  Embedding.cpp
  Activation.cpp
  Sequential.cpp
  Loss.cpp
  KVCache.cpp
  QuantizedKVCache.cpp
  BlockAllocator.cpp
  PagedKVPool.cpp
  PagedKVCache.cpp
  QuantizedPagedKVPool.cpp
  QuantizedPagedKVCache.cpp
  PagedAttention.cpp
  RMSNorm.cpp
  LayerNorm.cpp
  BatchNorm.cpp
  MultiHeadAttention.cpp
  FeedForward.cpp
  MoEFeedForward.cpp
  Mamba.cpp
  TransformerBlock.cpp
  RotaryEmbedding.cpp
)

add_library(tesseract_nn STATIC ${_tesseract_nn_sources})
add_library(Tesseract::nn ALIAS tesseract_nn)

target_include_directories(tesseract_nn
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(tesseract_nn
  PUBLIC  tesseract_core tesseract_ops tesseract_autograd tesseract_quant
  PRIVATE Tesseract::compile_options fmt::fmt Tesseract::cuda
)

set_target_properties(tesseract_nn PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

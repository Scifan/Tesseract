# tesseract_cuda is configured first so `tesseract_core` can PRIVATE-link
# it by name (the core -> cuda link dep flips the static-library ordering
# so GNU ld can resolve `tesseract::detail::cuda_default_allocator` and
# the stream/event bridge symbols in a single left-to-right scan).
add_subdirectory(cuda)

add_subdirectory(core)
add_subdirectory(graph)
add_subdirectory(quant)
add_subdirectory(ops)
add_subdirectory(autograd)
add_subdirectory(nn)
add_subdirectory(optim)
add_subdirectory(io)
add_subdirectory(models)
add_subdirectory(distributed)

if(TESSERACT_ENABLE_MLIR)
  add_subdirectory(ir)
endif()

# Meta-target: everything the rest of the tree links against.
add_library(tesseract INTERFACE)
add_library(Tesseract::tesseract ALIAS tesseract)
target_link_libraries(tesseract INTERFACE
  tesseract_core
  tesseract_graph
  tesseract_graph_runtime
  tesseract_quant
  tesseract_ops
  tesseract_autograd
  tesseract_nn
  tesseract_optim
  tesseract_io
  tesseract_models
  tesseract_distributed
  tesseract_cuda
)

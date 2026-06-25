# Python frontend (M4 Track B1 / B-041). Builds the `tesseract._core` pybind11
# extension and stages the pure-Python `tesseract/` package next to it so an
# in-tree `PYTHONPATH=<build>/python pytest` works without an install step.

pybind11_add_module(tesseract_core_ext MODULE _core.cpp)
set_target_properties(tesseract_core_ext PROPERTIES
  OUTPUT_NAME _core
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  # Place the compiled extension inside the staged package directory.
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tesseract"
)
target_link_libraries(tesseract_core_ext PRIVATE
  Tesseract::tesseract
  Tesseract::compile_options
)

# Stage the pure-Python package sources beside the built extension so the
# package is importable straight from the build tree.
file(GLOB _tesseract_py_sources "${CMAKE_CURRENT_SOURCE_DIR}/tesseract/*.py")
foreach(_py ${_tesseract_py_sources})
  get_filename_component(_name "${_py}" NAME)
  configure_file("${_py}" "${CMAKE_CURRENT_BINARY_DIR}/tesseract/${_name}" COPYONLY)
endforeach()

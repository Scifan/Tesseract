include_guard(GLOBAL)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# --- fmt ---------------------------------------------------------------------
# Prefer a system-installed fmt; fall back to FetchContent for reproducibility.
find_package(fmt 9 QUIET)
if(NOT fmt_FOUND)
  message(STATUS "fmt not found on system; fetching via FetchContent")
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(fmt)
endif()

# --- Catch2 v3 ---------------------------------------------------------------
if(TESSERACT_BUILD_TESTS)
  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    message(STATUS "Catch2 v3 not found on system; fetching via FetchContent")
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.5.4
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  else()
    # When found via find_package, Catch.cmake lives next to the installed Config.
    list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
  endif()
endif()

# --- Eigen (optional) --------------------------------------------------------
# Only fetched/found when the user explicitly opts in via TESSERACT_USE_EIGEN.
# Eigen is header-only so we never build a library target; we just materialize
# an `Eigen3::Eigen` INTERFACE target that includes the headers. Prefer a
# system install (e.g. `libeigen3-dev`) and fall back to FetchContent so CI
# machines without a pre-installed copy still work.
if(TESSERACT_USE_EIGEN)
  find_package(Eigen3 3.4 NO_MODULE QUIET)
  if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found on system; fetching via FetchContent")
    # Disable Eigen's own test / benchmark / doc targets so they don't show
    # up in our build graph.
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      Eigen3
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG        3.4.0
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Eigen3)
  else()
    message(STATUS "Eigen3 found: ${Eigen3_VERSION} (${EIGEN3_INCLUDE_DIR})")
  endif()
endif()

# --- pybind11 (optional Python frontend) -------------------------------------
# Only pulled in under TESSERACT_BUILD_PYTHON=ON (M4 Track B1 / B-041). Prefer
# the pybind11 shipped with the active Python interpreter (`python -m pybind11
# --cmakedir`), fall back to a plain find_package, then FetchContent so a bare
# checkout still configures.
if(TESSERACT_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  if(NOT pybind11_DIR)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE _pybind11_cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _pybind11_probe_rc)
    if(_pybind11_probe_rc EQUAL 0 AND EXISTS "${_pybind11_cmakedir}")
      set(pybind11_DIR "${_pybind11_cmakedir}" CACHE PATH "pybind11 cmake dir")
    endif()
  endif()
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found on system; fetching via FetchContent")
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.13.6
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
  else()
    message(STATUS "pybind11 found: ${pybind11_VERSION} (${pybind11_DIR})")
  endif()
endif()

# --- OpenMP (optional) -------------------------------------------------------
if(TESSERACT_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX QUIET)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION} (${OpenMP_CXX_LIB_NAMES})")
  else()
    message(STATUS "OpenMP not found; CPU kernels will fall back to single-thread execution")
  endif()
endif()

# --- CUDA Toolkit (optional) -------------------------------------------------
# Pulled in only under TESSERACT_ENABLE_CUDA=ON. When OFF, this block is
# inert: no CUDA compiler is probed, no CUDA headers appear on any include
# path, and no `cublas*` / `cudart*` symbol is linked. This is the single
# build-system gate for everything under `src/cuda/` (see ADR-0005).
if(TESSERACT_ENABLE_CUDA)
  # `native` as a CUDA_ARCHITECTURES value requires CMake >= 3.24. Only bump
  # the minimum when someone actually enables the CUDA backend; CPU-only
  # developers keep the M0 3.22 floor.
  if(TESSERACT_CUDA_ARCHITECTURES STREQUAL "native" AND CMAKE_VERSION VERSION_LESS 3.24)
    message(FATAL_ERROR
      "TESSERACT_ENABLE_CUDA=ON with TESSERACT_CUDA_ARCHITECTURES='native' "
      "requires CMake >= 3.24. Got ${CMAKE_VERSION}. Either upgrade CMake or "
      "set TESSERACT_CUDA_ARCHITECTURES to an explicit list (e.g. '86;89;90').")
  endif()

  # Applies to every CUDA target created after this point; `src/cuda/`
  # doesn't need to respecify it per-target.
  set(CMAKE_CUDA_ARCHITECTURES "${TESSERACT_CUDA_ARCHITECTURES}" CACHE STRING
    "CUDA architectures to compile for" FORCE)

  find_package(CUDAToolkit 12.0 REQUIRED)
  enable_language(CUDA)

  # CUDA translation units compile as C++17 — this matches cuBLAS / cuBLASLt
  # / CUTLASS header expectations through CUDA 12.x. Host C++ stays at 20.
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
  endif()

  message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION} "
                 "(nvcc: ${CMAKE_CUDA_COMPILER})")
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# --- CUTLASS (optional, header-only) -----------------------------------------
# Pulled in only under TESSERACT_ENABLE_CUTLASS=ON (custom GEMM / grouped-GEMM,
# M4 perf-closeout Phase 3/4). Header-only: we materialize a `tesseract_cutlass`
# INTERFACE target carrying the include dirs. Prefer a user-provided
# CUTLASS_DIR, else FetchContent a pinned release.
if(TESSERACT_ENABLE_CUTLASS)
  if(NOT TESSERACT_ENABLE_CUDA)
    message(FATAL_ERROR "TESSERACT_ENABLE_CUTLASS requires TESSERACT_ENABLE_CUDA=ON")
  endif()
  if(DEFINED CUTLASS_DIR AND EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(STATUS "CUTLASS: using user CUTLASS_DIR=${CUTLASS_DIR}")
    set(_cutlass_inc "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/tools/util/include")
  else()
    message(STATUS "CUTLASS not provided; fetching via FetchContent")
    set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      cutlass
      GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
      GIT_TAG        v3.5.1
      GIT_SHALLOW    TRUE
    )
    # Header-only consumption: populate without configuring CUTLASS's own build.
    FetchContent_GetProperties(cutlass)
    if(NOT cutlass_POPULATED)
      FetchContent_Populate(cutlass)
    endif()
    set(_cutlass_inc "${cutlass_SOURCE_DIR}/include"
                     "${cutlass_SOURCE_DIR}/tools/util/include")
  endif()
  add_library(tesseract_cutlass INTERFACE)
  target_include_directories(tesseract_cutlass INTERFACE ${_cutlass_inc})
  target_compile_definitions(tesseract_cutlass INTERFACE TESSERACT_HAS_CUTLASS=1)
  message(STATUS "CUTLASS include dirs: ${_cutlass_inc}")
endif()

# --- NCCL (optional) ---------------------------------------------------------
# Pulled in only under TESSERACT_ENABLE_NCCL=ON (real multi-GPU collectives,
# M4 perf-closeout Phase 7). Locates nccl.h + libnccl.so, preferring a
# user-provided NCCL_ROOT, then the nvidia-nccl-cu12 pip package, then system.
if(TESSERACT_ENABLE_NCCL)
  if(NOT TESSERACT_ENABLE_CUDA)
    message(FATAL_ERROR "TESSERACT_ENABLE_NCCL requires TESSERACT_ENABLE_CUDA=ON")
  endif()
  # Probe the active Python for an nvidia-nccl-cu12 wheel layout.
  if(NOT DEFINED NCCL_ROOT)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
          "import os,nvidia.nccl as n;print(os.path.dirname(n.__file__))"
        OUTPUT_VARIABLE _nccl_pip_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _nccl_pip_rc
        ERROR_QUIET)
      if(_nccl_pip_rc EQUAL 0 AND EXISTS "${_nccl_pip_dir}/include/nccl.h")
        set(NCCL_ROOT "${_nccl_pip_dir}" CACHE PATH "NCCL root (pip wheel)")
      endif()
    endif()
  endif()
  find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ${NCCL_ROOT} ENV NCCL_ROOT
    PATH_SUFFIXES include)
  # pip wheels ship only the SONAME-versioned `libnccl.so.2` (no unversioned
  # dev symlink), so match that filename explicitly in addition to the plain
  # `nccl` name used by system/conda installs.
  find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2 libnccl.so
    HINTS ${NCCL_ROOT} ENV NCCL_ROOT
    PATH_SUFFIXES lib lib64)
  if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR
      "TESSERACT_ENABLE_NCCL=ON but nccl.h / libnccl not found. "
      "Set -DNCCL_ROOT=<dir> or `pip install nvidia-nccl-cu12`. "
      "(inc=${NCCL_INCLUDE_DIR} lib=${NCCL_LIBRARY})")
  endif()
  add_library(tesseract_nccl INTERFACE)
  target_include_directories(tesseract_nccl INTERFACE "${NCCL_INCLUDE_DIR}")
  target_link_libraries(tesseract_nccl INTERFACE "${NCCL_LIBRARY}")
  target_compile_definitions(tesseract_nccl INTERFACE TESSERACT_HAS_NCCL=1)
  message(STATUS "NCCL found: ${NCCL_LIBRARY} (include: ${NCCL_INCLUDE_DIR})")
endif()

# --- MLIR / LLVM (optional) --------------------------------------------------
if(TESSERACT_ENABLE_MLIR)
  # Look for an MLIR build produced by scripts/build_llvm.sh by default, but
  # honor an explicitly provided MLIR_DIR / LLVM_DIR on the command line.
  if(NOT DEFINED MLIR_DIR)
    set(_default_mlir_dir "${CMAKE_SOURCE_DIR}/third_party/llvm-install/lib/cmake/mlir")
    if(EXISTS "${_default_mlir_dir}")
      set(MLIR_DIR "${_default_mlir_dir}" CACHE PATH "" FORCE)
    endif()
  endif()
  if(NOT DEFINED LLVM_DIR)
    set(_default_llvm_dir "${CMAKE_SOURCE_DIR}/third_party/llvm-install/lib/cmake/llvm")
    if(EXISTS "${_default_llvm_dir}")
      set(LLVM_DIR "${_default_llvm_dir}" CACHE PATH "" FORCE)
    endif()
  endif()

  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
endif()

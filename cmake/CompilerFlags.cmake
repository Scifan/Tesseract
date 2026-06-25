include_guard(GLOBAL)

# Default to Release if not specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Default build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# A single INTERFACE target all Tesseract targets link against for consistent
# warning flags, language standard, visibility, etc.
add_library(tesseract_compile_options INTERFACE)
add_library(Tesseract::compile_options ALIAS tesseract_compile_options)

target_compile_features(tesseract_compile_options INTERFACE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(tesseract_compile_options INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wold-style-cast
    -Wcast-align
    -Woverloaded-virtual
    -Wformat=2
    -Wno-unused-parameter
    # -Wnon-virtual-dtor is intentionally omitted: Catch2's internal
    # BinaryExpr<T> classes trigger it under Clang/GCC. Re-enable once we
    # stop transitively including those through test TUs.
  )
  if(TESSERACT_WERROR)
    target_compile_options(tesseract_compile_options INTERFACE -Werror)
  endif()
  if(TESSERACT_NATIVE_ARCH)
    target_compile_options(tesseract_compile_options INTERFACE -march=native)
  endif()
elseif(MSVC)
  target_compile_options(tesseract_compile_options INTERFACE /W4 /permissive-)
  if(TESSERACT_WERROR)
    target_compile_options(tesseract_compile_options INTERFACE /WX)
  endif()
endif()

# Position-independent code everywhere so that static libs can be relinked into
# shared libs or Python extensions down the road.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

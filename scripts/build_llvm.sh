#!/usr/bin/env bash
#
# build_llvm.sh — build LLVM + MLIR 18.x into third_party/llvm-install/ without
# needing sudo. Meant to be run once per machine; completion takes 30–90 min
# depending on core count.
#
# The produced install is referenced from CMake via
#   -DMLIR_DIR=<repo>/third_party/llvm-install/lib/cmake/mlir
# (which Dependencies.cmake auto-detects when you flip TESSERACT_ENABLE_MLIR=ON).
#
# Usage:
#   ./scripts/build_llvm.sh                 # default: LLVM 18.1.8, llvm+mlir
#   LLVM_VERSION=18.1.7 ./scripts/build_llvm.sh
#   JOBS=8 ./scripts/build_llvm.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

LLVM_VERSION="${LLVM_VERSION:-18.1.8}"
LLVM_TAG="llvmorg-${LLVM_VERSION}"
LLVM_URL="${LLVM_URL:-https://github.com/llvm/llvm-project.git}"

THIRD_PARTY="${REPO_ROOT}/third_party"
SRC_DIR="${THIRD_PARTY}/llvm-project"
BUILD_DIR="${THIRD_PARTY}/llvm-build"
INSTALL_DIR="${THIRD_PARTY}/llvm-install"

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "[build_llvm] repo root:   ${REPO_ROOT}"
echo "[build_llvm] llvm tag:    ${LLVM_TAG}"
echo "[build_llvm] install dir: ${INSTALL_DIR}"
echo "[build_llvm] jobs:        ${JOBS}"
echo "[build_llvm] build type:  ${BUILD_TYPE}"

mkdir -p "${THIRD_PARTY}"

# --- 1. Fetch / refresh the LLVM source ---------------------------------------

if [[ ! -d "${SRC_DIR}/.git" ]]; then
  echo "[build_llvm] cloning ${LLVM_URL} @ ${LLVM_TAG}"
  git clone --branch "${LLVM_TAG}" --depth 1 "${LLVM_URL}" "${SRC_DIR}"
else
  echo "[build_llvm] source dir already present at ${SRC_DIR} (re-using)"
fi

# --- 2. Sanity-check ninja ----------------------------------------------------

if ! command -v ninja >/dev/null 2>&1; then
  echo "[build_llvm][error] 'ninja' is not on PATH; run scripts/bootstrap.sh first" >&2
  exit 1
fi

# --- 3. Configure -------------------------------------------------------------

mkdir -p "${BUILD_DIR}"
cmake -S "${SRC_DIR}/llvm" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_BUILD_TESTS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_PARALLEL_LINK_JOBS=2

# --- 4. Build + install -------------------------------------------------------

cmake --build "${BUILD_DIR}" --target install -j "${JOBS}"

echo "[build_llvm] done. To use:"
echo "  cmake -S . -B build -DTESSERACT_ENABLE_MLIR=ON"

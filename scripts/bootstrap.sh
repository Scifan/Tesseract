#!/usr/bin/env bash
#
# bootstrap.sh — prepares a user-local toolchain for building Tesseract + MLIR
# without requiring `sudo`.
#
# Performs three things:
#   1. Ensures `ninja` is available (installs it via `pip install --user`).
#   2. Creates `third_party/` and common subdirectories.
#   3. Prints a hint on next steps.
#
# Usage: ./scripts/bootstrap.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[bootstrap] repo root: ${REPO_ROOT}"

# --- 1. ninja -----------------------------------------------------------------

if command -v ninja >/dev/null 2>&1; then
  echo "[bootstrap] ninja already on PATH: $(command -v ninja)"
else
  echo "[bootstrap] ninja not found; installing user-local copy via pip"
  if ! command -v pip >/dev/null 2>&1 && ! command -v pip3 >/dev/null 2>&1; then
    echo "[bootstrap][error] neither pip nor pip3 is installed; aborting" >&2
    exit 1
  fi
  PIP="$(command -v pip3 || command -v pip)"
  "${PIP}" install --user --upgrade ninja
  USER_BASE="$(python3 -c 'import site, sys; print(site.getuserbase())')"
  USER_BIN="${USER_BASE}/bin"
  echo "[bootstrap] ninja installed to ${USER_BIN}/ninja"
  if ! echo ":${PATH}:" | grep -q ":${USER_BIN}:"; then
    echo "[bootstrap][warn] ${USER_BIN} is not on PATH; add the following to"
    echo "[bootstrap][warn]   your shell rc file:"
    echo "    export PATH=\"${USER_BIN}:\$PATH\""
  fi
fi

# --- 2. third_party layout ----------------------------------------------------

mkdir -p "${REPO_ROOT}/third_party"
mkdir -p "${REPO_ROOT}/third_party/.cache"

echo "[bootstrap] third_party/ ready"

# --- 3. next steps ------------------------------------------------------------

cat <<'EOF'

[bootstrap] done.

Next steps:
  * CPU-only (default; no LLVM needed):
        cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
        cmake --build build -j
        ctest --test-dir build --output-on-failure

  * Enable MLIR dialect (requires LLVM/MLIR install):
        ./scripts/build_llvm.sh                      # ~60 min on 16 cores
        cmake -S . -B build -DTESSERACT_ENABLE_MLIR=ON
        cmake --build build -j

EOF

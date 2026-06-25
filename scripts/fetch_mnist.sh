#!/usr/bin/env bash
# Download the MNIST dataset into data/mnist/ (relative to the project root).
# The raw IDX files are placed alongside this script so examples/mnist.cpp can
# find them with a single `data/mnist` argument.
set -euo pipefail

DEST="${1:-data/mnist}"
mkdir -p "${DEST}"
cd "${DEST}"

BASES=(
  "train-images-idx3-ubyte"
  "train-labels-idx1-ubyte"
  "t10k-images-idx3-ubyte"
  "t10k-labels-idx1-ubyte"
)

# Upstream mirrors. We try them in order.
MIRRORS=(
  "https://storage.googleapis.com/cvdf-datasets/mnist"
  "https://ossci-datasets.s3.amazonaws.com/mnist"
  "http://yann.lecun.com/exdb/mnist"
)

fetch() {
  local name="$1"
  if [[ -f "${name}" ]]; then
    echo "[skip] ${name} already present"
    return 0
  fi
  for m in "${MIRRORS[@]}"; do
    local url="${m}/${name}.gz"
    echo "[fetch] ${url}"
    if curl -fsSL "${url}" -o "${name}.gz"; then
      gunzip -f "${name}.gz"
      return 0
    fi
  done
  echo "[error] failed to download ${name} from all mirrors" >&2
  return 1
}

for f in "${BASES[@]}"; do
  fetch "${f}"
done

echo "[done] MNIST ready in $(pwd)"

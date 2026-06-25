#!/usr/bin/env bash
# M4 cross-cutting (B-046) — external decode-throughput comparison scaffold:
# Tesseract vs llama.cpp on CPU.
#
# This is the *fairly-alignable* external benchmark idea.md §6.2/§8.5 calls for
# (no Hopper / vLLM dependency). It runs the Tesseract side automatically and
# prints the exact `llama-bench` invocation for the llama.cpp side, then leaves
# a results table for you to fill in. See docs/design/external-benchmark.md for
# the methodology and the recorded gap.
#
# Usage:
#   scripts/bench_vs_llama_cpp.sh [--build-dir build-cpu] [--prompt 32] [--gen 64]
#                                 [--threads N]
#                                 [--llama-cpp /path/to/llama.cpp]   (auto-builds GGUF)
#                                 [--llama-bench /path/to/llama-bench]
#                                 [--gguf /path/to/model.gguf]
#
# The Tesseract bench uses a small synthetic Llama (4L/d256/8h/v4096) — throughput
# is weight-independent, so the comparison is about the *runtime*, not the model.
# For a strict apples-to-apples number, pass --llama-cpp <checkout>: the script
# generates a SAME-architecture GGUF via scripts/make_tiny_llama_gguf.py and uses
# <checkout>/build/bin/llama-bench. Alternatively pass --gguf + --llama-bench.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="build-cpu"
PROMPT=32
GEN=64
LLAMA_BENCH="${LLAMA_BENCH:-}"
LLAMA_CPP="${LLAMA_CPP:-}"
GGUF="${GGUF:-}"
PY="${PYTHON:-python3}"
THREADS="$(nproc 2>/dev/null || echo 8)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --gen) GEN="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --llama-cpp) LLAMA_CPP="$2"; shift 2 ;;
    --llama-bench) LLAMA_BENCH="$2"; shift 2 ;;
    --gguf) GGUF="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# If a llama.cpp checkout was given, derive the binary and auto-build a
# same-architecture GGUF so the comparison is strictly apples-to-apples.
if [[ -n "${LLAMA_CPP}" ]]; then
  [[ -z "${LLAMA_BENCH}" ]] && LLAMA_BENCH="${LLAMA_CPP}/build/bin/llama-bench"
  if [[ -z "${GGUF}" ]]; then
    GGUF="/tmp/tiny_llama_f32.gguf"
    echo "=== generating same-arch GGUF (${GGUF}) ==="
    "${PY}" "${SCRIPT_DIR}/make_tiny_llama_gguf.py" \
      --llama-cpp "${LLAMA_CPP}" --out "${GGUF}"
    echo
  fi
fi

BENCH_BIN="${BUILD_DIR}/benchmarks/bench_llama_decode_cpu"
if [[ ! -x "${BENCH_BIN}" ]]; then
  echo "error: ${BENCH_BIN} not found. Build it with:" >&2
  echo "  cmake ${BUILD_DIR} -DTESSERACT_BUILD_BENCHMARKS=ON" >&2
  echo "  cmake --build ${BUILD_DIR} --target bench_llama_decode_cpu" >&2
  exit 1
fi

echo "=== Tesseract (CPU, ${THREADS} threads) ==="
TS_LINE="$("${BENCH_BIN}" --prompt "${PROMPT}" --gen "${GEN}")"
echo "${TS_LINE}"
TS_DECODE="$(sed -n 's/.*decode_tok_s=\([0-9.]*\).*/\1/p' <<<"${TS_LINE}")"
TS_PREFILL="$(sed -n 's/.*prefill_tok_s=\([0-9.]*\).*/\1/p' <<<"${TS_LINE}")"

echo
echo "=== llama.cpp (CPU) ==="
LC_DECODE="(not run)"
LC_PREFILL="(not run)"
if [[ -n "${LLAMA_BENCH}" && -n "${GGUF}" && -x "${LLAMA_BENCH}" ]]; then
  # llama-bench reports pp (prefill, prompt-processing) and tg (decode,
  # token-generation) tok/s. -p = prompt tokens, -n = generated tokens.
  echo "running: ${LLAMA_BENCH} -m ${GGUF} -p ${PROMPT} -n ${GEN} -t ${THREADS}"
  LC_OUT="$("${LLAMA_BENCH}" -m "${GGUF}" -p "${PROMPT}" -n "${GEN}" -t "${THREADS}" 2>/dev/null || true)"
  echo "${LC_OUT}"
  # Extract the t/s value (column before '±') for the pp* and tg* rows.
  LC_PREFILL="$(sed -n "s/.*| *pp${PROMPT} *| *\([0-9.]*\).*/\1/p" <<<"${LC_OUT}" | head -1)"
  LC_DECODE="$(sed -n "s/.*| *tg${GEN} *| *\([0-9.]*\).*/\1/p" <<<"${LC_OUT}" | head -1)"
  [[ -z "${LC_PREFILL}" ]] && LC_PREFILL="(parse failed)"
  [[ -z "${LC_DECODE}" ]] && LC_DECODE="(parse failed)"
else
  cat <<EOF
llama.cpp not provided. To complete the comparison:
  1. Build llama.cpp:  git clone https://github.com/ggml-org/llama.cpp
                       cmake -B build && cmake --build build -j --target llama-bench
  2. Re-run (auto-builds a same-arch GGUF via make_tiny_llama_gguf.py):
       scripts/bench_vs_llama_cpp.sh --llama-cpp /path/to/llama.cpp
     or pass an explicit GGUF + binary:
       scripts/bench_vs_llama_cpp.sh \\
         --llama-bench llama.cpp/build/bin/llama-bench --gguf model.gguf
EOF
fi

echo
echo "=== Comparison (CPU, ${THREADS} threads, prompt=${PROMPT} gen=${GEN}) ==="
printf '%-12s | %-16s | %-16s\n' "runtime" "prefill_tok_s" "decode_tok_s"
printf '%-12s-+-%-16s-+-%-16s\n' "------------" "----------------" "----------------"
printf '%-12s | %-16s | %-16s\n' "tesseract" "${TS_PREFILL}" "${TS_DECODE}"
printf '%-12s | %-16s | %-16s\n' "llama.cpp" "${LC_PREFILL}" "${LC_DECODE}"
echo
echo "Record the filled-in table in docs/design/external-benchmark.md."

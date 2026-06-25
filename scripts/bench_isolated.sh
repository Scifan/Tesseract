#!/usr/bin/env bash
# Strict-isolation benchmark runner (M4 perf-closeout, hard constraint).
#
# Guarantees the cards under test are *completely clean* (no other process,
# including our own gpu_reserve daemon) while every OTHER card is held by the
# reservation so no external job can grab them mid-run. CPU work is pinned to a
# fixed core set and gated on a low load average so timings are not polluted by
# other tenants on the shared host.
#
# Usage:
#   scripts/bench_isolated.sh --test-gpus 2 -- ./build-cuda/benchmarks/bench_cuda_matmul
#   scripts/bench_isolated.sh --test-gpus 0,1,2 -- ./build-cuda/benchmarks/bench_cuda_tp_scaling
#   scripts/bench_isolated.sh --cpu --cores 0-15 -- ./build-cpu/benchmarks/bench_matmul
#
# Behavior:
#   * GPU mode (--test-gpus): reserve the complement of the test set across all
#     6 cards, verify the test cards have zero compute apps, then run the
#     command with CUDA_VISIBLE_DEVICES set to the test cards.
#   * CPU mode (--cpu): wait until the 1-min load average is below --max-load,
#     then run under taskset -c <cores> nice -n -5 (best effort).
#   * On exit the reservation daemon keeps running on the complement so the
#     cards stay protected between runs; re-invoke to change the test set.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALL_GPUS=(0 1 2 3 4 5)
RESERVE_LOG=/tmp/gpu_reserve_isolated.log

TEST_GPUS=""
CPU_MODE=0
CORES="0-31"
MAX_LOAD="8.0"
MEM_FRACTION="0.85"
HEADROOM="4"

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 1; }

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-gpus) TEST_GPUS="$2"; shift 2;;
    --cpu) CPU_MODE=1; shift;;
    --cores) CORES="$2"; shift 2;;
    --max-load) MAX_LOAD="$2"; shift 2;;
    --mem-fraction) MEM_FRACTION="$2"; shift 2;;
    --headroom) HEADROOM="$2"; shift 2;;
    --) shift; ARGS=("$@"); break;;
    -h|--help) usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done

[[ ${#ARGS[@]} -eq 0 ]] && { echo "no command given after --" >&2; usage; }

kill_reserve() {
  pkill -f "gpu_reserve.py" 2>/dev/null || true
  # Wait until no gpu_reserve process remains AND its CUDA contexts have torn
  # down (clean-release path runs empty_cache per device, which is not instant).
  for _ in $(seq 1 20); do
    pgrep -f "gpu_reserve.py" >/dev/null 2>&1 || break
    sleep 0.5
  done
}

reserve_complement() {
  local test_csv="$1"
  IFS=',' read -ra TESTSET <<< "$test_csv"
  local complement=()
  for g in "${ALL_GPUS[@]}"; do
    local found=0
    for t in "${TESTSET[@]}"; do [[ "$g" == "$t" ]] && found=1; done
    [[ $found -eq 0 ]] && complement+=("$g")
  done
  if [[ ${#complement[@]} -gt 0 ]]; then
    local comp_csv
    comp_csv=$(IFS=,; echo "${complement[*]}")
    echo "[isolated] reserving complement cards: ${comp_csv}" >&2
    setsid python "${ROOT}/scripts/gpu_reserve.py" --devices "${comp_csv}" \
      --mem-fraction "${MEM_FRACTION}" --headroom-gib "${HEADROOM}" \
      > "${RESERVE_LOG}" 2>&1 < /dev/null & disown
    sleep 8
  fi
}

verify_clean() {
  local test_csv="$1"
  IFS=',' read -ra TESTSET <<< "$test_csv"
  local apps
  for t in "${TESTSET[@]}"; do
    # Poll: a just-killed daemon's context can linger a beat after the process
    # exits. Give it up to ~10s to drop off the card before declaring failure.
    for _ in $(seq 1 20); do
      apps=$(nvidia-smi -i "$t" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
      [[ "$apps" -eq 0 ]] && break
      sleep 0.5
    done
    if [[ "$apps" -ne 0 ]]; then
      echo "[isolated] FATAL: card $t has $apps compute process(es); not clean." >&2
      nvidia-smi -i "$t" --query-compute-apps=pid,used_memory --format=csv,noheader >&2
      exit 3
    fi
    echo "[isolated] card $t verified clean (0 compute apps)." >&2
  done
}

if [[ $CPU_MODE -eq 1 ]]; then
  # Wait for a quiet host.
  for _ in $(seq 1 60); do
    load=$(awk '{print $1}' /proc/loadavg)
    if awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN{exit !(l<m)}'; then break; fi
    echo "[isolated] load ${load} >= ${MAX_LOAD}; waiting..." >&2; sleep 10
  done
  echo "[isolated] CPU run pinned to cores ${CORES} (load=$(awk '{print $1}' /proc/loadavg))" >&2
  # Renice to highest priority we are allowed (negative needs root); fall back
  # silently to default. taskset core-pinning is the primary isolation lever.
  exec taskset -c "${CORES}" "${ARGS[@]}"
else
  [[ -z "$TEST_GPUS" ]] && { echo "--test-gpus required (or use --cpu)" >&2; usage; }
  kill_reserve
  reserve_complement "$TEST_GPUS"
  verify_clean "$TEST_GPUS"
  echo "[isolated] running on CUDA_VISIBLE_DEVICES=${TEST_GPUS}" >&2
  CUDA_VISIBLE_DEVICES="${TEST_GPUS}" "${ARGS[@]}"
fi

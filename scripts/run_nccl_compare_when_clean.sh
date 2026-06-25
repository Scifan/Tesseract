#!/usr/bin/env bash
# Opportunistic, isolation-respecting NCCL TP head-to-head runner.
#
# Polls until a set of >=2 GPUs is free of ANY foreign tenant (a process owned
# by another user), then — and only then — releases our own gpu_reserve on
# those cards, re-verifies they are clean, and runs the Tesseract vs PyTorch
# NCCL tensor-parallel comparison on exactly those cards. Foreign jobs are
# never touched. Results are appended to the log and the results doc.
#
# Usage: scripts/run_nccl_compare_when_clean.sh [poll_seconds] [need_gpus]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLL="${1:-90}"
NEED="${2:-2}"
ME="$(id -un)"
NCCL_LIB=/home/data/qfshi/Brain/est_env/lib/python3.13/site-packages/nvidia/nccl/lib
TR=/home/data/qfshi/Brain/est_env/bin/torchrun
LOG=/tmp/nccl_compare.log
ALL=(0 1 2 3 4 5)

export LD_LIBRARY_PATH="${NCCL_LIB}:${LD_LIBRARY_PATH:-}"
unset CUDA_VISIBLE_DEVICES

# A card is "foreign-free" if every compute-app PID on it is owned by $ME.
card_foreign_free() {
  local c="$1" pid owner
  for pid in $(nvidia-smi -i "$c" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
    [[ -n "$owner" && "$owner" != "$ME" ]] && return 1
  done
  return 0
}

echo "[wait] polling every ${POLL}s for >=${NEED} foreign-free GPUs..." | tee -a "$LOG"
while true; do
  free=()
  for c in "${ALL[@]}"; do card_foreign_free "$c" && free+=("$c"); done
  if [[ ${#free[@]} -ge $NEED ]]; then
    test_csv="$(IFS=,; echo "${free[*]:0:3}")"   # use up to 3 clean cards
    echo "[wait] clean window: cards ${test_csv} — running comparison" | tee -a "$LOG"
    {
      echo "===== $(date -Is)  cards=${test_csv} ====="
      bash "${ROOT}/scripts/bench_isolated.sh" --test-gpus "${test_csv}" -- bash -c '
        B='"${ROOT}"'/build-nccl/benchmarks/bench_cuda_nccl_tp
        n=$(echo '"${test_csv}"' | awk -F, "{print NF}")
        echo "### Tesseract"
        $B --world 2 --d 4096 --dff 12288 --iters 50 2>/dev/null | grep "\[bench\]"
        [ "$n" -ge 3 ] && $B --world 3 --d 4096 --dff 12288 --iters 50 2>/dev/null | grep "\[bench\]"
        echo "### PyTorch NCCL"
        '"${TR}"' --standalone --nproc_per_node=2 '"${ROOT}"'/bench/external/torch_nccl_tp.py --d 4096 --dff 12288 --M 512 --iters 50 2>/dev/null | grep "\[bench\]"
        [ "$n" -ge 3 ] && '"${TR}"' --standalone --nproc_per_node=3 '"${ROOT}"'/bench/external/torch_nccl_tp.py --d 4096 --dff 12288 --M 512 --iters 50 2>/dev/null | grep "\[bench\]"
      '
    } 2>&1 | tee -a "$LOG"
    echo "[wait] DONE — results in $LOG" | tee -a "$LOG"
    exit 0
  fi
  sleep "$POLL"
done

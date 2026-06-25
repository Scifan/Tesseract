#!/usr/bin/env python3
"""Reserve GPUs by holding VRAM (and optionally light compute) so other users
do not schedule onto the cards we depend on.

Default target is devices 0,1,2. For each target the script allocates a large
CUDA tensor sized to occupy `--mem-fraction` of the card while leaving
`--headroom-gib` free for our own benchmarks, then parks. With `--compute` it
also runs a low-duty keepalive matmul so the cards show non-zero utilization.

The reservation is released cleanly on SIGINT / SIGTERM (Ctrl-C or `kill`), so
freeing the cards is just stopping the process.

Examples
--------
  # Hold 0,1,2 at ~90% VRAM, leave 6 GiB headroom, memory-only:
  nohup python scripts/gpu_reserve.py > /tmp/gpu_reserve.log 2>&1 &

  # Hold with a light compute keepalive (shows util):
  python scripts/gpu_reserve.py --compute

  # Temporarily free headroom to run our own GPU benchmark, e.g. 0.4 fraction:
  python scripts/gpu_reserve.py --mem-fraction 0.4
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

try:
    import torch
except Exception as exc:  # pragma: no cover - environment guard
    sys.stderr.write(f"gpu_reserve: PyTorch import failed: {exc}\n")
    sys.exit(1)


_RUNNING = True


def _handle_signal(signum, _frame):
    global _RUNNING
    _RUNNING = False
    sys.stderr.write(f"\ngpu_reserve: received signal {signum}; releasing...\n")


def parse_devices(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def reserve_one(dev: int, mem_fraction: float, headroom_gib: float) -> torch.Tensor:
    """Allocate a float32 tensor on `dev` occupying the target VRAM and return
    it (the caller keeps the reference alive)."""
    free_b, total_b = torch.cuda.mem_get_info(dev)
    gib = 1024.0 ** 3
    # Target bytes to *hold*: min(fraction*total, total-headroom), clamped to
    # what is currently free (minus a small safety margin for the allocator).
    target_total = min(mem_fraction * total_b, total_b - headroom_gib * gib)
    safety = 256 * 1024 * 1024  # 256 MiB allocator/context slack
    target_hold = max(0.0, min(target_total, free_b - safety))
    n_elems = int(target_hold // 4)
    if n_elems <= 0:
        sys.stderr.write(
            f"gpu_reserve: dev {dev} has no room to reserve "
            f"(free={free_b/gib:.1f} GiB); skipping allocation.\n"
        )
        return torch.empty(0, device=f"cuda:{dev}")
    block = torch.empty(n_elems, dtype=torch.float32, device=f"cuda:{dev}")
    held = block.numel() * block.element_size()
    new_free, _ = torch.cuda.mem_get_info(dev)
    print(
        f"gpu_reserve: dev {dev} held {held/gib:.1f} GiB "
        f"(total {total_b/gib:.1f} GiB, free now {new_free/gib:.1f} GiB)",
        flush=True,
    )
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description="Reserve GPUs by holding VRAM.")
    ap.add_argument("--devices", default="0,1,2",
                    help="comma-separated CUDA device indices (default 0,1,2)")
    ap.add_argument("--mem-fraction", type=float, default=0.9,
                    help="fraction of each card's total VRAM to hold (default 0.9)")
    ap.add_argument("--headroom-gib", type=float, default=6.0,
                    help="GiB to leave free on each card (default 6)")
    ap.add_argument("--compute", action="store_true",
                    help="run a low-duty keepalive matmul to show utilization")
    ap.add_argument("--compute-mib", type=int, default=512,
                    help="per-card matmul working-set size in MiB (default 512)")
    ap.add_argument("--duty", type=float, default=0.1,
                    help="keepalive duty cycle in [0,1] (default 0.1)")
    ap.add_argument("--poll", type=float, default=30.0,
                    help="status/idle poll interval seconds (default 30)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.stderr.write("gpu_reserve: CUDA not available.\n")
        return 1

    devices = parse_devices(args.devices)
    ndev = torch.cuda.device_count()
    for d in devices:
        if d < 0 or d >= ndev:
            sys.stderr.write(
                f"gpu_reserve: device {d} out of range (have {ndev}).\n")
            return 1

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    holds: dict[int, torch.Tensor] = {}
    for d in devices:
        holds[d] = reserve_one(d, args.mem_fraction, args.headroom_gib)

    # Optional keepalive matmul tensors.
    work: dict[int, torch.Tensor] = {}
    if args.compute:
        for d in devices:
            # n*n float32 ~= compute_mib; pick n accordingly.
            n = int(((args.compute_mib * 1024 * 1024) / 4) ** 0.5)
            n = max(256, n)
            work[d] = torch.randn(n, n, device=f"cuda:{d}", dtype=torch.float32)
        print(f"gpu_reserve: keepalive matmul enabled "
              f"(duty={args.duty}, {args.compute_mib} MiB/card)", flush=True)

    print(f"gpu_reserve: holding devices {devices}. "
          f"Stop with Ctrl-C or `kill {_safe_pid()}`.", flush=True)

    period = 2.0  # keepalive cycle length (s)
    while _RUNNING:
        if args.compute and work:
            t0 = time.time()
            for d, w in work.items():
                with torch.cuda.device(d):
                    _ = w @ w
            torch.cuda.synchronize()
            busy = time.time() - t0
            # Sleep to honor the duty cycle.
            idle = max(0.0, (busy / max(args.duty, 1e-3)) - busy)
            time.sleep(min(idle, period))
        else:
            time.sleep(args.poll)

    # Release.
    holds.clear()
    work.clear()
    for d in devices:
        with torch.cuda.device(d):
            torch.cuda.empty_cache()
    print("gpu_reserve: released all reservations.", flush=True)
    return 0


def _safe_pid() -> int:
    import os
    return os.getpid()


if __name__ == "__main__":
    raise SystemExit(main())

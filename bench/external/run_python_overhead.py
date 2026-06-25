#!/usr/bin/env python3
"""Run both sides of the Python<->C++ overhead measurement and diff them.

Native: benchmarks/bench_python_overhead (C++). Python: python_overhead.py
through the pybind11 bindings. Reports per-call overhead (Python - C++) and
overhead% = overhead / C++ time for each op. The point: the frontend is a thin
shim — a fixed ~µs/call crossing cost that is large only for trivially small
ops and negligible for real compute.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_results(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        if line.startswith("RESULT,"):
            _, name, us = line.split(",")
            out[name] = float(us)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build-cpu")
    args = ap.parse_args()
    build = (ROOT / args.build_dir).resolve()

    native_bin = build / "benchmarks" / "bench_python_overhead"
    native = parse_results(
        subprocess.run([str(native_bin)], capture_output=True, text=True,
                       check=True).stdout)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(build / "python") + os.pathsep + env.get(
        "PYTHONPATH", "")
    py = parse_results(
        subprocess.run([sys.executable,
                        str(Path(__file__).with_name("python_overhead.py"))],
                       capture_output=True, text=True, check=True,
                       env=env).stdout)

    print(f"{'op':<22}{'C++ us':>12}{'Python us':>12}"
          f"{'overhead us':>14}{'overhead %':>12}")
    print("-" * 72)
    for name in native:
        if name not in py:
            continue
        c = native[name]
        p = py[name]
        ov = p - c
        pct = ov / c * 100.0 if c > 0 else float("nan")
        print(f"{name:<22}{c:>12.3f}{p:>12.3f}{ov:>14.3f}{pct:>11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

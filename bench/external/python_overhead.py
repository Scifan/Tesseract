#!/usr/bin/env python3
"""Python side of the Python<->C++ frontend-overhead measurement (B-041).

Calls the exact same ops as benchmarks/bench_python_overhead.cpp through the
pybind11 bindings, with matching iteration counts. Prints RESULT lines; the
wrapper (run_python_overhead.py) diffs them against the native run to compute
the per-call overhead the Python frontend adds.

Requires PYTHONPATH to include the build's python/ dir (where tesseract._core
lives), e.g. PYTHONPATH=build-cpu/python.
"""
from __future__ import annotations

import time

import numpy as np

import tesseract as ts


def us_per_call(iters: int, fn) -> float:
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> int:
    for n, iters in [(64, 5000), (256, 2000), (512, 1000), (1024, 300)]:
        a = ts.Tensor(np.ones((n, n), dtype="float32"))
        b = ts.Tensor(np.ones((n, n), dtype="float32"))
        us = us_per_call(iters, lambda: ts.ops.matmul(a, b))
        print(f"RESULT,matmul_{n},{us:.4f}")

    seq = ts.nn.Sequential([ts.nn.Linear(512, 512), ts.nn.ReLU(),
                            ts.nn.Linear(512, 512)])
    seq.eval()
    x = ts.Tensor(np.ones((64, 512), dtype="float32"))
    us = us_per_call(1000, lambda: seq(x))
    print(f"RESULT,mlp_fwd_b64_d512,{us:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

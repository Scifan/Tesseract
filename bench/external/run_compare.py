#!/usr/bin/env python3
"""Run the PyTorch baselines and the matched Tesseract CUDA benchmarks back to
back on the same GPU, save raw outputs + parsed JSON under results/, and print a
combined comparison so the external-benchmark table can be filled honestly.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2 python bench/external/run_compare.py \
      --build-dir build-cuda --device 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = HERE / "results"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def torch_json(bench: str, device: int) -> dict:
    p = run([sys.executable, str(HERE / "torch_baseline.py"), bench,
             "--device", str(device)])
    if p.returncode != 0:
        return {"bench": bench, "error": p.stderr.strip()[-500:]}
    return json.loads(p.stdout)


def run_tesseract(build_dir: Path, name: str) -> str:
    binpath = build_dir / "benchmarks" / name
    if not binpath.exists():
        return f"(missing: {binpath})"
    p = run([str(binpath)])
    return p.stdout + ("\n" + p.stderr if p.stderr else "")


def parse_matmul_ts(out: str) -> dict:
    """Pull (dtype,N)->ours_TF from bench_cuda_matmul's table."""
    res = {}
    for line in out.splitlines():
        m = re.match(r"\s*(fp32|fp16)\s+(\d+)\s+\|", line)
        if not m:
            continue
        cols = re.findall(r"[-+]?\d+\.\d+", line)
        # columns: raw_min bridge_min ours_min raw_TF bridge_TF ours_TF ratio dispatch
        if len(cols) >= 6:
            res[(m.group(1), int(m.group(2)))] = float(cols[5])  # ours_TF
    return res


def parse_decode_ts(out: str) -> dict:
    """Pull variant->step_us from bench_cuda_llama_decode's table."""
    res = {}
    for line in out.splitlines():
        m = re.match(r"\s*(FP32|INT8|INT4G[^|]*)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            res[m.group(1).strip()] = float(m.group(2))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build-cuda")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    build_dir = (ROOT / args.build_dir).resolve()

    combined = {"device": args.device, "torch": {}, "tesseract_raw": {}}

    # --- torch baselines ---
    for b in ["matmul", "decode", "attention"]:
        combined["torch"][b] = torch_json(b, args.device)

    # --- tesseract benches ---
    ts_benches = ["bench_cuda_matmul", "bench_cuda_llama_decode",
                  "bench_cuda_fused_attention", "bench_cuda_transformer_block"]
    for b in ts_benches:
        out = run_tesseract(build_dir, b)
        combined["tesseract_raw"][b] = out
        (RESULTS / f"{b}.log").write_text(out)

    (RESULTS / "combined.json").write_text(json.dumps(combined, indent=2))

    # --- print headline comparison ---
    print("=" * 64)
    print("MATMUL TFLOPS (higher is better) — Tesseract ops::matmul vs torch.mm")
    ts_mm = parse_matmul_ts(combined["tesseract_raw"]["bench_cuda_matmul"])
    th_mm = {(r["dtype"], r["n"]): r["tflops"]
             for r in combined["torch"]["matmul"].get("results", [])}
    print(f"{'dtype':5} {'N':6} {'tesseract_TF':>14} {'torch_TF':>12} {'ratio':>8}")
    for key in sorted(th_mm, key=lambda k: (k[0], k[1])):
        ts = ts_mm.get(key)
        th = th_mm[key]
        ratio = (ts / th) if (ts and th) else float("nan")
        ts_s = f"{ts:.2f}" if ts else "n/a"
        print(f"{key[0]:5} {key[1]:6} {ts_s:>14} {th:12.2f} {ratio:8.3f}")

    print("=" * 64)
    print("DECODE step_us (lower is better) — one Llama-7B block step")
    ts_dec = parse_decode_ts(combined["tesseract_raw"]["bench_cuda_llama_decode"])
    th_dec = {r["dtype"]: r["step_us"]
              for r in combined["torch"]["decode"].get("results", [])}
    print(f"  tesseract: {ts_dec}")
    print(f"  torch    : {th_dec}")
    print("=" * 64)
    print(f"Saved raw logs + combined.json under {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

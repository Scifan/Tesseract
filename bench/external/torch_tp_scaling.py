#!/usr/bin/env python3
"""PyTorch multi-GPU TP-scaling reference for bench_cuda_tp_scaling.

Same config (d_model=4096, d_ff=12288, 4096 tokens), same Megatron SwiGLU MLP
modeled as a sum of d_ff/N shards each on its own card, same cross-device
all-reduce (.to('cuda:0') + sum). Single-process, one CUDA stream per device
(PyTorch default) — mirrors the Tesseract benchmark's structure so the per-GPU
memory and aggregate-throughput scaling are apples-to-apples.

Run with CUDA_VISIBLE_DEVICES=0,1,2.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNShard(nn.Module):
    def __init__(self, d_model: int, d_ff_shard: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff_shard, bias=False)
        self.up = nn.Linear(d_model, d_ff_shard, bias=False)
        self.down = nn.Linear(d_ff_shard, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


@torch.no_grad()
def run(world: int, d_model: int, d_ff: int, T: int) -> tuple[float, float, float]:
    d_ff_shard = d_ff // world
    shards = [FFNShard(d_model, d_ff_shard).eval().to(f"cuda:{r}")
              for r in range(world)]
    x = [torch.zeros(T, d_model, device=f"cuda:{r}") for r in range(world)]
    per_gpu_bytes = sum(p.numel() * p.element_size()
                        for p in shards[0].parameters())

    def step():
        partials = [shards[r](x[r]) for r in range(world)]
        acc = partials[0]
        for r in range(1, world):
            acc = acc + partials[r].to("cuda:0")
        return acc

    for _ in range(5):
        step()
    for r in range(world):
        torch.cuda.synchronize(r)

    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    for r in range(world):
        torch.cuda.synchronize(r)
    step_us = (time.perf_counter() - t0) / iters * 1e6
    return step_us, T / (step_us * 1e-6), per_gpu_bytes / (1024 ** 2)


def main() -> int:
    d_model, d_ff, T = 4096, 12288, 4096
    n = torch.cuda.device_count()
    max_world = min(n, 3)
    print(f"[torch_tp_scaling] cards={max_world} d_model={d_model} "
          f"d_ff={d_ff} tokens={T}")
    results = []
    for world in range(1, max_world + 1):
        if d_ff % world != 0:
            continue
        step_us, tok_s, mb = run(world, d_model, d_ff, T)
        results.append((world, step_us, tok_s, mb))
        print(f"  TP={world} : step={step_us:8.1f} us  "
              f"throughput={tok_s:9.0f} tok/s  per-GPU weight={mb:6.1f} MB")
    if results:
        base = results[0][1]
        base_mb = results[0][3]
        print("  scaling vs TP=1:")
        for world, step_us, _, mb in results:
            print(f"    TP={world} : {base/step_us:.2f}x latency, "
                  f"{base_mb/mb:.2f}x per-GPU memory reduction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

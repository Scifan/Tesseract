#!/usr/bin/env python3
"""Real multi-process NCCL tensor-parallel reference for Tesseract's
`bench_cuda_nccl_tp` (M4 Phase 7).

Unlike `torch_tp_scaling.py` (single process, `.to('cuda:0')+sum` faking the
collective), this uses **true** `torch.distributed` NCCL: one process per GPU,
`dist.all_reduce(SUM)` over the wire — the exact analogue of Tesseract's
`NcclCommBackend`. Same Megatron SwiGLU MLP (gate/up column-parallel kept
sharded, down row-parallel -> one all-reduce), same shapes, same metrics
(per-GPU weight MB, forward+allreduce ms, isolated allreduce ms).

Launch (isolated cards only):
    torchrun --standalone --nproc_per_node=2 torch_nccl_tp.py --d 4096 --dff 12288 --M 512
    torchrun --standalone --nproc_per_node=3 torch_nccl_tp.py --d 4096 --dff 12288 --M 512
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--dff", type=int, default=12288)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    assert args.dff % world == 0, "dff must divide world"
    p = args.dff // world
    dev = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(1234 + rank)

    # Shard weights: gate/up [d, p] (column-parallel), down [p, d] (row-parallel).
    gate = (torch.randn(args.d, p, device=dev) * 0.02)
    up = (torch.randn(args.d, p, device=dev) * 0.02)
    down = (torch.randn(p, args.d, device=dev) * 0.02)
    x = torch.rand(args.M, args.d, device=dev)

    @torch.no_grad()
    def forward_local():
        g = x @ gate
        u = x @ up
        h = F.silu(g) * u
        return h @ down  # [M, d] partial

    # Warm up + NCCL channel setup.
    for _ in range(5):
        y = forward_local()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
    dist.barrier()
    torch.cuda.synchronize()

    # Timed: forward + all-reduce.
    t0 = time.perf_counter()
    for _ in range(args.iters):
        y = forward_local()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1e3 / args.iters

    # Isolated all-reduce.
    probe = forward_local()
    torch.cuda.synchronize()
    ta = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    ar_ms = (time.perf_counter() - ta) * 1e3 / args.iters

    shard_mb = (3 * args.d * p) * 4 / 1e6
    dense_mb = (3 * args.d * args.dff) * 4 / 1e6
    if rank == 0:
        print(
            f"[bench] pytorch nccl_tp  world={world} "
            f"cfg=M{args.M}_d{args.d}_dff{args.dff}  "
            f"shard_MB={shard_mb:.1f} dense_MB={dense_mb:.1f}  "
            f"fwd_ms={fwd_ms:.3f} allreduce_ms={ar_ms:.3f}"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

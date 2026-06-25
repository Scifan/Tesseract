#!/usr/bin/env python3
"""PyTorch baselines matched to Tesseract's CUDA benchmarks, for external
apples-to-apples comparison on the same GPU.

Each subcommand emits a JSON object on stdout with the measured numbers so a
driver (run_compare.py) can line them up against the Tesseract bench output.

Subcommands
-----------
  matmul   square GEMM TFLOPS sweep (matches benchmarks/bench_cuda_matmul.cpp:
           N in {512,1024,2048,4096,8192}, fp32 [TF32] + fp16).
  decode   one Llama decode step (MHA forward_step w/ KV prefix + SwiGLU FFN),
           matches benchmarks/bench_cuda_llama_decode.cpp shapes
           (d_model=4096, H=32, Dh=128, d_ff=11008, S_k=129, B=1), fp32 + fp16.
  attention scaled-dot-product attention, eager vs torch fused SDPA.

All timing uses CUDA events, a warmup, and best-of-N (min) to match the
Tesseract harness's min-over-samples reporting.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F


def _device(dev: int) -> torch.device:
    if not torch.cuda.is_available():
        sys.stderr.write("torch_baseline: CUDA not available\n")
        sys.exit(2)
    return torch.device(f"cuda:{dev}")


def time_us(fn, *, warmup: int = 10, iters: int = 50) -> float:
    """Return the min wall time (microseconds) of `fn` over `iters` runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0)  # ms -> us
    return best


def bench_matmul(dev: int) -> dict:
    d = _device(dev)
    # Match bench_cuda_matmul: TF32 on for fp32 (Ada), fp16 tensor cores.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sizes = [512, 1024, 2048, 4096, 8192]
    out = []
    for dtype, name in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
        for n in sizes:
            a = torch.randn(n, n, device=d, dtype=dtype)
            b = torch.randn(n, n, device=d, dtype=dtype)
            t_us = time_us(lambda: torch.mm(a, b))
            tflops = 2.0 * n * n * n / (t_us * 1e-6) / 1e12
            out.append({"dtype": name, "n": n, "min_us": t_us,
                        "tflops": tflops})
            del a, b
            torch.cuda.empty_cache()
    return {"bench": "matmul", "framework": "pytorch",
            "torch_version": torch.__version__, "results": out}


class _LlamaBlock(torch.nn.Module):
    """One Llama-style block (no biases): RMSNorm-free Linear-heavy decode path
    matching bench_cuda_llama_decode (which omits residual/norm to isolate the
    matmul-heavy path)."""

    def __init__(self, d_model, n_heads, d_ff, dtype, dev):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        lin = lambda i, o: torch.nn.Linear(i, o, bias=False).to(dev, dtype)
        self.wq = lin(d_model, d_model)
        self.wk = lin(d_model, d_model)
        self.wv = lin(d_model, d_model)
        self.wo = lin(d_model, d_model)
        self.gate = lin(d_model, d_ff)
        self.up = lin(d_model, d_ff)
        self.down = lin(d_ff, d_model)

    def decode_step(self, x_step, k_cache, v_cache, pos):
        # x_step: [B,1,D]. k/v_cache: [B,H,MAX,Dh] pre-filled to `pos`.
        b = x_step.shape[0]
        q = self.wq(x_step).view(b, 1, self.h, self.dh).transpose(1, 2)
        k = self.wk(x_step).view(b, 1, self.h, self.dh).transpose(1, 2)
        v = self.wv(x_step).view(b, 1, self.h, self.dh).transpose(1, 2)
        k_cache[:, :, pos:pos + 1, :] = k
        v_cache[:, :, pos:pos + 1, :] = v
        kk = k_cache[:, :, :pos + 1, :]
        vv = v_cache[:, :, :pos + 1, :]
        scale = 1.0 / (self.dh ** 0.5)
        scores = torch.matmul(q, kk.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, vv).transpose(1, 2).reshape(b, 1, -1)
        h = self.wo(ctx)
        g = F.silu(self.gate(h)) * self.up(h)
        y = self.down(g)
        return y


def bench_decode(dev: int) -> dict:
    d = _device(dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    d_model, H, d_ff = 4096, 32, 11008
    Dh = d_model // H
    MAX, pos, B = 256, 128, 1
    out = []
    for dtype, name in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
        block = _LlamaBlock(d_model, H, d_ff, dtype, d).eval()
        x = torch.zeros(B, 1, d_model, device=d, dtype=dtype)
        kc = torch.zeros(B, H, MAX, Dh, device=d, dtype=dtype)
        vc = torch.zeros(B, H, MAX, Dh, device=d, dtype=dtype)
        with torch.no_grad():
            t_us = time_us(lambda: block.decode_step(x, kc, vc, pos))
        out.append({"dtype": name, "step_us": t_us})
        del block, x, kc, vc
        torch.cuda.empty_cache()
    return {"bench": "decode", "framework": "pytorch",
            "config": {"d_model": d_model, "H": H, "Dh": Dh, "d_ff": d_ff,
                       "S_k": pos + 1, "B": B},
            "results": out}


def bench_attention(dev: int) -> dict:
    d = _device(dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    # Decode shapes match bench_cuda_fused_attention's (B,H,Sq,Sk,D); fp16.
    decode_shapes = [(1, 32, 1, 2048, 128), (1, 32, 1, 4096, 128),
                     (4, 32, 1, 2048, 128), (8, 32, 1, 2048, 128)]
    out = []
    for (B, H, Sq, Sk, Dh) in decode_shapes:
        q = torch.randn(B, H, Sq, Dh, device=d, dtype=torch.float16)
        k = torch.randn(B, H, Sk, Dh, device=d, dtype=torch.float16)
        v = torch.randn(B, H, Sk, Dh, device=d, dtype=torch.float16)
        scale = 1.0 / (Dh ** 0.5)

        def eager():
            s = torch.matmul(q, k.transpose(-1, -2)) * scale
            p = torch.softmax(s, dim=-1)
            return torch.matmul(p, v)

        def sdpa():
            return F.scaled_dot_product_attention(q, k, v)

        with torch.no_grad():
            t_eager = time_us(eager)
            t_sdpa = time_us(sdpa)
        out.append({"shape": [B, H, Sq, Sk, Dh], "dtype": "fp16",
                    "eager_us": t_eager, "sdpa_us": t_sdpa})
        del q, k, v
        torch.cuda.empty_cache()
    return {"bench": "attention_decode", "framework": "pytorch",
            "results": out}


def bench_matmul_fair(dev: int) -> dict:
    """fp32 GEMM with TF32 OFF (true FP32) — apples-to-apples vs Tesseract's
    CUBLAS_COMPUTE_32F path."""
    d = _device(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = []
    for n in [1024, 2048, 4096, 8192]:
        a = torch.randn(n, n, device=d, dtype=torch.float32)
        b = torch.randn(n, n, device=d, dtype=torch.float32)
        t_us = time_us(lambda: torch.mm(a, b))
        out.append({"dtype": "fp32_true", "n": n, "min_us": t_us,
                    "tflops": 2.0 * n * n * n / (t_us * 1e-6) / 1e12})
        del a, b
        torch.cuda.empty_cache()
    return {"bench": "matmul_fair", "framework": "pytorch", "results": out}


def bench_moe(dev: int) -> dict:
    """Mixtral-style SwiGLU MoE in eager PyTorch — the same per-expert loop
    HF Transformers uses (boolean-mask each expert's tokens, run its FFN,
    scatter-add the gate-weighted result). FP32, TF32 tensor cores on, to
    match Tesseract's CUBLAS_COMPUTE_32F fused path. Reported as the absolute
    full-MoE-layer latency for the head-to-head vs bench_cuda_moe."""
    d = _device(dev)
    # True FP32 (TF32 off) to match Tesseract's CUBLAS_COMPUTE_32F fused path
    # — apples-to-apples same-precision comparison.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # (T, D, dff, E, k) matching bench_cuda_moe.
    shapes = [(512, 1024, 2048, 8, 2), (2048, 1024, 2048, 8, 2),
              (4096, 1024, 4096, 8, 2), (4096, 1024, 2048, 16, 1)]
    out = []
    for (T, D, dff, E, k) in shapes:
        x = torch.randn(T, D, device=d, dtype=torch.float32)
        gate = torch.randn(D, E, device=d, dtype=torch.float32)
        Wg = torch.randn(E, dff, D, device=d, dtype=torch.float32) * 0.02
        Wu = torch.randn(E, dff, D, device=d, dtype=torch.float32) * 0.02
        Wd = torch.randn(E, D, dff, device=d, dtype=torch.float32) * 0.02

        def moe():
            logits = x @ gate                      # [T, E]
            probs = torch.softmax(logits, dim=-1)
            tw, ti = torch.topk(probs, k, dim=-1)  # [T, k]
            tw = tw / tw.sum(dim=-1, keepdim=True)
            y = torch.zeros_like(x)
            for e in range(E):
                # rows (token, slot) routed to expert e
                hit = (ti == e)                    # [T, k] bool
                tok = hit.any(dim=-1)              # [T] bool
                if not bool(tok.any()):
                    continue
                xe = x[tok]                        # [n_e, D]
                h = torch.nn.functional.silu(xe @ Wg[e].t()) * (xe @ Wu[e].t())
                ye = h @ Wd[e].t()                 # [n_e, D]
                # gate weight for those tokens on expert e
                w = (tw * hit).sum(dim=-1)[tok].unsqueeze(-1)
                y[tok] += w * ye
            return y

        with torch.no_grad():
            t_us = time_us(moe)
        out.append({"shape": [T, D, dff, E, k], "dtype": "fp32", "moe_us": t_us})
        del x, gate, Wg, Wu, Wd
        torch.cuda.empty_cache()
    return {"bench": "moe", "framework": "pytorch", "results": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["matmul", "matmul_fair", "decode",
                                      "attention", "moe"])
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    fn = {"matmul": bench_matmul, "matmul_fair": bench_matmul_fair,
          "decode": bench_decode, "attention": bench_attention,
          "moe": bench_moe}[args.bench]
    print(json.dumps(fn(args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

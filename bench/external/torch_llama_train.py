#!/usr/bin/env python3
"""PyTorch Llama training-parity baseline for examples/llama_train.cpp (B-042).

Same config (vocab=64, hidden=64, 2 layers, 4 heads, ffn=128), same optimizer
(Adam, lr=3e-3), same task (memorize a fixed [B,S] batch via next-token
cross-entropy). Architecture mirrors Tesseract's LlamaModel: RMSNorm + RoPE MHA
(causal) + SwiGLU FFN + final norm + untied lm_head.

Init and the exact token batch differ across frameworks (RNGs differ), so this
checks *behavioral* parity: same config + optimizer must drive a fixed-batch
loss to ~0 on a similar trajectory. Prints "step,loss" lines.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, cos, sin):
    # x: [B,H,S,Dh] interleaved-pair rotation matching Tesseract.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    c = cos[..., 0::2]
    s = sin[..., 0::2]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


class Block(nn.Module):
    def __init__(self, d, h, dff):
        super().__init__()
        self.h, self.dh = h, d // h
        self.n1 = nn.Parameter(torch.ones(d))
        self.n2 = nn.Parameter(torch.ones(d))
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.wg = nn.Linear(d, dff, bias=False)
        self.wu = nn.Linear(d, dff, bias=False)
        self.wd = nn.Linear(dff, d, bias=False)

    def forward(self, x, cos, sin):
        B, S, D = x.shape
        h = rms_norm(x, self.n1)
        q = self.wq(h).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.wk(h).view(B, S, self.h, self.dh).transpose(1, 2)
        v = self.wv(h).view(B, S, self.h, self.dh).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, S, D)
        x = x + self.wo(o)
        h = rms_norm(x, self.n2)
        x = x + self.wd(F.silu(self.wg(h)) * self.wu(h))
        return x


class Llama(nn.Module):
    def __init__(self, vocab, d, layers, h, dff, max_seq):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h, dff) for _ in range(layers)])
        self.norm = nn.Parameter(torch.ones(d))
        self.lm_head = nn.Linear(d, vocab, bias=False)
        dh = d // h
        pos = torch.arange(max_seq).float()[:, None]
        idx = torch.arange(0, dh, 2).float()[None, :]
        theta = pos * (10000.0 ** (-idx / (dh / 2)))
        ang = torch.zeros(max_seq, dh)
        ang[:, 0::2] = theta
        ang[:, 1::2] = theta
        self.register_buffer("cos", ang.cos()[None, None])
        self.register_buffer("sin", ang.sin()[None, None])

    def forward(self, tokens):
        S = tokens.shape[1]
        x = self.emb(tokens)
        cos = self.cos[:, :, :S]
        sin = self.sin[:, :, :S]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = rms_norm(x, self.norm)
        return self.lm_head(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.manual_seed(1234)
    dev = torch.device(args.device)

    vocab, d, layers, h, dff = 64, 64, 2, 4, 128
    model = Llama(vocab, d, layers, h, dff, max_seq=128).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    tokens = torch.randint(0, vocab, (args.batch, args.seq), device=dev)
    inp = tokens[:, :-1]
    tgt = tokens[:, 1:].reshape(-1)

    print(f"[torch_llama_train] device={dev} layers={layers} d_model={d} "
          f"batch={args.batch} seq={args.seq} steps={args.steps} lr={args.lr}")
    first = last = None
    for step in range(args.steps):
        opt.zero_grad()
        logits = model(inp).reshape(-1, vocab)
        loss = F.cross_entropy(logits, tgt)
        loss.backward()
        opt.step()
        last = loss.item()
        if step == 0:
            first = last
        if step == 0 or (step + 1) % 10 == 0:
            print(f"  step {step + 1}/{args.steps}  loss={last:.6g}")
    print(f"[torch_llama_train] loss: {first:.6g} -> {last:.6g} "
          f"(drop {first - last:.6g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Upcast a fp16 safetensors checkpoint to fp32 so Tesseract's strict-dtype
loader (`LlamaModel::from_pretrained`, exact dtype match — B-022) can load it
into an fp32 model. vLLM upcasts the original fp16 file on its own via
`--dtype float32`, so this keeps both engines on identical fp32 weights for a
fair same-precision serving comparison.

    python convert_safetensors_fp32.py <in.safetensors> <out.safetensors>
"""
import sys
import torch
from safetensors.torch import load_file, save_file


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sd = load_file(src)
    out = {k: v.to(torch.float32).contiguous() for k, v in sd.items()}
    save_file(out, dst, metadata={"format": "pt"})
    print(f"wrote {dst}: {len(out)} tensors, fp32")


if __name__ == "__main__":
    main()

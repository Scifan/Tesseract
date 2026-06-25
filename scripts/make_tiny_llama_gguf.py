#!/usr/bin/env python3
"""Write a GGUF whose architecture exactly matches the Tesseract CPU decode bench.

Mirrors LlamaConfig in benchmarks/bench_llama_decode_cpu.cpp so the Tesseract vs
llama.cpp comparison (M4 / B-046) is same-architecture and therefore fair:

  vocab=4096 hidden=256 layers=4 heads=8 kv_heads=8 intermediate=688
  max_pos=1024 act=silu (SwiGLU) + RMSNorm + RoPE, FP32.

Rather than going through transformers + convert_hf_to_gguf.py (whose tokenizer
auto-detection rejects a synthetic vocab), this writes the GGUF directly with
gguf-py: random FP32 weights and a trivial 4096-token vocab. Throughput is
weight-/vocab-content-independent, so this is a fair *runtime* comparison.

Usage:
  python3 scripts/make_tiny_llama_gguf.py \
      --llama-cpp /home/data/qfshi/llama.cpp --out /tmp/tiny_llama_f32.gguf
"""
import argparse
import sys

import numpy as np


# Architecture — keep in lock-step with benchmarks/bench_llama_decode_cpu.cpp.
VOCAB = 4096
N_EMBD = 256
N_LAYER = 4
N_HEAD = 8
N_KV = 8
N_FF = 688
CTX = 1024
HEAD_DIM = N_EMBD // N_HEAD  # 32
RMS_EPS = 1e-5
ROPE_BASE = 10000.0


def rnd(*shape: int) -> np.ndarray:
    # Small-variance init; magnitudes are irrelevant to throughput.
    return (np.random.randn(*shape) * 0.02).astype(np.float32)


def main() -> int:
    global VOCAB, N_EMBD, N_LAYER, N_HEAD, N_KV, N_FF, CTX, HEAD_DIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", required=True,
                    help="Path to a llama.cpp checkout (for its gguf-py)")
    ap.add_argument("--out", default="/tmp/tiny_llama_f32.gguf")
    ap.add_argument("--seed", type=int, default=0)
    # Architecture overrides (default: the tiny B-046 model). Pass these to
    # build a real-size model (e.g. TinyLlama-1.1B) for the Phase-6 W8A8
    # head-to-head; throughput is weight-content independent so random
    # weights give a fair *runtime* comparison.
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--embd", type=int, default=N_EMBD)
    ap.add_argument("--layers", type=int, default=N_LAYER)
    ap.add_argument("--heads", type=int, default=N_HEAD)
    ap.add_argument("--kv-heads", type=int, default=N_KV)
    ap.add_argument("--ff", type=int, default=N_FF)
    ap.add_argument("--ctx", type=int, default=CTX)
    args = ap.parse_args()

    VOCAB, N_EMBD, N_LAYER = args.vocab, args.embd, args.layers
    N_HEAD, N_KV, N_FF, CTX = args.heads, args.kv_heads, args.ff, args.ctx
    HEAD_DIM = N_EMBD // N_HEAD

    sys.path.insert(0, f"{args.llama_cpp}/gguf-py")
    import gguf

    np.random.seed(args.seed)
    w = gguf.GGUFWriter(args.out, arch="llama")

    w.add_name("tiny-llama-bench")
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_file_type(0)  # ALL_F32

    # Trivial SPM-style vocab of exactly VOCAB tokens (content irrelevant).
    tokens = [f"<t{i}>".encode("utf-8") for i in range(VOCAB)]
    scores = [0.0] * VOCAB
    toktypes = [gguf.TokenType.NORMAL] * VOCAB
    toktypes[0] = gguf.TokenType.UNKNOWN
    toktypes[1] = gguf.TokenType.CONTROL  # bos
    toktypes[2] = gguf.TokenType.CONTROL  # eos
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(toktypes)
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)

    # Tensors (PyTorch [out, in] convention; ggml reads ne reversed).
    w.add_tensor("token_embd.weight", rnd(VOCAB, N_EMBD))
    for i in range(N_LAYER):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(N_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(N_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"GGUF written: {args.out}  "
          f"(L{N_LAYER} d{N_EMBD} h{N_HEAD} v{VOCAB} ff{N_FF})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

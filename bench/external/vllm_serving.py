#!/usr/bin/env python3
"""vLLM serving-latency harness — TTFT / TPOT (M4 Phase 9).

The one external axis the rest of the scoreboard does not cover: online serving
latency. vLLM is a continuous-batching, paged-attention *server*, so the fair
metric is not a kernel micro-bench but per-request:

  * TTFT  — time-to-first-token  (prefill latency the user waits before output)
  * TPOT  — time-per-output-token (steady-state decode latency, = 1/decode-tok/s)

This script drives vLLM's offline engine with a stream of requests and reports
TTFT/TPOT (mean / p50 / p99). It is meant to run in the ISOLATED venv
(`/home/data/qfshi/vllm_venv`) so vLLM's own torch never collides with the
shared `torch 2.10.0+cu128` the C++ GPU benches link against:

    CUDA_VISIBLE_DEVICES=0 /home/data/qfshi/vllm_venv/bin/python \
        bench/external/vllm_serving.py --model <hf-or-local> --requests 64 \
        --prompt-len 128 --gen-len 128

Tesseract's matched single-stream decode TTFT/TPOT comes from
`bench_cuda_llama_decode` (prefill step = TTFT, per-decode-step = TPOT); the
two are compared in `results/vllm_serving.md`. Strict isolation: pin to one
reserved, foreign-free card via CUDA_VISIBLE_DEVICES.
"""

import argparse
import json
import statistics
import time


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF id or local path of the model vLLM should serve")
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=128,
                    help="approx prompt tokens per request")
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    # Imported here so --help works without vLLM installed.
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype=args.dtype,
              max_model_len=args.max_model_len, enforce_eager=False)

    # Build a deterministic synthetic prompt of ~prompt_len tokens. Using the
    # model's own tokenizer keeps the prefill length honest.
    tok = llm.get_tokenizer()
    base = "The quick brown fox jumps over the lazy dog. "
    text = base
    while len(tok(text).input_ids) < args.prompt_len:
        text += base
    ids = tok(text).input_ids[: args.prompt_len]
    prompt = tok.decode(ids)

    # 1) TTFT: generate exactly one token; the request latency is the prefill.
    ttft_ms = []
    sp1 = SamplingParams(max_tokens=1, temperature=0.0)
    for _ in range(args.requests):
        t0 = time.perf_counter()
        llm.generate([prompt], sp1, use_tqdm=False)
        ttft_ms.append((time.perf_counter() - t0) * 1e3)

    # 2) TPOT: generate gen_len tokens; (total - ttft) / (gen_len - 1) per token.
    tpot_ms = []
    e2e_ms = []
    spN = SamplingParams(max_tokens=args.gen_len, temperature=0.0,
                         ignore_eos=True)
    for _ in range(args.requests):
        t0 = time.perf_counter()
        out = llm.generate([prompt], spN, use_tqdm=False)
        dt = (time.perf_counter() - t0) * 1e3
        e2e_ms.append(dt)
        n_out = len(out[0].outputs[0].token_ids)
        if n_out > 1:
            # subtract a representative prefill, then per-token decode.
            tpot_ms.append((dt - statistics.median(ttft_ms)) / (n_out - 1))

    result = {
        "engine": "vllm",
        "model": args.model,
        "requests": args.requests,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "dtype": args.dtype,
        "ttft_ms": {"mean": statistics.mean(ttft_ms),
                    "p50": pct(ttft_ms, 50), "p99": pct(ttft_ms, 99)},
        "tpot_ms": {"mean": statistics.mean(tpot_ms) if tpot_ms else None,
                    "p50": pct(tpot_ms, 50), "p99": pct(tpot_ms, 99)},
        "decode_tok_s": (1000.0 / statistics.mean(tpot_ms)) if tpot_ms else None,
        "e2e_ms_mean": statistics.mean(e2e_ms),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

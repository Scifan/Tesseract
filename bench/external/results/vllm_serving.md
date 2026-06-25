# vLLM serving latency (TTFT / TPOT) — M4 Phase 9 + serving closeout

The last external axis: **online single-stream serving latency** against vLLM
0.11.0 (the purpose-built LLM serving engine). Unlike the rest of the scoreboard
(kernel/op micro-benches where Tesseract wins), this measures the *whole decode
loop* of a full model, which is exactly what vLLM is engineered for.

This was the one honest loss at the M4 closeout. After composing FP16 + a
whole-model CUDA-graph decode path + GQA-native fused attention, **Tesseract now
wins decode throughput, per-token latency, and end-to-end latency** at matched
FP16; the only sub-metric still behind vLLM is TTFT/prefill.

## Setup (strict isolation)

- **Model:** the *same* `TinyLlama-1.1B-Chat-v1.0` (Llama arch, 22 layers,
  hidden=2048, 32 q-heads / 4 kv-heads GQA, d_ff=5632, vocab=32000) on both
  engines. Downloaded via the `hf-mirror.com` domestic mirror; the bf16
  checkpoint was upcast to fp32 (`convert_safetensors_fp32.py`). Tesseract's
  loader now casts that fp32 checkpoint to FP16 on load (B-022,
  `Llama.cpp::cast_floating_cpu`); vLLM loads the bf16 file as fp16 itself.
- **Isolation:** vLLM in its own venv (`/home/data/qfshi/vllm_venv`,
  torch 2.8.0+cu128 to match the 12.8 driver). Each measurement ran on a single
  foreign-process-free RTX 5880 Ada card (verified idle before launch). Tesseract
  numbers from `bench_cuda_llama_serving --fp16 --cudagraph`; vLLM from
  `bench/external/vllm_serving.py --dtype float16`.
- **Workload:** prompt_len=128, gen_len=128. TTFT = prefill latency (one-token
  request). TPOT = steady-state per-output-token latency. decode_tok_s = 1000/TPOT.

## Results (FP16, same card class, same session)

| engine / config                     | TTFT (ms) | TPOT (ms) | decode tok/s | e2e (128 tok) |
|-------------------------------------|----------:|----------:|-------------:|--------------:|
| vLLM FP16 (native)                  | 5.47      | 3.275     | 305.8        | 420.1 ms      |
| **Tesseract FP16 + CUDA-graph**     | **5.86**  | **3.115** | **321.1**    | **~400 ms**   |
| Tesseract FP16 + graph (WMMA only)  | 6.59      | 3.115     | 321.1        | ~401 ms       |
| Tesseract FP16 + graph (pre-WMMA)   | 7.28      | 3.115     | 321.1        | ~403 ms       |
| Tesseract FP32 (eager, historical)  | 28.2      | 12.97     | 77.1         | —             |
| Tesseract INT8 W8A8 (eager, hist.)  | 225.0     | 8.60      | 116.3        | —             |

**Verdict:** Tesseract wins decode throughput (+5.0%), per-token latency
(3.115 vs 3.275 ms), and end-to-end latency (−5%). Two prefill optimizations
closed almost all of the TTFT gap: the WMMA tensor-core FlashAttention prefill
(B-024+, 7.28 → 6.59 ms) and the stride-aware / BSHD attention-layout pass that
removes the per-layer `contiguous`/transpose copies (B-024c, 6.59 → **5.86 ms**).
vLLM still leads TTFT by a hair (5.47 vs 5.86 ms, **1.07×**, down from 1.33×);
the residual gap is the shared cuBLAS GEMM floor plus un-fused norm/RoPE/residual
launches — see the updated breakdown below.

## What changed since the M4 honest loss

The earlier loss recorded Tesseract eager at 77 tok/s (FP32) / 116 tok/s (INT8)
vs vLLM 307. The blockers called out there are now all resolved:

1. **FP16 model construction** — the strict-dtype loader gained a float↔float
   cast path (B-022), so one on-disk fp32/bf16 checkpoint can seed a model
   declared in FP16. Halves the decode weight streaming vs fp32.
2. **Capture-safe Storage** — same-device memset / D2D copy are now
   async-on-the-current-stream (the old synchronous `cudaMemset` /
   `cudaMemcpy + synchronize()` aborted graph capture with "operation not
   permitted when stream is capturing"). The KV-cache append also moved from a
   per-`(b,h)` `cudaMemcpyAsync` loop (256 graph nodes at H=128) to a single
   `cudaMemcpy2DAsync` per K/V slab (2 nodes) — which additionally fixed a
   graph-replay bit-exactness drift (`test_nn_mha_cuda_graph`).
3. **Whole-model CUDA-graph capture/replay** — `--cudagraph` captures the entire
   `LlamaModel::forward_step` (both prefill and decode are fixed-shape here) and
   replays it, collapsing the ~hundreds of per-step kernel launches into one
   graph launch.
4. **GQA-native fused attention end-to-end** — decode routes to the split-K
   `decode_attention_gqa`; full-prompt prefill routes to a new GQA-native fused
   FlashAttention kernel (`ops::prefill_attention_gqa`), eliminating the
   composite path's 6-launch + `[S_q,S_k]` score-matrix HBM round trip AND the
   `repeat_kv` 8× KV materialization. TTFT fell 14.4 → 7.3 ms.

## Why decode wins but TTFT does not (honest physics)

- **Decode is HBM-bound.** Both engines stream the same ~2.2 GB of FP16 weights
  per token; the floor is ~2.29 ms at 960 GB/s. Tesseract's graph-captured loop
  runs at 3.115 ms (~1.36× floor) vs vLLM's 3.275 ms, so Tesseract edges ahead.
  A *large* decode win at matched precision is not physically available here —
  it would require reading fewer weight bytes (lower precision), which is a
  different comparison.
- **Prefill is cuBLAS-GEMM-bound.** The FFN + projection GEMMs at M=128 are
  ~5.2 ms on *both* engines (same vendor library, same weight bytes). This is a
  shared floor — neither engine can beat it at matched precision.
- **WMMA tensor-core FlashAttention shipped (B-024+, 2026-06-25).** The prefill
  attention now runs both matmuls on the FP16 tensor cores
  (`fused_attention_wmma_kernel`) instead of CUDA cores, with no score-matrix
  round trip. On the attention micro-bench it is 3.8–5.1× the cuBLASLt composite
  at 512–2048 prompt; end-to-end this cut TTFT 7.28 → 6.59 ms.
- **Stride-aware / BSHD attention layout shipped (B-024c, 2026-06-25).** nsys
  attribution had shown the largest non-GEMM prefill cost was `strided_copy` at
  **25.8 %** — the `[B,S,H,D] ↔ [B,H,S,D]` transpose/`contiguous` shuffle around
  `MultiHeadAttention::forward_step` (two 512 KB copies/layer: the RoPE-permuted
  Q and the output transpose). The prefill kernels now take optional Q/K/V/O
  element-strides, so `ops::prefill_attention_gqa_bshd` reads the KV-cache
  narrows in place (no `contiguous`) and writes output in BSHD (free head-merge
  reshape). This dropped `strided_copy` to **15.9 %** and cut TTFT 6.59 → 5.86 ms.
- **The residual gap is the shared GEMM floor + un-fused elementwise.** The
  largest remaining `strided_copy` is the RoPE contiguous-ification of the
  permuted Q (B-024e: a BSHD-native RoPE would remove it). Past that, Tesseract
  and vLLM share the cuBLAS GEMM floor; the last difference is that vLLM runs
  norm/RoPE/residual inductor-fused while Tesseract launches them separately.
  Closing the final ~0.4 ms (to *win* TTFT outright) needs that epilogue fusion
  — deferred; TTFT is the only sub-metric still behind, now within 7 %.

## Reproduce

```bash
# Tesseract (clean card):
CUDA_VISIBLE_DEVICES=0 ./build-cuda/benchmarks/bench_cuda_llama_serving \
    --config /home/data/qfshi/models/tinyllama_fp32/config.json \
    --safetensors /home/data/qfshi/models/tinyllama_fp32/model_fp32.safetensors \
    --fp16 --cudagraph --prompt-len 128 --gen-len 128 --reps 30

# vLLM (clean card, isolated venv):
CUDA_VISIBLE_DEVICES=1 /home/data/qfshi/vllm_venv/bin/python \
    bench/external/vllm_serving.py --model /home/data/qfshi/models/tinyllama-1.1b-chat \
    --requests 32 --prompt-len 128 --gen-len 128 --dtype float16
```

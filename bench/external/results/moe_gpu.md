# Phase 4 — fused GPU MoE: sparse dispatch now beats dense on GPU

Strict isolation (`scripts/bench_isolated.sh --test-gpus 2`), RTX 5880 Ada.

## The win

The MoE layer now runs as one fully-fused device pipeline
(`src/cuda/MoeForward.cu`, `launch_moe_grouped_ffn`), taken automatically by
`nn::MoEFeedForward::forward` for FP32 inference on CUDA:

1. **device top-k routing** (`MoeRoute.cu`) — softmax + top-k + mask in one
   kernel, no host round-trip;
2. **device permutation** — histogram tokens per expert, exclusive prefix
   sum → offsets, scatter into expert-contiguous order (no host sort, no
   `index_select`/`cat`);
3. **one grouped GEMM** across all experts for `gate_proj` and `up_proj`
   (`cublasGemmGroupedBatchedEx`, each expert its own variable-row group),
   **fused SiLU·up**;
4. **one grouped GEMM** for `down_proj`;
5. **fused scatter-combine** — gate-weight each routed row and
   atomic-accumulate into the token's output.

## Result — fused sparse vs dense all-experts (GPU)

| T    | D    | dff  | E  | k | sparse µs | dense µs | ratio | ideal k/E | speedup |
|-----:|-----:|-----:|---:|--:|----------:|---------:|------:|----------:|--------:|
| 512  | 1024 | 2048 | 8  | 2 | 718.6     | 2490     | 0.289 | 0.250     | **3.46×** |
| 2048 | 1024 | 2048 | 8  | 2 | 2442.8    | 11758    | 0.208 | 0.250     | **4.81×** |
| 4096 | 1024 | 4096 | 8  | 2 | 7485.9    | 40886    | 0.183 | 0.250     | **5.46×** |
| 4096 | 1024 | 2048 | 16 | 1 | 2775.9    | 45500    | 0.061 | 0.062     | **16.39×** |

The sparse/dense ratio now tracks the ideal k/E almost exactly (and beats it
where the dense path's own overhead is higher) — the routed FFN compute
scales with the active experts, not the full set, as MoE intends. Before the
fusion the same sparse path was 1.1–3.0× *slower* than dense (host-bound
per-expert small GEMMs + `index_select`/`cat`/`gather`); the fused path is
~10× faster than that old sparse path at every shape.

## vs PyTorch eager MoE (same precision, true FP32)

`bench/external/torch_baseline.py moe` runs the Mixtral/HF-style eager MoE
(softmax → topk → per-expert boolean-index → SwiGLU → scatter-add), TF32 off
to match Tesseract's `CUBLAS_COMPUTE_32F`. Tesseract's fused path wins every
shape:

| T    | E  | k | Tesseract µs | PyTorch µs | speedup |
|-----:|---:|--:|-------------:|-----------:|--------:|
| 512  | 8  | 2 | 718.6        | 2099       | **2.92×** |
| 2048 | 8  | 2 | 2442.8       | 3399       | **1.39×** |
| 4096 | 8  | 2 | 7485.9       | 11480      | **1.53×** |
| 4096 | 16 | 1 | 2775.9       | 4940       | **1.78×** |

(With TF32 on, PyTorch's per-expert GEMMs speed up and the large-T shapes get
close; the win above is the fair same-precision comparison. A TF32 Tesseract
path would shift both columns together.)

## Correctness

`tests/nn/test_moe.cpp` `[nn][moe][cuda]`: the fused GPU forward (taken under
`NoGradGuard`) matches the CPU reference to < 1e-4 across (E,k) =
(8,2),(4,1),(16,4),(6,6). The generic autograd path is unchanged and still
used for training, CPU, quantized experts, or biased experts (the fused path
falls back cleanly in those cases).

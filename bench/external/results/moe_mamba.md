# MoE sparsity + Mamba O(1) decode — architectural quantitative results

Addresses issue.md A1 (MoE was correctness-only) and A2 (Mamba O(1) claim was
never quantified). These are architecture wins: device-independent in shape, so
the contrast holds against any framework's Transformer (incl. PyTorch eager
attention, which is also O(L) per decode step like our Llama baseline).

## Mamba O(1) decode vs Llama O(L) (CPU, d_model=256, 4 layers)

| L    | Llama ms/step | Llama tok/s | Llama state_MiB | Mamba ms/step | Mamba tok/s | Mamba state_MiB |
|------|--------------:|------------:|----------------:|--------------:|------------:|----------------:|
| 256  | 16.53 | 60.5 | 2.0  | 12.54 | 79.7 | 0.148 |
| 512  | 18.62 | 53.7 | 4.0  | 12.47 | 80.2 | 0.148 |
| 1024 | 21.11 | 47.4 | 8.0  | 12.43 | 80.4 | 0.148 |
| 2048 | 27.97 | 35.7 | 16.0 | 12.58 | 79.5 | 0.148 |

- **Latency**: Mamba is flat (~12.5 ms/step, O(1)); Llama grows with L (O(L)).
  At L=2048 Mamba is **2.2x faster** (79.5 vs 35.7 tok/s) and the gap widens with L.
- **State memory**: Mamba's recurrent state is constant 0.148 MiB; Llama's KV
  cache grows linearly to 16 MiB at L=2048 — a **108x** reduction. This is the
  same O(1) vs O(L) advantage Tesseract's Mamba has over PyTorch eager attention
  decode (also O(L)).

## MoE sparse dispatch (CPU, honest)

`bench_moe_sparse` (true top-k token dispatch via index_select/gather, vs dense
all-experts). The FLOPs are genuinely sparse (only k of E experts run per token,
== k/E of dense compute), but on CPU the per-expert GEMMs are small and the
small-matmul overhead + routing/permutation cost eats most of the theoretical
k/E saving. Measured speedups ranged 0.9-1.3x under load (vs an ideal of 4x for
E=8/k=2), i.e. **FLOP-real but not yet latency-real on CPU**.

The win lands on GPU with a grouped/batched expert GEMM (one batched-matmul over
the per-expert token groups instead of E small launches) — tracked as the MoE
follow-up. We report the CPU result honestly rather than claim a CPU win.

## Headline GPU win is elsewhere

The clean, measured GPU win vs PyTorch is quantized decode (see
`pytorch_headtohead.md`): Tesseract INT8 GEMV 2.45x faster than torch FP16 at the
decode shape. MoE/Mamba here are the architecture-efficiency story.

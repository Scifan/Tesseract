# GPU bring-up — real numbers on RTX 5880 Ada (SM 8.9)

Collected on the reserved cards (`CUDA_VISIBLE_DEVICES=2`, daemon holding the
rest of 0,1,2). All Tesseract numbers from `build-cuda/benchmarks/*`; PyTorch
2.10.0+cu128 from `bench/external/torch_baseline.py`. Min-over-samples.

## matmul TFLOPS (Tesseract `ops::matmul` vs `torch.mm`)

Both call cuBLAS(Lt); a near-1.0 ratio is the honest expected outcome. Note the
fp32 column is NOT apples-to-apples here: torch had TF32 enabled, Tesseract's
bench uses true FP32 (`CUBLAS_COMPUTE_32F`) — see the fair fp32 re-run in
`pytorch_headtohead.md`. FP16 is directly comparable (tensor cores both sides).

| dtype | N    | Tesseract TF | torch TF | ratio (TS/torch) |
|-------|------|-------------:|---------:|-----------------:|
| fp16  | 512  | 41.97        | 37.45    | 1.12 |
| fp16  | 1024 | 100.35       | 84.10    | 1.19 |
| fp16  | 2048 | 134.22       | 138.65   | 0.97 |
| fp16  | 4096 | 150.98       | 162.69   | 0.93 |
| fp16  | 8192 | 132.32       | 98.62    | 1.34 |

Read: Tesseract's GEMM is on par with PyTorch (wins at 1024/8192, within ~7% at
4096/2048). No op-layer overhead over the vendor library PyTorch also uses.

## Llama decode step (one 7B block: d_model=4096 H=32 d_ff=11008 S_k=129)

| variant            | Tesseract step_us | weight_MB |
|--------------------|------------------:|----------:|
| FP32               | 1123.7            | 809.8     |
| INT8               | 510.2 (0.45x)     | 202.8     |
| INT4G(g=128)       | 726.0 (0.65x)     | 107.8     |

PyTorch eager same config: FP32 904.2 us, FP16 478.0 us.

Read: Tesseract **INT8 decode (510 us, 203 MB)** matches PyTorch **FP16 (478 us,
~400 MB)** at half the weight memory; INT4G gives 7.5x weight reduction. The
fp32-vs-fp32 latency gap is the TF32 setting (PyTorch eager uses TF32 matmuls by
default-on here); fair re-run in `pytorch_headtohead.md`.

## Fused attention (FP16) — Tesseract fused kernel vs its composite path

| (B,H,Sq,Sk,D)        | fused_us | composite_us | speedup |
|----------------------|---------:|-------------:|--------:|
| (8,32,1,2048,128) decode | 636.5 | 1595.5 | **2.51x** |
| (4,32,1,2048,128) decode | 611.6 | 803.4  | 1.31x |

Hard bar (fused speedup >= 1.50 at B*H>=256): PASS (2.51x). This is the fused
decode-attention kernel that beats the multi-kernel composite path.

## Known issue (pre-existing, not on this path)

`bench_cuda_transformer_block` (BERT-base fwd+bwd, B=16 S=1024) OOMs even with
44 GiB free: the timed loop retains autograd graphs across iterations (a 536 MB
scores tensor alloc fails after ~44 GiB accumulates). Pre-existing bench bug,
filed for follow-up; does not affect the inference/decode comparison numbers.

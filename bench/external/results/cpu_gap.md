# CPU decode vs llama.cpp — honest re-measure (B-046)

Same-architecture model (4L/d256/8h/v4096/ff688 F32 GGUF built via
`scripts/make_tiny_llama_gguf.py`), prompt=32 gen=64, 8 threads.

| runtime    | prefill tok/s | decode tok/s |
|------------|--------------:|-------------:|
| Tesseract  | 126.0         | 73.5         |
| llama.cpp (CPU, -ngl 0) | 13196 | 4877 |
| gap        | ~105x         | **~66x**     |

## Feasible CPU optimization attempted

Thread sweep (OMP_NUM_THREADS / EIGEN_NUM_THREADS): decode 70.3 (1t) → 73.2 (8t)
→ 75.8 (16t) tok/s — only ~8%. The decode step is a batch-1 GEMV on a tiny
(d=256) model, so it is overhead/latency bound, not throughput bound: extra
threads don't help and Tesseract's per-op tensor allocate/dispatch cost dominates.

## Honest read (this is the "do not claim a win" case)

llama.cpp is a hand-tuned CPU inference engine (SIMD GEMV kernels, fused ops,
zero per-op allocation, cache-resident tiny models). Tesseract's eager CPU path
pays per-op Tensor scaffolding + Eigen blocked-GEMM overhead that does not amortize
at batch-1 / tiny-d decode. Closing 66x would require a GEMV SIMD kernel + op
fusion + allocation elision — a major effort whose payoff is on CPU, which is not
our battleground. We report the gap honestly and compete on GPU instead (see
`pytorch_headtohead.md`: INT8 decode 2.45x faster than torch FP16).

Note Tesseract DOES win the FP32 GEMV on GPU vs torch FP32 and the INT8 GEMV vs
torch FP16 — the per-op overhead that sinks the CPU tiny-model case is negligible
once the kernel is HBM-bound on a real-size GPU shape.

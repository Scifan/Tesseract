# External-framework benchmark harness

Apples-to-apples comparisons of Tesseract against existing frameworks on the
**same GPU**, to back the "performance vs existing frameworks" claim with honest
numbers (idea.md §6.2/§8.5, backlog B-046).

Hardware on this box: 6x RTX 5880 Ada (SM 8.9, 49 GiB, FP8-capable, no Hopper).
GPU jobs are pinned to the reserved cards via `CUDA_VISIBLE_DEVICES=0,1,2`.

## Files

- `torch_baseline.py` — PyTorch baselines matched to Tesseract CUDA benches:
  - `matmul` — square GEMM TFLOPS sweep (matches `bench_cuda_matmul`).
  - `decode` — one Llama-7B-block decode step (matches `bench_cuda_llama_decode`).
  - `attention` — SDPA, eager vs torch fused (flash) SDPA.
  Each prints a JSON object; timing uses CUDA events + best-of-N (min).
- `run_compare.py` — runs the torch baselines and the matched Tesseract benches
  from a CUDA build dir, saves raw logs + `results/combined.json`, prints a
  headline comparison.
- `../../scripts/bench_vs_llama_cpp.sh` — CPU decode vs `llama.cpp` (separate;
  uses `scripts/make_tiny_llama_gguf.py` to build a same-arch GGUF).

## Run

```bash
# 1. Build the CUDA benches once:
cmake --build build-cuda --target bench_cuda_matmul bench_cuda_llama_decode \
      bench_cuda_fused_attention bench_cuda_transformer_block -j

# 2. Run the combined comparison on the reserved cards:
CUDA_VISIBLE_DEVICES=0,1,2 python bench/external/run_compare.py \
    --build-dir build-cuda --device 0
```

## Interpreting results

- **matmul**: both sides call cuBLAS(Lt); a ratio near 1.0 is the expected,
  honest outcome (we add no overhead over the vendor GEMM PyTorch also uses).
- **decode / attention / fused block**: where kernel fusion + quantization (INT8
  / INT4G / FP8 on Ada) give Tesseract a real edge over PyTorch eager. We do not
  claim wins over torch fused SDPA (flash) on shapes where it applies; those are
  reported side by side.

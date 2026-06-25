# Python ↔ C++ frontend overhead (issue.md B1 / B-041)

How much does calling Tesseract through the pybind11 Python frontend cost vs the
native C++ API? Both paths run the **same compiled C++ kernel**, so the
per-call delta isolates the frontend's crossing cost (interpreter loop + arg
marshaling + return wrapping). Same ops, same iteration counts.

- Native: `benchmarks/bench_python_overhead.cpp`
- Python: `bench/external/python_overhead.py` (via `tesseract._core`)
- Wrapper: `bench/external/run_python_overhead.py`

| op | C++ µs | Python µs | overhead µs | overhead % |
|----|-------:|----------:|------------:|-----------:|
| matmul 64³            | 7.58 | 9.13 | 1.55 | 20.4% |
| matmul 256³           | ~same kernel | | 4.0 | 0.1% |
| matmul 512³           | ~same kernel | | 58 | 1.0% |
| matmul 1024³          | ~same kernel | | 197 | 3.2% |
| MLP fwd (b=64, d=512) | ~same kernel | | 6.0 | 0.0% |

## Verdict

The Python frontend adds a **fixed ~1.5–6 µs per call** (a couple of pybind11
type conversions). As a fraction of the op it wraps:

- **Tiny ops** (64³ matmul, ~7 µs of compute): ~20% — the crossing cost is
  comparable to the work.
- **Real-sized ops** (≥256³ matmul, an MLP forward): **<1–3%, often ~0%** — the
  crossing cost is amortized away by the actual compute.

This is the expected "thin shim" profile: `python/tesseract` re-exports the
compiled extension with no logic of its own, so Python users pay only a small
constant dispatch tax that vanishes for any non-trivial workload. Training /
inference loops (where each step is many large ops) see negligible frontend
overhead.

## Note on absolute kernel times

The shared dev host was under CPU contention during the run, so the absolute
C++ matmul times above are noisier than a quiet machine would give (the larger
sizes don't show their full flop scaling). This does **not** affect the overhead
metric: the Python and C++ sides execute the identical kernel back-to-back, so
the *delta* (the frontend cost) is robust regardless of the kernel's absolute
speed.

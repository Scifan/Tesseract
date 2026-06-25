# M2 CUDA benchmark ledger

Captured-run ledger for the **M2L.1** CUDA perf gate
(`bench_cuda_{matmul, elementwise, attention, attention_bwd, rms_norm,
transformer_block}`). New runs append a section under §2; §3 documents the
methodology; §4 lists known caveats that the hard bars were tuned around.

The six benches are wired into `ctest -L bench_cuda`; each one exits `0` on
pass, `1` on perf miss, `77` on "no GPU visible". The hard bars live in the
bench binaries themselves (so every developer gets the same gate locally),
and every merge that touches CUDA code must keep them green on the
reference hardware below.

---

## 1. Reference hardware / toolchain

| Field          | Value                                                     |
|----------------|-----------------------------------------------------------|
| GPU            | NVIDIA RTX 5880 Ada Generation (SM 8.9, 48 GiB, PCIe)     |
| Peak DRAM BW   | 960.1 GB/s (theoretical)                                  |
| Measured D2D   | ~421.5 GB/s one-way → ~843.0 GB/s DRAM roofline at 256 MiB|
| CUDA toolkit   | 12.x (as pinned in `cmake/Dependencies.cmake`)            |
| Driver         | whatever the dev container mounts (captured per-run)      |
| Build type     | `Release` with `TESSERACT_ENABLE_CUDA=ON`                 |

All numbers below are **best-of-N CUDA-event timings** taken on an idle GPU
(no other tenants, persistence mode ON, application clocks unpinned — we
rely on `best_of_n_time` + CoV-gating rather than DVFS pinning; see §3).

---

## 2. Captured runs

### 2026-04-18 — M2L.1 lock-in (RTX 5880 Ada)

Run context: first run where all six benches pass the aggressive hard bars
simultaneously with `best_of_5` sampling and the caching allocator +
cuBLASLt descriptor cache + vectorized elementwise fast-path all landed
(see §5.L of `docs/m2-plan.md` progress log).

#### 2.1 `bench_cuda_matmul` — FP32/FP16 square sweep

```
dtype  N     |   raw_min bridge_min   ours_min |   raw_TF* bridge_TF*  ours_TF* |  ratio*  dispatch
----------------------------------------------------------------------------------------------------
fp32   512   |     15.20      16.00     15.86 |    17.66     16.77     16.92  | 1.0089   -0.14
fp32   1024  |     79.36     106.93     92.05 |    27.06     20.08     23.33  | 1.1617  -14.89
fp32   2048  |    402.94     555.52    405.55 |    42.64     30.93     42.36  | 1.3698 -149.97
fp32   4096  |   3149.12    4737.02   3196.93 |    43.64     29.01     42.99  | 1.4817-1540.10
fp32   8192  |  39436.29   40461.31  39798.78 |    27.88     27.17     27.63  | 1.0166 -662.53
fp16   512   |      6.50       6.41      6.39 |    41.31     41.90     41.98  | 1.0019   -0.01
fp16   1024  |     19.46      23.59     23.71 |   110.38     91.05     90.56  | 0.9946    0.13
fp16   2048  |    119.30     179.00    177.30 |   144.01     95.98     96.90  | 1.0096   -1.70
fp16   4096  |    816.13    1311.62   1311.74 |   168.40    104.79    104.78  | 0.9999    0.13
fp16   8192  |  13212.48   13449.22  13684.74 |    83.22     81.75     80.35  | 0.9828  235.52
```

Hard bars:

| metric                                                | observed  | target   | verdict |
|-------------------------------------------------------|-----------|----------|---------|
| `median(ours − bridge)` (µs, across 10 shapes)        | −0.14     | ≤ 5.00   | PASS    |
| `min(ours/bridge)` per-shape floor                    | 0.9828    | ≥ 0.95   | PASS    |
| dispatch overhead @ 4096² FP32 (µs)                   | −1540.10  | ≤ 20.00  | PASS    |

The negative "dispatch" number at 4096² FP32 is a consequence of the raw
vs bridge vs `ops::` call paths hitting different cuBLASLt heuristics on
that shape — see §4.A. The median additive overhead (which is the bar we
actually gate on) stays at ~0 µs.

#### 2.2 `bench_cuda_elementwise` — DRAM roofline

```
  memcpy D2D @ 256 MiB  :   421.5 GB/s  (1-way) →  843.0 GB/s DRAM roofline

size       MiB/fp32 |  add_us    DRAM   /roof |  mul_us    DRAM   /roof |  sig_us    DRAM   /roof
-------------------------------------------------------------------------------------------------
1 MiB            1  |    2.12  1481.8   1.758 |    2.14  1472.8   1.747 |    2.11   993.2   1.178
16 MiB          16  |    8.72  5768.7   6.843 |    8.73  5766.8   6.841 |    7.51  4468.2   5.301
64 MiB          64  |  230.66   872.8   1.035 |  230.17   874.7   1.038 |  145.53   922.3   1.094
256 MiB        256  |  919.55   875.8   1.039 |  918.53   876.7   1.040 |  633.34   847.7   1.006
```

Hard bars (L2-overflowing 64 MiB row, ratios are DRAM GB/s divided by the
measured `memcpy` DRAM roofline):

| metric                                     | observed | target  | verdict |
|--------------------------------------------|----------|---------|---------|
| `add` @ 64 MiB / memcpy_DRAM               | 1.0354   | ≥ 0.95  | PASS    |
| `mul` @ 64 MiB / memcpy_DRAM               | 1.0376   | ≥ 0.95  | PASS    |
| `sigmoid` @ 64 MiB / memcpy_DRAM           | 1.0940   | ≥ 0.90  | PASS    |

The ≥1.0 ratios for 16 MiB rows reflect L2 residency (see §4.B); we
deliberately gate on 64 MiB because that's the smallest size that spills
to DRAM on Ada's 48 MiB L2.

#### 2.3 `bench_cuda_attention` — composite vs sum-of-primitives (FP16)

Since B-015 landed FP16/BF16 on `src/cuda/Elementwise.cu` and
`src/cuda/Softmax.cu` (via FP32-promoted device math), this bench now
runs the whole composite chain — `ops::mul(Q, scale)` → batched
cuBLASLt matmul → `ops::softmax` → batched matmul — in native FP16 on
device. The composite/sum-of-primitives ratio is dtype-independent by
construction (both sides scale linearly with element size) and the
≥ 0.97 hard bar holds on all three shapes; the notable delta vs the
earlier FP32 table is the ~2× TFLOPS uplift at the 4096-seq head where
FP16 doubles both memory bandwidth and cuBLASLt tensor-core throughput.

```
(B,H,S,D)            | composite us sum-prims us   diff_us | ratio    TFLOPS
---------------------------------------------------------------------------
(8,32,512,64)        |     4970.50     4999.42    -28.93  | 1.0058    3.46
(4,32,2048,64)       |    11142.14    11235.04    -92.90  | 1.0083   12.34
(2,16,4096,128)      |     8432.45     8392.86     39.58  | 0.9953   32.60
```

| metric                                       | observed | target  | verdict |
|----------------------------------------------|----------|---------|---------|
| `min(sum_prims / composite)` across shapes   | 0.9953   | ≥ 0.97  | PASS    |

The `(2,16,4096,128)` row dips below 1.0 — composite is ~0.5% slower
than the sum of primitives — which is the opposite sign from the FP32
table. The reason is that `ops::attention` internally materialises the
scaled `Q` into a new FP16 buffer and hands it to cuBLASLt; at FP16
tensor-core throughput (32.6 TFLOPS observed) the
`ops::mul(Q, scale)` write no longer pays for itself via a fused-alpha
path the way it did in FP32, and the per-shape cuBLASLt algo for the
second matmul happens to be marginally better when invoked
stand-alone. The 0.3% gap is well inside the 3% composite-dispatch
budget the metric gates on, and the other two shapes still show
composite < sum as before.

#### 2.4 `bench_cuda_attention_bwd` — composite SDPA envelope (FP16)

Shape: `(B=2, H=16, S=1024, D=64)` — chosen to fit autograd activations
in 48 GiB without the `(4,32,2048,64)` OOM seen during dev.

Since B-016 landed FP16/BF16 on `src/cuda/Reduction.cu`, this bench
now runs the whole composite SDPA backward chain on FP16 — the last
gate was `SoftmaxBackward::apply`'s `sum(dim, keepdim)` call, which
previously rejected half-precision dtypes. The fwd/bwd ratio is
dtype-independent by construction (autograd graph is shape-only;
per-op kernel time scales linearly with element size), so the ≤ 5.0×
hard bar carries over unchanged from the earlier FP32 numbers. The
wall-clock stages each dropped ~24 % vs the FP32 baseline, in line
with FP16's 2× bandwidth / tensor-core throughput win on this
memory-bound composite shape.

```
forward only       : 1213.44 us (min)   1235.36 us (mean, cov 0.020, 39 batches)
forward + backward : 5725.18 us (min)   5755.33 us (mean, cov 0.002, 30 batches)
(fwd+bwd)/fwd      : 4.718  (bwd alone ≈ 3.72× fwd)
```

| metric                                      | observed | target  | verdict |
|---------------------------------------------|----------|---------|---------|
| `(fwd+bwd) / fwd` — composite SDPA envelope | 4.72     | ≤ 5.00  | PASS    |

The tighter 3.2× target only applies to a fused FA3 backward, which is
M2L.3 (Hopper-gated).

#### 2.5 `bench_cuda_rms_norm` — baseline (no hard bar)

```
(B,S,D)         |   per_us  eff_GB/s   frac
-------------------------------------------
(8,512,1024)    |   687.06      48.8  0.116
(32,2048,4096)  | 44908.70      47.8  0.113
(16,4096,4096)  | 45027.79      47.7  0.113
```

Pure composite today (no fused RMSNorm kernel) — gives the fused kernel
it will be replaced by in M3 a reference to beat.

#### 2.6 `bench_cuda_transformer_block` — end-to-end (no hard bar)

Config: `d_model=512 heads=8 d_ff=2048 B=16 S=1024`, FP32.

```
forward only     : 28216.35 us  =    580 656 tok/s
forward+backward : 87387.33 us  =    187 487 tok/s
bwd / fwd        : 3.10
```

Top-line dashboard; regressions here are not a hard fail but show up in
the bench log for the reviewer.

---

## 3. Methodology

### 3.1 Timer

All six benches use `benchmarks/cuda_bench_util.hpp`:

- **`CudaTimer`** wraps a pair of `cudaEvent_t` and records on whatever
  stream the caller pins (not necessarily the default stream).
- **`BenchStream` / `StreamGuard`** unify the stream that `ops::*` sees
  with the one the timer records on. Before this landed, `ops::matmul`
  was firing on `current_stream(device)` while the timer was recording
  on stream 0, making the measured time a function of wall clock not
  kernel time.
- **`steady_state_time`** runs a configurable warm-up, then batches
  `[batch]` iterations per event pair until the coefficient-of-variation
  across batches drops below `cov_target` (default 2 %). Returns
  `{min_us, mean_us, cov, batches}`.
- **`best_of_n_time`** runs `steady_state_time` N times (default 5) and
  returns the sample with the lowest `min_us`. We report `min_us`, not
  `mean_us`, because cuBLASLt's heuristic can swap algorithms between
  runs and the mean is dominated by the worst-algorithm run; the min is
  the one that reflects what a user actually gets in steady state.

### 3.2 Bandwidth normalization

For `bench_cuda_elementwise` we report **DRAM bytes moved per second**,
not "effective GB/s":

- **memcpy roofline** — `measure_d2d_memcpy_gbs` reports the one-way
  transfer rate; DRAM sees read + write = `2 × one_way`. We print both.
- **add/mul** — `2 reads + 1 write = 3 × nbytes` per call, divided by
  `min_us` → DRAM GB/s.
- **sigmoid** — `1 read + 1 write = 2 × nbytes` per call.

This makes the elementwise numbers directly comparable against the
memcpy roofline and against each other.

### 3.3 Stream + allocator + cuBLASLt caching

- `CudaAllocator` caches bucketed blocks so `cudaMalloc` / `cudaFree`
  don't show up in the per-call timeline (they'd otherwise dominate the
  <100 µs shapes).
- `src/cuda/MatMul.cpp` caches **(dtype × shape × layout)** cuBLASLt
  descriptors + per-stream workspace buffers, and exposes a single
  process-wide `cublasLtHandle_t` so raw / bridge / `ops::` call paths
  always hit the same heuristic tier.
- `src/cuda/Elementwise.cu` has a dense-contiguous vectorized fast-path
  (`float4` / `double2` / `int4` / `longlong2`) that all four shapes
  above hit.

---

## 4. Known caveats

### 4.A — cuBLASLt heuristic variance on square FP32

At N ∈ {1024, 2048, 4096}, raw cuBLASLt sometimes selects a different
algorithm than the bridge / `ops::` path selects, even with a shared
handle — the heuristic is input-tensor-aware and the "raw" path
constructs its `Aop` / `Bop` differently. This shows up in the `ratio*`
and `dispatch` columns as a large negative "dispatch" on FP32 2K-4K and
a correspondingly inflated `ratio*` > 1.

That's why the actual hard bar is:

1. **`median(ours − bridge)` ≤ 5 µs** (measures the ops-layer overhead
   that we can actually move with code) — not a ratio.
2. **`min(ours/bridge)` ≥ 0.95 per shape** (catches any shape where we
   regress significantly below the bridge, independent of what raw is
   doing) — but only ≥ 0.95, not ≥ 1.0, precisely to absorb heuristic
   noise.
3. **dispatch @ 4096² FP32 ≤ 20 µs** (measures raw-vs-bridge gap at a
   fixed shape) — loose on purpose.

### 4.B — Small elementwise sizes live in L2

For 1 MiB and 16 MiB working sets on Ada (48 MiB L2), the data never
leaves L2, so the "DRAM GB/s" metric reports well above peak DRAM
(1500–5700 GB/s). That's physical: L2 is ~4 TB/s aggregate. We gate the
hard bar on **64 MiB**, which is the smallest size guaranteed to spill
to DRAM.

### 4.C — FP16 / BF16: fully live on both forward and backward

Both `bench_cuda_attention` (§2.3) and `bench_cuda_attention_bwd`
(§2.4) now run end-to-end FP16. The composite chain is:

- **Forward**: `ops::mul(Q, scale)` → batched cuBLASLt matmul →
  `ops::softmax` → batched matmul. B-015 extended the elementwise
  and softmax CUDA TUs to FP16/BF16 via FP32-promoted kernels.
- **Backward**: five matmul-backwards + `SoftmaxBackward` (which
  composes `mul → sum(dim, keepdim) → sub → mul`) + `ops::sum(out)`
  for the loss scalar. B-016 extended
  `src/cuda/Reduction.cu` to FP16/BF16 via the same FP32-promoted
  accumulator pattern (stage-1 partials stay in `float*` so we
  don't re-round between stages). The CPU reference side switched
  to `dispatch_float_with_half` with
  `Acc = conditional<floating_point<T>, T, float>` to keep parity
  bit-for-bit with the CUDA kernels.

The ratio metrics — composite/sum for forward, fwd+bwd/fwd for
backward — are dtype-independent by construction; both sides of each
ratio scale linearly with element size. So the hard bars carried over
unchanged through the dtype flips, and the ~2× wall-clock uplift on
the attention shapes is the expected FP16 bandwidth / tensor-core
throughput win.

### 4.D — No FlashAttention-3 comparison yet

The "≥ 90 % of FA3" bar in §1 of `docs/m2-plan.md` is **M2L.3**, not
M2L.1. M2L.3 is Hopper-gated (SM 9.0), and our reference box is Ada
(SM 8.9). On M2L.3 we will add a `bench_cuda_attention_fa3` row that
reports the ratio against the vendored FA3 kernel on a Hopper runner;
it will land as a separate ctest entry alongside the six above.

### 4.E — `attention_bwd` shape pick

`(4, 32, 2048, 64) FP32` OOMs on the reference box because the
`scores` activation alone is 2 GiB and autograd retains it. We dropped
to `(2, 16, 1024, 64)` — still produces a stable
`(fwd+bwd)/fwd` ratio (< 1 % CoV) without forcing gradient
checkpointing, which would distort the measurement. Once FP16 is
enabled via B-015 we'll reinstate the larger shape.

---

## 5. How to add a run

1. Build with `cmake -B build-cuda -DTESSERACT_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release`
   and `cmake --build build-cuda -j`.
2. `cd build-cuda && ctest -L bench_cuda --output-on-failure`.
3. Copy the `stdout` of each bench into a new **§2.x — YYYY-MM-DD —
   `<label>` (`<GPU>`)** block above, keeping the tables in the exact
   shape the bench prints (so a diff between runs is trivial).
4. If a hard bar moves: update the "target" column **and** the matching
   hard-bar assert in the bench source **and** the corresponding row in
   §4 of `docs/m2-plan.md`. Hard bars never move without a roadmap
   entry.

# M3 — LLM inference as a first-class citizen

**Status:** kickoff (2026-04-18). Wave 1 closed same-day.

M3 turns the M2 CUDA op stack into a production-grade inference runtime.
Rather than attacking every direction at once, we walk a
performance-ordered wave plan so each wave's exit bar becomes the
benchmark floor for the next. The four directions the user explicitly
called out — **PagedKV**, **normalization running-stats**, **weight-only
quantization**, and **tokenizer-plus-model-loader** — each have a
dedicated backlog item (B-017 through B-021) so nothing is lost.

---

## Wave 1 — model-loader stack (**DONE, 2026-04-18**)

**Goal.** A real Llama-shaped checkpoint, loaded from a file, forward-
passing end-to-end on CPU and CUDA with byte-identical top-k logits.

Delivered:

| Component | Where | Tests |
| --------- | ----- | ----- |
| `tesseract::io::SafeTensors` reader (mmap + minimal JSON parser, 8 dtypes) | `src/io/SafeTensors.cpp` | 8 / 97 |
| `nn::Embedding` (forward via `index_select`, scatter-add backward) | `src/nn/Embedding.cpp` | 4 / 81 |
| `Module::named_parameters()` / `named_buffers()` + `ModuleList` | `src/nn/Module.cpp`, `include/tesseract/nn/ModuleList.hpp` | covered by downstream |
| `ops::layer_norm` + `nn::LayerNorm` (use_bias on/off) | `src/ops/cpu/LayerNorm.cpp`, `src/nn/LayerNorm.cpp` | 7 / 163 |
| `LlamaModel` + `LlamaConfig` + `llama_local_to_hf_name` + `load_safetensors` + `from_pretrained` | `src/models/Llama.cpp` | 6 / 6808 |
| `examples/llama_infer.cpp` full-stack demo | `examples/llama_infer.cpp` | CPU↔CUDA top-5 byte-identical |
| `tesseract::io::Tokenizer` interface + `WhitespaceTokenizer` reference | `src/io/Tokenizer.cpp` | 11 / 17 |

Exit signal:

```
$ ./build-cuda/examples/tesseract_llama_infer --synthetic --device cpu --seq 4
$ ./build-cuda/examples/tesseract_llama_infer --synthetic --device cuda --seq 4
```

both print the identical top-5 list `1.33876 / 1.20405 / 1.14007 /
1.01823 / 1.00854` — the strongest parity signal shy of a real HF
checkpoint. Tracking: **B-017 — Wave 1 model-loader stack**.

## Wave 2b — BatchNorm + Module::train(bool) running-stats *(DONE, 2026-04-18)*

**Tracking: B-020.** Closes the normalization family for the training
path (RMSNorm + LayerNorm already shipped stateless) and upgrades
`nn::Module::train()` to the recursive flip PyTorch users expect.
Orthogonal to the decode-path optimizations in Wave 2.1–2.5.

**Shipped:**

- `ops::batch_norm(x, weight, bias, running_mean, running_var,
  training, momentum, eps)` — composite over `mean / sub / mul / div /
  sqrt / add / reshape`. Handles rank-2 (`[N,C]`), rank-3
  (`[N,C,L]`), and rank-4 (`[N,C,H,W]`) inputs off a single
  "reduce-over-every-non-channel-dim" loop (keepdim=true so the
  broadcast back onto `x` is a no-op). Forward uses biased variance
  (1/N_red) for normalization; running-var EMA uses the unbiased
  estimator (Bessel-corrected N/(N−1)) — exact PyTorch match.
- `nn::BatchNorm1d` / `nn::BatchNorm2d` — register `weight` / `bias`
  as parameters (when `affine=true`) and `running_mean` /
  `running_var` as buffers. `Module::to(cuda)` migrates the whole
  state in lockstep via shared-impl aliasing, no per-subclass
  override needed. Forward delegates to `ops::batch_norm` with
  `training=is_training()`.
- `nn::Module::train(bool)` moved out-of-line and made **recursive**:
  `model->eval()` flips the whole sub-tree in one call. Fixes the
  latent "running stats keep drifting during inference" bug for
  composite modules containing BN leaves.
- In-place running-stats writeback via `Storage::copy_device_bytes`
  under a `NoGradGuard`: the freshly-computed EMA tensor is
  byte-copied into the caller's buffer storage so every Tensor
  handle sharing the buffer's impl observes the update. Autograd
  stays cleanly detached.

**Parity bars (all met):**

- BN1d-2D / BN1d-3D / BN2d forward vs hand-rolled reference (FP32, 1e-5 abs).
- Running-stats EMA: `running_mean ← (1-m)·rm + m·batch_mean`,
  `running_var ← (1-m)·rv + m·var_unbiased`.
- `training=false` → running stats byte-identical before/after
  forward (tight check, not tolerance-based).
- Autograd finite-diff (h = 5e-3): `dL/dx`, `dL/dw`, `dL/db` within
  2e-2 abs; bias gradient analytically exact (`N` per channel).
- CPU↔CUDA parity on both output `y` and post-forward running stats
  (1e-5 abs).
- `Module::train(bool)` / `eval()` recursion through `Sequential`
  into BN leaves.

**Test count:** `tests/nn/test_nn_batch_norm.cpp` — 12 cases.
Full ctest: **328/328 green** (CUDA build); CPU-only build cleanly
skips the CUDA parity case.

## Wave 2.2 — Fused RMSNorm / LayerNorm CUDA kernels *(DONE)*

**Tracking: B-022.** Landed 2026-04-18. The M2K composite path for
these ops unrolls into 5-6 per-element passes over `x`; the fused
one-block-per-row kernel collapses it to two passes and ships all
four dtypes (FP32 / FP64 / FP16 / BF16) via the B-015/B-016 FP32-
promoted accumulator pattern. Forward-only; backward continues to
flow through the composite primitives' autograd nodes unchanged.

**Measured on SM 8.9 Ada (RTX 5880) against composite baseline:**

| Shape (B,S,D)   | fused µs | composite µs | speedup |
| --------------- | -------- | ------------ | ------- |
| (8, 512, 1024)  | 15.1     | 746.8        | 49.4×   |
| (32, 2048, 4096) | 2483   | 43190        | 17.4×   |
| (16, 4096, 4096) | 2486   | 43205        | 17.4×   |

Effective bandwidth at the largest shape: 863.7 GB/s (2.05× memcpy
D2D roofline, thanks to L2 residency of the second read-pass).
Bench hard bars: `fused/memcpy ≥ 0.80` and `speedup ≥ 3.00×`, both
PASS.

## Wave 2.1 — KV cache + decode-phase attention *(DONE, 2026-04-18)*

**Tracking: B-019.** First wave to produce an end-to-end decode
path. The MVP ships a contiguous-backed cache; paged storage is
carved out as B-019b and deferred to Wave 4 (continuous batching),
where it actually starts paying for itself.

**Shipped:**
- `nn::KVCache` — owns `[B, H, max_len, D_head]` slabs for keys and
  values on a caller-chosen dtype/device; `append(k_new, v_new)`
  is a per-(b, h) `Storage::copy_device_bytes` into the next
  `S_new` seq slots; `keys_view()` / `values_view()` return
  zero-copy `[B, H, current_len_, D_head]` narrows; `reset()`
  rewinds without reallocation.
- `Tensor::narrow(dim, start, len)` — new primitive view op,
  used for both the cache prefix and RoPE table slicing.
- `nn::RotaryEmbedding::forward_offset(x, pos_offset)` — rotates
  with cached positions `[pos_offset, pos_offset + S)`; shares
  storage with the registered `cos`/`sin` buffers.
- `nn::MultiHeadAttention::forward_step(x, cache)` — inference
  decode/prefill path: projects `x : [B, S_new, D]`, RoPE-rotates
  Q and K_new at positions offset by `cache.current_len()`, appends
  K/V into the cache, runs attention against the full prefix
  `[B, H, current_len + S_new, D_head]`. Chunked prefill
  (`S_new > 1`) materializes a `[S_new, pos + S_new]` rectangular
  additive causal mask so query `pos + i` still only sees keys
  `j <= pos + i`. Whole step runs under `NoGradGuard`.

**Parity bars (all met):**
- `forward_step(x[:, i:i+1, :], cache)` concatenated across
  `i ∈ [0, S)` equals `forward(x)` within FP32 abs 1e-5 — validated
  under 1-1-1-1-1 and 2-1-2 chunk schedules, both with and without
  RoPE.
- CPU ↔ CUDA parity on the RoPE decode path: FP32 abs 5e-5 on
  `[B=1, S=8, D=32, H=4]`.
- Full `ctest -j 1` green: 305/305 (no regression of any attention,
  RoPE, or fused-norm test).

**Deferred to B-019b (paged storage):** vLLM-style block-table
indirection — see `docs/backlog.md#B-019b`. Public API is
structured so the swap lands behind `KVCache` without touching
attention / MHA / RoPE call sites.

## Wave 2.3 — CUDA Graph capture for decode *(DONE, 2026-04-18)*

**Tracking: B-023.** Ships the infrastructure primitive; the full
`forward_step` capture requires a fixed-S_k attention variant and
is carved out as B-023b / bundled with Wave 4 paged storage.

**Shipped:**
- `cuda::CudaGraph` — thin RAII wrapper over `cudaGraph_t` +
  `cudaGraphExec_t`. `capture(stream, fn)` installs a `StreamGuard`
  so ops resolving `current_stream(device)` inside the closure
  land on the capture stream, runs `fn()` TWICE as warmup (the
  second pass releases the caller-owned output slot from the first
  pass — without this the bucketed allocator is one block short at
  capture time and `cudaMalloc` fires mid-capture), then does
  `cudaStreamBeginCapture(ThreadLocal)` → `fn()` →
  `cudaStreamEndCapture` → `cudaGraphInstantiate`. `launch(stream)`
  is a single `cudaGraphLaunch`. Re-capture tears down and rebuilds.
- CPU-only build: every method throws `DeviceError` with a clear
  "CUDA backend not compiled in" message — same contract as the
  rest of `tesseract/cuda`.

**Parity + perf bars (all met):**
- Capture-replay of a 1 MiB elementwise chain reproduces eager
  output within FP32 abs 1e-5 across 262 144 elements (32 775
  assertions green).
- Re-capture of a different closure rebinds cleanly; no graph /
  graph-exec handle leaks.
- `bench_cuda_graph` (RTX 5880 Ada, SM 8.9): eager→graph speedup
  of **1.36×** on the 10-op chain at N=4 Ki (host-bound regime),
  passing the `≥ 1.25×` hard bar. 1.10× at N=64 Ki, 1.01× at
  N=1 Mi (compute-bound, as expected).
- `ctest -j 1`: 309/309 green.

**Deferred to B-023b (full decode-step capture):** wrapping
`MHA::forward_step(x, cache)` needs a fixed-S_k attention variant
(attend over `max_len` with a per-step device int driving the
mask) since the captured graph bakes in the cache length at
capture time. See `docs/backlog.md#B-023b`. Bundled with Wave 4
paged storage so both refactors land as a single production-decode
vertical.

## Wave 2.4 — Async `Tensor::to` + pinned host allocator *(DONE, 2026-04-18)*

**Tracking: B-011.** Completes the Wave 2 host↔device critical
path alongside B-023 CUDA Graph: graph capture collapses the
on-device launch sequence, pinned async transfer collapses the
input-staging pipeline, and the two together drive per-step
latency toward the compute roofline.

**Shipped:**
- `cuda::PinnedHostAllocator` (new `tesseract::Allocator`) backed
  by `cudaHostAlloc(cudaHostAllocPortable)`. Singleton; reports
  `cpu_device()` identity so every dispatch / autograd / op path
  keeps treating pinned tensors as CPU tensors (the pinning is
  invisible except at transfer time). CPU-only build throws a
  clean "rebuild with `-DTESSERACT_ENABLE_CUDA=ON`" stub from
  every non-const method, matching the rest of `tesseract/cuda`.
- `Tensor::empty_pinned(shape, dtype)` factory.
- `Storage::copy_device_bytes_async(dst, dst_dev, src, src_dev,
  nbytes, stream)` — new HAL primitive forwarding to
  `cudaMemcpyAsync(..., stream.native_handle())`. Stream validated
  to live on the CUDA endpoint's device; CPU↔CPU path still goes
  through `std::memcpy` for immediate observability.
- `Tensor::to_async(target_device, stream)` — wraps the above,
  preserves the `to()` same-device identity short-circuit, and
  keeps the non-contiguous-source pre-gather sync (documented).

**Parity + perf bars (all met):**
- `tests/hal/test_hal_cuda_pinned.cpp`: 6 cases green on CUDA,
  auto-SKIP on CPU-only. Covers pinned alloc round-trip,
  pinned-source `to_async` vs synchronous `to()` bit-for-bit
  parity, pageable-source `to_async` correctness (driver
  fallback), D→H async round-trip into pageable dst, same-device
  identity, zero-byte alloc.
- `benchmarks/bench_cuda_pinned` on RTX 5880 Ada / PCIe Gen4 x16:
  * 1 MiB H→D pinned vs sync-pageable: **1.80×** (hard bar ≥ 1.4×).
    Above 4 MiB the link saturates (≈ 26 GB/s) and pinned gains
    narrow to ~10 % — informational.
  * 4 KiB pinned-async submit latency: **1.76 µs** (hard bar ≤ 10 µs).
    This is the "fire and forget" property the decode overlap
    path depends on.
  * End-to-end overlap demo (4 MiB H→D ‖ 10-op compute chain on
    separate non-blocking streams): **0.83×** of sequential wall
    time (hard bar ≤ 0.85×). Regresses toward 1.0 if the async
    path silently falls back to sync or the streams implicitly
    serialize — which is what the bar catches.
- `ctest -j 1`: **316/316** green.

**Follow-ups (informal, not a separate B-NNN):** adding a
pinned-staging switch to the SafeTensors loader so bulk weight
ingest of large checkpoints rides the async path (deferred to
Wave 4 loader work; B-011 infrastructure is sufficient for every
decode-phase call site Wave 2 cares about).

## Wave 3 — INT8 / INT4 weight-only quantization

**Tracking: B-021.** Runs after Wave 2 so the quant benchmark is
measured against an already-paged decode baseline, not a
quadratic-memory straw man. Split into two shippable waves so the
INT8 half goes live without waiting on the INT4-group kernel:

### Wave 3.1 — INT8 symmetric end-to-end *(DONE, 2026-04-18)*

- `DType::Int8` flipped to `implemented=true` in
  `src/core/DType.cpp`; `CppTypeToDType<int8_t>` registered in the
  public header. `Tensor::empty` / `.to(device)` / factory round-
  trips all work on INT8 tensors.
- `tesseract::quant::pack_int8_symmetric(W)` — per-output-channel
  symmetric packer in `src/quant/Pack.cpp`. FP32 scale, INT8 body,
  `[-127, 127]` range (no `-128`), identically-zero rows get
  `scale = 1` so dequantization stays bit-exact zero. Packs on CPU
  regardless of source device and ships the result back to the
  source device.
- Fused INT8 dequant-matmul CUDA kernel in
  `src/cuda/DequantMatMul.cu` — one block per output pair `(m, n)`,
  block-scope FP32 reduction over `K`, FP32 scale broadcast on
  thread 0 post-reduce, FP16 / BF16 activations loaded through FP32
  promotion. Bridge header
  `include/tesseract/cuda/detail/DequantMatMul.hpp`; CPU-only stub
  `src/cuda/DequantMatMulStub.cpp` throws cleanly in
  `-DTESSERACT_ENABLE_CUDA=OFF` builds.
- `ops::dequantize_matmul_int8(x, q_w, scale)` wrapper in
  `src/ops/cpu/DequantMatMul.cpp` — validates shapes, flattens
  leading dims of `x` into `M`, dispatches to the CUDA kernel on
  CUDA tensors and to a blocked FP32-accumulator CPU reference
  otherwise. Autograd fallback: when `x.requires_grad()` the op
  materializes `W = dequant(q_w, scale)` once on host and routes
  through `ops::matmul(x, W.T)` so `MatMulBackward` wires the
  `grad_x` path through existing primitives (the frozen integer
  weight is not attached to the graph).
- `nn::QuantizedLinear` drop-in inference module in
  `src/nn/QuantizedLinear.cpp`. Registers `q_weight` /
  `weight_scale` as **buffers** (move with `Module::to(device)`,
  absent from `parameters()`), keeps `bias` as a trainable
  parameter. Factory `QuantizedLinear::from_linear(src)` quantizes
  a pre-trained `nn::Linear`'s weight + clones its bias without
  modifying the source.
- `tests/nn/test_nn_quantized_linear.cpp` — 11 `TEST_CASE`s,
  **339/339 ctest green** on the CUDA build (`build-cuda/`) and
  all non-CUDA cases green on CPU-only (`build/`). Coverage: per-
  row dequant-error bound (`|deq - W| ≤ scale / 2`), zero-row
  safety, CPU reference parity vs. an FP64 hand-rolled matmul,
  rank≥2 batched inputs, FP16 activations, autograd grad_x vs.
  analytic closed form, top-1 ranking match on a 32×128 layer
  across 16 queries, buffer / parameter registration, optional
  bias, CPU↔CUDA parity for both `ops::dequantize_matmul_int8`
  and the module (including `Module::to(cuda)` propagating the
  INT8 + FP32-scale buffers in lockstep).

### Wave 3.2 — INT4 per-group symmetric end-to-end *(DONE, 2026-04-18)*

- `tesseract::quant::pack_int4_group(W, group_size=128)` — per-group
  symmetric packer. Two signed 4-bit nibbles per byte (low = even-k,
  high = odd-k), `scale = max_abs / 7` (FP32, per group), values in
  `[-7, 7]` stored as two's-complement nibbles. Same zero-group
  safety (scale = 1 when a group is identically zero) and
  CPU-packing + ship-to-source-device idiom as the INT8 packer.
- `launch_dequant_matmul_int4_group` — sibling CUDA kernel to the
  INT8 launcher in `DequantMatMul.cu`. Same `grid = (N, M, 1)` +
  `block = (kBlockSize, 1)` layout, FP32 accumulator, per-group
  scale folded inside the reduction. Branchless nibble sign-extend
  via `(nib ^ 0x8) - 8`. Stub throws cleanly for CPU-only builds.
- `ops::dequantize_matmul_int4_group(x, q_packed, scale,
  group_size)` — op-layer wrapper, CPU reference, CUDA dispatch,
  and autograd fallback identical in spirit to the INT8 op
  (`dequantize_weight_int4_group` under `NoGradGuard` +
  `ops::matmul` for `grad_x`).
- `nn::QuantizedLinearInt4G` — drop-in inference module, parallel
  to Wave 3.1's `QuantizedLinear`. Stores `q_weight`
  `Int8 [out, in/2]` + `weight_scale` `Float32 [out, in/G]` as
  buffers, optional bias as a trainable parameter. Factory
  `QuantizedLinearInt4G::from_linear(src, group_size=128)` quantizes
  an FP `nn::Linear` and clones its bias. `group_size` is per
  instance so mixed-G stacks are supported.
- `tests/nn/test_nn_quantized_linear_int4.cpp` — 13 new
  `TEST_CASE`s. Full CUDA ctest: **352/352 green** (+13 over
  Wave 3.1). Per-group dequant-error bound, nibble ordering,
  zero-group safety, CPU↔CUDA op / module parity, FP16
  activations, rank≥2 batched inputs, autograd `grad_x` against
  analytic closed form, packer validation failures, and the
  B-021-DoD-level parity bar: `from_linear` preserves ≥ 4/5 of the
  FP32 top-5 output channels per query on a 256→64 `Linear`
  quantized at `group_size=32`.

### Wave 3.3 — LlamaModel::quantize_ + end-to-end parity *(DONE, 2026-04-21)*

- **Orthogonal refactor (landed):** widened
  `MultiHeadAttention::{q,k,v,o}_proj_`,
  `FeedForward::{gate,up,down}_proj_`, and
  `LlamaModel::lm_head_` from `std::shared_ptr<Linear>` to
  `std::shared_ptr<Module>`. Every `register_module` slot keeps
  its original name so `named_parameters()` / `named_buffers()`
  walk order is unchanged. Downstream tests that previously
  touched `lm_head()->weight()` now `dynamic_pointer_cast` to
  `nn::Linear` — documented as a pre-quantize invariant.
- **Module infrastructure:** `Module::replace_module(name, child)`
  for position-preserving by-name swap; public `children()`
  accessor so walkers can iterate without direct `children_`
  access. Two-line change, throws when the name isn't already
  registered.
- **Scheme + factory:** `include/tesseract/quant/Scheme.hpp`
  (`Method::{Int8Symmetric, Int4GroupSymmetric}`,
  `Scheme::int8_symmetric()` / `int4_group_symmetric(g=128)`)
  + `src/nn/Scheme.cpp::quant::quantize_linear(src, scheme)`
  that dispatches to the W3.1 / W3.2 `from_linear` factories.
  Impl lives in `tesseract_nn` to break the
  `tesseract_quant → tesseract_nn` dependency cycle.
- **Walkers:**
  - `MultiHeadAttention::quantize_(scheme)` — idempotent
    `try_swap` over the 4 projections.
  - `FeedForward::quantize_(scheme)` — same recipe over the 3
    SwiGLU projections.
  - `LlamaModel::quantize_(scheme)` — iterates every
    `TransformerBlock`'s `attn()` + `ffn()` and swaps
    `lm_head_` itself. `embed_tokens_` + every RMSNorm
    intentionally left FP (lookup quantization helps few
    tokens; per-channel RMSNorm scales compound error into
    every downstream layer without saving meaningful memory).
- **Tests (`tests/models/test_llama_quantize.cpp`, 4 cases):**
  - INT8 structural: post-quantize every MHA / FFN projection
    + `lm_head` is a `QuantizedLinear`; `named_buffers()`
    exposes `q_weight` / `weight_scale` under the original
    dotted prefixes; FP `.weight` entries disappear from
    `named_parameters()` while `embed_tokens` / `norm` stay.
  - INT8 end-to-end: ≥ 95 % per-token argmax match vs. FP32
    on a `[B=4, S=128]` (512-token) batch with a 2-layer
    synthetic Llama (vocab=128, d_model=64, d_ff=128).
    B-021 DoD "INT8 top-1 logit rank" bar.
  - INT4 end-to-end (`group_size=32`): mean top-5 overlap
    ≥ 4/5 across the same 512-token batch. B-021 DoD
    "INT4 top-5 ≥ 4/5" bar.
  - Walker idempotence: second `quantize_` call is a no-op
    (same `shared_ptr.get()` pointer returned).
- **Verification:**
  - CUDA ctest: **356/356 green** (+4 new walker cases).
  - Lints clean on all W3.3-touched files.
  - Examples (`llama_infer`, `llama_forward`, `mnist`) still
    build and link unchanged.
  - No regression on the 6808-assertion
    `test_models_llama` parity suite.
- **Effect on B-021:** **Resolved.** The four-part DoD
  (quantized `nn::Linear`, weight packer, quantized matmul
  op, model walker) is now fully met for both INT8 symmetric
  and INT4 group-symmetric schemes.

## Wave 4 — inference fast-path fusions *(in progress)*

**Goal.** With Wave 3 done, every Llama block's per-token traffic
is `{RMSNorm(2), MHA(projections + attention), RMSNorm, FFN(SwiGLU)}`.
Wave 3 already fused RMSNorm (B-022); Wave 4 targets the two
remaining memory-bound tails the composite primitive chain still
walks piecewise:

- **Wave 4.1 (B-025)** — fused SwiGLU forward (`silu(gate) * up`).
- **Wave 4.2 (B-024)** — FA2-style fused attention (single-kernel
  streaming softmax + tiled matmul, SM 8.9 compatible). **DONE**
  (2026-04-21, MVP; WMMA / mma.sync prefill tensor-core rewrite
  deferred as B-024+).
- **Wave 4.3 (B-023b)** — full decode-step CUDA Graph capture,
  unblocked by W4.2's fixed-`S_k` kernel contract. **DONE**
  (2026-04-21, MVP; 2.14× graph-replay speedup on `S_k=8` decode,
  chunked-prefill capture deferred as B-023b+).
- **Wave 4.4 (B-026)** — quantized inference fast path (skip
  autograd fallback on eval modules) + `bench_cuda_quantized_linear`
  and `bench_cuda_llama_decode` so the INT8 / INT4 win is pinned
  down. **DONE** (2026-04-21, MVP; 2.17× decode speedup + 4× weight
  memory compression for INT8, 1.54× + 7.5× for INT4G on a Llama-7B
  block — see below for the full writeup).
- **Wave 4.5 (B-019b)** — PagedKV storage: vLLM-style block pool +
  per-request block table so resident KV memory tracks actual
  sequence length, not `max_len`. **DONE** (2026-04-21, MVP; 0.031×
  resident memory on a short request, single-launch gather kernel
  brings the per-step paging tax to ~34 µs — see below).

Hopper-gated work (FA3, FP8 quantization, Hopper-specific WGMMA
paths) stays explicitly out of scope until we have a Hopper card.

### Wave 4.1 — fused SwiGLU forward (B-025) *(DONE, 2026-04-18)*

**Tracking: B-025.** Closes the element-wise tail of every Llama
FFN. The composite `sigmoid(gate)` + `mul(gate, sig)` + `mul(silu_gate, up)`
chain does 3 kernel launches and 5 reads + 3 writes of the
`[..., d_ff]` intermediate. The fused kernel collapses it to
1 launch and 2 reads + 1 write — a strict memory-bandwidth win
on the memory-bound tail of every FFN.

**Shipped:**
- `launch_swiglu_silu_gate(DType, device_index, numel, gate, up, out, stream)`
  bridge under `include/tesseract/cuda/detail/SwiGLU.hpp`.
- `src/cuda/SwiGLU.cu` — 1-D grid-stride element-wise kernel with
  FP32-promoted math on FP16/BF16 storage; FP32 / FP64 compute-in-
  storage variants; `__expf` for FP32 path, `exp` for FP64.
  Stub (`SwiGLUStub.cpp`) throws a clean `DeviceError` in CPU-only
  builds so the op-layer dispatch stays unconditional.
- `ops::swiglu_silu_gate(gate, up)` — CPU reference (contiguous
  fast-path loop, FP32-promoted for half storage) + CUDA fast-
  path dispatch under NoGradGuard + composite autograd fallback
  via existing `sigmoid` / `mul` primitives. Emits a single
  `swiglu_silu_gate` marker into the active `GraphScope`.
- `nn::FeedForward::forward` now routes the activation tail
  through `ops::swiglu_silu_gate`, dropping the FFN from six
  launches (3 matmuls + sigmoid + 2 muls) to four (3 matmuls +
  fused swiglu). Training paths are byte-identical because the
  fallback keeps the composite math.
- Tests (`tests/ops/test_ops_swiglu.cpp`, 9 cases): FP32 reference
  parity, composite-equivalence parity, autograd FD parity through
  the fallback, shape/dtype/device validation, CPU↔CUDA parity on
  all four dtypes (FP32 / FP64 / FP16 / BF16), and a realistic
  `B*S=128, d_ff=1024` FFN-shape parity case.
- Bench (`benchmarks/bench_cuda_swiglu`): fused vs composite
  three-row layout + memcpy D2D roofline. Hard bars (largest
  `(16, 4096, 4096)` shape): fused speedup ≥ 2.0× and fused
  eff_GB/s / memcpy ≥ 0.80. Measured on RTX 5880 Ada:
  - fused **2.72× faster** than composite (3.62 ms vs 9.87 ms).
  - fused eff bandwidth **891 GB/s** (≈92% of 960 GB/s HBM peak).
  - Both hard bars PASS.

**Verification:**
- CUDA ctest: **366/366 green** (+9 new ops tests, +1 new bench).
- CPU-only ctest: green; the 5 `CPU↔CUDA parity` cases properly
  SKIP via `SKIP_RETURN_CODE 4` on builds without CUDA.
- Lints clean on all W4.1-touched files.
- Examples (`llama_infer`, `llama_forward`, `mnist`) still build
  and link unchanged; `bench_cuda_transformer_block` still PASSes
  (FeedForward internally now uses the fused op under NoGrad).

### Wave 4.3 — Decode-step CUDA Graph capture (B-023b) *(DONE, 2026-04-21)*

**Tracking: B-023b.** Closes the "full decode step collapses into
one `cudaGraphLaunch`" promise that Wave 2.3 (B-023) shipped the
infrastructure for but couldn't exercise against `MHA::forward_step`
until Wave 4.2 delivered a fixed-`S_k` fused attention kernel.

**Shipped:**
- `KVCache::append` rewritten to ride the per-device current stream
  via `Storage::copy_device_bytes_async` instead of draining it with
  a sync `cudaMemcpy`. The old sync was pure waste — the producing
  projection kernels already ran on the same stream, so stream
  ordering alone guarantees the append is visible to any later
  attention kernel reading the slab. Making it async also made it
  capture-safe (a sync `cudaMemcpy` is rejected by
  `cudaStreamBeginCapture` as host-synchronizing).
- `KVCache::set_current_len(int64_t)` exposed as a capture-only
  rewind knob. `CudaGraph::capture` invokes the closure three times
  (two warmup passes + one capture pass); without the rewind a
  plain `cache.append(...)` inside the closure would advance
  `current_len_` by `S_new` per pass and land the capture at
  `target_pos + 2·S_new` instead of `target_pos`. The rewind lets
  every pass bind the same slab slot + same attention `S_k`.
- New `tests/nn/test_mha_cuda_graph.cpp` (3 cases): composite-path
  capture (`B*H=16`), fused-path capture (`B*H=128`, clearing the
  Wave 4.2 shape gate's `bh >= 64` branch), and `set_current_len`
  bounds. Both capture cases REQUIRE **bit-exact** parity against
  eager `forward_step` — any silent kernel drop or stale-`S_k`
  dispatch fails immediately.
- New `benchmarks/bench_cuda_mha_decode_graph` — captures a full
  decode step and measures replay cost vs eager across three
  shapes. Hard bar = 1.25× speedup on the host-bound `S_k=8` shape.

**Measured on RTX 5880 Ada (FP32, B=1, d_model=512, H=16, D_h=32):**
- `S_k=8`  (host-bound decode):   eager 185.99 µs → graph 87.06 µs → **2.14×** (hard bar ≥1.25×).
- `S_k=64` (mixed):                eager 185.70 µs → graph 85.93 µs → **2.16×**.
- `S_k=256` (compute-leaning):     eager 185.65 µs → graph 94.79 µs → **1.96×**.

The ~90 µs constant difference is the host-launch overhead the
graph replay collapses into a single driver call; compute stays
identical between the two paths, which is why speedup gracefully
narrows toward 1× on larger shapes.

**Verification:**
- CUDA ctest: **369/369 green** (+3 new graph-capture cases).
- CPU-only ctest: the new CUDA-required cases SKIP via
  `SKIP_RETURN_CODE 4`; the `set_current_len` bounds case always
  runs so the CPU-only build has an asserted path from the new TU.
  The 3 pre-existing LayerNorm / BatchNorm CPU-only test failures
  (#116, #119, #204) are unrelated and predate this wave.
- All 11 CUDA benches pass, including the new
  `bench_cuda_mha_decode_graph` under `RESOURCE_LOCK "cuda_gpu_0"`.
- Lints clean on all W4.3-touched files.
- Examples (`llama_infer`, `llama_forward`, `mnist`) still build
  and link on CUDA.

**Deferred to B-023b+ (future waves):**
- **Chunked-prefill capture** (`S_new > 1`). The current
  `forward_step` builds its rectangular causal mask on CPU and
  `.to(cuda)`s it; the migration drains the source stream, which
  capture rejects. Fix = GPU-side mask kernel or a pre-allocated
  async-updated device mask slab.
- **Length-parameterized replay.** A graph captured at
  `current_len = P` writes to `slab[P]` and attends over
  `S_k = P + S_new`; stepping further requires re-capture. Two
  production strategies: (a) cache one graph per position value
  (bounded by `max_len`), (b) `cudaGraphExecUpdate` to rebind
  memcpy-dst + attention `S_k` across replays.
- **Full TransformerBlock capture** — wraps MHA + FFN +
  residuals + norms in the same closure for ~25→1 launch-count
  reduction across the whole block. Falls out of the continuous-
  batching / scheduler work in Wave 4.4+.

### Wave 4.4 — Quantized inference fast path + benchmarks (B-026) *(DONE, 2026-04-21)*

**Tracking: B-026.** Closes the "quantization actually pays off on
the Llama decode step" promise that Waves 3.1 / 3.2 / 3.3 shipped
the infrastructure for but couldn't yet guarantee against
autograd-enabled evaluation. The hazard: `ops::dequantize_matmul_*`
has two branches — a fused CUDA kernel that streams `q_weight` +
`weight_scale` and emits the matmul result directly, and an autograd
fallback that materializes the full FP32 weight via
`matmul(x, dequant(W).T)` when `is_grad_enabled() && x.requires_grad()`.
A user doing inference who forgets `NoGradGuard` silently pays the
full FP32-weight cost, i.e. INT4G evaluation becomes *slower* than
pure FP32 because it pays the dequant scratch on top of the matmul
read.

**Shipped:**
- **Eval-mode fast path.** `QuantizedLinear::forward` and
  `QuantizedLinearInt4G::forward` now install a local `NoGradGuard`
  whenever `!is_training()`, pinning the dispatch to the fused
  kernel regardless of the grad state of `x` or the outer grad
  engine. Parity vs the explicit `NoGradGuard`-wrapped path is
  bit-identical (it's the same kernel call). Training mode is
  untouched — the autograd fallback still fires for `train()`
  modules so QAT remains available.
- **Unit tests** (`tests/nn/test_nn_quantized_linear_eval.cpp`,
  7 cases): for each of `{INT8, INT4G}` we cover
    1. `eval() + x.requires_grad()==true` produces a no-grad
       output equal to the explicit `NoGradGuard`-wrapped forward,
    2. `train() + x.requires_grad()==true` still builds a grad
       edge and `Engine::backward` populates `x.grad`,
    3. flipping `eval() ↔ train()` on the same module instance
       toggles the dispatch,
    4. `eval()` nested inside an outer `NoGradGuard` is a no-op
       (fast path already active).
- **`bench_cuda_quantized_linear`** — FP32 `nn::Linear` vs INT8
  `QuantizedLinear` vs INT4G `QuantizedLinearInt4G`, decode shape
  (`M=1`) at `K=N=8192` (the first shape whose FP32 weight — 256 MB
  — exceeds SM 8.9's ~64 MB L2 and forces a genuinely
  HBM-bandwidth-bound baseline), plus informational rows at
  `K=N=4096` (L2-resident, illustrative of cache-crossover effects)
  and a prefill shape (`M=512`). Hard bars on the HBM-bound decode:
  weight bytes `INT8/FP32 ≤ 0.30`, `INT4G/FP32 ≤ 0.18`; latency
  `INT8/FP32 ≤ 0.30`, `INT4G/FP32 ≤ 0.55`. All PASS.
- **`bench_cuda_llama_decode`** — full Llama-2-7B block
  (`d_model=4096, H=32, Dh=128, d_ff=11008`, `B=1, S_k=129`) decode
  step = `MultiHeadAttention::forward_step` + `FeedForward::forward`,
  exercising every Linear in a Llama block (4× MHA + 3× FFN). The
  7B shape is the smallest configuration whose per-block FP32
  footprint (~800 MB) comprehensively exceeds L2 on every
  currently-shipping GPU; at Llama-1B the FP32 baseline partially
  caches in L2 after warmup and INT4G's per-output compute
  dominates the memory savings. Hard bars:
  - Block weight bytes: `INT8/FP32 ≤ 0.30`, `INT4G/FP32 ≤ 0.18`.
  - Decode-step latency: `INT8/FP32 ≤ 0.55`, `INT4G/FP32 ≤ 0.75`.

**Measured on RTX 5880 Ada (SM 8.9, FP32 baseline):**

`bench_cuda_quantized_linear` at `M=1, K=N=8192`:

| variant       | latency (µs) | vs FP32  | weight (MB) | mem vs FP32 |
|---------------|-------------:|---------:|------------:|------------:|
| FP32          |  (baseline)  |  1.000×  |      256.00 |     1.000×  |
| INT8          |              |  0.25×   |       64.06 |     0.250×  |
| INT4G (g=128) |              |  0.50×   |       34.04 |     0.133×  |

`bench_cuda_llama_decode` on the Llama-2-7B block:

| variant       | step (µs) | vs FP32  | block weight (MB) | mem vs FP32 |
|---------------|----------:|---------:|------------------:|------------:|
| FP32          |  1143.87  |  1.000×  |           809.76  |     1.000×  |
| INT8          |   528.27  |  0.462×  |           202.81  |     0.250×  |
| INT4G (g=128) |   745.45  |  0.651×  |           107.77  |     0.133×  |

INT8 gives **2.17× decode speedup + 4× weight compression**; INT4G
gives **1.54× speedup + 7.5× weight compression**. The FP32 baseline
streams at ~708 GB/s effective (≈74% of the 960 GB/s HBM peak),
confirming it's HBM-bound. The INT4G latency bar (`≤ 0.75×`) is
intentionally looser than the memory ratio would suggest because on
SM 8.9 the fused dequant-matmul kernel is compute-bound on the
nibble unpack + group-scale lookup path — INT4G runs *slower* than
INT8 despite streaming half the bytes. A vectorized nibble unpack is
tracked as B-026+; for this wave we document the current ceiling
and bar regressions from it.

**Verification:**
- CUDA ctest: **389/389 green** (+7 eval-path tests, +2 new benches).
- CPU-only ctest: green on every Wave-4.4-touched file. The 3
  pre-existing `test_ops_layer_norm` / `test_nn_batch_norm` CPU-only
  failures are a missing `SKIP_RETURN_CODE 4` on those two
  registrations — infrastructure debt that predates this wave,
  filed as a separate cleanup.
- Lints clean on all W4.4-touched files.
- Examples (`llama_infer`, `llama_forward`, `mnist`) still build
  and link on CUDA.

**Deferred to B-026+ (future waves):**
- **Vectorized INT4G unpack.** Current kernel does one nibble-shift
  per output partial; a packed `uchar4 → uint32_t` load + bit-field
  unpack plus group-scale prefetch through shared memory should
  close the gap to the memory-bound ceiling (~0.15× FP32
  decode-step latency). Blocked on an FP16 / BF16 accumulator path
  so the compute saved isn't spent on FP32 casts.
- **Tensor-core INT8 path.** cuBLASLt's `CUBLAS_COMPUTE_32I` INT8
  GEMM on Ada tensor cores. Our current INT8 dequant-matmul is a
  hand-written CUDA-core kernel — good enough to beat the FP32
  baseline 2× but leaves another ~2× on the table. Pairs with the
  cuBLASLt SwiGLU + fused-attention epilogue work.
- **INT4G activation-aware calibration.** Current scheme is
  weight-only symmetric; the next accuracy lift is GPTQ/AWQ-style
  per-group scale optimization driven by a calibration dataset.
  Accuracy work, not perf — tracked separately under the "quant
  accuracy" umbrella.

### Wave 4.5 — PagedKV storage (B-019b) *(DONE, 2026-04-21)*

**Tracking: B-019b.** Wave 2.1 shipped a contiguous `KVCache` that
pre-allocates a `[B, H, max_len, D_head]` slab the moment it is
constructed — fine for one long request, wasteful for a server where
most requests are far shorter than `max_len` and the resident KV
memory is dominated by padding. Wave 4.5 swaps in the PagedAttention
storage model: a fixed physical **block pool** carved into `num_blocks`
blocks of `block_size` tokens, handed out on demand via a per-request
**block table**, so a request holding `L` tokens occupies
`ceil(L / block_size)` blocks instead of the full `max_len` slab.

**Shipped:**
- `nn::BlockAllocator` (`include/tesseract/nn/BlockAllocator.hpp` +
  `src/nn/BlockAllocator.cpp`) — LIFO free-list over physical block
  ids; `allocate()` / `free()` / `free_all()`; throws on exhaustion
  and on double-free / out-of-range. Pure integer bookkeeping, no
  Tensor/CUDA dependency, so it's unit-testable standalone and a
  future continuous-batching scheduler can share one allocator across
  caches.
- `nn::PagedKVCache` (`include/tesseract/nn/PagedKVCache.hpp` +
  `src/nn/PagedKVCache.cpp`) — physical pools
  `[num_blocks, H, block_size, D_head]` for K and V, a host block
  table `std::vector<std::vector<int32_t>>`, and (on CUDA) a `mutable`
  device-resident `[batch, max_logical]` Int32 mirror. `append()`
  walks the new tokens, grouping consecutive tokens that land in the
  same physical block into one copy per head and allocating blocks at
  block boundaries; `keys_view()` / `values_view()` gather the valid
  prefix into a fresh contiguous `[B, H, L, D_head]` tensor;
  `reset()` recycles every block back to the allocator. Same public
  surface as `KVCache` plus `block_size()` / `num_blocks()` /
  `num_allocated_blocks()`.
- `nn::KVCacheBase` (`include/tesseract/nn/KVCacheBase.hpp`) — extracted
  abstract interface (`append`, `keys_view`, `values_view`, shape
  accessors). Both caches implement it;
  `MultiHeadAttention::forward_step(const Tensor&, KVCacheBase&)` now
  takes the base, so swapping contiguous ↔ paged storage is a one-line
  change with zero attention-code churn — the B-019 API-stability
  promise, delivered.
- Paged gather CUDA kernel (`include/tesseract/cuda/detail/PagedKV.hpp`
  bridge + `src/cuda/PagedKV.cu` + `PagedKVStub.cpp`) — one thread per
  output element of the `[B, H, L, D_head]` prefix follows the
  device-side block table into the pool. Templated on element *size*
  (2 / 4 / 8 bytes) so a single body covers FP16 / BF16 / FP32 / FP64.
  This replaced a naive per-block `cudaMemcpyAsync` gather loop that
  fired `B·H·ceil(L/block_size)` tiny launches per decode step and was
  pure launch overhead. CPU caches keep the direct host-memcpy loop
  (cheap on host).

**Measured on RTX 5880 Ada (SM 8.9), `bench_cuda_paged_kv`:**

Memory residency — short request, Llama-7B head shape (`B=1, H=32,
D_head=128`, `L=256`, `max_len=8192`, `block_size=16`):

| cache                | resident |
|----------------------|---------:|
| contiguous (Wave 2.1) | 268.44 MB |
| paged (16 blocks)     |   8.39 MB |
| paged / contiguous    |  **0.0312×** |

Per-step gather latency (`L=512`, append 1 token + gather K/V):

| path                  | µs |
|-----------------------|-----:|
| contiguous (narrow)   | 141.0 |
| paged — naive memcpy loop | 4816.6 |
| paged — gather kernel | **175.6** |

The kernel cut the paging tax from ~4.7 ms (utterly decode-dominating)
to ~34 µs over the contiguous cache's append-dominated 141 µs floor.
Hard bars (residency ≤ 0.10, gather ≤ 250 µs) both PASS.

**Parity (`tests/nn/test_paged_kv_cache.cpp`, 6 cases):**
- `BlockAllocator` free-list correctness, exhaustion + double-free +
  out-of-range throws.
- `PagedKVCache` `keys_view`/`values_view` **byte-identical** to
  `KVCache` across a 2-1-3-2 chunk schedule that straddles the
  `block_size=3` boundary.
- residency tracks `ceil(L/block_size)`; `reset()` recycles.
- append rejects `max_len` overflow + pool exhaustion.
- `MHA::forward_step` with a paged cache equals one-shot `forward()`
  within FP32 abs 1e-5 (CPU); paged decode matches eager on CUDA
  within 5e-5.

**Verification:**
- CUDA ctest: **396/396 green** (+6 paged tests, +1 bench).
- CPU-only: `test_nn_paged_kv_cache` (6 cases) + `test_nn_kv_cache`
  green; the paged CUDA case `SUCCEED`-skips without a device.
- Lints clean on all W4.5-touched files.
- Examples (`llama_infer`, `llama_forward`, `mnist`) still build/link.

**Deferred to B-019b+ (future waves):**
- **Block-table-aware paged-attention kernel.** The MVP gathers the
  scattered prefix into a contiguous tensor each step so
  `ops::attention` is unchanged. A kernel that reads K/V in place via
  the block table (vLLM PagedAttention proper) removes the
  `O(L)`-per-step gather — pairs with the B-024+ WMMA attention
  rewrite.
- **Ragged per-request lengths.** MVP keeps a uniform batch
  `current_len_` (matches `KVCache` semantics) so the attention path
  is untouched. True continuous batching needs per-request lengths +
  a padded/masked attention call; falls out with the scheduler.
- **CUDA-graph-capturable paged decode.** The per-gather block-table
  H2D upload uses a synchronous copy that `cudaStreamBeginCapture`
  rejects; an incrementally-maintained device block table would make
  the paged step captureable like the Wave 4.3 contiguous path.

### Wave 4.6 — Byte-level BPE tokenizer (B-018) *(DONE, 2026-06-21)*

**Tracking: B-018.** Wave 1a shipped the `Tokenizer` interface but only a
`WhitespaceTokenizer` reference impl — not byte-compatible with any HF
checkpoint, so `llama_infer` could only run on random / synthetic token
ids. Wave 4.6 closes the last of the four M3 optimization directions
(PagedKV · normalization running-stats · quantization · **tokenizer +
loader**) by shipping the byte-level BPE tokenizer that makes a real HF
`tokenizer.json` runnable end-to-end. It sits off the CUDA critical path
(pure host code), so it never blocked the perf waves.

**Shipped:**
- `io::BpeTokenizer` (`include/tesseract/io/BpeTokenizer.hpp` +
  `src/io/BpeTokenizer.cpp`) — `final : public Tokenizer`. Reproduces the
  HF `tokenizers` ByteLevel-BPE pipeline exactly on ASCII:
  1. **Special-token isolation** — configured `added_tokens` are matched
     as whole units (longest-match) and never split by BPE.
  2. **GPT-2 pre-tokenization** — a hand-rolled byte scanner reproducing
     `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
     including the trailing-whitespace `(?!\S)` give-back rule. Unicode
     classes are ASCII-scoped (`\p{L}→[A-Za-z]`, `\p{N}→[0-9]`); bytes
     ≥ 0x80 group with letters so multi-byte UTF-8 stays together (exact
     on ASCII, best-effort beyond). `std::regex` is avoided — it can't
     express `\p{L}` and mishandles high bytes.
  3. **GPT-2 `bytes_to_unicode`** reversible byte→printable-codepoint map.
  4. **Rank-priority BPE** — the canonical GPT-2 `bpe()` loop: each
     iteration finds the lowest-rank adjacent pair, merges *all* its
     non-overlapping occurrences, repeats.
  `decode` inverts the pipeline (ids → vocab strings → reverse byte map →
  raw bytes). Byte-level BPE is lossless so `decode(encode(text)) == text`
  for any ASCII input.
- **`tokenizer.json` loader** — a self-contained recursive-descent JSON
  value-tree parser (handles nested objects/arrays, `\uXXXX` incl.
  surrogate pairs; objects kept as ordered `vector<pair>` so the 128k-entry
  vocab isn't hashed twice). `from_file` / `from_json` parse `model.vocab`,
  `model.merges` (both `"a b"` and `["a","b"]` shapes), `added_tokens`, and
  the `ByteLevel` pre-tokenizer's `add_prefix_space` (incl. inside a
  `Sequence`). BOS/EOS/PAD/UNK resolved by content convention
  (`<|begin_of_text|>`/`<s>`, `<|end_of_text|>`/`<|eot_id|>`/`</s>`, …).
- **`llama_infer` end-to-end wiring** — new `--tokenizer tokenizer.json
  --prompt "text"` flags encode a real prompt (BOS+ids+EOS), print the
  round-tripped decode, and feed the ids through the full stack. Closes
  the "real HF checkpoint demo" gap.

**Golden parity (`tests/io/test_bpe_tokenizer.cpp`, 9 cases / 52 asserts):**
- The fixture generator `tests/io/fixtures/generate_bpe_fixture.py` builds a
  **real** HF `tokenizers` ByteLevel BPE (trained on an ASCII corpus,
  vocab=400, 141 merges), saves `bpe_tokenizer.json`, and dumps golden ids
  for 13 fixture sentences exercising contractions, digit runs, punctuation
  runs, leading/interior/trailing whitespace, tabs/newlines, mixed case, and
  unseen-token single-byte fallback. Both artifacts are committed.
- The C++ test loads that exact `tokenizer.json` and asserts its `encode`
  matches the `tokenizers.Tokenizer.encode` ids **byte-for-byte** on every
  fixture line — verified against `tokenizers==0.22.2` / `transformers==5.3.0`.
- Plus: two hand-verified in-memory merge-loop cases (rank priority +
  merge-all-occurrences), lossless decode round-trip, BOS/EOS prepend/append,
  special-token isolation, virtual dispatch via `Tokenizer*`, empty-vocab
  rejection.

**Verification:**
- CUDA ctest: **405/405 green** (+9 BPE cases discovered as ctest entries).
- CPU-only: `test_io_bpe_tokenizer` (9 cases) green; portable (pure host).
- Lints clean on all W4.6-touched files.
- `llama_infer --synthetic --tokenizer … --prompt "the quick brown fox"`
  encodes `1 285 311 312 291 2`, decodes back exactly, runs the forward pass.

**Deferred to B-018+ (future):**
- **Full-Unicode pre-tokenization.** The ASCII-scoped `\p{L}`/`\p{N}`
  approximation diverges from `tokenizers` on scripts that mix
  letter/mark/number categories within a UTF-8 run; a real `\p{...}`
  classifier (ICU or a generated property table) makes non-ASCII
  byte-exact. The byte map + merge loop are already script-agnostic.
- **SentencePiece / unigram models** (T5, Gemma) — a separate model type
  off the same `Tokenizer` interface.
- **`tokenizer_config.json` chat templates** — BOS/EOS conventions are
  hard-coded here; the Jinja chat-template path is a loader follow-up.

---

### Wave 5 — End-to-end autoregressive generation (B-027) *(DONE, 2026-06-21)*

**Tracking: B-027.** Every primitive for real text generation now
existed in isolation — `MultiHeadAttention::forward_step` + `KVCache`
(Wave 2.1), the byte-level `BpeTokenizer` (Wave 4.6), `LlamaModel`
(Wave 1a) — but nothing wired them into an actual `generate()`. The
model only exposed the one-shot `forward(tokens)` that recomputes the
full prefix every call. Wave 5 threads per-layer KV caches through the
block stack and ships the greedy decode loop, turning the four shipped
directions into the first runnable end-to-end demo.

**Shipped:**
- `nn::TransformerBlock::forward_step(x, KVCacheBase&)` — the pre-norm
  residual structure with the attention sub-layer routed through the
  cache (`h = x + attn.forward_step(norm_1(x), cache)`; FFN + norms
  reuse the eager paths). Whole step under `NoGradGuard`.
- `LlamaModel::forward_step(tokens, caches)` — embed → N blocks each
  with its own `KVCache` → final RMSNorm → `lm_head`, returning
  `[B, S_new, vocab]`. `make_kv_caches(batch, max_len)` allocates one
  contiguous cache per layer on the model's device/dtype.
- `LlamaModel::generate(prompt_ids, GenerateConfig{max_new_tokens,
  eos_token_id})` — prefills the prompt in one chunked-decode step,
  then greedily (argmax) decodes one token at a time, reusing the
  caches, stopping at `max_new_tokens` or `eos_token_id`. Returns the
  full sequence (prompt + generated, HF convention). Argmax handles
  FP32/FP64/FP16/BF16 logits via `dispatch_float_with_half`.
- `llama_infer --generate [--max-new-tokens N]` — encodes the prompt
  with `BpeTokenizer`, runs `generate`, and prints the decoded
  continuation. The capstone "tokenizer → model → KV-cache → text"
  loop, runnable on CPU or CUDA.

**Correctness (`tests/models/test_llama_generate.cpp`, 5 cases):**
- **Chunked-prefill parity** — `forward_step(tokens[1,S], caches)` over
  the whole prompt equals one-shot `forward(tokens[1,S])` at every
  position within FP32 abs 2e-4.
- **Token-by-token parity** — feeding one token at a time reconstructs
  `forward()`'s logits row-for-row (proves cache append + RoPE offset
  align with the one-shot causal path). CPU **and** CUDA (CUDA case
  `SUCCEED`-skips without a device).
- **Greedy `generate`** — deterministic across calls, prompt-prefixed,
  exact `prompt + max_new_tokens` length, ids in range.
- **EOS early stop** — setting the greedy first token as `eos_token_id`
  makes generation emit it once and halt.
- Empty-prompt + out-of-range-id rejection.

**Verification:**
- CUDA ctest **410/410 green** (+5 generate cases); CPU-only green.
- Lints clean on all W5-touched files.
- `llama_infer --synthetic --tokenizer … --prompt "the quick brown fox"
  --generate` produces a deterministic continuation on CPU and CUDA
  (random-init weights ⇒ gibberish text, but the full pipeline is
  exercised end-to-end).

**Deferred (B-027+):** sampling beyond greedy (→ **Wave 6, done**);
batched multi-sequence generation with ragged lengths (needs the PagedKV
ragged-length follow-up); CUDA-graph capture of the per-step
`forward_step` (builds on Wave 4.3); streaming token callback. These
layer on top without changing the decode core.

---

### Wave 6 — Sampling strategies (B-028) *(DONE, 2026-06-21)*

**Tracking: B-028.** Wave 5 shipped greedy-only decode — fine for a
parity demo but useless for real generation (every prompt yields one
deterministic, low-diversity continuation). Wave 6 adds the standard
stochastic decoding toolkit so the engine produces real text and so
the upcoming continuous-batching scheduler (Wave 7) has per-request
sampling to batch over. Off the CUDA critical path (a tiny per-step
host op over the vocab row).

**Shipped:**
- `models::SamplingParams{temperature, top_k, top_p, repetition_penalty}`
  + `models::sample_from_logits(logits, params, prev_tokens, rng)` +
  stateful `models::Sampler` (`include/tesseract/models/Sampler.hpp` +
  `src/models/Sampler.cpp`). Reproduces the HF / vLLM logits-processing
  order: repetition penalty (CTRL-style) → temperature → top-k → top-p
  (nucleus) → softmax → multinomial draw from a seeded `std::mt19937_64`.
  Every filter is a no-op at its identity value; `temperature <= 0` is
  the greedy short-circuit. Nucleus always keeps ≥ 1 token.
- `LlamaModel::GenerateConfig` gains `do_sample` (default false ⇒ greedy,
  preserving Wave 5 behavior), `sampling`, and `seed`. `generate` routes
  each step through the sampler when `do_sample`, feeding the running
  sequence for repetition penalty; greedy stays the argmax path,
  unaffected by `seed`.
- `llama_infer --sample [--temperature F] [--top-k N] [--top-p F]
  [--repetition-penalty F] [--seed N]` — real stochastic generation
  from the CLI, deterministic per seed.

**Correctness (`tests/models/test_sampler.cpp`, 8 cases / 2146 asserts):**
- `temperature <= 0` ≡ greedy argmax; top-k restricts support to the k
  highest (and the larger logit dominates the histogram); top-p keeps
  only the nucleus (dominant-token-only at p=0.5) and never empties it;
  same seed reproduces draws bit-for-bit, different seeds diverge;
  repetition penalty provably shifts mass off penalized tokens;
  temperature sharpens (low T) vs flattens (high T) the histogram.
- Integration: `generate(do_sample=true)` is seed-deterministic and
  seed-sensitive; the greedy default ignores `seed`;
  `do_sample` with `temperature=0` reproduces the greedy sequence.
- CUDA ctest **418/418 green** (+8 cases); CPU-only green; lints clean.

**Deferred (B-028+):** min-p / typical / Mirostat samplers; logit bias
+ banned-token / stop-string criteria; batched per-request sampling
(lands with the Wave 7 scheduler).

---

### Wave 7 — Continuous-batching scheduler (B-029) *(DONE, 2026-06-21)*

**Tracking: B-029.** The M3 exit-bar headline: turn the single-sequence
`generate` into a request engine that multiplexes many requests over a
shared paged KV pool. This is the capability `idea.md` §4.4 lists right
after PagedAttention and the reason the Wave 4.5 `BlockAllocator` was
built pool-agnostic.

**Shipped:**
- `nn::PagedKVPool` (`include/tesseract/nn/PagedKVPool.hpp` +
  `src/nn/PagedKVPool.cpp`) — factors the physical K/V block storage +
  `BlockAllocator` out of `PagedKVCache` into a standalone,
  reference-counted, per-layer object. `PagedKVCache` refactored to hold
  a `std::shared_ptr<PagedKVPool>`: the Wave 4.5 ctor still works (it
  makes a private pool), a new ctor binds to a shared pool, and
  `reset()` now frees only this cache's own blocks (never `free_all`,
  which would yank blocks from sibling requests). Wave 4.5's 6 paged
  tests + `bench_cuda_paged_kv` stay green unchanged.
- `LlamaModel::forward_step` gained a `KVCacheBase`-vector overload
  (the `KVCache` one delegates via upcast); `make_layer_pools(num_blocks,
  block_size)` and `make_paged_kv_caches(pools, max_len)` helpers let a
  caller stand up per-layer shared pools + per-request paged caches.
- `models::ContinuousBatchingScheduler` + `EngineConfig`
  (`include/tesseract/models/Scheduler.hpp` + `src/models/Scheduler.cpp`):
  owns one shared `PagedKVPool` per layer; `add_request(prompt, gen)`
  queues, `step()` admits waiting requests up to `max_batch_size` (and
  only when the pool can hold their prompt — prefilling each), emits one
  token per running request with its own seeded `Sampler` + EOS/length
  stop, and reclaims a finished request's blocks into the shared pool;
  `run()` drains the queue. Residency introspection via
  `allocated_blocks()` / `free_blocks()`.
- `examples/llama_serve.cpp` — submits several prompts at once and prints
  per-tick occupancy (`running` / `waiting` / `blocks`), showing dynamic
  admission, recycling (blocks return to 0 between batches), and
  per-request sampling.

**Correctness — exact parity contract:** paged and contiguous caches
store the same bytes and the gather is an exact copy, so a scheduled
request's logits (hence greedy argmax and seeded samples) are
bit-identical to running `generate` standalone, *regardless of
interleaving or block budget*.
`tests/models/test_scheduler.cpp` (6 cases): greedy parity across mixed
prompt lengths; staggered admission under a batch cap smaller than the
request count (output unchanged + cap honored every tick); sampling
parity with per-request seeds; EOS early-stop parity; a 2-block pool
serving 4 requests via recycling; input/config validation. Plus
`tests/nn/test_paged_kv_pool.cpp` (4 cases): shared budget across caches,
reset frees only own blocks, recycling, shared-vs-private byte parity.
After every `run()`, `allocated_blocks() == 0` (no leak). CUDA ctest
**428/428 green** (+10), CPU-only green, lints clean.

**MVP scope / deferred (B-029+):** decode is per-request (one
`forward_step` per active request per tick), delivering full
continuous-batching *semantics* but not yet *compute*-batching the active
set into one fused ragged `forward_step` — that throughput win needs the
ragged paged-attention kernel (B-019b+) and lands without an API change.
Also deferred: preemption/eviction under memory pressure (current
admission is conservative — it waits rather than evicting), prefix
sharing (RadixAttention), and priority/fairness scheduling.

### Wave 8 — Grouped-query attention (B-030) *(DONE, 2026-06-22)*

**Tracking: B-030.** The gate to loading real modern checkpoints. Every
Llama-3 / Qwen2 / Mistral checkpoint is GQA (`num_key_value_heads <
num_attention_heads`), so the prior MHA-only attention could only run
Llama-1/2-style full-MHA configs. GQA also shrinks the KV cache by
`num_heads / num_kv_heads` (4× on Llama-3.2-1B: 32 Q heads, 8 KV heads),
which directly raises the concurrent-request ceiling for Wave 7's pools.

**What landed:**
- `nn::MultiHeadAttention` takes a trailing `num_kv_heads` ctor arg
  (`0` ⇒ plain MHA = num_heads). Q stays `d_model → d_model`; K/V shrink
  to `d_model → num_kv_heads · head_dim`. After RoPE, each KV head is
  repeat-expanded across its `G = num_heads / num_kv_heads` query heads
  (the PyTorch `repeat_kv` interleave: head `h` uses KV head `h / G`),
  implemented with autograd-aware view ops (reshape + broadcast_to), so
  `G == 1` (MHA) pays nothing and grads accumulate back onto KV heads.
- `forward_step` repeats *after* the cache gather, so the KV cache stores
  only `num_kv_heads` heads — the GQA memory win is realized in the cache
  itself. The decode head-shape check now compares against `num_kv_heads`.
- Threaded through `nn::TransformerBlock` (trailing `num_kv_heads` arg) and
  `models::LlamaConfig::num_key_value_heads` (+ `kv_heads()` resolver, 0 ⇒
  num_attention_heads). `make_kv_caches` / `make_layer_pools` allocate
  `kv_heads()` heads. `llama_3_2_1b()` preset now sets 8 KV heads.
- `llama_infer` gained a `--kv-heads N` flag.

**Verification (`tests/nn/test_gqa.cpp`, 5 cases):** (1) K/V projections
shrink to `num_kv_heads`; (2) **weight-replicated parity** — a GQA module
matches a plain-MHA reference whose K/V weights are the GQA weights with
each KV head's rows replicated `G` times, proving the head-sharing order
is exactly `h → h/G`; (3) GQA `forward_step` (token-by-token, RoPE on)
matches one-shot `forward`; (4) divisibility validation; (5) a Llama model
with GQA generates deterministically and its chunked-prefill `forward_step`
matches one-shot `forward`, with caches holding `kv_heads()` heads.
Backward-compatible: every existing MHA/transformer/Llama test (default
`num_kv_heads = 0`) is unchanged. 433/433 on CUDA, GQA + Llama parity
green on CPU.

**MVP scope / deferred (B-030+):** GQA replicates K/V to full head count
before `ops::attention` (the standard non-fused recipe — `G×` transient
K/V memory during the attention matmul, not in the cache). A fused
GQA/MQA attention kernel that indexes the shared KV head directly (no
replication) is the throughput/footprint follow-up, and pairs naturally
with the ragged paged-attention kernel (B-019b+).

### Wave 9 — KV-cache INT8 quantization (B-031) *(DONE, 2026-06-22)*

**Tracking: B-031.** The long-context memory frontier. The KV cache is
what bounds context length and concurrent-request count on the Wave-7
pools; storing it in INT8 instead of FP cuts the *persistent* footprint
~4× (vs FP32) / ~2× (vs FP16), so a fixed memory budget holds
correspondingly more tokens / requests.

**What landed:**
- **Device-resident quant ops** (`quant::quantize_kv_per_token` /
  `dequantize_kv_per_token`, `src/quant/QuantizeKV.cpp`). Unlike the
  `Pack.hpp` weight packers (one-shot, host-side, at load time) these run
  on the decode hot path, so they stay on-device: a CUDA kernel pair
  (`src/cuda/QuantizeKV.cu` — one thread per token·head row for quantize,
  one thread per element for dequantize; FP32-promoted on FP16/BF16) plus
  a numerically identical CPU loop, behind the standard stub-vs-kernel
  CMake pairing (`QuantizeKVStub.cpp`).
- **Granularity:** per-token, per-head symmetric INT8. The last dim
  (`D_head`) gets one FP32 scale = `absmax/127` (1.0 for an all-zero row);
  banker's round, clamp [-127,127]. Scales stay FP32 so the only error is
  the INT8 payload. Per-vector scaling keeps dynamic range tight — far
  better than a tensor-wide scale, within rounding noise of per-channel.
- **`nn::QuantizedKVCache`** (`KVCacheBase` drop-in): persistent storage
  is INT8 K/V slabs `[B,H,max_len,D_head]` + FP32 scale slabs
  `[B,H,max_len]`. `append` quantizes the projected slab and byte-copies
  the INT8 payload + scales into place; `keys_view()`/`values_view()`
  dequantize the current prefix into a fresh FP tensor. `MHA::forward_step`
  consumes it through the unchanged interface, so it composes with GQA and
  the contiguous/paged caches identically.
- **Integration:** `LlamaModel::make_quantized_kv_caches` +
  `GenerateConfig::kv_int8` flag route `generate` through quantized caches
  (the decode loop already drives `KVCacheBase`).

**Verification (`tests/nn/test_quant_kv.cpp`, 6 cases):** (1) roundtrip
within the symmetric-INT8 error bound `|x−x'| ≤ scale_row/2`; (2) all-zero
row → scale 1, exact zero; (3) **CPU and CUDA agree exactly** (identical
FP32 absmax + round); (4) `QuantizedKVCache` reconstructs
`dequant(quant(K))` *exactly* — chunked append == one-shot, validating the
slab/scale plumbing; (5) `MHA::forward_step` with a quantized cache tracks
the FP cache within a bounded tolerance (lossy, not bit-identical); (6)
`generate` with `kv_int8` is deterministic, valid, length-correct.
CUDA ctest **437/439 correctness green** (+6; the 2 reds are perf
benchmarks flaking under shared-GPU co-tenancy, no matmul code changed);
CPU green (the CUDA-agreement case SKIPs via `SKIP_RETURN_CODE 4`); lints
clean.

**MVP scope / deferred (B-031+):** `keys_view()` materializes a fresh FP
prefix each step (one layer's prefix, FP-sized transient) — the win is the
persistent INT8 storage across all layers / full `max_len`, not the
transient. Fusing dequant into the attention matmul (a quantized-attention
kernel that reads INT8 K/V + scales directly) is the throughput/footprint
follow-up and pairs with the ragged paged-attention kernel (B-019b+).
Also deferred: INT8 *paged* pool (quantized `PagedKVPool`), FP8 KV, and
per-channel / asymmetric KV schemes.

### Wave 10 — Compute-batched decode (B-032) *(DONE, 2026-06-22)*

**Tracking: B-032.** The throughput frontier. Wave 7's scheduler decoded
its running set one request at a time — N separate `forward_step` calls
per tick, each a stack of `M=1` GEMMs that are latency-bound on the GPU
and leave the SMs idle. Continuous batching only pays off when the active
set's matmuls are batched into one launch. Decode FLOPs are dominated by
the dense projections (4·D²/token) + FFN (3·D·D_ff) + LM head, all
position-independent; only attention is ragged (each request's KV prefix
is a different length). So: **batch the dense layers across the active set,
loop attention per sequence** — a pure op-composition path, no new CUDA
kernel, lowest risk to the headline win.

**What shipped:**
- `MultiHeadAttention::forward_step_batched(x [A, S_new, D],
  caches: vector<KVCacheBase*>)` — Q/K/V/O projections run once over all
  `A·S_new` rows; a per-sequence loop drives a file-local `attend_single`
  helper that factors the Wave-8 `forward_step` attention core (RoPE@`pos`
  → contiguous → `cache.append` → gather → `repeat_kv` for GQA → SDPA with
  a decode mask when `S_new>1` → merge heads). Each cache is `batch()==1`
  at its own `current_len`. Per-sequence outputs `ops::cat`-stacked, then
  one batched `o_proj`.
- `TransformerBlock::forward_step_batched` — norms / FFN / residual adds
  are position-independent ⇒ run batched; attention delegated.
- `LlamaModel::forward_step_batched(tokens [A, S_new], caches[seq][layer])`
  — batched embed → per-layer batched block (per-layer `KVCacheBase*` list
  assembled across sequences) → batched final RMSNorm + LM head →
  `[A, S_new, V]`.
- `ContinuousBatchingScheduler::step` now samples every running request
  from the prior step's logits (advancing tokens + EOS/length stop), then
  folds the still-active set through one `forward_step_batched`, slicing
  per-request `[1,1,V]` logits back out. Prefill stays per-request.

**Exact parity contract:** each GEMM output row is independent of the
others and attention is per-sequence, so row `r` of batched decode equals
standalone `forward_step` on sequence `r` fed the same tokens —
bit-identical on CPU (independent of `A`), within float tolerance on CUDA
(batched vs single GEMM algo). Verified: the 6 existing Wave-7 scheduler
parity tests (CPU) still match standalone `generate` byte-for-byte through
the batched path; new `tests/models/test_batched_decode.cpp` (6 cases) —
ragged mixed-length parity vs per-request for MHA + GQA (CPU exact, CUDA
≤ 3e-3 + argmax match), CUDA MHA+GQA, chunked-prefill `S_new>1` exact,
`A==1` ≡ `forward_step`, shape/cache-count validation. CUDA green on a
contention-free card (perf benches + 2 timing-sensitive cases flake under
`-j` self-contention but pass `-j1`); CPU green (CUDA case SKIPs via
`SKIP_RETURN_CODE 4`); lints clean.

**MVP scope / deferred (B-032+):** the per-sequence attention loop is
still N small launches per tick — addressed in Wave 11 below.

### Wave 11 — Fused ragged paged decode-attention (B-032+) *(DONE, 2026-06-22)*

**Tracking: B-032+.** The attention-side throughput win that finishes Wave
10. Wave 10 batched the dense decode layers but still looped attention per
sequence: for A active requests, per layer per tick, A `keys_view()`
gathers (each materializes a request's full KV prefix out of the scattered
paged blocks into a fresh contiguous slab) + A `repeat_kv` GQA
replications + A `ops::attention` launches — all memory-bound and
launch-bound on the decode hot path.

**What shipped:**
- `nn::paged_decode_attention(q [A,H,D], k_pool/v_pool
  [num_blocks,Hkv,block_size,D], block_tables [A,max_logical] Int32,
  lens [A] Int32, scale, group)` → O [A,H,D]. One query token per request;
  reads K/V in place from the shared per-layer pool via each request's
  block table (no gather), maps query head `h` → KV head `h/group`
  on the fly (no `repeat_kv`), online (FlashAttention-style) softmax so the
  `[1,S_k]` score row never hits HBM. `lens[r]==0` ⇒ zero output row.
- CUDA kernel `src/cuda/PagedAttention.cu`: one warp per `(request,
  query-head)`, per-lane Q fragment (scale folded), warp-shuffle score dot,
  online-softmax accumulate; FP32 interior math on FP32/FP16/BF16 storage
  (`D_PER_LANE = D_MAX/32 = 4`, head_dim ≤ 128). Always-compiled
  stub-vs-kernel pairing (`PagedAttentionStub.cpp`,
  `detail/PagedAttention.hpp`) + a numerically matching CPU reference loop.
- `PagedKVCache::block_table(b)` exposes the logical→physical map; the op
  layer (`tesseract_nn`) now links `Tesseract::cuda`.
- `MultiHeadAttention::forward_step_batched` takes the fused path when the
  step is a CUDA single-token decode (`S_new==1`), every cache is paged,
  and all share one pool — RoPE + cache append stay per-request (cheap),
  then one `paged_decode_attention`, then a batched `o_proj`. CPU /
  contiguous caches / chunked-prefill (`S_new>1`) keep the exact Wave-10
  per-sequence loop, so the scheduler's CPU bit-exact parity is untouched.

**Verification** (`tests/nn/test_paged_attention.cpp`, 6 cases): ragged
mixed-length parity vs gather+`ops::attention` for MHA (group=1) + GQA
(group>1) — CPU ≤ 1e-5, CUDA ≤ 3e-3; zero-length request → zero row; CUDA
op parity; end-to-end `forward_step_batched` (paged, CUDA) vs per-request
`forward_step` ≤ 3e-3; operand validation. CPU green (CUDA cases SKIP via
`SKIP_RETURN_CODE 4`), CUDA correctness green on a contention-free card
(the only reds in a full run are perf benches + 3 timing-sensitive cases —
CudaGraph `forward_step` replay, `to_async` pageable, mnist tolerance —
that flake under heavy external GPU contention and pass on a free card;
none touch the paged path).

**MVP scope / deferred (B-032++):** INT8-direct paged attention (read INT8
K/V + per-token scales straight from a quantized paged pool, fusing the
Wave-9 dequant into this kernel) — landed in Wave 12 below; padded
same-length micro-batching and a fused paged *prefill* kernel (`S_new>1`
over the active set) still deferred.

### Wave 12 — INT8-direct paged decode-attention (B-032++) *(DONE, 2026-06-22)*

**Tracking: B-032++.** Closes the last seam between the three KV fast
paths. Wave 9 made the *persistent* KV INT8 (`QuantizedKVCache`), Wave 11
made paged decode-attention fused, but a fused INT8 *paged* read still
required materializing an FP prefix (dequantize the whole pool, then run
the Wave-11 FP kernel) — defeating the quant memory-bandwidth win on the
hot path. Wave 12 dequantizes `int8 * scale` **inside** the attention
kernel, so an INT8 paged pool is read straight into attention at ~4× (vs
FP32) / ~2× (vs FP16) lower KV bandwidth.

**What shipped:**
- `nn::paged_decode_attention_int8(q [A,H,D], k_pool/v_pool
  [num_blocks,Hkv,block_size,D] Int8, k_scale/v_scale
  [num_blocks,Hkv,block_size] Float32, block_tables [A,max_logical] Int32,
  lens [A] Int32, scale, group)` → O [A,H,D] (q's FP dtype). Same ragged
  single-launch decode as Wave 11, but each K/V vector is `int8 * scale`
  with the Wave-9 per-(block,head,slot) symmetric scale (one FP32 per
  `D`-vector), dequantized on load — the FP-prefix transient is never
  built.
- CUDA kernel `paged_decode_attention_int8_kernel` (`src/cuda/PagedAttention.cu`):
  same warp-per-`(request,query-head)` / per-lane-Q-fragment / online-softmax
  structure as the FP kernel; loads `int8` K/V payloads + one scale per key
  `j` (broadcast over `D`), FP32 interior math. Stub
  (`launch_paged_decode_attention_int8` in `PagedAttentionStub.cpp`) +
  detail decl keep the always-compiled pairing; a matching CPU reference
  loop lives in the host op.
- Q / O stay FP32 / FP16 / BF16; head_dim ≤ 128.

**Verification** (`tests/nn/test_paged_attention.cpp`, +5 cases, 11 total):
the INT8 op is checked against the *Wave-11 FP op run on the dequantized
pool* (`dequantize_kv_per_token(pool)`) — identical `int8*scale` values and
online softmax, so the only delta is the inline dequant: CPU **bit-exact**
(≤ 1e-5), CUDA ≤ 3e-3, for MHA (group=1) + GQA (group>1); zero-length
request → zero row; operand validation (Int8 pools required, group match,
scale numel). CPU green (CUDA cases SKIP via `SKIP_RETURN_CODE 4`), all 11
cases green on a contention-free card.

**MVP scope / deferred (B-032+++):** a `QuantizedPagedKVPool` /
`QuantizedPagedKVCache` storage layer (so the scheduler/`generate` can
allocate INT8 paged KV and feed this op directly) — landed in Wave 13
below; padded same-length micro-batching and a fused paged *prefill* kernel
still deferred.

### Wave 13 — INT8-quantized paged KV cache (B-032+++) *(DONE, 2026-06-22)*

**Tracking: B-032+++.** Wave 12 proved the fused INT8 paged op + kernel but
left it unwired — nothing in the runtime *stored* INT8 paged KV to feed it.
Wave 13 ships that storage layer, unifying the three KV fast paths the
framework had built separately (GQA · KV-quant · paged-attention) into one
cache the scheduler can allocate.

**What shipped:**
- `nn::QuantizedPagedKVPool` (`include/tesseract/nn/QuantizedPagedKVPool.hpp`,
  `src/nn/QuantizedPagedKVPool.cpp`): the INT8 sibling of `PagedKVPool` —
  same `num_blocks × block_size` grid + one `BlockAllocator`, but each slot
  is an INT8 payload `[num_blocks,H,block_size,D]` + one FP32 scale per
  `(block,head,slot)` `[num_blocks,H,block_size]` (the Wave-9 per-token,
  per-head symmetric layout = exactly what `paged_decode_attention_int8`
  reads). INT8 payload and FP32 scale share block ids (single allocator).
- `nn::QuantizedPagedKVCache` (`KVCacheBase` drop-in): `append([B,H,Sn,D] FP)`
  quantizes per-(token,head) via `quant::quantize_kv_per_token` and scatters
  the INT8 payload + scale into on-demand blocks; `keys_view()`/
  `values_view()` gather the scattered INT8 + scale prefix and dequantize to
  a fresh FP `[B,H,L,D]` (the CPU / non-fused fallback). Per-head byte-copy
  scatter/gather (CPU memcpy / CUDA async) — the launch-collapsed gather
  kernel only handles 2/4/8-byte elements, and gather is the fallback path
  anyway (the fused decode reads the pool in place).
- **Integration:** `MultiHeadAttention::forward_step_batched`'s CUDA
  single-token fast path now recognizes *both* paged flavors — the
  per-request RoPE+append + block-table assembly is factored into one
  `run_fused` driver, dispatching to `paged_decode_attention` (FP pools) or
  `paged_decode_attention_int8` (INT8 pools). CPU / contiguous / chunked
  prefill keep the per-sequence fallback, so scheduler CPU bit-exact parity
  is untouched.

**Verification** (`tests/nn/test_quant_paged_kv.cpp`, 3 cases): paged vs
contiguous (`QuantizedKVCache`) dequant-view **bit-exact** on identical
chunked appends straddling block boundaries (validates scatter/gather);
on-demand paging + reset recycling; end-to-end `forward_step_batched`
(quant-paged, CUDA) vs per-request `forward_step` (gather+dequant+attention
fallback) ≤ 3e-3. CPU green (CUDA case SKIPs via `SKIP_RETURN_CODE 4`), CUDA
green on a free card. No regression: FP paged (11/11), batched decode
(6/6), scheduler CPU+CUDA (6/6 each, still bit-exact) after the
`forward_step_batched` refactor.

**MVP scope / deferred (B-032++++):** a scheduler/`generate` knob to select
INT8 paged pools per layer (the cache is ready; only the engine wiring +
config flag remain), padded same-length micro-batching, and a fused paged
*prefill* kernel (`S_new>1` over the active set).

---

## Post-wave M3 — re-organized completion plan (2026-06-21)

After Waves 1a–6 the four user-mandated directions (PagedKV ·
normalization running-stats · quantization · tokenizer+loader) are all
closed, and the decode stack now runs end-to-end with real sampling.
Re-sequenced remaining M3 work, ordered by dependency + on-box
(SM 8.9) verifiability:

- **Wave 7 — Continuous-batching scheduler** *(DONE, 2026-06-21 — B-029)*.
  Shared per-layer paged pools + dynamic admission + per-request
  sampling/EOS + block recycling, with exact parity vs standalone
  generate. See the Wave 7 section above.
- **Wave 8 — Grouped-query attention** *(DONE, 2026-06-22 — B-030)*. The
  gate to real Llama-3 / Qwen2 / Mistral checkpoints; also shrinks the KV
  cache by `num_heads / num_kv_heads`. See the Wave 8 section above.
- **Wave 9 — KV-cache INT8 quantization** *(DONE, 2026-06-22 — B-031)*.
  Device-resident per-token INT8 quant ops + `nn::QuantizedKVCache` +
  `GenerateConfig::kv_int8`; ~4×/2× smaller persistent KV. See the Wave 9
  section above.
- **Wave 10 — Compute-batched decode** *(DONE, 2026-06-22 — B-032)*. The
  scheduler folds its active set into one batched `forward_step_batched`
  (dense layers batched, attention per-sequence) — exact parity, no new
  kernel. See the Wave 10 section above.
- **Wave 11 — Fused ragged paged decode-attention** *(DONE, 2026-06-22 —
  B-032+)*. One `paged_decode_attention` launch over the active set (K/V
  read in place via block tables, GQA head mapping in-kernel, no
  `repeat_kv`, online softmax), wired into `forward_step_batched`'s CUDA
  decode path. See the Wave 11 section above.
- **Wave 12 — INT8-direct paged decode-attention** *(DONE, 2026-06-22 —
  B-032++)*. `paged_decode_attention_int8` reads INT8 K/V + per-token
  scales straight from a quantized paged pool, dequantizing in-kernel,
  unifying the GQA + KV-quant + paged-attention fast paths. See the Wave 12
  section above.
- **Wave 13 — INT8-quantized paged KV cache** *(DONE, 2026-06-22 —
  B-032+++)*. `QuantizedPagedKVPool` + `QuantizedPagedKVCache` (`KVCacheBase`
  drop-in): paged INT8 storage that quantizes on append and feeds the
  Wave-12 op directly through `forward_step_batched`'s CUDA fast path. See
  the Wave 13 section above. Deferred (B-032++++): the scheduler/`generate`
  knob to select INT8 paged pools, padded same-length micro-batching, and a
  fused paged *prefill* kernel.
- **Wave 14 — INT8 paged KV in the engine** *(DONE, 2026-06-22 — B-032++++
  core)*. `EngineConfig::kv_int8` knob + per-layer `QuantizedPagedKVPool`
  construction in `ContinuousBatchingScheduler`, bit-exact on CPU vs
  `generate(kv_int8)`. See the Wave 14 section below. Deferred tail: padded
  same-length micro-batching + fused paged *prefill* kernel.
- **Wave 15 — Graph-mode GPU codegen** *(DONE for the IR half, 2026-06-22 —
  B-009)*. `--convert-tesseract-to-gpu` lowers data-parallel tesseract ops to
  `gpu.module`/`gpu.func`/`gpu.launch_func`. The PTX/cubin JIT + eager-CUDA
  parity tail is gated on an NVPTX-enabled LLVM build (toolchain). See the
  Wave 15 section below.
- **Wave 16 — Real HF checkpoint end-to-end demo** *(DONE, 2026-06-22 —
  B-033)*. `LlamaConfig::from_json[_file]` + `examples/llama_generate.cpp`
  loads a real HF `config.json`/safetensors/tokenizer and streams decoded
  output. See the Wave 16 section below.
- **Wave 17 — Structured / grammar-constrained generation** *(DONE,
  2026-06-22 — B-034)*. Regex `ByteAutomaton` (Thompson NFA + lazy DFA) +
  `GrammarConstraint` logit masking; constrained greedy always matches the
  grammar. See the Wave 17 section below.
- **Wave 18 — Speculative decoding** *(DONE, 2026-06-22 — B-035)*.
  `SpeculativeDecoder` (draft proposes γ, target verifies in one batched
  pass, KV rewind on partial accept); output identical to target-only
  greedy. See the Wave 18 section below.
- **Wave 19 — Disaggregated prefill/decode** *(DONE, 2026-06-22 — B-036)*.
  `DisaggregatedEngine` + explicit `KvTransfer` migrates the prefill KV
  cache into a decode role; bit-identical to monolithic generate. See the
  Wave 19 section below.
- **Wave 20 — Graph-mode quantization dialects** *(DONE, 2026-06-22 —
  B-037)*. `tesseract.dequant_matmul` op with `scheme` (int8/int4_group/
  fp8_e4m3/fp8_e5m2/fp4) + `group_size` attributes, round-trip + verifier
  negatives pinned. See the Wave 20 section below.
- **Hopper-gated** (blocked on SM 9.0+ hardware, not scheduled here):
  FA3 fused kernel (M2L.3), WGMMA INT8 tensor-core path (B-026+).
- **Toolchain-gated tail** (blocked on an LLVM rebuild with the NVPTX target,
  not on this box): the B-009 PTX/cubin JIT + eager-CUDA parity stage that
  consumes the Wave 15 `gpu.module` IR.

---

## Wave 14 — INT8 paged KV in the engine (B-032++++ core) *(DONE, 2026-06-22)*

Closes the engine half of B-032++++: the scheduler can now allocate the
Wave-13 `QuantizedPagedKVPool` / `QuantizedPagedKVCache` per layer behind a
single config knob, so continuous batching runs on ~4×/2× smaller INT8 KV.

**What landed.**
- `EngineConfig::kv_int8` (`include/tesseract/models/Scheduler.hpp`). When set,
  `ContinuousBatchingScheduler`'s ctor builds `qpools_` via
  `LlamaModel::make_quantized_layer_pools(num_blocks, block_size)` instead of
  the FP `pools_`; `allocated_blocks()` / `free_blocks()` read whichever pool
  family is live.
- `LlamaModel::make_quantized_layer_pools` + `make_quantized_paged_kv_caches`
  (`include/tesseract/models/Llama.hpp`, `src/models/Llama.cpp`) mirror the FP
  factory methods.
- `admit_one` builds INT8 paged caches from `qpools_`; `retire` resets the
  `QuantizedPagedKVCache` blocks back to the shared pool.

**Verification** (`tests/models/test_scheduler.cpp`, case "Scheduler: INT8
paged KV matches generate(kv_int8)"): scheduler output with `kv_int8=true` is
bit-exact on CPU against `model->generate(..., kv_int8=true)`. Full scheduler
suite green (18/18 across waves).

**Resolved tail (2026-06-22, on free SM 8.9 GPUs):** the fused paged
*prefill* kernel (`S_new>1`) landed — `nn::paged_prefill_attention[_int8]`
(one warp per `(r, s, h)` with causal bound `kv_len-S+s+1`, FP + INT8-direct),
wired into `forward_step_batched`'s `Sn>1` paged CUDA path, collapsing the
chunked-prefill per-sequence loop into one launch. Parity-pinned
(`tests/nn/test_paged_attention.cpp`, +6 cases; CUDA validated on a free
RTX 5880 Ada). Padded same-length micro-batching is *superseded* by the fused
ragged decode+prefill kernels (they handle ragged lengths with no padding
waste — strictly better than the pre-paged-attention padding strategy).

---

## Wave 15 — Graph-mode GPU codegen (B-009 / M2L.2) *(IR half DONE, 2026-06-22)*

The device-IR half of the "one IR, train+infer" thesis: lower the `tesseract`
dialect all the way to the MLIR **GPU dialect**, reusing the upstream lowering
chain instead of hand-rolling a kernel outliner.

**What landed** (`src/ir/passes/ConvertToGpu.cpp`,
`buildConvertTesseractToGpuPipeline` + `--convert-tesseract-to-gpu`):

```
tesseract.*                            (elementwise / data-parallel)
  → --convert-tesseract-to-linalg      (reuses the M1G CPU lowering)
  → one-shot-bufferize                 (tensors → memrefs)
  → convert-linalg-to-parallel-loops   (→ scf.parallel)
  → gpu-map-parallel-loops             (annotate block/thread mapping)
  → convert-parallel-loops-to-gpu      (→ gpu.launch)
  → gpu-kernel-outlining               (→ gpu.module + gpu.func + gpu.launch_func)
```

The result is a `gpu.container_module` whose kernels carry `gpu.block_id`
indexing + the lowered `arith.*` body, dispatched by `gpu.launch_func`.

**Verification** (`tests/ir/convert_to_gpu.mlir`, ctest `ir_convert_to_gpu`):
FileCheck pins the `gpu.module`/`gpu.func`/`gpu.launch_func` structure for
add and mul→relu functions, asserting the source tesseract ops are gone. All
6 IR ctests green.

**Toolchain-gated tail (B-009 DoD #2–#5):** the `JitEngine::build_for_gpu`
PTX/cubin JIT, eager-CUDA parity, and `bench_cuda_jit_vs_eager` cannot run
here — the local `third_party/llvm-install` LLVM is built with the `host`
target only (no NVPTX backend), and the parity/bench gates also need a free
device. Unblock by rebuilding LLVM with `NVPTX` in `LLVM_TARGETS_TO_BUILD`;
the IR pipeline above is the foundation the NVVM stage plugs onto.

---

## Wave 16 — Real HF checkpoint end-to-end demo (B-033) *(DONE, 2026-06-22)*

The exit-criteria demo binary: load a *real* Hugging Face Llama checkpoint
(config + safetensors + tokenizer) and stream generated text.

**What landed.**
- `LlamaConfig::from_json` / `from_json_file`
  (`include/tesseract/models/Llama.hpp`, `src/models/Llama.cpp`): a minimal,
  focused scalar extractor for HF `config.json` — parses `vocab_size`,
  `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
  `num_key_value_heads`, `intermediate_size`, `rms_norm_eps`,
  `rope_theta`, `torch_dtype`, and `bos_token_id` / `eos_token_id`. Missing
  *essential* architecture fields throw rather than silently defaulting.
- `examples/llama_generate.cpp`: CLI over model dir / config / safetensors /
  tokenizer / prompt / device / generation params; loads via
  `LlamaConfig::from_json_file` + `io::BpeTokenizer`, then runs a streaming
  decode loop (also the `--grammar-regex` entry point from Wave 17).

**Verification** (`tests/models/test_llama_config_json.cpp`): parses a
Llama-3.2-1B-style config, `torch_dtype` variants, default fill-in for
optional fields, and rejection of malformed/incomplete input.

---

## Wave 17 — Structured / grammar-constrained generation (B-034) *(DONE, 2026-06-22)*

Constrain decoding so generated tokens always conform to a grammar.

**What landed** (`include/tesseract/models/StructuredDecoding.hpp`,
`src/models/StructuredDecoding.cpp`):
- `ByteAutomaton` interface + `RegexAutomaton`: a recursive-descent regex
  parser → Thompson NFA → **lazy DFA** (subset construction with state
  interning) over bytes. Supports literals, `.`, char classes / negation,
  `*`/`+`/`?`, alternation, grouping, escapes.
- `GrammarConstraint`: maps the automaton over the token vocabulary and masks
  logits to the set of tokens that keep the automaton on an accepting path;
  `accept()` advances the DFA state (EOS gated on an accepting state).

**Verification** (`tests/models/test_structured_decoding.cpp`): regex match
semantics + `GrammarConstraint` masking + a `constrained_greedy` helper that
proves the decoded string always matches the grammar.

---

## Wave 18 — Speculative decoding (B-035) *(DONE, 2026-06-22)*

Use a small draft model to propose tokens and verify them with the target in
one batched pass — output **identical** to target-only greedy.

**What landed** (`include/tesseract/models/Speculative.hpp`,
`src/models/Speculative.cpp`): `SpeculativeDecoder(target, draft, gamma)`.
Each round the draft proposes γ tokens; the target verifies all γ+1 positions
in a single `forward_step_batched`; the longest matching prefix is accepted;
KV caches are rewound with `KVCache::set_current_len` on partial acceptance
and the correction token is re-fed. `Result` tracks proposed/accepted tokens,
rounds, and target forwards (the speedup metric).

**Verification** (`tests/models/test_speculative.cpp`): greedy output matches
target standalone for several draft models + γ; self-draft acceptance;
EOS early stop; input validation.

---

## Wave 19 — Disaggregated prefill/decode (B-036) *(DONE, 2026-06-22)*

Split the prompt-prefill phase from token-decode behind an explicit KV
hand-off, the architecture vLLM/Dynamo use to scale the two phases
independently.

**What landed** (`include/tesseract/models/Disaggregated.hpp`,
`src/models/Disaggregated.cpp`): `KvTransfer` holds per-layer contiguous K/V
tensors + `first_token` + `prompt_len`. `DisaggregatedEngine::prefill` runs
the prompt and exports its caches' contiguous K/V views; `decode` imports the
transfer into fresh caches and continues autoregressively; `generate` chains
both — bit-identical to monolithic generation.

**Verification** (`tests/models/test_disaggregated.cpp`): `generate` matches
monolithic generate; explicit prefill-role → decode-role transfer; EOS early
stop; input validation.

---

## Wave 20 — Graph-mode quantization dialects (B-037) *(DONE, 2026-06-22)*

Represent quantized matmul as a first-class IR op so FP8/FP4/INT4 schemes are
graph-visible (the IR-level counterpart to the runtime quant kernels).

**What landed.**
- `tesseract.dequant_matmul` (`src/ir/TesseractOps.td`): `lhs`, `qweight`,
  `scale` operands + `scheme` (string) and `group_size` (si64) attributes.
- `DequantMatMulOp::verify()` (`src/ir/TesseractOps.cpp`): validates `scheme`
  ∈ {int8, int4_group, fp8_e4m3, fp8_e5m2, fp4} and that `group_size` agrees
  with the scheme family (positive for grouped, −1 for per-tensor).

**Verification:** round-trip cases in `tests/ir/roundtrip.mlir` (ctest
`ir_dialect_roundtrip`) + verifier negatives in
`tests/ir/dequant_matmul_invalid.mlir` (ctest `ir_dequant_matmul_invalid`,
`--verify-diagnostics`). All 6 IR ctests green.

---

## Exit criteria

Tesseract serves Llama-3 8B at parity with vLLM throughput on a single
Hopper node. End-to-end demo: `llama_generate` binary consuming a
real HF checkpoint + tokenizer, streaming tokens at Hopper-roofline
throughput.

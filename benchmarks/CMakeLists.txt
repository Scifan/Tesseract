# Micro-benchmarks. Enabled via -DTESSERACT_BUILD_BENCHMARKS=ON.

add_executable(bench_matmul bench_matmul.cpp)
target_link_libraries(bench_matmul
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
)
if(TESSERACT_USE_EIGEN)
  # Gives the bench direct access to `Eigen::Matrix` / `Eigen::Map` so we
  # can print an `[eigen]` row that reflects the raw ceiling of the library
  # — independent of how `ops::matmul` dispatches internally.
  target_link_libraries(bench_matmul PRIVATE Eigen3::Eigen)
  target_compile_definitions(bench_matmul PRIVATE TESSERACT_HAS_EIGEN=1)
endif()

# M4 cross-cutting (B-046): CPU end-to-end decode tok/s for LlamaModel, the
# Tesseract side of the vs-llama.cpp external comparison. Informational
# (no hard bar); driven by scripts/bench_vs_llama_cpp.sh.
add_executable(bench_llama_decode_cpu bench_llama_decode_cpu.cpp)
target_link_libraries(bench_llama_decode_cpu
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
)

# M4 Track A1 (B-038+): MoE sparse top-k dispatch vs dense all-experts — proves
# the routed FFN compute scales with k, not E. CPU-only. See moe-sparse.md.
# M4 Phase 6 (perf closeout): CPU decode throughput with the AVX-512-VNNI
# W8A8 GEMV + zero-alloc arena — the kernel that closes the gap to llama.cpp.
add_executable(bench_cpu_decode_vnni bench_cpu_decode_vnni.cpp)
target_link_libraries(bench_cpu_decode_vnni
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
)
if(TESSERACT_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(bench_cpu_decode_vnni PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(bench_cpu_decode_vnni PRIVATE TESSERACT_HAS_OPENMP=1)
endif()
# CPU perf gate: the W8A8 VNNI decode kernel must sustain its bandwidth floor
# (the bench self-fails under 40 GB/s when VNNI is present). `ctest -L bench_cpu`.
# A small --gen keeps CI fast; the floor is far above the FP32 eager path and
# well below the clean ~290 GB/s so it only trips on a real regression.
add_test(NAME bench_cpu_decode_vnni COMMAND bench_cpu_decode_vnni --gen 24)
set_tests_properties(bench_cpu_decode_vnni PROPERTIES
  LABELS "bench_cpu"
  TIMEOUT 180
  RESOURCE_LOCK "cpu_bench"
)

add_executable(bench_moe_sparse bench_moe_sparse.cpp)
target_link_libraries(bench_moe_sparse
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
)

# M4 Track A2 (B-039): the SSM architectural-differentiation metric — Mamba's
# O(1) decode-step latency + O(1) resident state vs Llama's O(L) attention, swept
# over context length. CPU-only, informational. See docs/design/mamba-scaling.md.
add_executable(bench_mamba_vs_llama_scaling bench_mamba_vs_llama_scaling.cpp)
target_link_libraries(bench_mamba_vs_llama_scaling
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
)

add_executable(bench_graph_vs_eager bench_graph_vs_eager.cpp)
target_link_libraries(bench_graph_vs_eager
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
)

# M4 Track B1 (B-041): native side of the Python↔C++ frontend-overhead
# measurement. Times the same ops the pybind11 frontend exposes, in pure C++,
# with matching iteration counts; bench/external/python_overhead.py runs the
# Python side and the wrapper diffs them. CPU-only, informational.
add_executable(bench_python_overhead bench_python_overhead.cpp)
target_link_libraries(bench_python_overhead
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
)
if(TESSERACT_ENABLE_MLIR)
  # Pulls in `tesseract::ir::JitEngine` so the benchmark can add an
  # "mlir-jit" column alongside eager + graph-interpreter. Linking is
  # gated so the benchmark still builds (without the JIT column) when
  # MLIR support is off.
  target_link_libraries(bench_graph_vs_eager PRIVATE Tesseract::ir)
  target_compile_definitions(bench_graph_vs_eager
    PRIVATE TESSERACT_ENABLE_MLIR=1)
endif()

# ---------------------------------------------------------------------------
# M2L.1 — CUDA micro-benchmarks.
#
# These gate the M2.γ exit bars on actual measured throughput on the dev
# GPU (see docs/m2-plan.md §5.3 for the hard bars). Each bench exits
#   0   on PASS,
#   1   on perf miss (ctest FAIL),
#   77  if no CUDA device is visible (ctest SKIP, via SKIP_RETURN_CODE).
#
# Grouped under the ctest label `bench_cuda` so CI can run them with
# `ctest -L bench_cuda`. The bench executables live next to their CPU
# peers but pull in `CUDA::cudart` + `CUDA::cublasLt` directly because
# they call the CUDA runtime + cuBLASLt API — our public surface
# hides those handles intentionally.
# ---------------------------------------------------------------------------
if(TESSERACT_ENABLE_CUDA)
  set(_cuda_benches
    bench_cuda_matmul
    bench_cuda_elementwise
    bench_cuda_attention
    bench_cuda_attention_bwd
    bench_cuda_rms_norm
    bench_cuda_transformer_block
    bench_cuda_graph
    bench_cuda_pinned
    # Wave 4.1 (B-025): fused SwiGLU forward (silu(gate) * up) vs
    # primitive composite (sigmoid + mul + mul). Three-row layout
    # mirrors `bench_cuda_rms_norm`; hard bars = 2.0× composite
    # speedup and ≥0.80 × memcpy D2D roofline at the largest shape.
    bench_cuda_swiglu
    # Wave 4.2 (B-024): fused FA2-style SDPA vs composite
    # (matmul → softmax → matmul). Three-row layout at S=512/1K/2K.
    # Hard bar = 2.0× composite speedup at the largest shape (2K×
    # long-context prefill) — the composite's score-matrix round
    # trip dominates HBM traffic and the fused kernel sidesteps it.
    bench_cuda_fused_attention
    # Wave 4.3 (B-023b): capture a full `MHA::forward_step` decode
    # pass into a single `cudaGraphExec_t` and compare replay cost
    # vs eager. Hard bar = 1.25× speedup at the `S_k=8` host-bound
    # shape — the regime where the graph's "collapse 20+ launches
    # into one" property actually matters. `S_k=64` / `S_k=256`
    # rows are informational; at those sizes compute dominates.
    bench_cuda_mha_decode_graph
    # Wave 4.4 (B-026): quantized inference fast path + memory
    # footprint. FP32 `nn::Linear` vs INT8 `QuantizedLinear` vs
    # INT4G `QuantizedLinearInt4G` on decode- / prefill-shape
    # matmul. Hard bars cover both latency (on HBM-bound decode)
    # and deterministic weight-byte ratios.
    bench_cuda_quantized_linear
    # Wave 4.4 (B-026): end-to-end one-transformer-block decode
    # step — FP32 vs `mha+ffn quantize_(INT8)` vs `quantize_(INT4G,
    # g=128)`. Hard bars enforce both memory reduction and a
    # latency win on the HBM-bound decode regime.
    bench_cuda_llama_decode
    # Wave 4.5 (B-019b): paged vs contiguous KV cache. Hard bars on
    # (a) memory residency proportional to actual tokens, not max_len,
    # and (b) the per-step gather overhead staying a small fraction of
    # a real decode step.
    bench_cuda_paged_kv
    # M4 Track B3 (occupancy-unlocked): real multi-GPU tensor-parallel
    # scaling on physical cards 0..N-1. Each rank's FFN shard lives on its
    # own device; a real cross-device all-reduce combines partials. Reports
    # per-GPU weight memory = dense/N and aggregate throughput. Informational
    # (no hard bar) — scaling depends on PCIe topology / NVLink presence.
    bench_cuda_tp_scaling
    # M4 perf-closeout Phase 3: FP8 (E4M3) tensor-core GEMM vs FP16
    # cuBLASLt GEMM (the same path PyTorch eager uses on Ada). Hard bar =
    # FP8 wins the dense linear-layer line at the large shapes.
    bench_cuda_fp8_gemm
    # M4 perf-closeout Phase 4: GPU MoE — sparse top-k dispatch (device-side
    # routing) vs dense all-experts. Hard bar = sparse cheaper than dense.
    bench_cuda_moe
    # M4 perf-closeout Phase 5: GPU Mamba O(1) decode vs attention O(L) decode
    # scaling. Hard bar = Mamba per-step faster than attention at the max L.
    bench_cuda_mamba_scaling
    # M4 perf-closeout Phase 9: full-model Llama serving latency (TTFT / TPOT)
    # on CUDA — the Tesseract side of the vLLM head-to-head. Loads the same
    # TinyLlama-1.1B (config.json + safetensors) vLLM serves, or a synthetic
    # TinyLlama-shape model with no args. Informational (no hard bar): the
    # external comparison lives in bench/external/results/vllm_serving.md.
    bench_cuda_llama_serving
  )
  foreach(_b IN LISTS _cuda_benches)
    add_executable(${_b} ${_b}.cpp)
    target_link_libraries(${_b}
      PRIVATE
        Tesseract::tesseract
        Tesseract::cuda
        Tesseract::compile_options
        CUDA::cudart
        CUDA::cublasLt
    )
  endforeach()

  # M4 perf-closeout Phase 7: REAL multi-process NCCL tensor-parallel scaling +
  # forward parity. Forks one process per GPU and runs ncclAllReduce across
  # cards. Separate from the generic foreach because it needs >=2 GPUs (not the
  # single `cuda_gpu_0` lock) and the NCCL link. Skips (77) without NCCL/>=2 GPU.
  add_executable(bench_cuda_nccl_tp bench_cuda_nccl_tp.cpp)
  target_link_libraries(bench_cuda_nccl_tp
    PRIVATE
      Tesseract::tesseract
      Tesseract::cuda
      Tesseract::compile_options
      CUDA::cudart
      CUDA::cublasLt
  )
  if(TESSERACT_ENABLE_NCCL AND TARGET tesseract_nccl)
    target_link_libraries(bench_cuda_nccl_tp PRIVATE tesseract_nccl)
  endif()

  # `bench_cuda_all` is an aggregate convenience target so
  # `cmake --build . --target bench_cuda_all` compiles all six in one
  # shot — useful during local tuning.
  add_custom_target(bench_cuda_all DEPENDS ${_cuda_benches} bench_cuda_nccl_tp)

  if(TESSERACT_BUILD_TESTS)
    # ctest registration: one test per bench, same SKIP/FAIL contract as
    # the M2E/F/G/H/I/K bridges. All share label `bench_cuda` so
    # `ctest -L bench_cuda` is the perf gate. A generous TIMEOUT (300 s)
    # covers the attention + transformer benches at their larger
    # shapes; most run in under 10 s.
    foreach(_b IN LISTS _cuda_benches)
      add_test(
        NAME ${_b}
        COMMAND $<TARGET_FILE:${_b}>
      )
      set_tests_properties(${_b} PROPERTIES
        SKIP_RETURN_CODE 77
        LABELS "bench_cuda"
        TIMEOUT 300
        # Serialize every CUDA perf bench on the same resource key so
        # `ctest -j N` doesn't run two GPU benches concurrently —
        # they'd contend for SM time and invalidate the `_us` / `GB/s`
        # hard bars (matmul dispatch_us, rms_norm speedup, etc.). One
        # logical GPU slot is enough since every bench already pins
        # device 0. This is ctest's built-in mutual-exclusion
        # mechanism (introduced in CMake 3.16).
        RESOURCE_LOCK "cuda_gpu_0"
      )
    endforeach()

    # Multi-GPU NCCL TP bench: own resource lock for *two* GPU slots so ctest
    # won't co-schedule it with single-GPU benches. Longer timeout covers NCCL
    # init across processes.
    add_test(NAME bench_cuda_nccl_tp COMMAND $<TARGET_FILE:bench_cuda_nccl_tp>)
    set_tests_properties(bench_cuda_nccl_tp PROPERTIES
      SKIP_RETURN_CODE 77
      LABELS "bench_cuda"
      TIMEOUT 300
      RESOURCE_LOCK "cuda_gpu_0;cuda_gpu_1"
    )
  endif()
endif()

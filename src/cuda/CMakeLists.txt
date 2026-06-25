# tesseract_cuda — the HAL entry point for the CUDA backend.
#
# This library is created unconditionally (so that downstream code can link
# against `Tesseract::cuda` without probing a CMake option), but its
# contents depend on TESSERACT_ENABLE_CUDA:
#
#   * OFF (default): only the plain-C++ routing file (`CudaRuntime.cpp`) is
#     compiled. Every public probe function short-circuits to a "no CUDA"
#     return value. No CUDA Toolkit symbol is referenced, no nvcc runs.
#
#   * ON:  the `.cu` translation units are added to the target, nvcc compiles
#     them with CUDA_ARCHITECTURES=${TESSERACT_CUDA_ARCHITECTURES}, and the
#     library links against `CUDA::cudart`. The routing file picks up the
#     real implementations via the TESSERACT_HAS_CUDA compile define.
#
# Public headers live in `include/tesseract/cuda/` and never include
# `<cuda_runtime.h>`; see ADR-0005 for the layering rationale.

set(TESSERACT_CUDA_SOURCES
  CudaRuntime.cpp
  Allocator.cpp
  Stream.cpp
  Memcpy.cpp
  # Wave 2.3 (B-023): CUDA Graph capture + replay primitive. Host-only
  # TU — uses `cudaGraph_t` / `cudaGraphExec_t` / `cudaStreamBeginCapture`
  # from the runtime API, no device code, so it ships as a plain `.cpp`
  # even though it's part of the CUDA-only source set. CPU-only build
  # gets the same file with every method throwing a clean DeviceError.
  CudaGraph.cpp
  # Wave 2.4 (B-011): page-locked host memory allocator. Same always-
  # compiled pattern as `Allocator.cpp` / `Stream.cpp` / `CudaGraph.cpp`:
  # CPU-only build ships throwing stubs, CUDA build forwards to
  # `cudaHostAlloc` / `cudaFreeHost`. Pinned memory is the prerequisite
  # for real async H→D overlap — see `Tensor::to_async`.
  PinnedHostAllocator.cpp
  # *Stub.cpp compiles to an empty TU when TESSERACT_HAS_CUDA is
  # defined; each body only emits throwing stubs in the CPU-only
  # build. Paired with the matching `.cu` files below so we never
  # double-define the `detail::launch_*` symbols — see each stub
  # source for the rationale.
  ElementwiseStub.cpp
  ReductionStub.cpp
  SoftmaxStub.cpp
  LossStub.cpp
  # M2G: cuBLASLt matmul bridge. Unlike the other *.cu kernels it's a
  # plain `.cpp` — cuBLASLt's entry points are host C++ (no device
  # code of our own), so we don't need nvcc for this TU. The real
  # impl ships in `MatMul.cpp` under TESSERACT_ENABLE_CUDA; this stub
  # only emits throwing symbols in the CPU-only build.
  MatMulStub.cpp
  # M2H: shape-copy + indexing bridges. Same stub-vs-kernel pairing
  # as the earlier M2E/M2F launchers: the stubs below only have a
  # non-empty body under `!TESSERACT_HAS_CUDA`, and the matching
  # `.cu` TUs are added below when the CUDA backend is compiled in.
  ShapeStub.cpp
  IndexingStub.cpp
  # M2I: fused Adam step. The stub throws a clear DeviceError in
  # CPU-only builds; the real kernel ships next to it as `Optim.cu`.
  OptimStub.cpp
  # B-014: rotary position embedding bridge. Stub-or-kernel pairing
  # mirrors the earlier M2E/M2F launchers — throwing symbol in
  # CPU-only builds, real CUDA kernel wired in below when the
  # backend is compiled in.
  RotaryEmbeddingStub.cpp
  # Wave 2 (B-022): fused RMSNorm / LayerNorm forward kernels. Same
  # stub-vs-kernel pairing as the earlier M2E/F/H/B-014 bridges.
  RMSNormStub.cpp
  # Wave 3.1 (B-021): INT8 weight-only dequantize-matmul bridge.
  # Same always-compiled stub-vs-kernel pattern. Real kernel lands
  # in `DequantMatMul.cu` below when TESSERACT_ENABLE_CUDA=ON.
  DequantMatMulStub.cpp
  # Wave 4.1 (B-025): fused SwiGLU forward (silu(gate) * up). Same
  # always-compiled stub-vs-kernel pattern. Real kernel ships in
  # `SwiGLU.cu` below when TESSERACT_ENABLE_CUDA=ON; the stub
  # throws a clean DeviceError in CPU-only builds so the op layer
  # can keep the dispatch branch unconditional.
  SwiGLUStub.cpp
  # Wave 4.2 (B-024): fused FlashAttention-2-style SDPA forward.
  # Same always-compiled stub-vs-kernel pattern. Real kernel ships
  # in `FusedAttention.cu` below when TESSERACT_ENABLE_CUDA=ON; the
  # stub throws a clean DeviceError in CPU-only builds so the
  # `ops::attention` dispatch branch can be declared unconditionally.
  FusedAttentionStub.cpp
  # Wave 4.5 (B-019b): paged KV-cache gather. Single-launch
  # element-wise gather replacing PagedKVCache's per-block memcpy
  # loop. Real kernel ships in `PagedKV.cu` below when
  # TESSERACT_ENABLE_CUDA=ON; the stub throws in CPU-only builds (the
  # CPU gather path never calls it).
  PagedKVStub.cpp
  # Wave 9 (B-031): KV-cache INT8 quantize / dequantize. Hot-path
  # per-token symmetric quantization backing `nn::QuantizedKVCache`.
  # Real kernels ship in `QuantizeKV.cu` below when
  # TESSERACT_ENABLE_CUDA=ON; the stub throws in CPU-only builds.
  QuantizeKVStub.cpp
  # Wave 11 (B-032+): fused ragged paged decode-attention. One warp per
  # (request, query-head); streams the KV prefix straight from the paged
  # pool via the block table (no gather / no repeat_kv) with an online
  # softmax. Real kernel ships in `PagedAttention.cu` below when
  # TESSERACT_ENABLE_CUDA=ON; the stub throws in CPU-only builds.
  PagedAttentionStub.cpp
  # M4 Track A2 (B-039): selective state-space scan (Mamba / S6). One thread
  # per (batch, inner channel); sequential t-recurrence with FP32 interior.
  # Real kernel ships in `SelectiveScan.cu` below when TESSERACT_ENABLE_CUDA=ON.
  SelectiveScanStub.cpp
  # M4 perf-closeout Phase 4: device-side MoE top-k routing. Real kernel in
  # `MoeRoute.cu` below when TESSERACT_ENABLE_CUDA=ON; stub throws otherwise.
  MoeRouteStub.cpp
  # M4 perf-closeout Phase 4: fully fused GPU MoE inference (device permute +
  # grouped GEMM + fused SiLU + scatter-combine). Real kernel in
  # `MoeForward.cu` below when TESSERACT_ENABLE_CUDA=ON; stub throws otherwise.
  MoeForwardStub.cpp
  # M4 perf-closeout Phase 5: causal depthwise conv1d for Mamba. Real kernel in
  # `CausalConv1d.cu` below when TESSERACT_ENABLE_CUDA=ON; stub throws otherwise.
  CausalConv1dStub.cpp
)

if(TESSERACT_ENABLE_CUDA)
  list(APPEND TESSERACT_CUDA_SOURCES
    Probe.cu
    Elementwise.cu
    # M2F: reduction / softmax / loss kernels. Each one is a
    # self-contained TU — they share only the device-side helpers in
    # `KernelUtils.cuh` (ShapePod + flat_to_offset + DeviceGuard),
    # which is header-only so there's no link-order coupling.
    Reduction.cu
    Softmax.cu
    Loss.cu
    # M2G: cuBLASLt matmul. Host-only translation unit (no device
    # code); compiled by the regular C++ compiler even though it's
    # part of the CUDA-only source set. Kept separate from
    # `MatMulStub.cpp` above so the stub can compile in CPU-only
    # builds without linking cuBLASLt.
    MatMul.cpp
    # M2H: strided copy / scatter-add + index_select / gather kernels.
    # Both are nvcc-compiled `.cu` TUs like the M2E/F kernels; they
    # share `KernelUtils.cuh` with the earlier bridges.
    Shape.cu
    Indexing.cu
    # M2I: fused Adam update kernel. Elementwise, no cross-param
    # dependency; single `.cu` TU driving both FP32 and FP64 via a
    # template on element type.
    Optim.cu
    # B-014: rotary position embedding. Lightweight per-pair kernel,
    # same KernelUtils.cuh scaffolding as the other M2E/M2F bridges
    # (DeviceGuard + check_launch + kBlockSize). FP32/FP64 native;
    # FP16/BF16 via the B-015/B-016 FP32-promotion pattern.
    RotaryEmbedding.cu
    # Wave 2 (B-022): fused RMSNorm / LayerNorm forward. One block
    # per row, block-scope sum / sum-of-squares reduction, FP32-
    # promoted on FP16/BF16 storage. Backward continues to flow
    # through the composite primitives' autograd nodes — the fused
    # path is forward-only, which is the Wave-2 scope.
    RMSNorm.cu
    # Wave 3.1 (B-021): INT8 weight-only dequantize-matmul. One
    # block per output pair (m, n), block-reduce over K, FP32
    # accumulator; FP16/BF16 activations loaded through FP32
    # promotion. Forward-only — autograd path falls back to the
    # composite `matmul(x, dequant(q_w, scale).T)` in
    # `src/ops/cpu/DequantMatMul.cpp`.
    DequantMatMul.cu
    # Wave 4.1 (B-025): fused SwiGLU forward (silu(gate) * up).
    # 1-D grid-stride element-wise kernel, FP32-accumulated on
    # FP16/BF16 storage. Backward continues to flow through the
    # composite `sigmoid` + `mul` + `mul` primitives' autograd
    # nodes — forward-only, mirrors the RMSNorm scope.
    SwiGLU.cu
    # Wave 4.2 (B-024): fused FlashAttention-2-style SDPA forward.
    # One block per `(b*h, q)` output row, online-softmax + tiled
    # K/V streaming so the `[S_q, S_k]` score matrix is never
    # materialized. FP32-accumulated on FP16/BF16 storage; FP64
    # goes through a separate native kernel for reference parity.
    # Backward continues to flow through the composite
    # `matmul → softmax → matmul` primitives' autograd nodes —
    # forward-only, mirrors the RMSNorm / SwiGLU scope.
    FusedAttention.cu
    # Wave 4.5 (B-019b): paged KV-cache gather. One thread per output
    # element of the [B, H, L, D_head] prefix, following a device-side
    # block table into the physical pool. Templated on element size
    # (2/4/8 bytes) — one body covers FP16/BF16/FP32/FP64.
    PagedKV.cu
    # Wave 9 (B-031): KV-cache INT8 quantize (one thread per token·head
    # row, FP32 absmax + scale) / dequantize (one thread per element).
    # FP32 native, FP16/BF16 via FP32-promoted load + narrow-on-store.
    QuantizeKV.cu
    # Wave 11 (B-032+): fused ragged paged decode-attention. One warp per
    # (request, query-head), online softmax over the request's KV prefix
    # read in place from the paged pool; GQA head h → KV head h/group.
    PagedAttention.cu
    # M4 Track A2 (B-039): selective state-space scan (Mamba / S6) kernel.
    SelectiveScan.cu
    # M4 perf-closeout Phase 3: FP8 (E4M3) tensor-core GEMM (cuBLASLt).
    # Wins the dense linear-layer line vs PyTorch FP16 by using Ada's FP8
    # tensor cores (~2× FP16 TC throughput). Includes an FP32→E4M3
    # conversion kernel so callers can quantize activations/weights.
    Fp8MatMul.cu
    # M4 perf-closeout Phase 4: device-side MoE top-k routing kernel.
    # Removes the host round-trip (D→H softmax/top-k, H→D mask) from
    # nn::MoEFeedForward so routing stays on-device for the GPU MoE path.
    MoeRoute.cu
    MoeForward.cu
    CausalConv1d.cu
  )
endif()

add_library(tesseract_cuda STATIC ${TESSERACT_CUDA_SOURCES})
add_library(Tesseract::cuda ALIAS tesseract_cuda)

target_include_directories(tesseract_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tesseract_cuda
  PUBLIC
    fmt::fmt
  PRIVATE
    Tesseract::compile_options
)

set_target_properties(tesseract_cuda PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

if(TESSERACT_ENABLE_CUDA)
  # TESSERACT_HAS_CUDA is the single define the rest of the codebase checks
  # to decide whether the CUDA backend is compiled in. Kept PRIVATE so the
  # define doesn't leak into translation units that include our public
  # headers (e.g. users, tests, downstream projects) — those only ever see
  # the probe API's stub-or-real decision through the linked library.
  target_compile_definitions(tesseract_cuda PRIVATE TESSERACT_HAS_CUDA=1)

  # CUDA::cudart for every CUDA-touching TU; CUDA::cublasLt only for
  # `MatMul.cpp` (M2G). Keeping cuBLASLt scoped to the one file that
  # actually calls it means link-time rebuilds after touching other
  # kernels don't re-resolve cuBLASLt symbols, and CPU-only builds
  # never reference a cuBLASLt target at all.
  # CUDA::cublas adds the legacy cuBLAS v2 API used by the fused MoE
  # (cublasGemmGroupedBatchedEx, CUDA 12.5+) in `MoeForward.cu`.
  target_link_libraries(tesseract_cuda PRIVATE CUDA::cudart CUDA::cublasLt CUDA::cublas)

  # Apply CUDA-specific properties only to the CUDA-language sources. This
  # keeps the C++ routing file compiled by the host C++ compiler (GCC /
  # Clang) and the `.cu` files compiled by nvcc.
  #
  # CUDA_STANDARD 17 is deliberate and portable down to CMake 3.22
  # (Ubuntu 22.04). The M2E elementwise bridge is hand-built not to
  # require C++20 device code: kernel entry points take raw `int64_t*`
  # + `int ndim` instead of `const Shape&`, so nvcc never parses
  # `std::span` or explicit-template lambdas. Host-side code under
  # `tesseract_cuda` still compiles with `CMAKE_CXX_STANDARD=20`
  # inherited from the project.
  set_target_properties(tesseract_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
  )
endif()

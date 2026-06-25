include(Catch)

function(tesseract_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name}
    PRIVATE
      Tesseract::tesseract
      Tesseract::compile_options
      Catch2::Catch2WithMain
  )
  # SKIP_RETURN_CODE 4 maps Catch2's SKIP() (exit code 4) to a ctest SKIP
  # instead of a FAILURE. Harmless for tests that never skip; required for
  # TUs whose CUDA-parity cases SKIP on CPU-only builds (e.g. layer_norm,
  # batch_norm) so a CPU-only `ctest` run is genuinely green.
  catch_discover_tests(${name}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES SKIP_RETURN_CODE 4
  )
endfunction()

tesseract_add_test(test_smoke     smoke/test_smoke.cpp)
tesseract_add_test(test_dtype     core/test_dtype.cpp)
tesseract_add_test(test_device    core/test_device.cpp)
tesseract_add_test(test_shape     core/test_shape.cpp)
tesseract_add_test(test_storage   core/test_storage.cpp)
tesseract_add_test(test_tensor    core/test_tensor.cpp)

tesseract_add_test(test_ops_gemv_vnni ops/test_gemv_vnni.cpp)
tesseract_add_test(test_core_to_autograd core/test_to_autograd.cpp)
tesseract_add_test(test_io_safetensors io/test_safetensors_roundtrip.cpp)
tesseract_add_test(test_io_tokenizer   io/test_tokenizer.cpp)
tesseract_add_test(test_io_bpe_tokenizer io/test_bpe_tokenizer.cpp)
target_compile_definitions(test_io_bpe_tokenizer
  PRIVATE TESSERACT_BPE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/io/fixtures")
tesseract_add_test(test_models_llama   models/test_llama_parity.cpp)
tesseract_add_test(test_models_llama_config_json models/test_llama_config_json.cpp)
tesseract_add_test(test_models_llama_quantize models/test_llama_quantize.cpp)
tesseract_add_test(test_models_llama_generate models/test_llama_generate.cpp)
tesseract_add_test(test_models_llama_train models/test_llama_train.cpp)
tesseract_add_test(test_models_sampler models/test_sampler.cpp)
tesseract_add_test(test_models_structured_decoding models/test_structured_decoding.cpp)
tesseract_add_test(test_models_speculative models/test_speculative.cpp)
tesseract_add_test(test_models_disaggregated models/test_disaggregated.cpp)
tesseract_add_test(test_models_scheduler models/test_scheduler.cpp)
# Wave 10 (B-032) compute-batched decode parity. The CUDA case SKIPs on
# CPU-only builds; register manually so catch_discover_tests can attach
# SKIP_RETURN_CODE (otherwise ctest reads the Catch2 skip as a failure).
add_executable(test_models_batched_decode models/test_batched_decode.cpp)
target_link_libraries(test_models_batched_decode
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_models_batched_decode
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)
tesseract_add_test(test_nn_paged_kv_pool nn/test_paged_kv_pool.cpp)
tesseract_add_test(test_nn_gqa nn/test_gqa.cpp)
tesseract_add_test(test_nn_moe nn/test_moe.cpp)
tesseract_add_test(test_nn_mamba nn/test_mamba.cpp)
tesseract_add_test(test_distributed_tp distributed/test_tensor_parallel.cpp)
# Wave 11 (B-032+) fused paged decode-attention. CUDA case SKIPs on
# CPU-only builds; register manually for SKIP_RETURN_CODE.
add_executable(test_nn_paged_attention nn/test_paged_attention.cpp)
target_link_libraries(test_nn_paged_attention
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_nn_paged_attention
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)
# Wave 13 (B-032+++) INT8-quantized paged KV cache. CUDA wiring case SKIPs
# on CPU-only builds; register manually for SKIP_RETURN_CODE.
add_executable(test_nn_quant_paged_kv nn/test_quant_paged_kv.cpp)
target_link_libraries(test_nn_quant_paged_kv
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_nn_quant_paged_kv
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)
# Wave 9 (B-031): INT8 KV-cache quantization. Same SKIP_RETURN_CODE 4
# contract as the other CUDA-parity TUs — the "CPU and CUDA agree" case
# SKIPs cleanly on CPU-only builds; the roundtrip / cache / generate
# cases always run. Registered manually so catch_discover_tests can
# attach the property to each discovered case.
add_executable(test_nn_quant_kv nn/test_quant_kv.cpp)
target_link_libraries(test_nn_quant_kv
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_nn_quant_kv
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

tesseract_add_test(test_ops_arithmetic ops/test_arithmetic.cpp)
tesseract_add_test(test_ops_matmul     ops/test_matmul.cpp)
tesseract_add_test(test_ops_reduction  ops/test_reduction.cpp)
tesseract_add_test(test_ops_activation ops/test_activation.cpp)
tesseract_add_test(test_ops_softmax    ops/test_softmax.cpp)
tesseract_add_test(test_ops_loss       ops/test_loss.cpp)
tesseract_add_test(test_ops_indexing   ops/test_indexing.cpp)
tesseract_add_test(test_ops_fp16       ops/test_fp16.cpp)
tesseract_add_test(test_ops_layer_norm ops/test_layer_norm.cpp)
tesseract_add_test(test_ops_attention  ops/test_ops_attention.cpp)

# Wave 4.1 (B-025) — fused SwiGLU activation (silu(gate) * up). Same
# SKIP_RETURN_CODE 4 contract as the M2E/F/G CUDA parity TUs: the
# CUDA parity cases SKIP cleanly on CPU-only builds, and the CPU-
# reference / autograd / validation cases always run. We register
# manually (not via `tesseract_add_test`) so `catch_discover_tests`
# can attach the SKIP_RETURN_CODE property to each discovered test.
add_executable(test_ops_swiglu ops/test_ops_swiglu.cpp)
target_link_libraries(test_ops_swiglu
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_swiglu
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# Wave 4.2 (B-024): fused FlashAttention-2-style SDPA parity tests.
# Same SKIP_RETURN_CODE 4 treatment as test_ops_swiglu — every CUDA
# test case exits with SKIP() on CPU-only builds, which CTest
# interprets as "skipped" rather than "failed" via this property.
add_executable(test_ops_cuda_fused_attention ops/test_ops_cuda_fused_attention.cpp)
target_link_libraries(test_ops_cuda_fused_attention
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_cuda_fused_attention
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

tesseract_add_test(test_autograd_gradcheck autograd/test_gradcheck.cpp)

tesseract_add_test(test_nn_embedding nn/test_embedding.cpp)
tesseract_add_test(test_nn_fit_linear nn/test_fit_linear.cpp)
tesseract_add_test(test_nn_mnist_smoke nn/test_mnist_smoke.cpp)
tesseract_add_test(test_nn_multihead_mixing nn/test_multihead_mixing.cpp)
tesseract_add_test(test_nn_kv_cache nn/test_kv_cache.cpp)
tesseract_add_test(test_nn_paged_kv_cache nn/test_paged_kv_cache.cpp)

# Wave 4.3 (B-023b) — CUDA Graph capture of a full `MHA::forward_step`
# decode pass. Registered manually so the GPU-required test cases mark
# as ctest SKIP via SKIP_RETURN_CODE 4 on CPU-only builds rather than
# hard-failing. The plain `set_current_len` bounds case always runs,
# which keeps the CPU-only build with at least one asserted path from
# this TU.
add_executable(test_nn_mha_cuda_graph nn/test_mha_cuda_graph.cpp)
target_link_libraries(test_nn_mha_cuda_graph
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_nn_mha_cuda_graph
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)
tesseract_add_test(test_nn_batch_norm nn/test_nn_batch_norm.cpp)
tesseract_add_test(test_nn_quantized_linear nn/test_nn_quantized_linear.cpp)
tesseract_add_test(test_nn_quantized_linear_int4 nn/test_nn_quantized_linear_int4.cpp)
# Wave 4.4 (B-026): eval-mode fast path guard — pins
# `module.eval()` → fused kernel regardless of `x.requires_grad()`.
tesseract_add_test(test_nn_quantized_linear_eval nn/test_nn_quantized_linear_eval.cpp)

tesseract_add_test(test_graph_smoke graph/test_graph_smoke.cpp)
tesseract_add_test(test_graph_autograd graph/test_graph_autograd.cpp)
tesseract_add_test(test_graph_train_parity graph/test_graph_train_parity.cpp)

# M2B — CUDA HAL probe. Builds + runs in both TESSERACT_ENABLE_CUDA=OFF
# and =ON configurations; the GPU-touching test case internally SKIPs via
# Catch2 when the build has no CUDA support or no GPU is visible. Catch2
# exits with status 4 when *every* assertion in a test case was skipped,
# so the custom catch_discover_tests call below marks that code as a
# ctest SKIP rather than FAILURE (SKIP_RETURN_CODE 4).
add_executable(test_hal_cuda_probe hal/test_hal_cuda_probe.cpp)
target_link_libraries(test_hal_cuda_probe
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
if(TESSERACT_ENABLE_CUDA)
  target_compile_definitions(test_hal_cuda_probe PRIVATE TESSERACT_ENABLE_CUDA_BUILD_FLAG=1)
endif()
catch_discover_tests(test_hal_cuda_probe
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# M2C — CUDA HAL allocator + streams. Same SKIP_RETURN_CODE contract as
# the probe test: every GPU-touching test case SKIPs cleanly when no
# CUDA is available, and Catch2 propagates exit code 4 to ctest.
add_executable(test_hal_cuda_alloc hal/test_hal_cuda_alloc.cpp)
target_link_libraries(test_hal_cuda_alloc
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
if(TESSERACT_ENABLE_CUDA)
  target_compile_definitions(test_hal_cuda_alloc PRIVATE TESSERACT_HAS_CUDA=1)
  target_link_libraries(test_hal_cuda_alloc PRIVATE CUDA::cudart)
endif()
catch_discover_tests(test_hal_cuda_alloc
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

add_executable(test_hal_cuda_stream hal/test_hal_cuda_stream.cpp)
target_link_libraries(test_hal_cuda_stream
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_hal_cuda_stream
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# M2D — HAL byte-copy primitives. CPU-only tests always run; CUDA
# round-trip tests SKIP cleanly when no GPU is visible.
add_executable(test_hal_cuda_copy hal/test_hal_cuda_copy.cpp)
target_link_libraries(test_hal_cuda_copy
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_hal_cuda_copy
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# M2D — Tensor-level CUDA round-trip. Satisfies the M2.α exit-bar
# `test_hal_cuda` contract in docs/m2-plan.md §5.1.
add_executable(test_hal_cuda_pinned hal/test_hal_cuda_pinned.cpp)
target_link_libraries(test_hal_cuda_pinned
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_hal_cuda_pinned
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

add_executable(test_hal_cuda_tensor hal/test_hal_cuda_tensor.cpp)
target_link_libraries(test_hal_cuda_tensor
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_hal_cuda_tensor
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# Wave 2.3 (B-023) — CUDA Graph capture + replay primitive. Always
# compiled; GPU-touching cases SKIP cleanly via `cuda::device_count()`
# guards when no GPU is visible (same contract as the `test_hal_*`
# suites above). Intentionally kept separate from the ops/ folder
# because this exercises the infrastructure layer (cudaGraph_t +
# stream capture), not any single op.
add_executable(test_cuda_graph cuda/test_cuda_graph.cpp)
target_link_libraries(test_cuda_graph
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_cuda_graph
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# M2E — CPU↔CUDA parity for the elementwise suite (add/sub/mul/div,
# neg/relu/sigmoid/tanh/exp/log, fill). Every test guards on GPU
# availability internally and SKIPs cleanly; the CPU-only smoke case
# at the tail of the file always runs so we have at least one
# asserted path in the CPU-only build.
add_executable(test_ops_cuda_elementwise ops/test_ops_cuda_elementwise.cpp)
target_link_libraries(test_ops_cuda_elementwise
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_cuda_elementwise
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# B-015 — CPU↔CUDA parity for the Float16 / BFloat16 elementwise path.
# Same `SKIP_RETURN_CODE 4` contract as the FP32/FP64 TU above. Kept in
# its own executable (rather than bolted onto `test_ops_cuda_elementwise`)
# so that any regression shows up on a dedicated ctest line.
add_executable(test_ops_cuda_elementwise_f16 ops/test_ops_cuda_elementwise_f16.cpp)
target_link_libraries(test_ops_cuda_elementwise_f16
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_cuda_elementwise_f16
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# M2F — CPU↔CUDA parity for reduction / softmax / cross-entropy. Same
# contract as `test_ops_cuda_elementwise`: every GPU-touching case
# SKIPs cleanly when no CUDA is available, and each TU keeps one
# always-running CPU smoke case so the CPU-only build still has an
# asserted path from the file.
add_executable(test_ops_cuda_reduction ops/test_ops_cuda_reduction.cpp)
target_link_libraries(test_ops_cuda_reduction
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_cuda_reduction
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

add_executable(test_ops_cuda_softmax ops/test_ops_cuda_softmax.cpp)
target_link_libraries(test_ops_cuda_softmax
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_cuda_softmax
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

add_executable(test_ops_cuda_loss ops/test_ops_cuda_loss.cpp)
target_link_libraries(test_ops_cuda_loss
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_cuda_loss
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# M2G — CPU↔CUDA parity for the cuBLASLt-backed matmul. Covers rank-2,
# batched (native + broadcast), transposed operands (the autograd
# backward's `mat_transpose` pattern), and the FP16 / BF16 Tensor-Core
# paths. One CPU-only smoke stays unconditional so the CPU build keeps
# at least one asserted path out of this TU.
add_executable(test_ops_cuda_matmul ops/test_ops_cuda_matmul.cpp)
target_link_libraries(test_ops_cuda_matmul
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_cuda_matmul
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# M2H — CPU↔CUDA parity for the shape / indexing suite (strided copy +
# atomic scatter-add + B-003 `index_select` / `gather`). Same
# SKIP_RETURN_CODE contract as the M2E/F/G bridges: every GPU-touching
# test case SKIPs cleanly when no CUDA is available, and one
# CPU-only smoke case always runs so the CPU build has at least one
# asserted path out of each TU.
add_executable(test_ops_cuda_shape ops/test_ops_cuda_shape.cpp)
target_link_libraries(test_ops_cuda_shape
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_cuda_shape
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

add_executable(test_ops_cuda_indexing ops/test_ops_cuda_indexing.cpp)
target_link_libraries(test_ops_cuda_indexing
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_cuda_indexing
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# M2J — CPU↔CUDA parity for `ops::attention(q, k, v, mask, causal)`.
# Exercises the composite attention path (Q·Kᵀ → softmax → ·V, Q
# pre-scaled by 1/√d) on both forward and autograd backward. Same
# SKIP_RETURN_CODE contract as the M2E/F/G/H bridges: every GPU-
# touching test case SKIPs cleanly when no CUDA is available, and one
# CPU-only smoke case always runs so the CPU build has at least one
# asserted path out of this TU.
add_executable(test_ops_cuda_attention ops/test_ops_cuda_attention.cpp)
target_link_libraries(test_ops_cuda_attention
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_ops_cuda_attention
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# M2I — step-by-step loss parity between the CPU and CUDA training
# stacks. Runs the mnist-smoke architecture (2-layer MLP + ReLU + CE
# + Adam) on a hermetic synthetic dataset so ctest stays download-
# free, then asserts per-step loss parity within 5e-3 absolute
# between the CPU and CUDA runs. SKIPs cleanly when no CUDA is
# available; one CPU-only convergence smoke always runs so the CPU
# build keeps an asserted path out of this TU.
add_executable(test_cuda_mnist_parity nn/test_cuda_mnist_parity.cpp)
target_link_libraries(test_cuda_mnist_parity
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_cuda_mnist_parity
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# M2K — single-block Llama-style transformer (RMSNorm + MHA + SwiGLU
# + residuals). Exercises the whole M2K nn stack + the refreshed
# `ops::rms_norm` / `ops::attention` composites. One CPU-only test and
# one CPU-only autograd test always run; CPU↔CUDA parity cases SKIP
# cleanly when no GPU is visible.
add_executable(test_transformer_block nn/test_transformer_block.cpp)
target_link_libraries(test_transformer_block
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_transformer_block
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

# B-014 — rotary position embedding (RoPE) parity suite. Exercises
# `ops::rotary_embedding` (CPU reference, CPU↔CUDA on FP32/64/FP16/BF16,
# autograd finite-diff) and `nn::RotaryEmbedding` (cached-table spec +
# Module::to-migrates-buffers smoke). Same SKIP_RETURN_CODE 4 contract
# as the other CUDA parity TUs.
add_executable(test_rotary_embedding ops/test_rotary_embedding.cpp)
target_link_libraries(test_rotary_embedding
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    Catch2::Catch2WithMain
)
catch_discover_tests(test_rotary_embedding
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES SKIP_RETURN_CODE 4
)

if(TESSERACT_ENABLE_MLIR)
  # Placeholder: MLIR round-trip tests are driven by the tesseract-opt binary,
  # not Catch2. Register a minimal ctest that exercises the dialect by running
  # the binary on a canned .mlir file.
  add_test(
    NAME ir_dialect_roundtrip
    COMMAND $<TARGET_FILE:tesseract-opt>
            ${CMAKE_CURRENT_SOURCE_DIR}/ir/roundtrip.mlir
  )

  # Wave 20 (B-037): tesseract.dequant_matmul verifier negatives — unknown
  # scheme + group_size that disagrees with the scheme family. Uses MLIR's
  # --verify-diagnostics so the expected-error markers must all fire.
  add_test(
    NAME ir_dequant_matmul_invalid
    COMMAND $<TARGET_FILE:tesseract-opt>
            ${CMAKE_CURRENT_SOURCE_DIR}/ir/dequant_matmul_invalid.mlir
            --split-input-file --verify-diagnostics
  )

  # M4 Track C2 (B-045): paged KV-cache + dynamic-shape op verifier negatives.
  add_test(
    NAME ir_paged_kv_invalid
    COMMAND $<TARGET_FILE:tesseract-opt>
            ${CMAKE_CURRENT_SOURCE_DIR}/ir/paged_kv_invalid.mlir
            --split-input-file --verify-diagnostics
  )

  # M1G lowering smoke test: pipe the input through tesseract-opt and
  # FileCheck the output against the expected sequence.
  find_program(TESSERACT_FILECHECK FileCheck
    HINTS ${LLVM_TOOLS_BINARY_DIR}
          ${CMAKE_PREFIX_PATH}/bin
          ${CMAKE_SOURCE_DIR}/third_party/llvm-install/bin)
  if(TESSERACT_FILECHECK)
    add_test(
      NAME ir_convert_to_linalg
      COMMAND bash -c
      "$<TARGET_FILE:tesseract-opt> --convert-tesseract-to-linalg --verify-each \
       ${CMAKE_CURRENT_SOURCE_DIR}/ir/convert_to_linalg.mlir \
       | ${TESSERACT_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/ir/convert_to_linalg.mlir"
    )
    # M4 Track C1 (B-044): transformer FFN + normalization primitive lowerings
    # (sigmoid / exp / log / sqrt / tanh / mean) used by rms_norm + swiglu.
    add_test(
      NAME ir_convert_transformer
      COMMAND bash -c
      "$<TARGET_FILE:tesseract-opt> --convert-tesseract-to-linalg --verify-each \
       ${CMAKE_CURRENT_SOURCE_DIR}/ir/convert_transformer.mlir \
       | ${TESSERACT_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/ir/convert_transformer.mlir"
    )
    # M4 Track C1 tail (B-044): single-head SDPA — softmax + the matmul /
    # transpose / mul / add chain that attention records — lowers to linalg.
    add_test(
      NAME ir_convert_attention
      COMMAND bash -c
      "$<TARGET_FILE:tesseract-opt> --convert-tesseract-to-linalg --verify-each \
       ${CMAKE_CURRENT_SOURCE_DIR}/ir/convert_attention.mlir \
       | ${TESSERACT_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/ir/convert_attention.mlir"
    )
    # M4 Track C2 (B-045): paged KV-cache + dynamic-shape ops round-trip.
    add_test(
      NAME ir_paged_kv_roundtrip
      COMMAND bash -c
      "$<TARGET_FILE:tesseract-opt> ${CMAKE_CURRENT_SOURCE_DIR}/ir/paged_kv.mlir \
       | $<TARGET_FILE:tesseract-opt> \
       | ${TESSERACT_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/ir/paged_kv.mlir"
    )
    add_test(
      NAME ir_backward
      COMMAND bash -c
      "$<TARGET_FILE:tesseract-opt> --tesseract-backward --verify-each \
       ${CMAKE_CURRENT_SOURCE_DIR}/ir/backward.mlir \
       | ${TESSERACT_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/ir/backward.mlir"
    )
    # Pins the M1I.2.c loop-order rewrite that closed B-007 — every
    # `linalg.matmul` is expected to become a `linalg.generic` with
    # `iterator_types = ["parallel", "reduction", "parallel"]` (M-K-N
    # loop order). The benchmark's 7–15x speedup over eager is
    # directly downstream of this property holding.
    add_test(
      NAME ir_interchange_matmul
      COMMAND bash -c
      "$<TARGET_FILE:tesseract-opt> --tesseract-interchange-matmul --verify-each \
       ${CMAKE_CURRENT_SOURCE_DIR}/ir/interchange_matmul.mlir \
       | ${TESSERACT_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/ir/interchange_matmul.mlir"
    )
    # Wave 15 (B-009 / M2L.2): tesseract → GPU dialect lowering. Pins the
    # device-IR structure (gpu.module / gpu.func / gpu.launch_func) produced
    # by the --convert-tesseract-to-gpu pipeline. The PTX/cubin JIT + parity
    # tail is gated on an NVPTX-enabled LLVM build (see ConvertToGpu.cpp).
    add_test(
      NAME ir_convert_to_gpu
      COMMAND bash -c
      "$<TARGET_FILE:tesseract-opt> --convert-tesseract-to-gpu \
       ${CMAKE_CURRENT_SOURCE_DIR}/ir/convert_to_gpu.mlir \
       | ${TESSERACT_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/ir/convert_to_gpu.mlir"
    )
    # M4 Phase 8 (B-009 tail): device compilation — gpu.module -> NVVM -> PTX
    # -> cubin via ptxas. Pins that --gpu-module-to-binary produces a real
    # gpu.binary with a #gpu.object for sm_89. Requires ptxas on PATH (ships
    # with the CUDA toolkit); no GPU needed at test time.
    if(TESSERACT_ENABLE_CUDA)
      add_test(
        NAME ir_gpu_to_cubin
        COMMAND bash -c
        "$<TARGET_FILE:tesseract-opt> \
         --convert-tesseract-to-gpu \
         --nvvm-attach-target='chip=sm_89 O=3' \
         --convert-gpu-to-nvvm --convert-nvvm-to-llvm \
         --reconcile-unrealized-casts --gpu-module-to-binary \
         ${CMAKE_CURRENT_SOURCE_DIR}/ir/gpu_to_cubin.mlir \
         | ${TESSERACT_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/ir/gpu_to_cubin.mlir"
      )
    endif()
  else()
    message(WARNING
      "FileCheck not found next to LLVM install; skipping "
      "ir_convert_to_linalg test. Build LLVM with "
      "scripts/build_llvm.sh to pick it up.")
  endif()

  # Catch2 test driving the Graph -> MLIR emitter. We link tesseract_ir in
  # addition to the main target because emit_mlir is MLIR-gated and only
  # exposed when TESSERACT_ENABLE_MLIR=ON.
  add_executable(test_ir_emit_mlir ir/test_emit_mlir.cpp)
  target_link_libraries(test_ir_emit_mlir
    PRIVATE
      Tesseract::tesseract
      Tesseract::ir
      Tesseract::compile_options
      Catch2::Catch2WithMain
  )
  catch_discover_tests(test_ir_emit_mlir
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # M1I stage-1: end-to-end parity test driving the MNIST MLP through
  # GraphScope → emit_mlir → --tesseract-backward → --convert-tesseract
  # -to-linalg entirely in-process. Keeps tesseract_ir on the link line
  # because we build + run the pass pipeline programmatically here.
  add_executable(test_graph_mnist_mode graph/test_graph_mnist_mode.cpp)
  target_link_libraries(test_graph_mnist_mode
    PRIVATE
      Tesseract::tesseract
      Tesseract::ir
      Tesseract::compile_options
      Catch2::Catch2WithMain
  )
  catch_discover_tests(test_graph_mnist_mode
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # M1I.2.b: MLIR ExecutionEngine JIT parity. Builds tiny graphs, runs
  # them through both the C++ graph interpreter and the JIT, asserts
  # numerical agreement.
  add_executable(test_ir_jit_forward ir/test_jit_forward.cpp)
  target_link_libraries(test_ir_jit_forward
    PRIVATE
      Tesseract::tesseract
      Tesseract::ir
      Tesseract::compile_options
      Catch2::Catch2WithMain
  )
  catch_discover_tests(test_ir_jit_forward
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # M4 Track C1 (B-044): transformer FFN + normalization primitives through
  # the JIT — sigmoid, SwiGLU activation, and the RMSNorm primitive chain
  # compared to the eager ops within FP32 tolerance.
  add_executable(test_ir_jit_transformer ir/test_jit_transformer.cpp)
  target_link_libraries(test_ir_jit_transformer
    PRIVATE
      Tesseract::tesseract
      Tesseract::ir
      Tesseract::compile_options
      Catch2::Catch2WithMain
  )
  catch_discover_tests(test_ir_jit_transformer
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # M1I.2.b-Phase-2: full-training parity — the JIT engine must match
  # the graph interpreter's loss curve when driving a real MLP training
  # loop (Adam, CE, 40 steps, shape-specialized MLP).
  add_executable(test_graph_jit_parity ir/test_graph_jit_parity.cpp)
  target_link_libraries(test_graph_jit_parity
    PRIVATE
      Tesseract::tesseract
      Tesseract::ir
      Tesseract::compile_options
      Catch2::Catch2WithMain
  )
  catch_discover_tests(test_graph_jit_parity
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # M4 Phase 8 — numerical backward parity (graph build_backward vs eager
  # autograd) for the softmax/sum any-dim rules. Interpreter-only (no JIT), so
  # it runs even where the ExecutionEngine can't, but it is an IR-pass gate.
  add_executable(test_ir_backward_parity ir/test_backward_parity.cpp)
  target_link_libraries(test_ir_backward_parity
    PRIVATE
      Tesseract::tesseract
      Tesseract::ir
      Tesseract::compile_options
      Catch2::Catch2WithMain
  )
  catch_discover_tests(test_ir_backward_parity
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # M4 Phase 8 (B-009 tail) — GPU JIT (cubin) vs eager numerical parity.
  # Only meaningful when CUDA is enabled (GpuJitEngine.cpp is compiled in);
  # the test self-skips at run-time when no device is visible.
  if(TESSERACT_ENABLE_CUDA)
    add_executable(test_ir_gpu_jit_parity ir/test_gpu_jit_parity.cpp)
    target_link_libraries(test_ir_gpu_jit_parity
      PRIVATE
        Tesseract::tesseract
        Tesseract::ir
        Tesseract::compile_options
        Catch2::Catch2WithMain
    )
    catch_discover_tests(test_ir_gpu_jit_parity
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
  endif()
endif()

# B-006 — end-to-end MNIST smoke test. Runs the real `tesseract_mnist`
# binary against the four IDX files in `data/mnist/` (shipped by
# `scripts/fetch_mnist.sh`) for a small `--max-steps` window. Gated on
# `TESSERACT_BUILD_EXAMPLES=ON` (the binary doesn't exist otherwise)
# and skips at run-time with exit code 77 when the data directory is
# missing, so developers / CI workers without the fixture still get a
# clean `ctest` run. Defaults to the in-tree `data/mnist/`; override
# with the `TESSERACT_MNIST_DATA` env var.
if(TESSERACT_BUILD_EXAMPLES)
  set(_tesseract_mnist_default_dir "${CMAKE_SOURCE_DIR}/data/mnist")
  add_test(
    NAME example_mnist_smoke
    COMMAND bash -c
    "dir=\"\${TESSERACT_MNIST_DATA:-${_tesseract_mnist_default_dir}}\"; \
     for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
              t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte; do \
       if [ ! -f \"\$dir/\$f\" ]; then \
         echo \"[skip] MNIST fixture missing: \$dir/\$f (populate with scripts/fetch_mnist.sh)\"; \
         exit 77; \
       fi; \
     done; \
     exec $<TARGET_FILE:mnist> \"\$dir\" --epochs 1 --max-steps 50"
  )
  set_tests_properties(example_mnist_smoke PROPERTIES
    LABELS "slow;example"
    SKIP_RETURN_CODE 77
    TIMEOUT 300
  )
endif()

# Example programs. Enabled via -DTESSERACT_BUILD_EXAMPLES=ON.

add_executable(mnist mnist.cpp)
target_link_libraries(mnist
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    fmt::fmt
)
set_target_properties(mnist PROPERTIES OUTPUT_NAME "tesseract_mnist")

# When MLIR is available, fold the `--mode graph` path in: it captures
# one forward+loss step into `tesseract.graph`, synthesizes the backward
# pass, and verifies the lowered IR — see examples/mnist.cpp for the
# exact sequence. Without MLIR, the `--mode graph` flag errors out.
if(TESSERACT_ENABLE_MLIR)
  target_link_libraries(mnist PRIVATE Tesseract::ir)
  target_compile_definitions(mnist PRIVATE TESSERACT_ENABLE_MLIR=1)
endif()

# M2K: single-layer Llama-style transformer block demo. Exercises the
# full M2K primitive chain (ops::rms_norm / ops::attention / SwiGLU +
# residuals) in eager mode on both CPU and CUDA. Intentionally the
# smallest viable transformer binary — more layers / sampling / KV-cache
# land with `llama_inference` in M3.
add_executable(llama_forward llama_forward.cpp)
target_link_libraries(llama_forward
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    fmt::fmt
)
set_target_properties(llama_forward PROPERTIES OUTPUT_NAME "tesseract_llama_forward")

# Wave-1a: full-stack Llama inference demo. Stacks `tesseract::models::LlamaModel`
# (embed_tokens + N × TransformerBlock + RMSNorm + lm_head) end-to-end on CPU
# or CUDA and prints top-k logits for a random token sequence. Can either
# load an HF safetensors checkpoint or exercise a synthetic random-init model
# (the second mode is the default so `ctest` / CI can smoke the binary
# without shipping a real checkpoint).
add_executable(llama_infer llama_infer.cpp)
target_link_libraries(llama_infer
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    fmt::fmt
)
set_target_properties(llama_infer PROPERTIES OUTPUT_NAME "tesseract_llama_infer")

# Wave 7 (B-029): continuous-batching serving demo. Submits several
# prompts to the ContinuousBatchingScheduler at once and drives the
# engine to completion over a shared paged KV pool — showing dynamic
# admission, per-request sampling, and block recycling.
add_executable(llama_serve llama_serve.cpp)
target_link_libraries(llama_serve
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    fmt::fmt
)
set_target_properties(llama_serve PROPERTIES OUTPUT_NAME "tesseract_llama_serve")

# M4 Track B2 (B-042): single-GPU/CPU LLM training loop. Trains a tiny
# LlamaModel with next-token cross-entropy + Adam and shows the loss of a
# fixed synthetic batch collapsing to ~0 — the convergence prerequisite for
# distributed training (B3). Doubles as a CI-fast smoke via --max-steps /
# --target-loss.
add_executable(llama_train llama_train.cpp)
target_link_libraries(llama_train
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    fmt::fmt
)
set_target_properties(llama_train PROPERTIES OUTPUT_NAME "tesseract_llama_train")

# Wave 16 (B-033): real HF checkpoint end-to-end generation. Reads
# architecture from config.json, loads *.safetensors + tokenizer.json, and
# streams the decoded continuation token-by-token. The "actually load a real
# model and talk to it" demo for the M3 exit criteria.
add_executable(llama_generate llama_generate.cpp)
target_link_libraries(llama_generate
  PRIVATE
    Tesseract::tesseract
    Tesseract::compile_options
    fmt::fmt
)
set_target_properties(llama_generate PROPERTIES OUTPUT_NAME "tesseract_llama_generate")

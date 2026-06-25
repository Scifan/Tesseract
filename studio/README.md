# Tesseract Studio

A Scratch-like **visual block builder** for the Tesseract framework. Drag,
drop, and wire graphical blocks to build a model, train it (with a live loss
curve), run text generation, experiment with tensors + autograd, or inspect the
**tesseract-dialect IR** the blocks lower to — no code required. Studio is a
single, self-contained native C++ executable that **embeds the engine directly**
(no Python), so the blocks call the real `nn` / `models` / `optim` / `ops` C++
objects.

The canvas is built on [LiteGraph.js](https://github.com/jagenjo/litegraph.js)
(MIT) — a mature canvas node editor — vendored and embedded into the binary, so
drag / connect / pan / zoom and typed slots are robust and offline. The slot
types and node palette are generated from the C++ block catalog at runtime.

It doubles as a visual front-end onto the framework's "one IR, train+infer"
idea: a block program is a dataflow graph that lowers to the same `tesseract`
MLIR dialect the C++/Python APIs and `tests/ir/*.mlir` use.

## Build

Studio is gated behind a CMake option (default OFF):

```bash
cmake -S . -B build-studio -DTESSERACT_BUILD_STUDIO=ON -DTESSERACT_BUILD_TESTS=ON
cmake --build build-studio --target tesseract_studio -j
```

Add `-DTESSERACT_ENABLE_CUDA=ON` to make the `CUDA` run device available.

## Run

```bash
./build-studio/studio/tesseract_studio --port 8770          # then open the URL
./build-studio/studio/tesseract_studio --device cuda         # default run device
```

It prints `http://localhost:<port>`; open that in a browser. The UI is served
and embedded by the binary itself (the browser is just the display surface, so
this works on a headless GPU box). All compute runs in the embedded C++ engine.

## The canvas

- **Palette (left):** click a block to drop it on the canvas, or **drag** it
  onto the canvas to place it. Filter with the search box. Categories: Data,
  Layers, Tensor, Model, Loss, Optimizer, Tokenizer, Train, Inference.
- **Wiring:** drag from an **output** slot (right of a block) to an **input**
  slot (left of another). Only type-compatible ports connect (Tensor, Model,
  Optimizer, Loss, Dataset, Tokenizer, …); LiteGraph rejects mismatched wires.
  Pan with the canvas drag, zoom with the wheel, double-click for a node search.
- **Inspector (right):** edit the selected block's parameters; the graph
  re-validates and re-infers tensor shapes on every change (shapes render on
  each block, problems in the Diagnostics tab, an error/warning dot on the node).
- **Run ▶ / Stop ■:** execute the graph on the selected device. The bottom
  dock streams a live loss curve + accuracy (training), generated tokens
  (inference), or tensor **heatmaps** of values + gradients (tensor playground).
- **IR:** lower the current blocks to a `tesseract`-dialect MLIR module.
- **C++ / Python:** generate runnable source for the current graph.
- **Examples ▾:** load the MLP-training, Llama-generation, or tensor-playground
  starter graphs.
- **Save / Open:** save the block program as a `.tsb` file, or open a `.tsb` —
  or even a Studio-generated `.cpp`/`.py` (re-imported via its embedded
  `@tsb-graph` header).

## Starter flows (Examples ▾ menu)

- **Train an MLP** (`studio/examples/mlp_train.tsb`):
  `Input → Linear → ReLU → Linear → SequentialModel`, plus
  `SyntheticClassification`, `Adam`, `CrossEntropyLoss`, `TrainLoop`. Run it and
  watch the loss collapse to ~0 / accuracy to 100% (CPU or CUDA).
- **Generate from a Llama** (`studio/examples/llama_generate.tsb`):
  `LoadLlama → Generate`. Add a `BpeTokenizer` and set a text prompt to decode
  real text; set `model_dir` on `LoadLlama` to load HF safetensors weights.
- **Tensor & autograd playground:** `TensorConst · TensorConst → TMatMul →
  TReLU → Tensor Inspect`. The Inspect block evaluates the expression, shows the
  output tensor as a heatmap, runs `backward` on `Σ(output)`, and visualizes the
  gradient w.r.t. each leaf tensor. Element-wise (`TAdd/TSub/TMul`), `TMatMul`,
  and unary (`TReLU/TSigmoid/TTanh/TExp`) ops are available.

## How it is organized

| Path | What |
| ---- | ---- |
| `studio/core/` | `tesseract_studio_core` — block graph, `.tsb` JSON, catalog, validation + shape inference, C++/Python codegen, IR emission, and the executor (train / generate / tensor). GUI-independent and unit-tested headless (`test_studio`). |
| `studio/app/` | The executable: a minimal in-tree HTTP server, the control plane (`main.cpp`), and the embedded web UI (`app/web/`, incl. the vendored `app/web/vendor/litegraph.*`). |
| `studio/examples/` | Sample `.tsb` programs. |

Adding a block = one `BlockSpec` in
[`core/src/BlockCatalog.cpp`](core/src/BlockCatalog.cpp) plus a case in the
codegen and executor.

## Tests

```bash
cmake --build build-studio --target test_studio -j && ./build-studio/studio/test_studio
```

Covers JSON/`.tsb` round-trip, catalog, validation + shape inference, codegen
round-trip, IR emission, and the executor training an MLP, generating from a
Llama, and evaluating + back-propagating a tensor graph through the real engine
(10 cases, verified on CPU and CUDA).

## Notes / limits

- Generation runs `model.generate()` then replays the tokens; true per-token
  streaming via `forward_step` is a follow-up (B-047+).
- RMSNorm / LayerNorm / FeedForward / attention blocks are emitted in C++ codegen
  but not in Python codegen (the current pybind surface doesn't expose them).
- The IR view is display-only: it mirrors the dialect's real op set/assembly
  (so it reads like genuine `tesseract`-dialect MLIR) but is not parsed back.
  Complex layers (Embedding/FeedForward/MHA/Transformer) are shown as a
  placeholder comment rather than their full lowering.
- Linux-only for now (the HTTP server uses POSIX sockets).

# ADR-0007 — Tesseract Studio: a native, engine-embedded visual block builder

- Status: accepted (2026-06-25)
- Milestone: M5 (adoption); backlog B-047
- Supersedes / relates: ADR-0006 (M4 adoption track established the Python
  frontend B-041)

## Context

A pre-M5 review asked whether the framework is usable "fluently and completely
like PyTorch/TensorFlow" from Python. It is not yet: the pybind11 frontend
(B-041) is a deliberately thin shim — CPU-only construction (no Python-side
`.to(device)`), a narrow op/module surface, no serialization, no data pipeline.

Separately, M5's adoption goal calls for a **visual block builder** (Scratch
style): drag-drop graphical blocks for building, training, and running models so
beginners can use the framework without writing code. We want this shipped as a
runnable executable.

Three design forks were considered for how the UI drives the engine:

1. Web UI + local **Python** backend — requires first expanding the Python
   bindings to near-PyTorch coverage.
2. Native desktop app **embedding the C++ engine directly** (no Python).
3. Code-generation-first (blocks emit source you run elsewhere).
4. Web + WASM (engine compiled into the browser).

The user selected **(2)**: a single self-contained native C++ executable,
embedding the engine directly, no Python.

## Decision

Build **Tesseract Studio** as a native C++ program under `studio/`, gated by the
CMake option `TESSERACT_BUILD_STUDIO` (default OFF). It has two layers:

- **`tesseract_studio_core`** (static lib): GUI-independent logic — the block
  graph model, `.tsb` JSON (bidirectional), catalog, validation + shape
  inference, code generation (C++/Python), and the executor that instantiates
  the real `nn`/`models`/`optim` objects and runs training/generation. It links
  only the engine, so it is fully unit-testable headless.
- **`tesseract_studio`** (executable): a thin presentation layer. It embeds the
  block-canvas UI (compiled into the binary as byte arrays) and exposes the core
  through a tiny localhost HTTP/JSON control plane.

### Rendering: web-served UI, not GLFW/ImGui

The plan's first instinct was Dear ImGui + GLFW + OpenGL3. The dev/deploy box
is a **headless GPU server** with no X11/OpenGL dev headers and no `sudo`, so a
GLFW window can neither build nor run there. To honor the intent (a native,
self-contained, engine-embedded executable with a drag-drop canvas) while
staying buildable and verifiable on the actual hardware, the executable instead
**serves a self-contained block-canvas web UI** (embedded in the binary) over
localhost; the browser is only the display surface. All ML work runs in the
embedded C++ engine. This keeps every constraint the user set — native C++
executable, engine embedded, no Python — and adds headless operability (useful
on remote GPU boxes) as a bonus. If a desktop-window build is wanted later, the
GUI-independent `studio_core` already isolates all logic so an ImGui/Qt
front-end can be added without touching it.

### Node editor: LiteGraph.js (vendored), not hand-rolled SVG

The first canvas was hand-rolled (vanilla JS + SVG). It was crude and had a
drag/connect bug (re-rendering nodes mid-interaction invalidated the dragged
element). Rather than reinvent a node editor, Studio now builds on
**LiteGraph.js** (MIT) — a mature canvas node editor with robust
drag/connect/pan/zoom, typed slots, and link-type validation that map cleanly
onto the block catalog's `PortType`s. The library is downloaded once and
**vendored** under `app/web/vendor/` and **embedded into the binary** (byte
arrays, same as the rest of the UI), so the executable stays self-contained and
offline. The catalog (palette, slot types, params) is generated into LiteGraph
node types at runtime, so adding a `BlockSpec` still requires no UI code.

### Dependencies

The backend adds **no third-party C++ dependency**: a small hand-rolled JSON
module and a minimal POSIX HTTP server live in-tree, matching the framework's
"dependency-free parser" convention. The only vendored asset is the embedded
LiteGraph.js front-end (build-time download via CDN/mirror; GitHub `git`
FetchContent is what is blocked from the box, not HTTPS CDNs).

## Why this matters (significance that guides scope)

- A block program *is* a dataflow graph → Studio is a visual front-end onto the
  "one IR, train+infer" thesis. This is made literal by the **IR view**, which
  lowers the blocks to the same `tesseract` MLIR dialect as `tests/ir/*.mlir`.
- It is the strongest adoption lever for the M5 open-source release.
- It gives beginners a concrete, runnable mental model of tensors / autograd /
  model assembly with zero boilerplate, on CPU or CUDA via the same blocks.

### Feature scope (all six proposed groups shipped)

The full proposed scope is implemented, not a subset: **build** (layer blocks
→ Sequential / Llama / Mamba), **train** (dataset + optimizer + loss + loop with
a live loss curve), **infer** (Generate + tokenizer), **tensor** (a tensor &
autograd playground: `TensorConst`, element-wise / matmul / unary ops, and a
`Tensor Inspect` sink that evaluates the expression, runs `backward`, and
heatmaps values + per-leaf gradients), **IR** (block graph → tesseract-dialect
MLIR view), and **code I/O** (C++/Python codegen + `.tsb` save/open, with
generated source re-openable via its `@tsb-graph` header).

## Consequences

- The executable runs headless; users open `http://localhost:<port>`.
- Block coverage is driven by one declarative catalog; adding a block is one
  `BlockSpec` plus a codegen/executor case.
- The Python-binding maturity gap is documented but **not** on Studio's critical
  path (native embedding sidesteps it); expanding the bindings remains a
  separate, optional adoption task.
- B-024+ WMMA tensor-core FlashAttention (the vLLM-TTFT closer) has since landed
  (2026-06-25), together with B-024c stride-aware/BSHD attention layout — TTFT
  7.28 → 5.86 ms (vLLM 5.47, gap now 1.07×). Studio was the productive work done
  while cards were busy; the remaining TTFT closer is B-024e (BSHD-native RoPE +
  elementwise fusion).

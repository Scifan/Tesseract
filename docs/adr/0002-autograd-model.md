# ADR-0002: Eager tape-based autograd for M0, IR-native in M1

- **Status:** Accepted (2026-04)
- **Supersedes:** none
- **Authors:** tesseract core team

## Context

Every deep learning framework has to commit to an autograd architecture early,
because the shape of that machinery constrains the entire op set: how ops
record tape entries, where gradients are stored, what a leaf tensor means, and
how functorch-style higher-order transforms are layered on top. The
mainstream designs are:

- **PyTorch:** eager tape (`AutogradMeta` per tensor, `Node`/`Edge` DAG,
  `Engine::execute` topo-sort), extended by `torch.func` for functional
  transforms.
- **JAX:** no tape. Traces pure Python functions into an IR (`jaxpr`) and
  runs reverse-mode AD as an IR-to-IR transform.
- **TensorFlow 2:** `tf.GradientTape` — an opt-in tape only inside the
  `with` block.
- **MLIR / Enzyme / OpenXLA:** AD is a compiler pass over SSA IR.

Tesseract's roadmap calls for an IR-native AD pass at M1. Building that
immediately would have meant shipping M0 without being able to use
`loss.backward()` at all — unacceptable for the MNIST demo and for every test
we want to write. So M0 ships a tape, but one designed to **retire gracefully**
once the IR lands.

## Decision

M0 ships an eager tape-based autograd with the following structural choices:

1. **Per-tensor metadata.** Every `Tensor` may carry an `AutogradMeta`
   (`requires_grad`, `grad_fn`, accumulated `grad`, `output_nr`) allocated
   lazily in `TensorImpl`. Tensors without `requires_grad` pay zero overhead.

2. **Explicit `Node` DAG.** Each differentiable op constructs a subclass of
   `autograd::Node` in its forward function, wires its `next_edges` to the
   inputs' `grad_fn`s (or leaf impls), saves only the tensors it needs, and
   attaches itself to the output via `attach_grad_fn`. This mirrors PyTorch's
   `Function` API closely enough that porting kernels later is mechanical.

3. **Iterative topological execution.** `Engine::backward(root)` performs an
   iterative DFS from the root `grad_fn`, accumulates input gradients per
   node, and dispatches to `Node::apply`. Gradient accumulation on leaves
   uses `ops::add` inside `NoGradGuard`.

4. **Scoped grad enable/disable.** `NoGradGuard` flips a thread-local flag
   consulted by every differentiable forward; this keeps evaluation paths,
   optimizer steps, and backward node execution itself cheap and safe.

5. **Views are autograd-aware, but through the `ops::` layer.** The raw
   `Tensor::{view, reshape, permute, transpose, contiguous, clone}` methods
   stay dependency-free (they live in `tesseract_core`, which cannot depend
   on autograd). The parallel `tesseract::ops::{view, reshape, permute,
   transpose, contiguous, clone}` entry points delegate to those methods
   for the forward and then, when `is_grad_enabled()`, attach the
   corresponding backward `Node` (`ReshapeBackward`, `PermuteBackward`,
   `TransposeBackward`, `IdentityBackward`). Downstream modules
   (`nn::Linear`, the future IR-backed eager fallback) must use the
   `ops::` versions — this is the trade-off for keeping `core` free of a
   cycle into `autograd`. PyTorch-style `[out_features, in_features]`
   weight storage is feasible because `ops::transpose` preserves the edge
   back to the original parameter.

## Consequences

### Positive

- **Low conceptual overhead.** Contributors familiar with PyTorch autograd
  feel at home immediately.
- **Eagerly debuggable.** A broken backward is isolated to one `Node::apply`
  function; stack traces are native C++.
- **Incrementally testable.** `tests/autograd/test_gradcheck.cpp` uses finite
  differences to verify every forward/backward pair against its analytic
  counterpart to 1e-3 tolerance (float32).

### Negative

- **Tape cost at runtime.** Every autograd-enabled op allocates a `Node` +
  saves tensors; this inflates step latency vs a fully-fused graph run.
  Acceptable for M0's CPU MNIST target, retired once M1's IR AD pass takes
  over.
- **View ops live in two places.** The value-level method on `Tensor`
  *and* the wrapper in `ops::` both exist, and callers have to pick the
  right one. Enforced by review + the grep-friendly `ops::` prefix.
- **Two AD implementations to maintain during M1.** During the M0→M1
  transition we will run both paths side-by-side in tests to guarantee
  numerical parity before deleting the tape.

### Mitigations

- The `Node` API (`apply(grad_output) -> vector<Tensor>`) matches the shape
  expected by an IR-level AD pass operating on SSA values, so M1 can reuse
  the per-op backward logic with minimal rewriting.
- `AutogradMeta` is deliberately opaque to users; all access goes through
  `Tensor::grad()` / `Tensor::requires_grad()` / `Tensor::mutable_autograd_meta()`,
  so M1 can swap the implementation without breaking source compatibility.

## Alternatives considered

- **Skip autograd in M0 and go straight to IR AD.** Rejected: delays the
  training demo by months and prevents any test that depends on gradients.
- **Port PyTorch's `autograd` wholesale.** Rejected: too much code surface
  for M0 (engine queues, reentrant backward, distributed hooks). We take the
  shape of the API but keep the implementation ~500 LOC.
- **Dual-number / forward-mode AD in M0.** Rejected: good for Jacobians,
  bad for training; reverse-mode is the only useful direction at M0.

## References

- `include/tesseract/autograd/Node.hpp`, `src/autograd/Engine.cpp`
- `tests/autograd/test_gradcheck.cpp`
- ADR-0001 (MLIR as shared IR): motivates the eventual migration target.

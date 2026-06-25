# ADR-0003: Naming conventions

- **Status:** Accepted (2026-04)
- **Supersedes:** none
- **Authors:** tesseract core team

## Context

The C++ deep learning ecosystem is a collage of naming styles: PyTorch uses
`snake_case` methods on `CamelCase` classes, Eigen uses `CamelCase` for
everything, and Abseil and Google C++ style are in tension with LLVM style.
Tesseract lives next to MLIR/LLVM (imported into the same translation units)
and will be consumed from Python later (where `snake_case` is expected), so
we must state a policy up front to keep the public headers legible.

## Decision

Tesseract uses a hybrid style, encoded in one sentence: **types follow LLVM,
ops follow PyTorch, macros follow the `TESSERACT_` screaming-snake
convention.** The concrete rules:

### Identifiers

| Entity                       | Style              | Example                              |
|------------------------------|--------------------|--------------------------------------|
| Namespaces                   | `snake_case`       | `tesseract::nn`, `tesseract::ops`    |
| Types / classes / structs    | `CamelCase`        | `Tensor`, `AutogradMeta`, `SGD`      |
| Enums and enum values        | `CamelCase` + `CamelCase` | `DType::Float32`, `DeviceKind::Cuda` |
| Free functions / methods     | `snake_case`       | `ops::matmul`, `Tensor::is_contiguous` |
| Member variables             | `snake_case_`      | `weight_`, `next_edges`              |
| Constants / `constexpr`      | `kCamelCase`       | `kMaxTensorRank`                     |
| Template parameters          | `CamelCase`        | `template <typename T>`              |
| Macros                       | `TESSERACT_*`      | `TESSERACT_CHECK`, `TESSERACT_ENABLE_MLIR` |
| CMake options                | `TESSERACT_*`      | `TESSERACT_BUILD_TESTS`              |

Trailing underscore on private/protected member variables is mandatory. It
distinguishes `this->weight_` from a locally-scoped `weight` without having to
type `this->` everywhere and matches LLVM/Abseil.

### Files

- Headers: `include/tesseract/<subsystem>/<Type>.hpp` — one primary type per
  file, `CamelCase` filename matching the primary type (e.g.
  `include/tesseract/nn/Linear.hpp`).
- Source: `src/<subsystem>/<Type>.cpp` or `src/<subsystem>/cpu/<Op>.cpp`
  (backend subfolder). Tests mirror the source tree: `tests/nn/test_<x>.cpp`.
- Include guard style: `#pragma once` only; no macro guards.

### Include order

Within each `.cpp` the block order is:

1. The matching `.hpp` (`#include "tesseract/nn/Linear.hpp"`).
2. C++ standard library.
3. Third-party (`fmt`, `Catch2`, `mlir/...`).
4. Other Tesseract headers (`#include "tesseract/ops/...hpp"`).

Each block is alphabetized and separated by a blank line.

### Operators

User-facing ops live in `tesseract::ops` and are implemented twice:

- `ops::foo_forward(...)` — computes the output, no autograd wiring.
- `ops::foo(...)` — the autograd-aware entry point; attaches `FooBackward`
  when `is_grad_enabled()` is true.

Backward classes are named `<Op>Backward`, stored next to the forward, and
exist in the anonymous namespace of the `.cpp` file unless tests need them.

### Error handling

`TESSERACT_CHECK(cond, fmt_string, args...)` for invariants.
`TESSERACT_THROW(fmt_string, args...)` for unconditional failure.
Never use `assert()` in framework code; never silently log.

## Consequences

- Consistent public surface across `Tensor`, `nn`, `ops`, `optim`.
- Mechanical grep-ability: `TESSERACT_` always refers to a macro or a CMake
  option, `tesseract::ops::` always refers to a kernel.
- Python bindings (M4) can expose `Tensor::is_contiguous()` directly as
  `tensor.is_contiguous()` without a translation layer.

### Negative

- Hybrid style is harder to enforce than a single convention; we rely on
  `clang-format` and code review rather than a formal linter rule.

## References

- LLVM coding standards: <https://llvm.org/docs/CodingStandards.html>
- Google C++ style: <https://google.github.io/styleguide/cppguide.html>
- PyTorch C++ style (informal): the PyTorch ATen headers.

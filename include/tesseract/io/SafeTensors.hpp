#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"

// Hugging Face `.safetensors` reader.
//
// File layout (spec: huggingface/safetensors):
//
//   [0 .. 8)              : uint64 little-endian `N` = JSON header length in bytes
//   [8 .. 8 + N)          : UTF-8 JSON header
//   [8 + N .. file_end)   : raw tensor bytes, concatenated
//
// The JSON header is an object whose keys are tensor names and whose values
// describe the tensor:
//
//   {
//     "__metadata__": { "format": "pt", ... },            // optional, string→string
//     "tensor_name": {
//       "dtype": "F32" | "F16" | "BF16" | "F64" | "I32" | "I64" | "I8" | "BOOL",
//       "shape": [int, ...],                              // empty for scalar
//       "data_offsets": [begin, end]                      // bytes, relative to start of data region
//     },
//     ...
//   }
//
// This loader reads the header, validates every tensor's bounds, and exposes
// a typed `load(name)` that materializes a fresh `Tensor` on the requested
// device (CPU by default; CUDA forces a host→device copy). The raw file bytes
// are held via `mmap` for the lifetime of the `SafeTensors` handle, so
// `load()` is a single `memcpy` on CPU and a single `cudaMemcpy` on CUDA —
// no per-tensor disk read.
//
// Thread safety: `SafeTensors` is read-only after construction; `load()` is
// safe to call concurrently from multiple threads against the same handle
// (each call allocates its own destination Tensor).

namespace tesseract::io {

class SafeTensors {
 public:
  // Descriptor for one tensor in the file. `byte_offset` is absolute within
  // the mmap'd region (i.e. includes the `8 + N` header prefix).
  struct View {
    DType dtype{DType::Float32};
    Shape shape;
    std::size_t byte_offset{0};
    std::size_t nbytes{0};
  };

  // Parse the file at `path`. Throws on malformed header / out-of-bounds
  // offsets / unsupported dtypes.
  static SafeTensors open(const std::string& path);

  SafeTensors();
  ~SafeTensors();
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;
  SafeTensors(SafeTensors&&) noexcept;
  SafeTensors& operator=(SafeTensors&&) noexcept;

  // Tensor names in the order they appear in the JSON header. `__metadata__`
  // is stripped out and surfaced via `metadata()` instead.
  const std::vector<std::string>& keys() const noexcept { return keys_; }
  bool contains(std::string_view name) const noexcept;

  // Descriptor lookup; throws if `name` is absent.
  const View& view(std::string_view name) const;

  // Raw byte pointer for a tensor — valid for the lifetime of *this*.
  // Use this when you want to build a zero-copy `Tensor::from_blob`
  // view and explicitly manage the SafeTensors lifetime yourself.
  const std::byte* raw(std::string_view name) const;

  // Materialize a fresh Tensor on `device`. Always allocates new storage
  // and copies from the mmap'd region; for CUDA destinations this is a
  // single `cudaMemcpy` per call.
  Tensor load(std::string_view name, Device device = cpu_device()) const;

  // Whole-file metadata pulled from the JSON `__metadata__` key (string→string).
  const std::unordered_map<std::string, std::string>& metadata() const noexcept {
    return metadata_;
  }

  // Exposed mainly for tests / diagnostics.
  std::size_t header_size() const noexcept { return header_size_; }
  std::size_t file_size() const noexcept { return mapped_size_; }

 private:
  void open_impl(const std::string& path);
  void close_impl() noexcept;

  // mmap state
  int fd_{-1};
  const std::byte* mapped_{nullptr};
  std::size_t mapped_size_{0};
  std::size_t header_size_{0};  // length of JSON header in bytes (not including the 8-byte length prefix)

  // Parsed descriptors
  std::unordered_map<std::string, View> views_;
  std::vector<std::string> keys_;
  std::unordered_map<std::string, std::string> metadata_;
};

// Convert a safetensors dtype string (e.g. "F32", "BF16") to the framework's
// `DType`. Throws `NotImplementedError` for dtype strings we don't yet map.
DType safetensors_dtype_from_string(std::string_view s);

// Inverse of `safetensors_dtype_from_string`. Returns the canonical HF name
// (e.g. "F32"). Throws for dtypes not representable in safetensors.
std::string_view safetensors_dtype_to_string(DType dt);

}  // namespace tesseract::io

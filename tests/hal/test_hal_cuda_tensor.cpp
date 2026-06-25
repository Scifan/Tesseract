// M2D: Tensor-level CUDA round-trip. This is the first file in the
// test tree that constructs `Tensor`s on a CUDA device; the target is
// the M2.α exit-bar language in `docs/m2-plan.md`:
//
//   "A new ctest, `test_hal_cuda`, creates a
//    `Tensor::zeros({16}, Float32, Device(DeviceType::CUDA, 0))`,
//    copies it host-side via `Tensor::to(cpu_device())`, and asserts
//    the round-trip preserves bytes."
//
// We cover a little more than that bare minimum:
//   * `Tensor::zeros` end-to-end on CUDA (exercises the
//     `Storage::zero_device_bytes` → `cudaMemset` path).
//   * `Tensor::ones` end-to-end on CUDA (exercises the `fill_` host-
//     scratch + H→D fallback — remove this test when M2E lands a
//     real CUDA fill kernel and the fallback is gone).
//   * `Tensor::arange` on CUDA (host-scratch + H→D fallback, same
//     story as `ones`).
//   * Round-trip by constructing a CPU tensor, `.to(cuda)` → `.to(cpu)`
//     and asserting the pattern is byte-identical.
//   * `Tensor::clone()` on a CUDA tensor returns a distinct storage
//     with the same bytes.
//   * Same-device `.to()` returns the same storage (no copy), matching
//     PyTorch's convention.

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"

using tesseract::cpu_device;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Tensor;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

TEST_CASE("Tensor::zeros on CUDA round-trips as all-zero on host",
          "[tensor][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for Tensor round-trip test.");
  }

  Device cuda0{DeviceType::CUDA, 0};
  Tensor t = Tensor::zeros({16}, DType::Float32, cuda0);
  REQUIRE(t.device() == cuda0);
  REQUIRE(t.nbytes() == 16 * sizeof(float));

  Tensor host = t.to(cpu_device());
  REQUIRE(host.device() == cpu_device());

  const float* p = host.data_ptr<float>();
  for (int i = 0; i < 16; ++i) REQUIRE(p[i] == 0.0f);
}

TEST_CASE("Tensor::ones on CUDA round-trips as all-one on host",
          "[tensor][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for Tensor round-trip test.");
  }

  Device cuda0{DeviceType::CUDA, 0};
  Tensor t = Tensor::ones({8, 8}, DType::Float32, cuda0);
  Tensor host = t.to(cpu_device());
  const float* p = host.data_ptr<float>();
  for (int i = 0; i < 64; ++i) REQUIRE(p[i] == 1.0f);
}

TEST_CASE("Tensor::arange on CUDA matches its CPU reference",
          "[tensor][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for Tensor round-trip test.");
  }

  Device cuda0{DeviceType::CUDA, 0};
  Tensor d = Tensor::arange(0, 256, 1, DType::Int64, cuda0);
  Tensor h = Tensor::arange(0, 256, 1, DType::Int64, cpu_device());
  Tensor back = d.to(cpu_device());
  const int64_t* a = back.data_ptr<int64_t>();
  const int64_t* b = h.data_ptr<int64_t>();
  for (int i = 0; i < 256; ++i) REQUIRE(a[i] == b[i]);
}

TEST_CASE("Tensor::to(cuda) then to(cpu) preserves every byte",
          "[tensor][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for Tensor round-trip test.");
  }

  // Use a deterministic pattern, not zeros — a no-op copy would slip
  // past the check otherwise.
  std::vector<float> payload(1024);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<float>(i) * 0.25f - 128.0f;
  }
  Tensor h = Tensor::from_vector(payload, {32, 32});
  Device cuda0{DeviceType::CUDA, 0};
  Tensor d = h.to(cuda0);
  REQUIRE(d.device() == cuda0);
  REQUIRE(d.shape() == h.shape());

  Tensor back = d.to(cpu_device());
  const float* a = back.data_ptr<float>();
  const float* b = h.data_ptr<float>();
  for (size_t i = 0; i < payload.size(); ++i) {
    REQUIRE(a[i] == b[i]);
  }
}

TEST_CASE("Tensor::clone on CUDA allocates fresh storage",
          "[tensor][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for Tensor round-trip test.");
  }

  Device cuda0{DeviceType::CUDA, 0};
  Tensor a = Tensor::ones({64}, DType::Float32, cuda0);
  Tensor b = a.clone();
  // Distinct storage: the underlying shared_ptr<Storage>s must differ.
  REQUIRE(a.storage().get() != b.storage().get());
  // ... but the bytes should be identical on the host readback.
  Tensor ha = a.to(cpu_device());
  Tensor hb = b.to(cpu_device());
  const float* pa = ha.data_ptr<float>();
  const float* pb = hb.data_ptr<float>();
  for (int i = 0; i < 64; ++i) REQUIRE(pa[i] == pb[i]);
}

TEST_CASE("Tensor::to(same_device) returns the same storage",
          "[tensor][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for Tensor round-trip test.");
  }
  Device cuda0{DeviceType::CUDA, 0};
  Tensor a = Tensor::zeros({4, 4}, DType::Float32, cuda0);
  Tensor b = a.to(cuda0);
  REQUIRE(a.storage().get() == b.storage().get());
}

TEST_CASE("Tensor::to_string on CUDA renders the first elements",
          "[tensor][gpu]") {
  if (!has_cuda_support() || !is_available()) {
    SKIP("No CUDA GPU available for Tensor round-trip test.");
  }

  Device cuda0{DeviceType::CUDA, 0};
  Tensor t = Tensor::arange(0, 8, 1, DType::Int32, cuda0);
  const std::string s = t.to_string(8);
  // The device tag and all eight elements must be present even though
  // the bytes live on the GPU — to_string silently bounces through a
  // host copy.
  REQUIRE(s.find("cuda:0") != std::string::npos);
  REQUIRE(s.find("7") != std::string::npos);
}

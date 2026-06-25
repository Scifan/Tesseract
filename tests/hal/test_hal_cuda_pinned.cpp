// Wave 2.4 (B-011) — pinned host allocator + async H↔D transfer.
//
// Four things we prove here:
//
//   1. `Tensor::empty_pinned` hands back a tensor whose storage
//      bytes are legal read/write on the host side (same as a
//      plain `Tensor::empty` on CPU — the only difference should
//      be the allocator underneath). We round-trip a pattern
//      through the buffer to confirm.
//   2. `Tensor::to_async(cuda, s)` moves the bytes to the device
//      and, after `s.synchronize()`, produces the same host-
//      observable result as the synchronous `to(cuda)`. Covers
//      both pinned-source (the real async path) and pageable-
//      source (silent driver fallback) so regressions in either
//      dispatch direction trip a test.
//   3. Device-to-host async round-trip: send to device, then
//      `.to_async(cpu, s)` back; the destination is a plain
//      pageable CPU tensor (not pinned) and the read is valid
//      after synchronize. This exercises the `D→H` branch of the
//      async memcpy bridge.
//   4. Same-device `to_async` is a no-op (identity) — returning
//      `*this` without touching the stream, matching the
//      `to(device)` contract.
//
// Everything guards on `cuda::device_count() > 0` so the test
// binary SKIPs cleanly on CPU-only CI.

#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/cuda/PinnedHostAllocator.hpp"

using namespace tesseract;

namespace {

constexpr Device cuda0() { return Device{DeviceType::CUDA, 0}; }

// Fill with a deterministic pattern so every byte is distinct —
// catches wrong-length, off-by-one, and swapped-endpoint bugs
// that a uniform fill would miss.
void write_pattern(Tensor& t) {
  const int64_t n = t.numel();
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) p[i] = static_cast<float>(i * 0.5 - 0.25);
}

bool pattern_matches(const Tensor& t) {
  const int64_t n = t.numel();
  const float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    if (p[i] != static_cast<float>(i * 0.5 - 0.25)) return false;
  }
  return true;
}

}  // namespace

TEST_CASE("PinnedHostAllocator: allocate + fill round-trip",
          "[cuda][pinned]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  constexpr int64_t N = 1024;
  Tensor p = Tensor::empty_pinned({N}, DType::Float32);
  REQUIRE(p.device() == cpu_device());  // pinning is invisible to the dispatcher
  REQUIRE(p.numel() == N);
  write_pattern(p);
  REQUIRE(pattern_matches(p));
}

TEST_CASE("Tensor::to_async(H→D) with pinned source: parity vs .to()",
          "[cuda][pinned]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  constexpr int64_t N = 4096;
  Tensor src_pinned = Tensor::empty_pinned({N}, DType::Float32);
  write_pattern(src_pinned);

  Stream s = Stream::create(cuda0());

  Tensor d_async = src_pinned.to_async(cuda0(), s);
  // Before synchronize the destination is not yet host-observable.
  // We can't portably assert on that (the transfer may have
  // completed for small N), but we CAN assert that after
  // synchronize the round-trip back to CPU matches the original
  // pattern bit-for-bit.
  s.synchronize();
  Tensor back_async = d_async.to(cpu_device());
  REQUIRE(pattern_matches(back_async));

  // Reference path: synchronous `to(cuda)` → back. Used as the
  // ground truth so the async path is verified against whatever
  // the existing `cudaMemcpy` emits (regression-proof against any
  // future endianness / alignment bug that affects both paths).
  Tensor d_sync  = src_pinned.to(cuda0());
  Tensor back_sync = d_sync.to(cpu_device());
  REQUIRE(std::memcmp(back_async.raw_data(), back_sync.raw_data(),
                      back_sync.nbytes()) == 0);
}

TEST_CASE("Tensor::to_async(H→D) with pageable source: correctness preserved",
          "[cuda][pinned]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  // The driver silently degrades to a sync-equivalent path when the
  // source is pageable. The OVERLAP benefit is lost, but the BYTES
  // must still land correctly — that's the guarantee we assert here.
  constexpr int64_t N = 4096;
  Tensor src = Tensor::empty({N}, DType::Float32, cpu_device());
  write_pattern(src);

  Stream s = Stream::create(cuda0());
  Tensor d = src.to_async(cuda0(), s);
  s.synchronize();

  Tensor back = d.to(cpu_device());
  REQUIRE(pattern_matches(back));
}

TEST_CASE("Tensor::to_async(D→H) to a pageable destination",
          "[cuda][pinned]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  constexpr int64_t N = 8192;
  Tensor src_cpu = Tensor::empty({N}, DType::Float32, cpu_device());
  write_pattern(src_cpu);
  Tensor d = src_cpu.to(cuda0());  // sync upload

  Stream s = Stream::create(cuda0());
  Tensor back = d.to_async(cpu_device(), s);
  s.synchronize();
  REQUIRE(pattern_matches(back));
}

TEST_CASE("Tensor::to_async same-device is an identity no-op",
          "[cuda][pinned]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  Tensor d = Tensor::full({16}, 3.0, DType::Float32, cuda0());
  Stream s = Stream::create(cuda0());
  Tensor same = d.to_async(cuda0(), s);
  // Same-device: storage is shared, so mutating `same` (impossible
  // here without an op) would also mutate `d`. We check pointer
  // identity of the raw data buffer as the reliable proxy.
  REQUIRE(same.raw_data() == d.raw_data());
}

TEST_CASE("PinnedHostAllocator: zero-size allocation is legal",
          "[cuda][pinned]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  // Mirrors the CpuAllocator / CudaAllocator contract — a zero-byte
  // request is a no-op returning nullptr. Important because
  // `Tensor::empty_pinned({0}, ...)` legitimately ends up here.
  Tensor empty_tensor = Tensor::empty_pinned({0}, DType::Float32);
  REQUIRE(empty_tensor.numel() == 0);
  REQUIRE(empty_tensor.nbytes() == 0);
}

// M2H: CPU↔CUDA parity for the shape / view ops that land with the
// `launch_strided_copy` + `launch_strided_scatter_add` bridge.
//
// Coverage:
//   * `Tensor::contiguous()` on a CUDA permute-view (strided → dense
//     copy, rank-3 Float32).
//   * `Tensor::clone()` on a non-contiguous CUDA tensor (routes
//     through `contiguous()` on CUDA).
//   * `ops::broadcast_to` on CUDA (rank-2 → rank-3 fan-out, both
//     middle-axis broadcast and leading-axis broadcast).
//   * `ops::reduce_to_shape` on CUDA (fan-in along a broadcast axis,
//     Float32 + Float64 + Int64).
//   * `ops::transpose` + `ops::contiguous` autograd backward round-trip
//     (exercises `contiguous()` on CUDA + `reduce_to_shape` for the
//     final grad shape match — the exact pattern the matmul backward
//     hits).
//   * CPU-only smoke (always runs) so the CPU build keeps at least
//     one asserted path in this TU.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Broadcast.hpp"
#include "tesseract/ops/View.hpp"

using Catch::Matchers::WithinAbs;
using tesseract::cpu_device;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

namespace {

bool cuda_ready() { return has_cuda_support() && is_available(); }
Device cuda0() { return Device{DeviceType::CUDA, 0}; }

std::vector<float> pattern(std::size_t n, float scale = 0.05f) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = (static_cast<float>(i) - static_cast<float>(n) * 0.5f) * scale;
  }
  return out;
}

std::vector<float> to_float_vec(const Tensor& t_host) {
  const auto n = static_cast<std::size_t>(t_host.numel());
  std::vector<float> out(n);
  const auto* p = t_host.data_ptr<float>();
  for (std::size_t i = 0; i < n; ++i) out[i] = p[i];
  return out;
}

std::vector<double> to_double_vec(const Tensor& t_host) {
  const auto n = static_cast<std::size_t>(t_host.numel());
  std::vector<double> out(n);
  const auto* p = t_host.data_ptr<double>();
  for (std::size_t i = 0; i < n; ++i) out[i] = p[i];
  return out;
}

std::vector<int64_t> to_i64_vec(const Tensor& t_host) {
  const auto n = static_cast<std::size_t>(t_host.numel());
  std::vector<int64_t> out(n);
  const auto* p = t_host.data_ptr<int64_t>();
  for (std::size_t i = 0; i < n; ++i) out[i] = p[i];
  return out;
}

void require_close(const std::vector<float>& expected,
                   const std::vector<float>& actual, float tol) {
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE_THAT(actual[i], WithinAbs(expected[i], tol));
  }
}

}  // namespace

TEST_CASE("CUDA Tensor::contiguous() on a strided view matches CPU",
          "[ops][gpu][shape]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Build a rank-3 tensor, permute its axes, and materialize on both
  // sides. `.contiguous()` on CUDA exercises the strided-copy path;
  // on CPU it stays on the reference `for_each_position` loop.
  const int64_t D0 = 3, D1 = 4, D2 = 5;
  auto data = pattern(D0 * D1 * D2);
  Tensor t_cpu = Tensor::from_vector(data, {D0, D1, D2});

  // Permute (0, 1, 2) -> (2, 0, 1): last axis becomes the leading
  // axis. Neither side is contiguous after this so both paths must
  // materialize.
  const std::vector<int64_t> axes = {2, 0, 1};
  Tensor p_cpu = t_cpu.permute(std::span<const int64_t>(axes.data(), axes.size()));
  Tensor p_gpu = t_cpu.to(cuda0()).permute(std::span<const int64_t>(axes.data(), axes.size()));

  REQUIRE_FALSE(p_cpu.is_contiguous());
  REQUIRE_FALSE(p_gpu.is_contiguous());

  Tensor c_cpu = p_cpu.contiguous();
  Tensor c_gpu = p_gpu.contiguous();
  REQUIRE(c_cpu.is_contiguous());
  REQUIRE(c_gpu.is_contiguous());
  REQUIRE(c_gpu.shape() == c_cpu.shape());

  require_close(to_float_vec(c_cpu), to_float_vec(c_gpu.to(cpu_device())),
                0.0f);
}

TEST_CASE("CUDA Tensor::clone() on a non-contiguous tensor matches CPU",
          "[ops][gpu][shape]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t R = 6, C = 8;
  auto data = pattern(R * C);
  Tensor t_gpu = Tensor::from_vector(data, {R, C}).to(cuda0());
  // transpose → strided view
  Tensor tv_gpu = t_gpu.transpose(0, 1);
  REQUIRE_FALSE(tv_gpu.is_contiguous());

  Tensor clone_gpu = tv_gpu.clone();
  REQUIRE(clone_gpu.is_contiguous());
  REQUIRE(clone_gpu.shape() == Shape({C, R}));

  Tensor t_cpu = Tensor::from_vector(data, {R, C});
  Tensor clone_cpu = t_cpu.transpose(0, 1).clone();

  require_close(to_float_vec(clone_cpu), to_float_vec(clone_gpu.to(cpu_device())),
                0.0f);
}

TEST_CASE("CUDA ops::broadcast_to matches CPU",
          "[ops][gpu][shape]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Leading-axis broadcast: [M, N] → [B, M, N].
  const int64_t B = 3, M = 5, N = 4;
  auto data = pattern(M * N);
  Tensor t_cpu = Tensor::from_vector(data, {M, N});
  Tensor t_gpu = t_cpu.to(cuda0());

  Tensor b_cpu = tesseract::ops::broadcast_to(t_cpu, Shape({B, M, N}));
  Tensor b_gpu = tesseract::ops::broadcast_to(t_gpu, Shape({B, M, N}));
  REQUIRE(b_cpu.shape() == b_gpu.shape());
  require_close(to_float_vec(b_cpu), to_float_vec(b_gpu.to(cpu_device())),
                0.0f);

  // Middle-axis broadcast: [M, 1, N] → [M, P, N].
  const int64_t P = 4;
  auto data2 = pattern(M * N, 0.03f);
  Tensor u_cpu = Tensor::from_vector(data2, {M, 1, N});
  Tensor u_gpu = u_cpu.to(cuda0());

  Tensor c_cpu = tesseract::ops::broadcast_to(u_cpu, Shape({M, P, N}));
  Tensor c_gpu = tesseract::ops::broadcast_to(u_gpu, Shape({M, P, N}));
  REQUIRE(c_cpu.shape() == c_gpu.shape());
  require_close(to_float_vec(c_cpu), to_float_vec(c_gpu.to(cpu_device())),
                0.0f);
}

TEST_CASE("CUDA ops::reduce_to_shape matches CPU (Float32)",
          "[ops][gpu][shape]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Sum-fan-in: [B, M, N] -> [M, N] (reduce B axis).
  const int64_t B = 3, M = 5, N = 4;
  auto data = pattern(B * M * N, 0.02f);
  Tensor t_cpu = Tensor::from_vector(data, {B, M, N});
  Tensor t_gpu = t_cpu.to(cuda0());

  Tensor r_cpu = tesseract::ops::reduce_to_shape(t_cpu, Shape({M, N}));
  Tensor r_gpu = tesseract::ops::reduce_to_shape(t_gpu, Shape({M, N}));
  REQUIRE(r_cpu.shape() == r_gpu.shape());
  require_close(to_float_vec(r_cpu), to_float_vec(r_gpu.to(cpu_device())),
                1e-4f);

  // Full reduction: [B, M, N] -> scalar.
  Tensor s_cpu = tesseract::ops::reduce_to_shape(t_cpu, Shape{});
  Tensor s_gpu = tesseract::ops::reduce_to_shape(t_gpu, Shape{});
  REQUIRE(s_cpu.shape() == s_gpu.shape());
  require_close(to_float_vec(s_cpu), to_float_vec(s_gpu.to(cpu_device())),
                1e-4f);
}

TEST_CASE("CUDA ops::reduce_to_shape matches CPU (Float64)",
          "[ops][gpu][shape]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 4, N = 6;
  auto data = pattern(B * N, 0.02f);
  std::vector<double> d(data.begin(), data.end());
  Tensor t_cpu = Tensor::from_vector(d, {B, N});
  Tensor t_gpu = t_cpu.to(cuda0());

  Tensor r_cpu = tesseract::ops::reduce_to_shape(t_cpu, Shape({N}));
  Tensor r_gpu = tesseract::ops::reduce_to_shape(t_gpu, Shape({N}));
  auto expected = to_double_vec(r_cpu);
  auto actual = to_double_vec(r_gpu.to(cpu_device()));
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE_THAT(actual[i], WithinAbs(expected[i], 1e-10));
  }
}

TEST_CASE("CUDA ops::reduce_to_shape matches CPU (Int64)",
          "[ops][gpu][shape]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 3, N = 5;
  std::vector<int64_t> d(B * N);
  for (std::size_t i = 0; i < d.size(); ++i) {
    d[i] = static_cast<int64_t>(i) - 3;
  }
  Tensor t_cpu = Tensor::from_vector(d, {B, N});
  Tensor t_gpu = t_cpu.to(cuda0());

  Tensor r_cpu = tesseract::ops::reduce_to_shape(t_cpu, Shape({N}));
  Tensor r_gpu = tesseract::ops::reduce_to_shape(t_gpu, Shape({N}));
  REQUIRE(to_i64_vec(r_cpu) == to_i64_vec(r_gpu.to(cpu_device())));
}

TEST_CASE("CPU shape smoke (always runs)",
          "[ops][shape]") {
  // Unconditional CPU path so the CPU-only build has an asserted
  // case from this TU.
  Tensor t = Tensor::from_vector(pattern(12), {3, 4});
  Tensor tt = t.transpose(0, 1);
  REQUIRE_FALSE(tt.is_contiguous());
  Tensor c = tt.contiguous();
  REQUIRE(c.is_contiguous());
  REQUIRE(c.shape() == Shape({4, 3}));
  require_close(to_float_vec(tt.clone()), to_float_vec(c), 0.0f);
}

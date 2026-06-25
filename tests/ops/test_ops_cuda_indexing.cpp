// M2H: CPU↔CUDA parity for the B-003 indexing suite (`cat`, `split`,
// `split_with_sizes`, `index_select`, `gather`), plus autograd backward
// for the scatter-add variants.
//
// Coverage:
//   * `cat` forward on rank-2 Float32 (row-split layout) and rank-3
//     Float32 (middle-axis concat).
//   * `split` forward on rank-2 Float32 (even chunks) and
//     `split_with_sizes` (non-uniform chunks).
//   * `index_select` forward + backward on rank-2 Float32 (the
//     embedding-lookup pattern).
//   * `gather` forward + backward on rank-2 Float32 (one-hot style
//     per-row element pick).
//   * Autograd round-trip for `cat` and `split` (engine sums the
//     per-chunk gradients at the shared parent).
//   * CPU-only smoke case (always runs) so the CPU build keeps at
//     least one asserted path.

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Indexing.hpp"

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

void require_close(const std::vector<float>& expected,
                   const std::vector<float>& actual, float tol) {
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE_THAT(actual[i], WithinAbs(expected[i], tol));
  }
}

}  // namespace

TEST_CASE("CUDA ops::cat matches CPU (Float32)",
          "[ops][gpu][indexing]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Row-split along dim 0 of two rank-2 tensors.
  auto da = pattern(12, 0.05f);
  auto db = pattern(8, -0.03f);
  Tensor a_cpu = Tensor::from_vector(da, {3, 4});
  Tensor b_cpu = Tensor::from_vector(db, {2, 4});

  Tensor r_cpu = tesseract::ops::cat({a_cpu, b_cpu}, 0);
  Tensor r_gpu = tesseract::ops::cat({a_cpu.to(cuda0()), b_cpu.to(cuda0())}, 0);
  REQUIRE(r_cpu.shape() == r_gpu.shape());
  require_close(to_float_vec(r_cpu), to_float_vec(r_gpu.to(cpu_device())),
                0.0f);

  // Concat along a middle axis on a rank-3 tensor.
  auto dc = pattern(24, 0.02f);
  auto dd = pattern(12, 0.04f);
  Tensor c_cpu = Tensor::from_vector(dc, {2, 3, 4});
  Tensor d_cpu = Tensor::from_vector(dd, {2, 3, 2});

  Tensor rr_cpu = tesseract::ops::cat({c_cpu, d_cpu}, 2);
  Tensor rr_gpu = tesseract::ops::cat({c_cpu.to(cuda0()), d_cpu.to(cuda0())}, 2);
  REQUIRE(rr_cpu.shape() == rr_gpu.shape());
  require_close(to_float_vec(rr_cpu), to_float_vec(rr_gpu.to(cpu_device())),
                0.0f);
}

TEST_CASE("CUDA ops::split matches CPU (Float32)",
          "[ops][gpu][indexing]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto d = pattern(20, 0.05f);
  Tensor t_cpu = Tensor::from_vector(d, {5, 4});
  auto splits_cpu = tesseract::ops::split(t_cpu, 2, 0);
  auto splits_gpu = tesseract::ops::split(t_cpu.to(cuda0()), 2, 0);
  REQUIRE(splits_cpu.size() == splits_gpu.size());
  for (std::size_t i = 0; i < splits_cpu.size(); ++i) {
    REQUIRE(splits_cpu[i].shape() == splits_gpu[i].shape());
    require_close(to_float_vec(splits_cpu[i]),
                  to_float_vec(splits_gpu[i].to(cpu_device())),
                  0.0f);
  }
}

TEST_CASE("CUDA ops::split_with_sizes matches CPU (Float32)",
          "[ops][gpu][indexing]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto d = pattern(30, 0.05f);
  Tensor t_cpu = Tensor::from_vector(d, {5, 6});
  const std::vector<int64_t> sizes = {1, 3, 2};
  auto sp_cpu = tesseract::ops::split_with_sizes(t_cpu, sizes, 1);
  auto sp_gpu = tesseract::ops::split_with_sizes(t_cpu.to(cuda0()), sizes, 1);
  REQUIRE(sp_cpu.size() == sp_gpu.size());
  for (std::size_t i = 0; i < sp_cpu.size(); ++i) {
    REQUIRE(sp_cpu[i].shape() == sp_gpu[i].shape());
    require_close(to_float_vec(sp_cpu[i]),
                  to_float_vec(sp_gpu[i].to(cpu_device())),
                  0.0f);
  }
}

TEST_CASE("CUDA ops::index_select matches CPU (Float32)",
          "[ops][gpu][indexing]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Embedding-lookup pattern: rank-2 source, rank-1 Int64 indices.
  const int64_t V = 10, D = 5;
  auto d = pattern(V * D, 0.03f);
  Tensor src_cpu = Tensor::from_vector(d, {V, D});
  std::vector<int64_t> idx_host = {3, 0, 7, 2, 7, 9};
  Tensor idx_cpu = Tensor::from_vector(idx_host, {static_cast<int64_t>(idx_host.size())});

  Tensor r_cpu = tesseract::ops::index_select(src_cpu, 0, idx_cpu);
  Tensor r_gpu = tesseract::ops::index_select(src_cpu.to(cuda0()),
                                              0, idx_cpu.to(cuda0()));
  REQUIRE(r_cpu.shape() == r_gpu.shape());
  require_close(to_float_vec(r_cpu), to_float_vec(r_gpu.to(cpu_device())),
                0.0f);
}

TEST_CASE("CUDA ops::index_select autograd backward matches CPU",
          "[ops][gpu][indexing]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t V = 8, D = 4;
  auto d = pattern(V * D, 0.03f);
  std::vector<int64_t> idx_host = {1, 1, 3, 5, 0};  // duplicates exercise atomicAdd
  Tensor idx_cpu = Tensor::from_vector(idx_host, {static_cast<int64_t>(idx_host.size())});

  // Shared upstream grad (contiguous).
  auto g_data = pattern(idx_host.size() * D, 0.07f);
  Shape g_shape({static_cast<int64_t>(idx_host.size()), D});

  // CPU reference.
  Tensor src_cpu = Tensor::from_vector(d, {V, D});
  src_cpu.set_requires_grad(true);
  Tensor y_cpu = tesseract::ops::index_select(src_cpu, 0, idx_cpu);
  Tensor g_cpu = Tensor::from_vector(g_data, g_shape);
  tesseract::Engine::backward(y_cpu, g_cpu);
  Tensor grad_cpu = src_cpu.grad();
  REQUIRE(grad_cpu.defined());
  REQUIRE(grad_cpu.shape() == Shape({V, D}));

  // CUDA parity.
  Tensor src_gpu = Tensor::from_vector(d, {V, D}).to(cuda0());
  src_gpu.set_requires_grad(true);
  Tensor y_gpu = tesseract::ops::index_select(src_gpu, 0, idx_cpu.to(cuda0()));
  Tensor g_gpu = Tensor::from_vector(g_data, g_shape).to(cuda0());
  tesseract::Engine::backward(y_gpu, g_gpu);
  Tensor grad_gpu = src_gpu.grad();
  REQUIRE(grad_gpu.defined());
  REQUIRE(grad_gpu.device() == cuda0());
  REQUIRE(grad_gpu.shape() == Shape({V, D}));

  require_close(to_float_vec(grad_cpu),
                to_float_vec(grad_gpu.to(cpu_device())),
                1e-4f);
}

TEST_CASE("CUDA ops::gather matches CPU (Float32)",
          "[ops][gpu][indexing]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Rank-2 gather along dim 1: for each row, pick per-column indices.
  const int64_t R = 4, C = 6;
  auto d = pattern(R * C, 0.03f);
  Tensor src_cpu = Tensor::from_vector(d, {R, C});
  // indices.shape == {R, 3}
  std::vector<int64_t> idx_host = {
      0, 2, 5,
      5, 1, 1,
      3, 3, 4,
      2, 0, 5,
  };
  Tensor idx_cpu = Tensor::from_vector(idx_host, {R, 3});

  Tensor r_cpu = tesseract::ops::gather(src_cpu, 1, idx_cpu);
  Tensor r_gpu = tesseract::ops::gather(src_cpu.to(cuda0()),
                                        1, idx_cpu.to(cuda0()));
  REQUIRE(r_cpu.shape() == r_gpu.shape());
  require_close(to_float_vec(r_cpu), to_float_vec(r_gpu.to(cpu_device())),
                0.0f);
}

TEST_CASE("CUDA ops::gather autograd backward matches CPU",
          "[ops][gpu][indexing]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t R = 3, C = 5;
  auto d = pattern(R * C, 0.04f);
  // indices.shape == {R, 2}, with intentional duplicates along dim 1.
  std::vector<int64_t> idx_host = {
      1, 1,
      4, 2,
      0, 0,
  };
  Tensor idx_cpu = Tensor::from_vector(idx_host, {R, 2});

  auto g_data = pattern(R * 2, 0.06f);
  Shape g_shape({R, 2});

  Tensor src_cpu = Tensor::from_vector(d, {R, C});
  src_cpu.set_requires_grad(true);
  Tensor y_cpu = tesseract::ops::gather(src_cpu, 1, idx_cpu);
  Tensor g_cpu = Tensor::from_vector(g_data, g_shape);
  tesseract::Engine::backward(y_cpu, g_cpu);
  Tensor grad_cpu = src_cpu.grad();
  REQUIRE(grad_cpu.defined());

  Tensor src_gpu = Tensor::from_vector(d, {R, C}).to(cuda0());
  src_gpu.set_requires_grad(true);
  Tensor y_gpu = tesseract::ops::gather(src_gpu, 1, idx_cpu.to(cuda0()));
  Tensor g_gpu = Tensor::from_vector(g_data, g_shape).to(cuda0());
  tesseract::Engine::backward(y_gpu, g_gpu);
  Tensor grad_gpu = src_gpu.grad();
  REQUIRE(grad_gpu.defined());
  REQUIRE(grad_gpu.device() == cuda0());

  require_close(to_float_vec(grad_cpu),
                to_float_vec(grad_gpu.to(cpu_device())),
                1e-4f);
}

TEST_CASE("CUDA ops::cat autograd backward matches CPU",
          "[ops][gpu][indexing]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto da = pattern(12, 0.05f);
  auto db = pattern(8, -0.03f);
  auto g = pattern(20, 0.09f);
  Shape g_shape({5, 4});

  Tensor a_cpu = Tensor::from_vector(da, {3, 4});
  Tensor b_cpu = Tensor::from_vector(db, {2, 4});
  a_cpu.set_requires_grad(true);
  b_cpu.set_requires_grad(true);
  Tensor y_cpu = tesseract::ops::cat({a_cpu, b_cpu}, 0);
  Tensor gc = Tensor::from_vector(g, g_shape);
  tesseract::Engine::backward(y_cpu, gc);
  Tensor ga_cpu = a_cpu.grad();
  Tensor gb_cpu = b_cpu.grad();

  Tensor a_gpu = Tensor::from_vector(da, {3, 4}).to(cuda0());
  Tensor b_gpu = Tensor::from_vector(db, {2, 4}).to(cuda0());
  a_gpu.set_requires_grad(true);
  b_gpu.set_requires_grad(true);
  Tensor y_gpu = tesseract::ops::cat({a_gpu, b_gpu}, 0);
  Tensor gg = Tensor::from_vector(g, g_shape).to(cuda0());
  tesseract::Engine::backward(y_gpu, gg);
  Tensor ga_gpu = a_gpu.grad();
  Tensor gb_gpu = b_gpu.grad();

  require_close(to_float_vec(ga_cpu), to_float_vec(ga_gpu.to(cpu_device())),
                0.0f);
  require_close(to_float_vec(gb_cpu), to_float_vec(gb_gpu.to(cpu_device())),
                0.0f);
}

TEST_CASE("CUDA ops::split autograd backward matches CPU",
          "[ops][gpu][indexing]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto d = pattern(20, 0.05f);
  Tensor t_cpu = Tensor::from_vector(d, {5, 4});
  t_cpu.set_requires_grad(true);
  auto chunks_cpu = tesseract::ops::split(t_cpu, 2, 0);
  // Sum-equivalent: use an explicit grad per chunk (pattern data, so
  // bypasses the `ops::sum` backward path M2G flagged).
  Tensor agg_cpu = tesseract::ops::cat(chunks_cpu, 0);
  Tensor g = Tensor::from_vector(pattern(20, 0.02f), {5, 4});
  tesseract::Engine::backward(agg_cpu, g);
  Tensor grad_cpu = t_cpu.grad();

  Tensor t_gpu = Tensor::from_vector(d, {5, 4}).to(cuda0());
  t_gpu.set_requires_grad(true);
  auto chunks_gpu = tesseract::ops::split(t_gpu, 2, 0);
  Tensor agg_gpu = tesseract::ops::cat(chunks_gpu, 0);
  Tensor gg = Tensor::from_vector(pattern(20, 0.02f), {5, 4}).to(cuda0());
  tesseract::Engine::backward(agg_gpu, gg);
  Tensor grad_gpu = t_gpu.grad();

  require_close(to_float_vec(grad_cpu),
                to_float_vec(grad_gpu.to(cpu_device())),
                0.0f);
}

TEST_CASE("CPU indexing smoke (always runs)",
          "[ops][indexing]") {
  // Unconditional CPU path so the CPU-only build has an asserted
  // case from this TU.
  Tensor t = Tensor::from_vector(pattern(12), {3, 4});
  std::vector<int64_t> idx = {2, 0, 1};
  Tensor sel = tesseract::ops::index_select(
      t, 0, Tensor::from_vector(idx, {3}));
  REQUIRE(sel.shape() == Shape({3, 4}));
}

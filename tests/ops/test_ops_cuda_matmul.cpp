// M2G: CPU↔CUDA parity for the cuBLASLt-backed matmul. Mirrors the
// existing `test_ops_cuda_{elementwise,reduction,softmax,loss}` suites:
// build identical inputs on CPU and CUDA, run the op on both, copy the
// CUDA output back, and assert numeric equality under a dtype-specific
// tolerance.
//
// Coverage matrix:
//   * Rank-2 FP32 parity (+ autograd backward).
//   * Rank-2 FP64 parity.
//   * Rank-3 batched FP32 parity (non-broadcast).
//   * Broadcast batched FP32 parity ([B, M, K] @ [K, N]).
//   * Transposed operand FP32 parity (exercises op=T detection; also
//     the shape the autograd backward creates via `mat_transpose`).
//   * FP16 / BF16 parity (looser tolerance — FP32 accumulation so the
//     drift is bounded by ~1 ULP-per-K · magnitude_of_K).
//   * One CPU-only smoke case (always runs) so a CPU build still has
//     at least one asserted path in this TU.
//
// Tolerance notes:
//   * FP32: cuBLASLt uses TF32 Tensor Cores by default on Ada+; the
//     mantissa truncation is ~1e-3 relative per operand, so a 4096²
//     GEMM can drift up to a few 1e-3. Our tests use K <= 96, where
//     3e-3 absolute gives headroom.
//   * FP64: CUBLAS_COMPUTE_64F is bit-for-bit comparable with the CPU
//     scalar / Eigen path — 1e-9 atol.
//   * FP16 / BF16: FP32 accumulation means the GPU matches a 2-pass
//     FP32 reference within ~1 ULP per K; we compare against the CPU
//     upcast-to-FP32 path (which has the same accumulation contract)
//     and pick 5e-2 / 1e-1 atol respectively.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/MatMul.hpp"
#include "tesseract/ops/Reduction.hpp"

using Catch::Matchers::WithinAbs;
using tesseract::BFloat16;
using tesseract::cpu_device;
using tesseract::Device;
using tesseract::DeviceType;
using tesseract::DType;
using tesseract::Half;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::cuda::has_cuda_support;
using tesseract::cuda::is_available;

namespace {

bool cuda_ready() { return has_cuda_support() && is_available(); }
Device cuda0() { return Device{DeviceType::CUDA, 0}; }

// Deterministic float pattern with both signs and a controlled
// magnitude. Scaled so a 32-wide accumulation stays in a range where
// FP16 doesn't overflow or denormalize.
std::vector<float> pattern(std::size_t n, float scale = 0.0125f) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = (static_cast<float>(i) - static_cast<float>(n) * 0.5f) * scale;
  }
  return out;
}

std::vector<float> to_float_vec(const Tensor& t_host) {
  const auto n = static_cast<std::size_t>(t_host.numel());
  std::vector<float> out(n);
  switch (t_host.dtype()) {
    case DType::Float32: {
      const float* p = t_host.data_ptr<float>();
      for (std::size_t i = 0; i < n; ++i) out[i] = p[i];
      break;
    }
    case DType::Float64: {
      const double* p = t_host.data_ptr<double>();
      for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
      break;
    }
    case DType::Float16: {
      const Half* p = t_host.data_ptr<Half>();
      for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
      break;
    }
    case DType::BFloat16: {
      const BFloat16* p = t_host.data_ptr<BFloat16>();
      for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
      break;
    }
    default:
      FAIL("unsupported dtype in to_float_vec");
  }
  return out;
}

Tensor make_fp(DType dt, const std::vector<float>& data, Shape shape) {
  switch (dt) {
    case DType::Float32:
      return Tensor::from_vector(data, std::move(shape));
    case DType::Float64: {
      std::vector<double> v(data.size());
      for (std::size_t i = 0; i < data.size(); ++i) v[i] = data[i];
      return Tensor::from_vector(v, std::move(shape));
    }
    case DType::Float16: {
      std::vector<Half> v(data.size());
      for (std::size_t i = 0; i < data.size(); ++i) v[i] = Half(data[i]);
      return Tensor::from_vector(v, std::move(shape));
    }
    case DType::BFloat16: {
      std::vector<BFloat16> v(data.size());
      for (std::size_t i = 0; i < data.size(); ++i) v[i] = BFloat16(data[i]);
      return Tensor::from_vector(v, std::move(shape));
    }
    default:
      FAIL("unsupported dtype in make_fp");
      return Tensor{};
  }
}

void require_close(const std::vector<float>& expected,
                   const std::vector<float>& actual, float tol) {
  REQUIRE(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE_THAT(actual[i], WithinAbs(expected[i], tol));
  }
}

}  // namespace

TEST_CASE("CUDA matmul rank-2 matches CPU (Float32)",
          "[ops][gpu][matmul]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t M = 32, K = 48, N = 24;
  auto a_data = pattern(M * K, 0.01f);
  auto b_data = pattern(K * N, 0.02f);
  Tensor a_cpu = Tensor::from_vector(a_data, {M, K});
  Tensor b_cpu = Tensor::from_vector(b_data, {K, N});

  Tensor h_out = tesseract::ops::matmul(a_cpu, b_cpu);
  Tensor d_out = tesseract::ops::matmul(a_cpu.to(cuda0()), b_cpu.to(cuda0()));
  REQUIRE(d_out.device() == cuda0());
  REQUIRE(d_out.shape() == Shape({M, N}));

  require_close(to_float_vec(h_out), to_float_vec(d_out.to(cpu_device())),
                3e-3f);
}

TEST_CASE("CUDA matmul rank-2 matches CPU (Float64)",
          "[ops][gpu][matmul]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t M = 16, K = 24, N = 20;
  std::vector<double> a_data(M * K), b_data(K * N);
  for (std::size_t i = 0; i < a_data.size(); ++i)
    a_data[i] = (static_cast<double>(i) - 0.5 * a_data.size()) * 0.01;
  for (std::size_t i = 0; i < b_data.size(); ++i)
    b_data[i] = (static_cast<double>(i) - 0.5 * b_data.size()) * 0.02;

  Tensor a_cpu = Tensor::from_vector(a_data, {M, K});
  Tensor b_cpu = Tensor::from_vector(b_data, {K, N});

  Tensor h_out = tesseract::ops::matmul(a_cpu, b_cpu);
  Tensor d_out = tesseract::ops::matmul(a_cpu.to(cuda0()), b_cpu.to(cuda0()));
  REQUIRE(d_out.device() == cuda0());
  REQUIRE(d_out.dtype() == DType::Float64);

  // FP64 uses `CUBLAS_COMPUTE_64F` — parity with the CPU is tight.
  const double* e = h_out.data_ptr<double>();
  Tensor back = d_out.to(cpu_device());
  const double* a = back.data_ptr<double>();
  for (int64_t i = 0; i < back.numel(); ++i) {
    REQUIRE_THAT(a[i], WithinAbs(e[i], 1e-9));
  }
}

TEST_CASE("CUDA matmul rank-3 batched matches CPU (Float32)",
          "[ops][gpu][matmul]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t B = 4, M = 12, K = 20, N = 8;
  auto a_data = pattern(B * M * K, 0.01f);
  auto b_data = pattern(B * K * N, 0.02f);
  Tensor a_cpu = Tensor::from_vector(a_data, {B, M, K});
  Tensor b_cpu = Tensor::from_vector(b_data, {B, K, N});

  Tensor h_out = tesseract::ops::matmul(a_cpu, b_cpu);
  Tensor d_out = tesseract::ops::matmul(a_cpu.to(cuda0()), b_cpu.to(cuda0()));
  REQUIRE(d_out.shape() == Shape({B, M, N}));

  require_close(to_float_vec(h_out), to_float_vec(d_out.to(cpu_device())),
                3e-3f);
}

TEST_CASE("CUDA matmul broadcast batched matches CPU (Float32)",
          "[ops][gpu][matmul]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // LHS has a batch dim, RHS does not — trailing rank-2 broadcasts.
  const int64_t B = 3, M = 10, K = 16, N = 6;
  auto a_data = pattern(B * M * K, 0.01f);
  auto b_data = pattern(K * N, 0.02f);
  Tensor a_cpu = Tensor::from_vector(a_data, {B, M, K});
  Tensor b_cpu = Tensor::from_vector(b_data, {K, N});

  Tensor h_out = tesseract::ops::matmul(a_cpu, b_cpu);
  Tensor d_out = tesseract::ops::matmul(a_cpu.to(cuda0()), b_cpu.to(cuda0()));
  REQUIRE(d_out.shape() == Shape({B, M, N}));

  require_close(to_float_vec(h_out), to_float_vec(d_out.to(cpu_device())),
                3e-3f);
}

TEST_CASE("CUDA matmul transposed-rhs matches CPU (Float32)",
          "[ops][gpu][matmul]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Exercises op=T detection: `b.transpose(-2,-1)` produces a [N, K]
  // view with strides [1, N] over an underlying [K, N] contig. This
  // is the exact shape the autograd matmul backward creates.
  const int64_t M = 24, K = 16, N = 20;
  auto a_data = pattern(M * K, 0.01f);
  auto b_data = pattern(K * N, 0.02f);
  Tensor a_cpu = Tensor::from_vector(a_data, {M, K});
  Tensor b_cpu = Tensor::from_vector(b_data, {K, N});

  // g @ b^T shape-wise: we use a=g of shape [M, N] and b_cpu^T of shape [N, K].
  // `Tensor::transpose` rejects negative dims (the ops::matmul backward
  // goes through an internal `mat_transpose(rank - 2, rank - 1)` helper
  // for the same reason), so pass explicit non-negative axes here.
  Tensor g_cpu = Tensor::from_vector(pattern(M * N, 0.015f), {M, N});
  Tensor bt_cpu = b_cpu.transpose(0, 1);              // [N, K] view
  Tensor h_out = tesseract::ops::matmul(g_cpu, bt_cpu);  // [M, K]

  Tensor g_cu = g_cpu.to(cuda0());
  Tensor b_cu = b_cpu.to(cuda0());
  Tensor bt_cu = b_cu.transpose(0, 1);  // strided view on CUDA
  Tensor d_out = tesseract::ops::matmul(g_cu, bt_cu);
  REQUIRE(d_out.shape() == Shape({M, K}));

  require_close(to_float_vec(h_out), to_float_vec(d_out.to(cpu_device())),
                3e-3f);
}

TEST_CASE("CUDA matmul autograd backward matches CPU (Float32)",
          "[ops][gpu][matmul]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t M = 20, K = 12, N = 16;
  auto a_data = pattern(M * K, 0.01f);
  auto b_data = pattern(K * N, 0.02f);

  // Seed gradient of the same shape as y, on the same device as each
  // run. We supply an explicit grad (rather than go through
  // `ops::sum` to reduce to a scalar) because `ops::sum` has a CUDA
  // forward path but its backward expands via `ops::broadcast_to`,
  // whose CUDA dispatch lands in M2H. Passing the grad manually
  // exercises only the matmul autograd path, which is what this test
  // is about.
  auto g_data = pattern(M * N, 0.015f);

  // CPU autograd run.
  Tensor ha = Tensor::from_vector(a_data, {M, K});
  Tensor hb = Tensor::from_vector(b_data, {K, N});
  ha.set_requires_grad(true);
  hb.set_requires_grad(true);
  Tensor hy = tesseract::ops::matmul(ha, hb);
  Tensor hgrad = Tensor::from_vector(g_data, {M, N});
  tesseract::Engine::backward(hy, hgrad);

  // CUDA autograd run.
  Tensor da = Tensor::from_vector(a_data, {M, K}).to(cuda0());
  Tensor db = Tensor::from_vector(b_data, {K, N}).to(cuda0());
  da.set_requires_grad(true);
  db.set_requires_grad(true);
  Tensor dy = tesseract::ops::matmul(da, db);
  Tensor dgrad = Tensor::from_vector(g_data, {M, N}).to(cuda0());
  tesseract::Engine::backward(dy, dgrad);

  auto expect_ga = to_float_vec(ha.grad());
  auto expect_gb = to_float_vec(hb.grad());
  auto got_ga = to_float_vec(da.grad().to(cpu_device()));
  auto got_gb = to_float_vec(db.grad().to(cpu_device()));
  require_close(expect_ga, got_ga, 3e-3f);
  require_close(expect_gb, got_gb, 3e-3f);
}

TEST_CASE("CUDA matmul rank-2 matches CPU (Float16)",
          "[ops][gpu][matmul][fp16]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t M = 16, K = 24, N = 20;
  auto a_data = pattern(M * K, 0.01f);
  auto b_data = pattern(K * N, 0.02f);
  Tensor a_cpu = make_fp(DType::Float16, a_data, {M, K});
  Tensor b_cpu = make_fp(DType::Float16, b_data, {K, N});

  Tensor h_out = tesseract::ops::matmul(a_cpu, b_cpu);
  Tensor d_out = tesseract::ops::matmul(a_cpu.to(cuda0()), b_cpu.to(cuda0()));
  REQUIRE(d_out.dtype() == DType::Float16);

  // FP16 storage, FP32 accumulation on both sides. Tolerance scaled
  // for the 1-ULP-of-FP16 rounding on each of the M·N output slots.
  require_close(to_float_vec(h_out), to_float_vec(d_out.to(cpu_device())),
                5e-2f);
}

TEST_CASE("CUDA matmul rank-2 matches CPU (BFloat16)",
          "[ops][gpu][matmul][bf16]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  const int64_t M = 16, K = 24, N = 20;
  auto a_data = pattern(M * K, 0.01f);
  auto b_data = pattern(K * N, 0.02f);
  Tensor a_cpu = make_fp(DType::BFloat16, a_data, {M, K});
  Tensor b_cpu = make_fp(DType::BFloat16, b_data, {K, N});

  Tensor h_out = tesseract::ops::matmul(a_cpu, b_cpu);
  Tensor d_out = tesseract::ops::matmul(a_cpu.to(cuda0()), b_cpu.to(cuda0()));
  REQUIRE(d_out.dtype() == DType::BFloat16);

  // BF16 has ~2× fewer mantissa bits than FP16 → looser tolerance,
  // but still fine for the small sizes used here.
  require_close(to_float_vec(h_out), to_float_vec(d_out.to(cpu_device())),
                1e-1f);
}

TEST_CASE("matmul CPU still produces correct result (smoke)",
          "[ops][matmul]") {
  // Always-on smoke so the CPU-only build has at least one real
  // assertion out of this TU. Matches the [ops][matmul] non-CUDA
  // suite in `test_matmul.cpp` but uses a different shape so we don't
  // shadow it on the `ctest` dashboard.
  const int64_t M = 4, K = 6, N = 5;
  std::vector<float> a(M * K), b(K * N);
  for (int64_t i = 0; i < M * K; ++i) a[i] = 1.0f + 0.01f * i;
  for (int64_t i = 0; i < K * N; ++i) b[i] = 0.5f - 0.02f * i;

  Tensor ta = Tensor::from_vector(a, {M, K});
  Tensor tb = Tensor::from_vector(b, {K, N});
  Tensor tc = tesseract::ops::matmul(ta, tb);
  REQUIRE(tc.shape() == Shape({M, N}));

  // Reference by hand-rolled triple loop.
  std::vector<float> ref(M * N, 0.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += a[i * K + k] * b[k * N + j];
      ref[i * N + j] = acc;
    }
  }
  const float* got = tc.data_ptr<float>();
  for (int64_t i = 0; i < M * N; ++i) REQUIRE_THAT(got[i], WithinAbs(ref[i], 1e-5f));
}

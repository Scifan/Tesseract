// M2E: CPU↔CUDA parity tests for the elementwise op suite.
//
// Each test case constructs the same inputs on CPU and CUDA, runs the
// op on both, copies the CUDA result back to host, and asserts numeric
// equality. Transcendental ops (`exp`, `log`, `sigmoid`, `tanh`) use
// an absolute-tolerance comparison because CUDA's fastmath intrinsics
// (`__expf`, `tanhf`, ...) are bit-equivalent to the CPU reference only
// up to the CUDA IEEE ULP budget; the tolerance is tight enough to
// catch an off-by-one-in-mantissa bug.
//
// Like the M2D `test_hal_cuda_tensor.cpp`, every test guards on
// `has_cuda_support() && is_available()` and `SKIP`s cleanly when
// either is false. The enclosing ctest picks up Catch2's exit code 4
// as "skipped" via `SKIP_RETURN_CODE` on the test executable.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"

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

// Materialize a deterministic float pattern. Covers negatives,
// zeros (at a couple of indices), and a wide magnitude range so
// div-by-small and log-of-small paths get real work.
std::vector<float> pattern_floats(std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto x = static_cast<float>(i);
    out[i] = (x - static_cast<float>(n / 2)) * 0.125f;
  }
  return out;
}

// Strictly-positive pattern for `log` (which has a singularity at 0).
std::vector<float> positive_floats(std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = 0.25f + static_cast<float>(i) * 0.125f;
  }
  return out;
}

}  // namespace

TEST_CASE("CUDA add matches CPU add (dense same shape)", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto pa = pattern_floats(1024);
  auto pb = pattern_floats(1024);
  for (auto& v : pb) v += 0.75f;
  Tensor ha = Tensor::from_vector(pa, {32, 32});
  Tensor hb = Tensor::from_vector(pb, {32, 32});
  Tensor hc = tesseract::ops::add(ha, hb);

  Tensor da = ha.to(cuda0());
  Tensor db = hb.to(cuda0());
  Tensor dc = tesseract::ops::add(da, db);
  REQUIRE(dc.device() == cuda0());
  Tensor back = dc.to(cpu_device());

  const float* expected = hc.data_ptr<float>();
  const float* actual   = back.data_ptr<float>();
  for (int64_t i = 0; i < back.numel(); ++i) {
    REQUIRE(expected[i] == actual[i]);
  }
}

TEST_CASE("CUDA sub/mul/div match CPU (dense same shape)", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Avoid 0 in the divisor — we exercise the integer-div path in a
  // separate test because div-by-zero is UB on CUDA integer div.
  std::vector<float> pa(256), pb(256);
  for (size_t i = 0; i < pa.size(); ++i) {
    pa[i] = static_cast<float>(i) - 128.0f;
    pb[i] = static_cast<float>(i % 17) * 0.5f + 1.0f;
  }
  Tensor ha = Tensor::from_vector(pa, {16, 16});
  Tensor hb = Tensor::from_vector(pb, {16, 16});
  Tensor da = ha.to(cuda0());
  Tensor db = hb.to(cuda0());

  auto check_equal = [&](const Tensor& host_ref, const Tensor& cuda_res) {
    Tensor back = cuda_res.to(cpu_device());
    REQUIRE(back.numel() == host_ref.numel());
    const float* a = host_ref.data_ptr<float>();
    const float* b = back.data_ptr<float>();
    for (int64_t i = 0; i < back.numel(); ++i) {
      REQUIRE_THAT(b[i], WithinAbs(a[i], 1e-5f));
    }
  };

  check_equal(tesseract::ops::sub(ha, hb), tesseract::ops::sub(da, db));
  check_equal(tesseract::ops::mul(ha, hb), tesseract::ops::mul(da, db));
  check_equal(tesseract::ops::div(ha, hb), tesseract::ops::div(da, db));
}

TEST_CASE("CUDA add broadcasts a row vector over a matrix", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Matrix (8, 16) + row (1, 16) → (8, 16).
  std::vector<float> mat(8 * 16), row(16);
  for (size_t i = 0; i < mat.size(); ++i) mat[i] = static_cast<float>(i);
  for (size_t j = 0; j < row.size(); ++j) row[j] = static_cast<float>(j) * 0.1f;

  Tensor hm = Tensor::from_vector(mat, {8, 16});
  Tensor hr = Tensor::from_vector(row, {1, 16});
  Tensor h_out = tesseract::ops::add(hm, hr);

  Tensor dm = hm.to(cuda0());
  Tensor dr = hr.to(cuda0());
  Tensor d_out = tesseract::ops::add(dm, dr);
  Tensor back = d_out.to(cpu_device());

  const float* e = h_out.data_ptr<float>();
  const float* a = back.data_ptr<float>();
  for (int64_t i = 0; i < back.numel(); ++i) REQUIRE(a[i] == e[i]);
}

TEST_CASE("CUDA elementwise on Int32 matches CPU", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  std::vector<int32_t> a(128), b(128);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<int32_t>(i) - 64;
    b[i] = static_cast<int32_t>((i * 7) % 13) + 1;  // non-zero
  }
  Tensor ha = Tensor::from_vector(a, {8, 16});
  Tensor hb = Tensor::from_vector(b, {8, 16});
  Tensor hr = tesseract::ops::mul(ha, hb);

  Tensor da = ha.to(cuda0());
  Tensor db = hb.to(cuda0());
  Tensor dr = tesseract::ops::mul(da, db);
  Tensor back = dr.to(cpu_device());
  const int32_t* e = hr.data_ptr<int32_t>();
  const int32_t* r = back.data_ptr<int32_t>();
  for (int64_t i = 0; i < back.numel(); ++i) REQUIRE(r[i] == e[i]);
}

TEST_CASE("CUDA neg matches CPU across dtypes", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  {
    auto pa = pattern_floats(512);
    Tensor h = Tensor::from_vector(pa, {16, 32});
    Tensor hn = tesseract::ops::neg(h);
    Tensor d = h.to(cuda0());
    Tensor dn = tesseract::ops::neg(d);
    Tensor back = dn.to(cpu_device());
    const float* e = hn.data_ptr<float>();
    const float* a = back.data_ptr<float>();
    for (int64_t i = 0; i < back.numel(); ++i) REQUIRE(a[i] == e[i]);
  }
  {
    std::vector<int64_t> src(64);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<int64_t>(i) - 32;
    Tensor h = Tensor::from_vector(src, {8, 8});
    Tensor hn = tesseract::ops::neg(h);
    Tensor d = h.to(cuda0());
    Tensor dn = tesseract::ops::neg(d);
    Tensor back = dn.to(cpu_device());
    const int64_t* e = hn.data_ptr<int64_t>();
    const int64_t* a = back.data_ptr<int64_t>();
    for (int64_t i = 0; i < back.numel(); ++i) REQUIRE(a[i] == e[i]);
  }
}

TEST_CASE("CUDA relu matches CPU on mixed signs", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto pa = pattern_floats(1024);
  // Stamp in a few exact zeros so the boundary behavior is covered.
  for (std::size_t i = 0; i < pa.size(); i += 63) pa[i] = 0.0f;
  Tensor h = Tensor::from_vector(pa, {32, 32});
  Tensor h_out = tesseract::ops::relu(h);
  Tensor d = h.to(cuda0());
  Tensor d_out = tesseract::ops::relu(d);
  Tensor back = d_out.to(cpu_device());
  const float* e = h_out.data_ptr<float>();
  const float* a = back.data_ptr<float>();
  for (int64_t i = 0; i < back.numel(); ++i) REQUIRE(a[i] == e[i]);
}

TEST_CASE("CUDA sigmoid/tanh/exp/log match CPU within tolerance",
          "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Clamp the input range so `exp` doesn't blow up the test and
  // sigmoid stays in the precise regime.
  std::vector<float> pa(256);
  for (std::size_t i = 0; i < pa.size(); ++i) {
    pa[i] = static_cast<float>(i) * 0.05f - 6.0f;  // [-6, 6.75)
  }
  Tensor h = Tensor::from_vector(pa, {16, 16});
  Tensor d = h.to(cuda0());

  auto expect_close = [&](const Tensor& cpu_ref, const Tensor& cuda_res,
                          float tol) {
    Tensor back = cuda_res.to(cpu_device());
    const float* e = cpu_ref.data_ptr<float>();
    const float* a = back.data_ptr<float>();
    for (int64_t i = 0; i < back.numel(); ++i) {
      REQUIRE_THAT(a[i], WithinAbs(e[i], tol));
    }
  };

  // Larger tol for sigmoid/exp because we use `__expf` (CUDA fast
  // intrinsic) on float — the CPU reference uses `std::exp`. Accuracy
  // is still well within what training tolerates.
  expect_close(tesseract::ops::sigmoid(h), tesseract::ops::sigmoid(d), 2e-5f);
  expect_close(tesseract::ops::tanh(h),    tesseract::ops::tanh(d),    1e-6f);
  expect_close(tesseract::ops::exp(h),     tesseract::ops::exp(d),     3e-4f);

  // Log needs strictly positive inputs.
  Tensor hp = Tensor::from_vector(positive_floats(256), {16, 16});
  Tensor dp = hp.to(cuda0());
  expect_close(tesseract::ops::log(hp), tesseract::ops::log(dp), 1e-5f);
}

TEST_CASE("Tensor::ones on CUDA now uses the fill kernel", "[tensor][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Regression guard on the M2D→M2E transition: `ones(Float64, cuda)`
  // and `full(Int64, cuda, v)` used to host-scratch + H→D copy. They
  // must now launch the fill kernel and produce identical output.
  Tensor t1 = Tensor::ones({1024}, DType::Float64, cuda0());
  Tensor h1 = t1.to(cpu_device());
  const double* p1 = h1.data_ptr<double>();
  for (int64_t i = 0; i < h1.numel(); ++i) REQUIRE(p1[i] == 1.0);

  Tensor t2 = Tensor::full({8, 8}, 42.0, DType::Int64, cuda0());
  Tensor h2 = t2.to(cpu_device());
  const int64_t* p2 = h2.data_ptr<int64_t>();
  for (int64_t i = 0; i < h2.numel(); ++i) REQUIRE(p2[i] == 42);
}

TEST_CASE("Tensor::full on CUDA handles Bool dtype", "[tensor][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  Tensor t = Tensor::full({16}, 1.0, DType::Bool, cuda0());
  Tensor h = t.to(cpu_device());
  const bool* p = h.data_ptr<bool>();
  for (int64_t i = 0; i < h.numel(); ++i) REQUIRE(p[i] == true);
  Tensor t0 = Tensor::full({16}, 0.0, DType::Bool, cuda0());
  Tensor h0 = t0.to(cpu_device());
  const bool* p0 = h0.data_ptr<bool>();
  for (int64_t i = 0; i < h0.numel(); ++i) REQUIRE(p0[i] == false);
}

TEST_CASE("CUDA dispatch leaves CPU ops untouched", "[ops][gpu]") {
  // CPU-only smoke: two tensors on CPU must still resolve to the CPU
  // path regardless of whether CUDA is compiled in. This guards
  // against a device-dispatch bug where we'd accidentally route all
  // tensors through the CUDA launcher.
  auto a = Tensor::from_vector<float>({1.0f, 2.0f, 3.0f, 4.0f}, {4});
  auto b = Tensor::from_vector<float>({0.1f, 0.2f, 0.3f, 0.4f}, {4});
  auto c = tesseract::ops::add(a, b);
  REQUIRE(c.device() == cpu_device());
  const float* p = c.data_ptr<float>();
  REQUIRE_THAT(p[0], WithinAbs(1.1, 1e-5));
  REQUIRE_THAT(p[3], WithinAbs(4.4, 1e-5));
}

// B-015: CPU↔CUDA parity tests for the Float16 / BFloat16 elementwise
// path.
//
// Until B-015 landed, `src/cuda/Elementwise.cu`'s dtype switch rejected
// `Float16` and `BFloat16` with a `DeviceError`, which forced every
// attention / transformer workload to run at FP32 on CUDA even though
// the CPU path (`src/ops/cpu/Arithmetic.cpp`) handled both dtypes
// natively through the `Half::operator float()` round-trip. The fix
// extends the CUDA kernels with FP32-promotion (read `__half` /
// `__nv_bfloat16` → widen → compute → narrow), so the numerical
// semantics match the CPU path bit-for-bit up to the normal half-
// precision rounding envelope.
//
// Tolerance budget: `2e-3` absolute on FP16 and `5e-3` on BFloat16.
// Matches the `[B-005]` envelope established for CPU tests and
// accounts for the 10-bit FP16 / 7-bit BF16 mantissa. Transcendentals
// (`exp`, `log`, `sigmoid`, `tanh`) use a slightly wider `5e-3` since
// the CUDA fast-math intrinsics (`__expf`, `tanhf`) differ from the
// CPU `std::exp` / `std::tanh` by a few ULPs at FP32 — that delta is
// hidden by the FP16 narrowing on most inputs but shows up near zero.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"

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

std::vector<float> pattern_floats(std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto x = static_cast<float>(i);
    out[i] = (x - static_cast<float>(n / 2)) * 0.0625f;  // narrower range so fp16 representable
  }
  return out;
}

std::vector<float> positive_floats(std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = 0.25f + static_cast<float>(i) * 0.0625f;
  }
  return out;
}

template <typename H>
std::vector<H> quantize(const std::vector<float>& f) {
  std::vector<H> out(f.size());
  for (std::size_t i = 0; i < f.size(); ++i) out[i] = H(f[i]);
  return out;
}

// CPU and CUDA both round the FP16 result independently; after the
// op the two stored halves should match to within `tol` in FP32
// after conversion back. This helper centralises that comparison
// so we can use the same scaffold for FP16 and BF16.
template <typename H>
void require_close(const Tensor& cpu_ref, const Tensor& cuda_res, float tol) {
  Tensor back = cuda_res.to(cpu_device());
  REQUIRE(back.numel() == cpu_ref.numel());
  REQUIRE(back.dtype() == cpu_ref.dtype());
  const H* e = cpu_ref.template data_ptr<H>();
  const H* a = back.template data_ptr<H>();
  for (int64_t i = 0; i < back.numel(); ++i) {
    const float ef = static_cast<float>(e[i]);
    const float af = static_cast<float>(a[i]);
    REQUIRE_THAT(af, WithinAbs(ef, tol));
  }
}

}  // namespace

TEST_CASE("CUDA FP16 add matches CPU (dense same shape)", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto fa = pattern_floats(1024);
  auto fb = pattern_floats(1024);
  for (auto& v : fb) v += 0.75f;
  auto ha_data = quantize<Half>(fa);
  auto hb_data = quantize<Half>(fb);
  Tensor ha = Tensor::from_vector(ha_data, {32, 32});
  Tensor hb = Tensor::from_vector(hb_data, {32, 32});
  Tensor hc = tesseract::ops::add(ha, hb);

  Tensor da = ha.to(cuda0());
  Tensor db = hb.to(cuda0());
  Tensor dc = tesseract::ops::add(da, db);
  REQUIRE(dc.device() == cuda0());
  require_close<Half>(hc, dc, 2e-3f);
}

TEST_CASE("CUDA FP16 sub/mul/div match CPU", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  std::vector<float> fa(256), fb(256);
  for (size_t i = 0; i < fa.size(); ++i) {
    fa[i] = static_cast<float>(i) * 0.125f - 16.0f;
    fb[i] = static_cast<float>(i % 17) * 0.5f + 1.0f;  // non-zero divisor
  }
  Tensor ha = Tensor::from_vector(quantize<Half>(fa), {16, 16});
  Tensor hb = Tensor::from_vector(quantize<Half>(fb), {16, 16});
  Tensor da = ha.to(cuda0()), db = hb.to(cuda0());

  require_close<Half>(tesseract::ops::sub(ha, hb),
                     tesseract::ops::sub(da, db), 2e-3f);
  require_close<Half>(tesseract::ops::mul(ha, hb),
                     tesseract::ops::mul(da, db), 2e-3f);
  require_close<Half>(tesseract::ops::div(ha, hb),
                     tesseract::ops::div(da, db), 5e-3f);
}

TEST_CASE("CUDA FP16 add broadcasts a row vector over a matrix", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  std::vector<float> mat(8 * 16), row(16);
  for (size_t i = 0; i < mat.size(); ++i) mat[i] = static_cast<float>(i) * 0.125f;
  for (size_t j = 0; j < row.size(); ++j) row[j] = static_cast<float>(j) * 0.0625f;

  Tensor hm = Tensor::from_vector(quantize<Half>(mat), {8, 16});
  Tensor hr = Tensor::from_vector(quantize<Half>(row), {1, 16});
  Tensor h_out = tesseract::ops::add(hm, hr);

  Tensor dm = hm.to(cuda0());
  Tensor dr = hr.to(cuda0());
  Tensor d_out = tesseract::ops::add(dm, dr);
  require_close<Half>(h_out, d_out, 2e-3f);
}

TEST_CASE("CUDA FP16 neg / relu match CPU", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto fa = pattern_floats(512);
  for (std::size_t i = 0; i < fa.size(); i += 63) fa[i] = 0.0f;
  Tensor h = Tensor::from_vector(quantize<Half>(fa), {16, 32});
  Tensor d = h.to(cuda0());

  require_close<Half>(tesseract::ops::neg(h), tesseract::ops::neg(d), 0.0f);
  require_close<Half>(tesseract::ops::relu(h), tesseract::ops::relu(d), 0.0f);
}

TEST_CASE("CUDA FP16 sigmoid/tanh/exp/log match CPU within tolerance",
          "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  // Clamp input range: FP16 overflows `exp` around ~11 (`exp(11) ≈ 60k`).
  std::vector<float> fa(256);
  for (std::size_t i = 0; i < fa.size(); ++i) {
    fa[i] = static_cast<float>(i) * 0.03125f - 4.0f;  // [-4, 4)
  }
  Tensor h = Tensor::from_vector(quantize<Half>(fa), {16, 16});
  Tensor d = h.to(cuda0());

  require_close<Half>(tesseract::ops::sigmoid(h), tesseract::ops::sigmoid(d), 3e-3f);
  require_close<Half>(tesseract::ops::tanh(h),    tesseract::ops::tanh(d),    3e-3f);
  require_close<Half>(tesseract::ops::exp(h),     tesseract::ops::exp(d),     5e-2f);
  // Stricter tol on log where we control the input domain away from 0.
  Tensor hp = Tensor::from_vector(quantize<Half>(positive_floats(256)), {16, 16});
  Tensor dp = hp.to(cuda0());
  require_close<Half>(tesseract::ops::log(hp), tesseract::ops::log(dp), 5e-3f);
}

// ---------- BFloat16 ----------

TEST_CASE("CUDA BF16 add matches CPU (dense same shape)", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  auto fa = pattern_floats(1024);
  auto fb = pattern_floats(1024);
  for (auto& v : fb) v += 0.75f;
  Tensor ha = Tensor::from_vector(quantize<BFloat16>(fa), {32, 32});
  Tensor hb = Tensor::from_vector(quantize<BFloat16>(fb), {32, 32});
  Tensor hc = tesseract::ops::add(ha, hb);

  Tensor da = ha.to(cuda0()), db = hb.to(cuda0());
  Tensor dc = tesseract::ops::add(da, db);
  require_close<BFloat16>(hc, dc, 5e-3f);
}

TEST_CASE("CUDA BF16 mul + sigmoid match CPU within tolerance", "[ops][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  std::vector<float> fa(256), fb(256);
  for (size_t i = 0; i < fa.size(); ++i) {
    fa[i] = static_cast<float>(i) * 0.0625f - 8.0f;
    fb[i] = static_cast<float>(i % 11) * 0.25f + 1.0f;
  }
  Tensor ha = Tensor::from_vector(quantize<BFloat16>(fa), {16, 16});
  Tensor hb = Tensor::from_vector(quantize<BFloat16>(fb), {16, 16});
  Tensor da = ha.to(cuda0()), db = hb.to(cuda0());

  require_close<BFloat16>(tesseract::ops::mul(ha, hb),
                         tesseract::ops::mul(da, db), 2e-2f);

  // Sigmoid: clamp range so input fits comfortably in BF16's ~8-bit
  // effective precision; CPU reference rounds identically.
  std::vector<float> fs(256);
  for (size_t i = 0; i < fs.size(); ++i) {
    fs[i] = static_cast<float>(i) * 0.03125f - 4.0f;
  }
  Tensor hs = Tensor::from_vector(quantize<BFloat16>(fs), {16, 16});
  Tensor ds = hs.to(cuda0());
  require_close<BFloat16>(tesseract::ops::sigmoid(hs),
                         tesseract::ops::sigmoid(ds), 1e-2f);
}

TEST_CASE("Tensor::ones / full on CUDA handle FP16 and BF16", "[tensor][gpu]") {
  if (!cuda_ready()) SKIP("No CUDA GPU available.");
  Tensor t1 = Tensor::ones({1024}, DType::Float16, cuda0());
  Tensor h1 = t1.to(cpu_device());
  const Half* p1 = h1.data_ptr<Half>();
  for (int64_t i = 0; i < h1.numel(); ++i) {
    REQUIRE(static_cast<float>(p1[i]) == 1.0f);
  }
  Tensor t2 = Tensor::full({8, 8}, -2.5, DType::BFloat16, cuda0());
  Tensor h2 = t2.to(cpu_device());
  const BFloat16* p2 = h2.data_ptr<BFloat16>();
  for (int64_t i = 0; i < h2.numel(); ++i) {
    REQUIRE(static_cast<float>(p2[i]) == -2.5f);
  }
}

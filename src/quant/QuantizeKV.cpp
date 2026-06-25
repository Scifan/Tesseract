#include "tesseract/quant/QuantizeKV.hpp"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/detail/QuantizeKV.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::quant {

namespace {

// Banker's rounding to nearest even, clamped to [-127, 127]. Identical
// numerics to the CUDA `round_clip_i8` (rintf) and to `Pack.cpp`.
int8_t round_clip_i8(float v) {
  const float r = std::nearbyint(v);
  if (r >= 127.0f) return static_cast<int8_t>(127);
  if (r <= -127.0f) return static_cast<int8_t>(-127);
  return static_cast<int8_t>(r);
}

// Shape with the trailing dim dropped (the per-row scale shape).
Shape drop_last_dim(const Shape& s) {
  std::vector<int64_t> dims;
  dims.reserve(s.rank() - 1);
  for (int64_t i = 0; i + 1 < static_cast<int64_t>(s.rank()); ++i) {
    dims.push_back(s[i]);
  }
  return Shape(dims);
}

}  // namespace

std::pair<Tensor, Tensor> quantize_kv_per_token(const Tensor& x) {
  TESSERACT_CHECK(x.defined(), "quantize_kv_per_token: x is undefined");
  TESSERACT_CHECK(x.rank() >= 2,
                  "quantize_kv_per_token: x must be rank >= 2, got {}",
                  x.shape().to_string());
  const DType dtype = x.dtype();
  TESSERACT_CHECK(dtype == DType::Float32 || dtype == DType::Float16 ||
                  dtype == DType::BFloat16,
                  "quantize_kv_per_token: x dtype must be "
                  "Float32/Float16/BFloat16, got {}", dtype_name(dtype));
  TESSERACT_CHECK(x.is_contiguous(),
                  "quantize_kv_per_token: x must be contiguous");

  const int64_t dh   = x.shape()[x.rank() - 1];
  const int64_t rows = (dh > 0) ? (x.numel() / dh) : 0;

  Tensor q     = Tensor::empty(x.shape(), DType::Int8, x.device());
  Tensor scale = Tensor::empty(drop_last_dim(x.shape()), DType::Float32,
                               x.device());

  if (rows == 0 || dh == 0) return {std::move(q), std::move(scale)};

  if (x.device().is_cuda()) {
    Stream s = current_stream(x.device());
    cuda::detail::launch_quantize_kv_per_token(
        dtype, x.device().index, rows, dh, x.raw_data(),
        q.data_ptr<int8_t>(), scale.data_ptr<float>(), s.native_handle());
  } else {
    int8_t* qp = q.data_ptr<int8_t>();
    float*  sp = scale.data_ptr<float>();
    dispatch_float_with_half(dtype, [&]<typename T>() {
      const T* xp = x.data_ptr<T>();
      for (int64_t r = 0; r < rows; ++r) {
        const int64_t base = r * dh;
        float max_abs = 0.0f;
        for (int64_t d = 0; d < dh; ++d) {
          const float a = std::fabs(static_cast<float>(xp[base + d]));
          if (a > max_abs) max_abs = a;
        }
        const float sc = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        sp[r] = sc;
        const float inv = 1.0f / sc;
        for (int64_t d = 0; d < dh; ++d) {
          qp[base + d] = round_clip_i8(static_cast<float>(xp[base + d]) * inv);
        }
      }
    });
  }
  return {std::move(q), std::move(scale)};
}

Tensor dequantize_kv_per_token(const Tensor& q, const Tensor& scale,
                               DType out_dtype) {
  TESSERACT_CHECK(q.defined() && scale.defined(),
                  "dequantize_kv_per_token: q / scale must be defined");
  TESSERACT_CHECK(q.rank() >= 2,
                  "dequantize_kv_per_token: q must be rank >= 2, got {}",
                  q.shape().to_string());
  TESSERACT_CHECK(q.dtype() == DType::Int8,
                  "dequantize_kv_per_token: q dtype must be Int8, got {}",
                  dtype_name(q.dtype()));
  TESSERACT_CHECK(scale.dtype() == DType::Float32,
                  "dequantize_kv_per_token: scale dtype must be Float32, got {}",
                  dtype_name(scale.dtype()));
  TESSERACT_CHECK(out_dtype == DType::Float32 || out_dtype == DType::Float16 ||
                  out_dtype == DType::BFloat16,
                  "dequantize_kv_per_token: out_dtype must be "
                  "Float32/Float16/BFloat16, got {}", dtype_name(out_dtype));
  TESSERACT_CHECK(q.is_contiguous() && scale.is_contiguous(),
                  "dequantize_kv_per_token: q / scale must be contiguous");
  TESSERACT_CHECK(q.device() == scale.device(),
                  "dequantize_kv_per_token: q / scale device mismatch ({} / {})",
                  q.device().to_string(), scale.device().to_string());

  const int64_t dh   = q.shape()[q.rank() - 1];
  const int64_t rows = (dh > 0) ? (q.numel() / dh) : 0;
  TESSERACT_CHECK(scale.numel() == rows,
                  "dequantize_kv_per_token: scale numel {} != rows {} "
                  "(q.shape={})", scale.numel(), rows, q.shape().to_string());

  Tensor out = Tensor::empty(q.shape(), out_dtype, q.device());
  if (rows == 0 || dh == 0) return out;

  if (q.device().is_cuda()) {
    Stream s = current_stream(q.device());
    cuda::detail::launch_dequantize_kv_per_token(
        out_dtype, q.device().index, rows, dh, q.data_ptr<int8_t>(),
        scale.data_ptr<float>(), out.raw_data(), s.native_handle());
  } else {
    const int8_t* qp = q.data_ptr<int8_t>();
    const float*  sp = scale.data_ptr<float>();
    dispatch_float_with_half(out_dtype, [&]<typename T>() {
      T* op = out.data_ptr<T>();
      for (int64_t r = 0; r < rows; ++r) {
        const int64_t base = r * dh;
        const float sc = sp[r];
        for (int64_t d = 0; d < dh; ++d) {
          op[base + d] = static_cast<T>(static_cast<float>(qp[base + d]) * sc);
        }
      }
    });
  }
  return out;
}

}  // namespace tesseract::quant

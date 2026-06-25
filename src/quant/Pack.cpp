#include "tesseract/quant/Pack.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/core/Float16.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::quant {

namespace {

// Convert a storage element to `float` for quantization math. We do
// this through a plain switch so the caller doesn't have to template.
// The conversion is exact for FP16 / BF16 (widening) and the identity
// for FP32. Never instantiated for non-floating dtypes — the outer
// validation block rejects those.
float to_f32_load(const void* p, DType dtype, int64_t idx) {
  switch (dtype) {
    case DType::Float32:
      return static_cast<const float*>(p)[idx];
    case DType::Float16:
      return static_cast<float>(static_cast<const Half*>(p)[idx]);
    case DType::BFloat16:
      return static_cast<float>(static_cast<const BFloat16*>(p)[idx]);
    default:
      TESSERACT_THROW("tesseract::quant: unreachable dtype {}",
                      dtype_name(dtype));
  }
}

// Banker's rounding towards nearest even, clamped to `[-127, 127]`.
// Using `std::nearbyint` so the rounding follows the currently
// configured FE_TONEAREST mode (the C++ default). The clamp is
// defensive — with `scale = max_abs / 127` the dividend divided by
// the scale lands in `[-127, 127]` to within FP error, but a noisy
// row can still produce `127.5` which would round to 128.
int8_t round_clip_i8(float v) {
  const float r = std::nearbyint(v);
  if (r >=  127.0f) return static_cast<int8_t>(127);
  if (r <= -127.0f) return static_cast<int8_t>(-127);
  return static_cast<int8_t>(r);
}

// Banker's rounding to nearest even, clamped to `[-7, 7]`. The
// 4-bit signed range is `[-8, 7]`, but we deliberately exclude -8
// so dequantize is a pure `q * scale` multiply (same asymmetric
// trim as the INT8 `[-127, 127]` convention).
int8_t round_clip_i4(float v) {
  const float r = std::nearbyint(v);
  if (r >=  7.0f) return static_cast<int8_t>(7);
  if (r <= -7.0f) return static_cast<int8_t>(-7);
  return static_cast<int8_t>(r);
}

// Pack two signed 4-bit values (already clamped to [-7, 7]) into one
// byte. Low nibble = even-k, high nibble = odd-k. The signed nibble
// is stored as two's-complement (`-7 -> 0x9`, `-1 -> 0xF`, ... ,
// `7 -> 0x7`); `((byte >> shift) & 0xF)` followed by a 4->8-bit
// sign-extend recovers the original value.
uint8_t pack_nibbles(int8_t even, int8_t odd) {
  const uint8_t lo = static_cast<uint8_t>(static_cast<uint8_t>(even) & 0x0F);
  const uint8_t hi = static_cast<uint8_t>(static_cast<uint8_t>(odd)  & 0x0F);
  return static_cast<uint8_t>(lo | (hi << 4));
}

}  // namespace

std::pair<Tensor, Tensor> pack_int8_symmetric(const Tensor& weight) {
  TESSERACT_CHECK(weight.defined(),
                  "pack_int8_symmetric: weight tensor is undefined");
  TESSERACT_CHECK(weight.rank() >= 2,
                  "pack_int8_symmetric: weight must be rank >= 2, got {}",
                  weight.shape().to_string());

  const DType dtype = weight.dtype();
  TESSERACT_CHECK(dtype == DType::Float32 ||
                  dtype == DType::Float16 ||
                  dtype == DType::BFloat16,
                  "pack_int8_symmetric: expected Float32/Float16/BFloat16 "
                  "weight dtype, got {}", dtype_name(dtype));

  // Pull to host for the quantization pass. The packer is a one-shot
  // operation at model-load time so the H<-D copy cost is negligible
  // relative to inference serving over the packed weights.
  const Device src_device = weight.device();
  Tensor w_cpu = weight.to(cpu_device());
  w_cpu = w_cpu.contiguous();  // ensure row-major contiguous layout for the row-wise scan

  // Flatten all leading dims into a single "out channels" axis `out`,
  // keeping the trailing "in features" axis `in`. For a rank-2
  // `[out, in]` Linear weight this is the identity.
  const Shape shape = w_cpu.shape();
  const int64_t in_features = shape[shape.rank() - 1];
  int64_t out_channels = 1;
  for (int64_t i = 0; i + 1 < static_cast<int64_t>(shape.rank()); ++i) {
    out_channels *= shape[i];
  }
  TESSERACT_CHECK(in_features > 0,
                  "pack_int8_symmetric: in_features must be positive, got {}",
                  in_features);
  TESSERACT_CHECK(out_channels > 0,
                  "pack_int8_symmetric: out_channels must be positive, got {}",
                  out_channels);

  // Allocate outputs on CPU first — we'll ship them back to the source
  // device at the end.
  Tensor q_cpu     = Tensor::empty(shape,        DType::Int8,   cpu_device());
  Tensor scale_cpu = Tensor::empty({out_channels}, DType::Float32, cpu_device());

  const void*  w_data = w_cpu.raw_data();
  int8_t*      q_data = q_cpu.data_ptr<int8_t>();
  float*       s_data = scale_cpu.data_ptr<float>();

  // Two-pass per row:
  //   pass 1 — find max_abs on the row, derive scale = max_abs / 127.
  //   pass 2 — quantize every element by `round(v / scale)` clipped.
  // Dense loop; no OMP (model-load, not hot path). The compiler auto-
  // vectorizes the FP32 branch and the FP16/BF16 branches through the
  // `to_f32_load` dispatch.
  for (int64_t r = 0; r < out_channels; ++r) {
    const int64_t row_off = r * in_features;
    float max_abs = 0.0f;
    for (int64_t c = 0; c < in_features; ++c) {
      const float v = to_f32_load(w_data, dtype, row_off + c);
      const float a = std::fabs(v);
      if (a > max_abs) max_abs = a;
    }

    // Identically-zero rows: scale = 1.0 so dequant(q_w * scale) = 0
    // for every element. Using 0 here would make every downstream
    // multiply undefined.
    const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    s_data[r] = scale;

    const float inv_scale = 1.0f / scale;
    for (int64_t c = 0; c < in_features; ++c) {
      const float v = to_f32_load(w_data, dtype, row_off + c);
      q_data[row_off + c] = round_clip_i8(v * inv_scale);
    }
  }

  // Round-trip back to the source device so the tensors are ready for
  // `ops::dequantize_matmul_int8` on whatever the caller was using.
  if (src_device.is_cpu()) {
    return {std::move(q_cpu), std::move(scale_cpu)};
  }
  Tensor q_dev     = q_cpu.to(src_device);
  Tensor scale_dev = scale_cpu.to(src_device);
  return {std::move(q_dev), std::move(scale_dev)};
}

std::pair<Tensor, Tensor> pack_int4_group(const Tensor& weight,
                                          int64_t group_size) {
  TESSERACT_CHECK(weight.defined(),
                  "pack_int4_group: weight tensor is undefined");
  TESSERACT_CHECK(weight.rank() >= 2,
                  "pack_int4_group: weight must be rank >= 2, got {}",
                  weight.shape().to_string());

  const DType dtype = weight.dtype();
  TESSERACT_CHECK(dtype == DType::Float32 ||
                  dtype == DType::Float16 ||
                  dtype == DType::BFloat16,
                  "pack_int4_group: expected Float32/Float16/BFloat16 "
                  "weight dtype, got {}", dtype_name(dtype));

  TESSERACT_CHECK(group_size >= 2,
                  "pack_int4_group: group_size must be >= 2, got {}",
                  group_size);
  TESSERACT_CHECK(group_size % 2 == 0,
                  "pack_int4_group: group_size must be even (so a byte's "
                  "two nibbles live in the same group), got {}",
                  group_size);

  const Device src_device = weight.device();
  Tensor w_cpu = weight.to(cpu_device());
  w_cpu = w_cpu.contiguous();

  const Shape shape = w_cpu.shape();
  const int64_t in_features = shape[shape.rank() - 1];
  int64_t out_channels = 1;
  for (int64_t i = 0; i + 1 < static_cast<int64_t>(shape.rank()); ++i) {
    out_channels *= shape[i];
  }
  TESSERACT_CHECK(in_features > 0,
                  "pack_int4_group: in_features must be positive, got {}",
                  in_features);
  TESSERACT_CHECK(out_channels > 0,
                  "pack_int4_group: out_channels must be positive, got {}",
                  out_channels);
  TESSERACT_CHECK(in_features % group_size == 0,
                  "pack_int4_group: in_features ({}) must be a multiple of "
                  "group_size ({})", in_features, group_size);
  TESSERACT_CHECK(group_size <= in_features,
                  "pack_int4_group: group_size ({}) must be <= in_features ({})",
                  group_size, in_features);

  const int64_t packed_cols  = in_features / 2;
  const int64_t groups_per_row = in_features / group_size;

  // Output shapes:
  //   q_packed : [out_channels, packed_cols]                  Int8 bytes
  //   scale    : [out_channels, groups_per_row]               Float32
  // We intentionally keep the packed weight as a rank-2 tensor (not
  // rank-N) so the op kernel sees a flat `[out, in/2]` layout. For
  // higher-rank inputs the caller reshapes externally; that's the
  // same convention `pack_int8_symmetric` uses for its flattened
  // out-channel dim.
  Tensor q_cpu     = Tensor::empty({out_channels, packed_cols},
                                   DType::Int8,    cpu_device());
  Tensor scale_cpu = Tensor::empty({out_channels, groups_per_row},
                                   DType::Float32, cpu_device());

  const void*  w_data = w_cpu.raw_data();
  int8_t*      q_data = q_cpu.data_ptr<int8_t>();
  float*       s_data = scale_cpu.data_ptr<float>();

  for (int64_t r = 0; r < out_channels; ++r) {
    const int64_t w_row_off = r * in_features;
    const int64_t q_row_off = r * packed_cols;
    const int64_t s_row_off = r * groups_per_row;

    for (int64_t g = 0; g < groups_per_row; ++g) {
      const int64_t g_start = g * group_size;

      float max_abs = 0.0f;
      for (int64_t j = 0; j < group_size; ++j) {
        const float v = to_f32_load(w_data, dtype, w_row_off + g_start + j);
        const float a = std::fabs(v);
        if (a > max_abs) max_abs = a;
      }
      const float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
      s_data[s_row_off + g] = scale;

      const float inv_scale = 1.0f / scale;
      // Walk the group two elements at a time so each iteration
      // produces exactly one packed byte. `group_size` was
      // validated even so this is always well-formed.
      for (int64_t j = 0; j < group_size; j += 2) {
        const float v_even = to_f32_load(w_data, dtype,
                                         w_row_off + g_start + j);
        const float v_odd  = to_f32_load(w_data, dtype,
                                         w_row_off + g_start + j + 1);
        const int8_t q_even = round_clip_i4(v_even * inv_scale);
        const int8_t q_odd  = round_clip_i4(v_odd  * inv_scale);
        const int64_t byte_col = (g_start + j) / 2;
        q_data[q_row_off + byte_col] =
            static_cast<int8_t>(pack_nibbles(q_even, q_odd));
      }
    }
  }

  if (src_device.is_cpu()) {
    return {std::move(q_cpu), std::move(scale_cpu)};
  }
  Tensor q_dev     = q_cpu.to(src_device);
  Tensor scale_dev = scale_cpu.to(src_device);
  return {std::move(q_dev), std::move(scale_dev)};
}

}  // namespace tesseract::quant

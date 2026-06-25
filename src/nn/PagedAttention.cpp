#include "tesseract/nn/PagedAttention.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/core/Stream.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/detail/PagedAttention.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

Tensor paged_decode_attention(const Tensor& q, const Tensor& k_pool,
                              const Tensor& v_pool, const Tensor& block_tables,
                              const Tensor& lens, double scale, int64_t group) {
  TESSERACT_CHECK(q.defined() && k_pool.defined() && v_pool.defined() &&
                  block_tables.defined() && lens.defined(),
                  "paged_decode_attention: all operands must be defined");
  TESSERACT_CHECK(q.rank() == 3,
                  "paged_decode_attention: q must be [A, H, D], got {}",
                  q.shape().to_string());
  TESSERACT_CHECK(k_pool.rank() == 4 && v_pool.rank() == 4,
                  "paged_decode_attention: pools must be "
                  "[num_blocks, Hkv, block_size, D]");
  const DType dtype = q.dtype();
  TESSERACT_CHECK(dtype == DType::Float32 || dtype == DType::Float16 ||
                  dtype == DType::BFloat16,
                  "paged_decode_attention: q dtype must be FP32/FP16/BF16, "
                  "got {}", dtype_name(dtype));
  TESSERACT_CHECK(k_pool.dtype() == dtype && v_pool.dtype() == dtype,
                  "paged_decode_attention: pool dtype must match q");
  TESSERACT_CHECK(block_tables.dtype() == DType::Int32 &&
                  lens.dtype() == DType::Int32,
                  "paged_decode_attention: block_tables / lens must be Int32");
  TESSERACT_CHECK(q.is_contiguous() && k_pool.is_contiguous() &&
                  v_pool.is_contiguous() && block_tables.is_contiguous() &&
                  lens.is_contiguous(),
                  "paged_decode_attention: all operands must be contiguous");
  TESSERACT_CHECK(q.device() == k_pool.device() && q.device() == v_pool.device() &&
                  q.device() == block_tables.device() && q.device() == lens.device(),
                  "paged_decode_attention: all operands must share a device");

  const int64_t A          = q.shape()[0];
  const int64_t H          = q.shape()[1];
  const int64_t D          = q.shape()[2];
  const int64_t num_blocks = k_pool.shape()[0];
  const int64_t Hkv        = k_pool.shape()[1];
  const int64_t block_size = k_pool.shape()[2];
  TESSERACT_CHECK(k_pool.shape()[3] == D && v_pool.shape()[3] == D,
                  "paged_decode_attention: pool head_dim {} != q head_dim {}",
                  k_pool.shape()[3], D);
  TESSERACT_CHECK(v_pool.shape()[0] == num_blocks && v_pool.shape()[1] == Hkv &&
                  v_pool.shape()[2] == block_size,
                  "paged_decode_attention: k/v pool shape mismatch");
  TESSERACT_CHECK(group >= 1 && H == Hkv * group,
                  "paged_decode_attention: H ({}) must equal Hkv ({}) * group "
                  "({})", H, Hkv, group);
  TESSERACT_CHECK(D <= 128,
                  "paged_decode_attention: head_dim {} exceeds kernel D_MAX=128",
                  D);
  TESSERACT_CHECK(block_tables.rank() == 2 && block_tables.shape()[0] == A,
                  "paged_decode_attention: block_tables must be [A, max_logical]");
  TESSERACT_CHECK(lens.rank() == 1 && lens.shape()[0] == A,
                  "paged_decode_attention: lens must be [A]");
  const int64_t max_logical = block_tables.shape()[1];

  Tensor o = Tensor::empty(Shape({A, H, D}), dtype, q.device());
  if (A == 0 || H == 0 || D == 0) return o;

  const float scale_f = static_cast<float>(scale);

  if (q.device().is_cuda()) {
    Stream s = current_stream(q.device());
    cuda::detail::launch_paged_decode_attention(
        dtype, q.device().index, A, H, Hkv, D, block_size, num_blocks,
        max_logical, static_cast<int>(group), scale_f,
        q.raw_data(), k_pool.raw_data(), v_pool.raw_data(),
        block_tables.data_ptr<int32_t>(), lens.data_ptr<int32_t>(),
        o.raw_data(), s.native_handle());
    return o;
  }

  const int32_t* table = block_tables.data_ptr<int32_t>();
  const int32_t* lp    = lens.data_ptr<int32_t>();
  dispatch_float_with_half(dtype, [&]<typename T>() {
    const T* qp = q.data_ptr<T>();
    const T* kp = k_pool.data_ptr<T>();
    const T* vp = v_pool.data_ptr<T>();
    T* op       = o.data_ptr<T>();
    std::vector<float> scores;
    for (int64_t r = 0; r < A; ++r) {
      const int64_t len = lp[r];
      const int32_t* row_table = table + r * max_logical;
      for (int64_t h = 0; h < H; ++h) {
        const int64_t hkv  = h / group;
        const T* q_row     = qp + (r * H + h) * D;
        T* o_row           = op + (r * H + h) * D;
        if (len <= 0) {
          for (int64_t d = 0; d < D; ++d) o_row[d] = static_cast<T>(0.0f);
          continue;
        }
        scores.assign(static_cast<std::size_t>(len), 0.0f);
        float m = -INFINITY;
        for (int64_t j = 0; j < len; ++j) {
          int64_t p = row_table[j / block_size];
          if (p < 0 || p >= num_blocks) p = 0;
          const int64_t slot = j % block_size;
          const T* k_row = kp + ((p * Hkv + hkv) * block_size + slot) * D;
          float acc = 0.0f;
          for (int64_t d = 0; d < D; ++d)
            acc += static_cast<float>(q_row[d]) * static_cast<float>(k_row[d]);
          acc *= scale_f;
          scores[static_cast<std::size_t>(j)] = acc;
          if (acc > m) m = acc;
        }
        float l = 0.0f;
        for (int64_t j = 0; j < len; ++j) {
          const float e = std::exp(scores[static_cast<std::size_t>(j)] - m);
          scores[static_cast<std::size_t>(j)] = e;
          l += e;
        }
        const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        std::vector<float> out(static_cast<std::size_t>(D), 0.0f);
        for (int64_t j = 0; j < len; ++j) {
          int64_t p = row_table[j / block_size];
          if (p < 0 || p >= num_blocks) p = 0;
          const int64_t slot = j % block_size;
          const T* v_row = vp + ((p * Hkv + hkv) * block_size + slot) * D;
          const float pj = scores[static_cast<std::size_t>(j)];
          for (int64_t d = 0; d < D; ++d)
            out[static_cast<std::size_t>(d)] += pj * static_cast<float>(v_row[d]);
        }
        for (int64_t d = 0; d < D; ++d)
          o_row[d] = static_cast<T>(out[static_cast<std::size_t>(d)] * inv_l);
      }
    }
  });
  return o;
}

Tensor paged_decode_attention_int8(const Tensor& q, const Tensor& k_pool,
                                   const Tensor& k_scale, const Tensor& v_pool,
                                   const Tensor& v_scale,
                                   const Tensor& block_tables,
                                   const Tensor& lens, double scale,
                                   int64_t group) {
  TESSERACT_CHECK(q.defined() && k_pool.defined() && k_scale.defined() &&
                  v_pool.defined() && v_scale.defined() &&
                  block_tables.defined() && lens.defined(),
                  "paged_decode_attention_int8: all operands must be defined");
  TESSERACT_CHECK(q.rank() == 3,
                  "paged_decode_attention_int8: q must be [A, H, D], got {}",
                  q.shape().to_string());
  TESSERACT_CHECK(k_pool.rank() == 4 && v_pool.rank() == 4,
                  "paged_decode_attention_int8: pools must be "
                  "[num_blocks, Hkv, block_size, D]");
  const DType dtype = q.dtype();
  TESSERACT_CHECK(dtype == DType::Float32 || dtype == DType::Float16 ||
                  dtype == DType::BFloat16,
                  "paged_decode_attention_int8: q dtype must be FP32/FP16/BF16, "
                  "got {}", dtype_name(dtype));
  TESSERACT_CHECK(k_pool.dtype() == DType::Int8 && v_pool.dtype() == DType::Int8,
                  "paged_decode_attention_int8: K/V pools must be Int8");
  TESSERACT_CHECK(k_scale.dtype() == DType::Float32 &&
                  v_scale.dtype() == DType::Float32,
                  "paged_decode_attention_int8: scales must be Float32");
  TESSERACT_CHECK(block_tables.dtype() == DType::Int32 &&
                  lens.dtype() == DType::Int32,
                  "paged_decode_attention_int8: block_tables / lens Int32");
  TESSERACT_CHECK(q.is_contiguous() && k_pool.is_contiguous() &&
                  k_scale.is_contiguous() && v_pool.is_contiguous() &&
                  v_scale.is_contiguous() && block_tables.is_contiguous() &&
                  lens.is_contiguous(),
                  "paged_decode_attention_int8: all operands must be contiguous");
  TESSERACT_CHECK(q.device() == k_pool.device() && q.device() == k_scale.device() &&
                  q.device() == v_pool.device() && q.device() == v_scale.device() &&
                  q.device() == block_tables.device() && q.device() == lens.device(),
                  "paged_decode_attention_int8: all operands must share a device");

  const int64_t A          = q.shape()[0];
  const int64_t H          = q.shape()[1];
  const int64_t D          = q.shape()[2];
  const int64_t num_blocks = k_pool.shape()[0];
  const int64_t Hkv        = k_pool.shape()[1];
  const int64_t block_size = k_pool.shape()[2];
  TESSERACT_CHECK(k_pool.shape()[3] == D && v_pool.shape()[3] == D,
                  "paged_decode_attention_int8: pool head_dim {} != q head_dim {}",
                  k_pool.shape()[3], D);
  TESSERACT_CHECK(v_pool.shape()[0] == num_blocks && v_pool.shape()[1] == Hkv &&
                  v_pool.shape()[2] == block_size,
                  "paged_decode_attention_int8: k/v pool shape mismatch");
  const int64_t scale_numel = num_blocks * Hkv * block_size;
  TESSERACT_CHECK(k_scale.numel() == scale_numel && v_scale.numel() == scale_numel,
                  "paged_decode_attention_int8: scale numel {} / {} != "
                  "num_blocks*Hkv*block_size ({})", k_scale.numel(),
                  v_scale.numel(), scale_numel);
  TESSERACT_CHECK(group >= 1 && H == Hkv * group,
                  "paged_decode_attention_int8: H ({}) must equal Hkv ({}) * "
                  "group ({})", H, Hkv, group);
  TESSERACT_CHECK(D <= 128,
                  "paged_decode_attention_int8: head_dim {} exceeds D_MAX=128", D);
  TESSERACT_CHECK(block_tables.rank() == 2 && block_tables.shape()[0] == A,
                  "paged_decode_attention_int8: block_tables must be [A, max_logical]");
  TESSERACT_CHECK(lens.rank() == 1 && lens.shape()[0] == A,
                  "paged_decode_attention_int8: lens must be [A]");
  const int64_t max_logical = block_tables.shape()[1];

  Tensor o = Tensor::empty(Shape({A, H, D}), dtype, q.device());
  if (A == 0 || H == 0 || D == 0) return o;

  const float scale_f = static_cast<float>(scale);

  if (q.device().is_cuda()) {
    Stream s = current_stream(q.device());
    cuda::detail::launch_paged_decode_attention_int8(
        dtype, q.device().index, A, H, Hkv, D, block_size, num_blocks,
        max_logical, static_cast<int>(group), scale_f, q.raw_data(),
        k_pool.data_ptr<int8_t>(), k_scale.data_ptr<float>(),
        v_pool.data_ptr<int8_t>(), v_scale.data_ptr<float>(),
        block_tables.data_ptr<int32_t>(), lens.data_ptr<int32_t>(),
        o.raw_data(), s.native_handle());
    return o;
  }

  const int8_t* kq = k_pool.data_ptr<int8_t>();
  const int8_t* vq = v_pool.data_ptr<int8_t>();
  const float*  ksc = k_scale.data_ptr<float>();
  const float*  vsc = v_scale.data_ptr<float>();
  const int32_t* table = block_tables.data_ptr<int32_t>();
  const int32_t* lp    = lens.data_ptr<int32_t>();
  dispatch_float_with_half(dtype, [&]<typename T>() {
    const T* qp = q.data_ptr<T>();
    T* op       = o.data_ptr<T>();
    std::vector<float> scores;
    for (int64_t r = 0; r < A; ++r) {
      const int64_t len = lp[r];
      const int32_t* row_table = table + r * max_logical;
      for (int64_t h = 0; h < H; ++h) {
        const int64_t hkv  = h / group;
        const T* q_row     = qp + (r * H + h) * D;
        T* o_row           = op + (r * H + h) * D;
        if (len <= 0) {
          for (int64_t d = 0; d < D; ++d) o_row[d] = static_cast<T>(0.0f);
          continue;
        }
        scores.assign(static_cast<std::size_t>(len), 0.0f);
        float m = -INFINITY;
        for (int64_t j = 0; j < len; ++j) {
          int64_t p = row_table[j / block_size];
          if (p < 0 || p >= num_blocks) p = 0;
          const int64_t slot = j % block_size;
          const int64_t hbs  = (p * Hkv + hkv) * block_size + slot;
          const int8_t* k_row = kq + hbs * D;
          const float ks = ksc[hbs];
          float acc = 0.0f;
          for (int64_t d = 0; d < D; ++d)
            acc += static_cast<float>(q_row[d]) *
                   (static_cast<float>(k_row[d]) * ks);
          acc *= scale_f;
          scores[static_cast<std::size_t>(j)] = acc;
          if (acc > m) m = acc;
        }
        float l = 0.0f;
        for (int64_t j = 0; j < len; ++j) {
          const float e = std::exp(scores[static_cast<std::size_t>(j)] - m);
          scores[static_cast<std::size_t>(j)] = e;
          l += e;
        }
        const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        std::vector<float> out(static_cast<std::size_t>(D), 0.0f);
        for (int64_t j = 0; j < len; ++j) {
          int64_t p = row_table[j / block_size];
          if (p < 0 || p >= num_blocks) p = 0;
          const int64_t slot = j % block_size;
          const int64_t hbs  = (p * Hkv + hkv) * block_size + slot;
          const int8_t* v_row = vq + hbs * D;
          const float vs = vsc[hbs];
          const float pj = scores[static_cast<std::size_t>(j)];
          for (int64_t d = 0; d < D; ++d)
            out[static_cast<std::size_t>(d)] +=
                pj * (static_cast<float>(v_row[d]) * vs);
        }
        for (int64_t d = 0; d < D; ++d)
          o_row[d] = static_cast<T>(out[static_cast<std::size_t>(d)] * inv_l);
      }
    }
  });
  return o;
}

Tensor paged_prefill_attention(const Tensor& q, const Tensor& k_pool,
                               const Tensor& v_pool, const Tensor& block_tables,
                               const Tensor& kv_lens, double scale,
                               int64_t group) {
  TESSERACT_CHECK(q.defined() && k_pool.defined() && v_pool.defined() &&
                  block_tables.defined() && kv_lens.defined(),
                  "paged_prefill_attention: all operands must be defined");
  TESSERACT_CHECK(q.rank() == 4,
                  "paged_prefill_attention: q must be [A, S, H, D], got {}",
                  q.shape().to_string());
  TESSERACT_CHECK(k_pool.rank() == 4 && v_pool.rank() == 4,
                  "paged_prefill_attention: pools must be "
                  "[num_blocks, Hkv, block_size, D]");
  const DType dtype = q.dtype();
  TESSERACT_CHECK(dtype == DType::Float32 || dtype == DType::Float16 ||
                  dtype == DType::BFloat16,
                  "paged_prefill_attention: q dtype must be FP32/FP16/BF16, "
                  "got {}", dtype_name(dtype));
  TESSERACT_CHECK(k_pool.dtype() == dtype && v_pool.dtype() == dtype,
                  "paged_prefill_attention: pool dtype must match q");
  TESSERACT_CHECK(block_tables.dtype() == DType::Int32 &&
                  kv_lens.dtype() == DType::Int32,
                  "paged_prefill_attention: block_tables / kv_lens Int32");
  TESSERACT_CHECK(q.is_contiguous() && k_pool.is_contiguous() &&
                  v_pool.is_contiguous() && block_tables.is_contiguous() &&
                  kv_lens.is_contiguous(),
                  "paged_prefill_attention: all operands must be contiguous");
  TESSERACT_CHECK(q.device() == k_pool.device() && q.device() == v_pool.device() &&
                  q.device() == block_tables.device() &&
                  q.device() == kv_lens.device(),
                  "paged_prefill_attention: all operands must share a device");

  const int64_t A          = q.shape()[0];
  const int64_t S          = q.shape()[1];
  const int64_t H          = q.shape()[2];
  const int64_t D          = q.shape()[3];
  const int64_t num_blocks = k_pool.shape()[0];
  const int64_t Hkv        = k_pool.shape()[1];
  const int64_t block_size = k_pool.shape()[2];
  TESSERACT_CHECK(k_pool.shape()[3] == D && v_pool.shape()[3] == D,
                  "paged_prefill_attention: pool head_dim {} != q head_dim {}",
                  k_pool.shape()[3], D);
  TESSERACT_CHECK(v_pool.shape()[0] == num_blocks && v_pool.shape()[1] == Hkv &&
                  v_pool.shape()[2] == block_size,
                  "paged_prefill_attention: k/v pool shape mismatch");
  TESSERACT_CHECK(group >= 1 && H == Hkv * group,
                  "paged_prefill_attention: H ({}) must equal Hkv ({}) * group "
                  "({})", H, Hkv, group);
  TESSERACT_CHECK(D <= 128,
                  "paged_prefill_attention: head_dim {} exceeds kernel D_MAX=128",
                  D);
  TESSERACT_CHECK(block_tables.rank() == 2 && block_tables.shape()[0] == A,
                  "paged_prefill_attention: block_tables must be [A, max_logical]");
  TESSERACT_CHECK(kv_lens.rank() == 1 && kv_lens.shape()[0] == A,
                  "paged_prefill_attention: kv_lens must be [A]");
  const int64_t max_logical = block_tables.shape()[1];

  Tensor o = Tensor::empty(Shape({A, S, H, D}), dtype, q.device());
  if (A == 0 || S == 0 || H == 0 || D == 0) return o;

  const float scale_f = static_cast<float>(scale);

  if (q.device().is_cuda()) {
    Stream s = current_stream(q.device());
    cuda::detail::launch_paged_prefill_attention(
        dtype, q.device().index, A, S, H, Hkv, D, block_size, num_blocks,
        max_logical, static_cast<int>(group), scale_f, q.raw_data(),
        k_pool.raw_data(), v_pool.raw_data(),
        block_tables.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(),
        o.raw_data(), s.native_handle());
    return o;
  }

  const int32_t* table = block_tables.data_ptr<int32_t>();
  const int32_t* kvl   = kv_lens.data_ptr<int32_t>();
  dispatch_float_with_half(dtype, [&]<typename T>() {
    const T* qp = q.data_ptr<T>();
    const T* kp = k_pool.data_ptr<T>();
    const T* vp = v_pool.data_ptr<T>();
    T* op       = o.data_ptr<T>();
    std::vector<float> scores;
    for (int64_t r = 0; r < A; ++r) {
      const int64_t kv_len     = kvl[r];
      const int32_t* row_table = table + r * max_logical;
      for (int64_t sq = 0; sq < S; ++sq) {
        const int64_t len = kv_len - S + sq + 1;  // causal bound
        for (int64_t h = 0; h < H; ++h) {
          const int64_t hkv  = h / group;
          const T* q_row     = qp + ((r * S + sq) * H + h) * D;
          T* o_row           = op + ((r * S + sq) * H + h) * D;
          if (len <= 0) {
            for (int64_t d = 0; d < D; ++d) o_row[d] = static_cast<T>(0.0f);
            continue;
          }
          scores.assign(static_cast<std::size_t>(len), 0.0f);
          float m = -INFINITY;
          for (int64_t j = 0; j < len; ++j) {
            int64_t p = row_table[j / block_size];
            if (p < 0 || p >= num_blocks) p = 0;
            const int64_t slot = j % block_size;
            const T* k_row = kp + ((p * Hkv + hkv) * block_size + slot) * D;
            float acc = 0.0f;
            for (int64_t d = 0; d < D; ++d)
              acc += static_cast<float>(q_row[d]) * static_cast<float>(k_row[d]);
            acc *= scale_f;
            scores[static_cast<std::size_t>(j)] = acc;
            if (acc > m) m = acc;
          }
          float l = 0.0f;
          for (int64_t j = 0; j < len; ++j) {
            const float e = std::exp(scores[static_cast<std::size_t>(j)] - m);
            scores[static_cast<std::size_t>(j)] = e;
            l += e;
          }
          const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
          std::vector<float> out(static_cast<std::size_t>(D), 0.0f);
          for (int64_t j = 0; j < len; ++j) {
            int64_t p = row_table[j / block_size];
            if (p < 0 || p >= num_blocks) p = 0;
            const int64_t slot = j % block_size;
            const T* v_row = vp + ((p * Hkv + hkv) * block_size + slot) * D;
            const float pj = scores[static_cast<std::size_t>(j)];
            for (int64_t d = 0; d < D; ++d)
              out[static_cast<std::size_t>(d)] += pj * static_cast<float>(v_row[d]);
          }
          for (int64_t d = 0; d < D; ++d)
            o_row[d] = static_cast<T>(out[static_cast<std::size_t>(d)] * inv_l);
        }
      }
    }
  });
  return o;
}

Tensor paged_prefill_attention_int8(const Tensor& q, const Tensor& k_pool,
                                    const Tensor& k_scale, const Tensor& v_pool,
                                    const Tensor& v_scale,
                                    const Tensor& block_tables,
                                    const Tensor& kv_lens, double scale,
                                    int64_t group) {
  TESSERACT_CHECK(q.defined() && k_pool.defined() && k_scale.defined() &&
                  v_pool.defined() && v_scale.defined() &&
                  block_tables.defined() && kv_lens.defined(),
                  "paged_prefill_attention_int8: all operands must be defined");
  TESSERACT_CHECK(q.rank() == 4,
                  "paged_prefill_attention_int8: q must be [A, S, H, D], got {}",
                  q.shape().to_string());
  TESSERACT_CHECK(k_pool.rank() == 4 && v_pool.rank() == 4,
                  "paged_prefill_attention_int8: pools must be "
                  "[num_blocks, Hkv, block_size, D]");
  const DType dtype = q.dtype();
  TESSERACT_CHECK(dtype == DType::Float32 || dtype == DType::Float16 ||
                  dtype == DType::BFloat16,
                  "paged_prefill_attention_int8: q dtype FP32/FP16/BF16, got {}",
                  dtype_name(dtype));
  TESSERACT_CHECK(k_pool.dtype() == DType::Int8 && v_pool.dtype() == DType::Int8,
                  "paged_prefill_attention_int8: K/V pools must be Int8");
  TESSERACT_CHECK(k_scale.dtype() == DType::Float32 &&
                  v_scale.dtype() == DType::Float32,
                  "paged_prefill_attention_int8: scales must be Float32");
  TESSERACT_CHECK(block_tables.dtype() == DType::Int32 &&
                  kv_lens.dtype() == DType::Int32,
                  "paged_prefill_attention_int8: block_tables / kv_lens Int32");
  TESSERACT_CHECK(q.is_contiguous() && k_pool.is_contiguous() &&
                  k_scale.is_contiguous() && v_pool.is_contiguous() &&
                  v_scale.is_contiguous() && block_tables.is_contiguous() &&
                  kv_lens.is_contiguous(),
                  "paged_prefill_attention_int8: all operands contiguous");
  TESSERACT_CHECK(q.device() == k_pool.device() && q.device() == k_scale.device() &&
                  q.device() == v_pool.device() && q.device() == v_scale.device() &&
                  q.device() == block_tables.device() &&
                  q.device() == kv_lens.device(),
                  "paged_prefill_attention_int8: all operands share a device");

  const int64_t A          = q.shape()[0];
  const int64_t S          = q.shape()[1];
  const int64_t H          = q.shape()[2];
  const int64_t D          = q.shape()[3];
  const int64_t num_blocks = k_pool.shape()[0];
  const int64_t Hkv        = k_pool.shape()[1];
  const int64_t block_size = k_pool.shape()[2];
  TESSERACT_CHECK(k_pool.shape()[3] == D && v_pool.shape()[3] == D,
                  "paged_prefill_attention_int8: pool head_dim {} != q head_dim {}",
                  k_pool.shape()[3], D);
  TESSERACT_CHECK(v_pool.shape()[0] == num_blocks && v_pool.shape()[1] == Hkv &&
                  v_pool.shape()[2] == block_size,
                  "paged_prefill_attention_int8: k/v pool shape mismatch");
  const int64_t scale_numel = num_blocks * Hkv * block_size;
  TESSERACT_CHECK(k_scale.numel() == scale_numel && v_scale.numel() == scale_numel,
                  "paged_prefill_attention_int8: scale numel {} / {} != "
                  "num_blocks*Hkv*block_size ({})", k_scale.numel(),
                  v_scale.numel(), scale_numel);
  TESSERACT_CHECK(group >= 1 && H == Hkv * group,
                  "paged_prefill_attention_int8: H ({}) must equal Hkv ({}) * "
                  "group ({})", H, Hkv, group);
  TESSERACT_CHECK(D <= 128,
                  "paged_prefill_attention_int8: head_dim {} exceeds D_MAX=128", D);
  TESSERACT_CHECK(block_tables.rank() == 2 && block_tables.shape()[0] == A,
                  "paged_prefill_attention_int8: block_tables [A, max_logical]");
  TESSERACT_CHECK(kv_lens.rank() == 1 && kv_lens.shape()[0] == A,
                  "paged_prefill_attention_int8: kv_lens must be [A]");
  const int64_t max_logical = block_tables.shape()[1];

  Tensor o = Tensor::empty(Shape({A, S, H, D}), dtype, q.device());
  if (A == 0 || S == 0 || H == 0 || D == 0) return o;

  const float scale_f = static_cast<float>(scale);

  if (q.device().is_cuda()) {
    Stream s = current_stream(q.device());
    cuda::detail::launch_paged_prefill_attention_int8(
        dtype, q.device().index, A, S, H, Hkv, D, block_size, num_blocks,
        max_logical, static_cast<int>(group), scale_f, q.raw_data(),
        k_pool.data_ptr<int8_t>(), k_scale.data_ptr<float>(),
        v_pool.data_ptr<int8_t>(), v_scale.data_ptr<float>(),
        block_tables.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(),
        o.raw_data(), s.native_handle());
    return o;
  }

  const int8_t* kq = k_pool.data_ptr<int8_t>();
  const int8_t* vq = v_pool.data_ptr<int8_t>();
  const float*  ksc = k_scale.data_ptr<float>();
  const float*  vsc = v_scale.data_ptr<float>();
  const int32_t* table = block_tables.data_ptr<int32_t>();
  const int32_t* kvl   = kv_lens.data_ptr<int32_t>();
  dispatch_float_with_half(dtype, [&]<typename T>() {
    const T* qp = q.data_ptr<T>();
    T* op       = o.data_ptr<T>();
    std::vector<float> scores;
    for (int64_t r = 0; r < A; ++r) {
      const int64_t kv_len     = kvl[r];
      const int32_t* row_table = table + r * max_logical;
      for (int64_t sq = 0; sq < S; ++sq) {
        const int64_t len = kv_len - S + sq + 1;
        for (int64_t h = 0; h < H; ++h) {
          const int64_t hkv  = h / group;
          const T* q_row     = qp + ((r * S + sq) * H + h) * D;
          T* o_row           = op + ((r * S + sq) * H + h) * D;
          if (len <= 0) {
            for (int64_t d = 0; d < D; ++d) o_row[d] = static_cast<T>(0.0f);
            continue;
          }
          scores.assign(static_cast<std::size_t>(len), 0.0f);
          float m = -INFINITY;
          for (int64_t j = 0; j < len; ++j) {
            int64_t p = row_table[j / block_size];
            if (p < 0 || p >= num_blocks) p = 0;
            const int64_t slot = j % block_size;
            const int64_t hbs  = (p * Hkv + hkv) * block_size + slot;
            const int8_t* k_row = kq + hbs * D;
            const float ks = ksc[hbs];
            float acc = 0.0f;
            for (int64_t d = 0; d < D; ++d)
              acc += static_cast<float>(q_row[d]) *
                     (static_cast<float>(k_row[d]) * ks);
            acc *= scale_f;
            scores[static_cast<std::size_t>(j)] = acc;
            if (acc > m) m = acc;
          }
          float l = 0.0f;
          for (int64_t j = 0; j < len; ++j) {
            const float e = std::exp(scores[static_cast<std::size_t>(j)] - m);
            scores[static_cast<std::size_t>(j)] = e;
            l += e;
          }
          const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
          std::vector<float> out(static_cast<std::size_t>(D), 0.0f);
          for (int64_t j = 0; j < len; ++j) {
            int64_t p = row_table[j / block_size];
            if (p < 0 || p >= num_blocks) p = 0;
            const int64_t slot = j % block_size;
            const int64_t hbs  = (p * Hkv + hkv) * block_size + slot;
            const int8_t* v_row = vq + hbs * D;
            const float vs = vsc[hbs];
            const float pj = scores[static_cast<std::size_t>(j)];
            for (int64_t d = 0; d < D; ++d)
              out[static_cast<std::size_t>(d)] +=
                  pj * (static_cast<float>(v_row[d]) * vs);
          }
          for (int64_t d = 0; d < D; ++d)
            o_row[d] = static_cast<T>(out[static_cast<std::size_t>(d)] * inv_l);
        }
      }
    }
  });
  return o;
}

}  // namespace tesseract::nn

#include "tesseract/nn/RotaryEmbedding.hpp"

#include <cmath>
#include <vector>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/ops/RotaryEmbedding.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

namespace {

// Precompute the `[max_seq, d_head]` cos/sin tables on the host. The
// module constructor always allocates on CPU first; `Module::to()`
// migrates both buffers alongside the Linear weights of the
// enclosing attention block, which is why we don't take a `Device`
// argument here.
//
// Convention: adjacent-pair rotation. For each position `p` and each
// feature pair index `j ∈ [0, d_head/2)`, we set
//
//     cos[p, 2j]   = cos[p, 2j+1]   = cos(p · θⱼ)
//     sin[p, 2j]   = sin[p, 2j+1]   = sin(p · θⱼ)
//     θⱼ           = base^(-2j/d_head)
//
// The duplication across the pair is what lets the kernel stay a
// pure elementwise multiply-add per output slot (see
// `src/cuda/RotaryEmbedding.cu` for the math). Tables are built in
// FP64 regardless of the stored dtype so FP16 / BF16 tables start
// from the same high-precision source — the final down-cast on store
// absorbs at most one ulp of rounding per entry, which is well
// inside the half-precision parity envelope (~2e-3).
void fill_rope_tables(Tensor& cos_t, Tensor& sin_t,
                      int64_t max_seq, int64_t d_head, double base) {
  TESSERACT_CHECK(cos_t.device().is_cpu() && sin_t.device().is_cpu(),
                  "fill_rope_tables: tables must start on CPU");

  const int64_t half = d_head / 2;
  std::vector<double> inv_freq(half);
  const double d = static_cast<double>(d_head);
  for (int64_t j = 0; j < half; ++j) {
    inv_freq[j] = std::pow(base, -(2.0 * static_cast<double>(j)) / d);
  }

  dispatch_float_with_half(cos_t.dtype(), [&]<typename T>() {
    T* pc = cos_t.data_ptr<T>();
    T* ps = sin_t.data_ptr<T>();
    for (int64_t p = 0; p < max_seq; ++p) {
      for (int64_t j = 0; j < half; ++j) {
        const double angle = static_cast<double>(p) * inv_freq[j];
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        pc[p * d_head + 2 * j]     = static_cast<T>(c);
        pc[p * d_head + 2 * j + 1] = static_cast<T>(c);
        ps[p * d_head + 2 * j]     = static_cast<T>(s);
        ps[p * d_head + 2 * j + 1] = static_cast<T>(s);
      }
    }
  });
}

}  // namespace

RotaryEmbedding::RotaryEmbedding(int64_t d_head, double base,
                                 int64_t max_seq, DType dtype)
    : d_head_(d_head), max_seq_(max_seq), base_(base) {
  TESSERACT_CHECK(d_head > 0 && d_head % 2 == 0,
                  "RotaryEmbedding: d_head must be a positive even integer "
                  "(got {})", d_head);
  TESSERACT_CHECK(max_seq > 0,
                  "RotaryEmbedding: max_seq must be > 0 (got {})", max_seq);
  TESSERACT_CHECK(base > 0.0,
                  "RotaryEmbedding: base must be > 0 (got {})", base);
  TESSERACT_CHECK(dtype_is_floating(dtype),
                  "RotaryEmbedding: dtype must be floating-point, got {}",
                  dtype_name(dtype));

  cos_ = Tensor::empty({max_seq, d_head}, dtype);
  sin_ = Tensor::empty({max_seq, d_head}, dtype);
  fill_rope_tables(cos_, sin_, max_seq, d_head, base);
  register_buffer("cos", cos_);
  register_buffer("sin", sin_);
}

Tensor RotaryEmbedding::forward(const Tensor& x) {
  TESSERACT_CHECK(x.rank() >= 2,
                  "RotaryEmbedding::forward: expected rank >= 2 "
                  "[..., S, d_head], got {}", x.shape().to_string());
  const int64_t r = x.rank();
  const int64_t S = x.shape()[r - 2];
  const int64_t D = x.shape()[r - 1];
  TESSERACT_CHECK(D == d_head_,
                  "RotaryEmbedding::forward: last dim {} does not match "
                  "d_head {}", D, d_head_);
  TESSERACT_CHECK(S <= max_seq_,
                  "RotaryEmbedding::forward: S={} exceeds cached max_seq={}. "
                  "Rebuild the module with a larger max_seq.", S, max_seq_);

  // `ops::rotary_embedding` accepts `cos.shape()[0] >= S`, so we hand
  // it the full cached table without slicing. That keeps this path
  // allocation-free on every forward (no view tensor, no slab) and
  // matches the Llama inference pattern where the table is computed
  // once at model load and reused across every decode step.
  return ::tesseract::ops::rotary_embedding(x, cos_, sin_);
}

Tensor RotaryEmbedding::forward_offset(const Tensor& x, int64_t pos_offset) {
  TESSERACT_CHECK(x.rank() >= 2,
                  "RotaryEmbedding::forward_offset: expected rank >= 2 "
                  "[..., S, d_head], got {}", x.shape().to_string());
  const int64_t r = x.rank();
  const int64_t S = x.shape()[r - 2];
  const int64_t D = x.shape()[r - 1];
  TESSERACT_CHECK(D == d_head_,
                  "RotaryEmbedding::forward_offset: last dim {} != d_head {}",
                  D, d_head_);
  TESSERACT_CHECK(pos_offset >= 0,
                  "RotaryEmbedding::forward_offset: pos_offset must be >= 0 "
                  "(got {})", pos_offset);
  TESSERACT_CHECK(pos_offset + S <= max_seq_,
                  "RotaryEmbedding::forward_offset: pos_offset ({}) + S ({}) "
                  "exceeds cached max_seq ({}). Rebuild the module with a "
                  "larger max_seq.", pos_offset, S, max_seq_);

  // Narrow the `[max_seq, d_head]` tables down to the
  // `[S, d_head]` window starting at `pos_offset`. `ops::rotary_embedding`
  // already accepts `cos.shape()[0] >= S`, so we could pass the full
  // cached table with the positions encoded in `S`, but that would be
  // wrong — the kernel always uses row `0..S-1` of the table. Slicing
  // the table via `narrow` is the correct positional-offset
  // mechanism. The view is allocation-free (shares storage with the
  // registered buffer) and the underlying `ops::rotary_embedding`
  // treats its cos/sin inputs as row-major `[S_table, D]`, which a
  // dim-0 `narrow` preserves (stride along the first dim is unchanged).
  Tensor cos_win = cos_.narrow(/*dim=*/0, pos_offset, S);
  Tensor sin_win = sin_.narrow(/*dim=*/0, pos_offset, S);
  return ::tesseract::ops::rotary_embedding(x, cos_win, sin_win);
}

}  // namespace tesseract::nn

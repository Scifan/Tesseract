#include "tesseract/nn/Embedding.hpp"

#include <cmath>
#include <cstdint>

#include "tesseract/core/Dispatch.hpp"
#include "tesseract/ops/Indexing.hpp"
#include "tesseract/ops/View.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

namespace {

// Same xorshift-style deterministic-but-varying RNG Linear uses so
// successive Embeddings get distinct initial weights without explicit
// per-layer seeding.
thread_local uint64_t g_embedding_seed = 0xA5A5A5A53C3C3C3CULL;

void normal_init(Tensor& t, double stddev) {
  uint64_t s = (g_embedding_seed ^= (g_embedding_seed << 13));
  s ^= s >> 7;
  s ^= s << 17;
  g_embedding_seed = s;

  const int64_t n = t.numel();
  // `_with_half` so FP16 / BF16 models can be constructed directly. The
  // Box-Muller math stays in fp64 and only the final store down-casts.
  dispatch_float_with_half(t.dtype(), [&]<typename T>() {
    T* p = t.data_ptr<T>();
    // Box-Muller: pairs of uniform → pairs of normal.
    for (int64_t i = 0; i < n; i += 2) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      const double u1 =
          static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0;
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      const double u2 =
          static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0;
      const double r  = std::sqrt(-2.0 * std::log(std::max(u1, 1e-300)));
      const double z0 = r * std::cos(6.283185307179586 * u2);
      const double z1 = r * std::sin(6.283185307179586 * u2);
      p[i] = static_cast<T>(z0 * stddev);
      if (i + 1 < n) p[i + 1] = static_cast<T>(z1 * stddev);
    }
  });
}

}  // namespace

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, DType dtype)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
  TESSERACT_CHECK(num_embeddings > 0 && embedding_dim > 0,
                  "Embedding: num_embeddings ({}) and embedding_dim ({}) must be positive",
                  num_embeddings, embedding_dim);

  // PyTorch default: N(0, 1). Training typically scales this inside the
  // model (e.g. by 1/sqrt(d_model) for GPT-style), but keeping the bare
  // N(0, 1) here matches nn.Embedding() with no kwargs.
  weight_ = Tensor::empty({num_embeddings, embedding_dim}, dtype);
  normal_init(weight_, /*stddev=*/1.0);
  register_parameter("weight", weight_);
}

Tensor Embedding::forward(const Tensor& indices) {
  TESSERACT_CHECK(indices.defined(), "Embedding::forward: indices undefined");
  TESSERACT_CHECK(indices.dtype() == DType::Int64,
                  "Embedding::forward: indices must be Int64, got {}",
                  dtype_name(indices.dtype()));
  TESSERACT_CHECK(indices.rank() >= 1,
                  "Embedding::forward: expected rank >= 1 indices, got scalar");

  // ops::index_select wants a rank-1 Int64 index tensor. Flatten, look up,
  // then reshape back to [..., embedding_dim]. reshape is autograd-aware
  // and routes through contiguous() when needed — the backward path for
  // index_select already sums duplicate-row gradients into the weight.
  const Shape idx_shape = indices.shape();
  const int64_t numel = indices.numel();

  Tensor flat_idx = indices.is_contiguous() ? indices.view(Shape({numel}))
                                            : indices.contiguous().view(Shape({numel}));
  Tensor flat_out = ops::index_select(weight_, /*dim=*/0, flat_idx);  // [numel, D]

  Shape out_shape;
  for (std::size_t i = 0; i < idx_shape.rank(); ++i) out_shape.push_back(idx_shape[i]);
  out_shape.push_back(embedding_dim_);
  return ops::reshape(flat_out, out_shape);
}

}  // namespace tesseract::nn

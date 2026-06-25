#pragma once

#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// Dense lookup table for integer indices → dense vectors, the standard
// `nn.Embedding` primitive found at the input of every LLM. The learnable
// state is a `[num_embeddings, embedding_dim]` matrix stored as a regular
// leaf parameter so optimizers update it in place.
//
//   forward(indices: Int64, shape [...]) -> Tensor shape [..., embedding_dim]
//
// Implementation: flatten `indices` to rank-1, call `ops::index_select` on
// the weight along `dim=0`, then reshape back. `ops::index_select`'s
// backward is a scatter-add into the row axis of the weight, so the
// gradient update is the correct "sum-of-gradients-per-row" semantics
// (matching PyTorch's nn.Embedding, not `sparse_grad=True` — we always
// produce a dense gradient here, which is what transformer training uses).
//
// Weight-tying convention (Llama): callers with tied embeddings should
// load `embed_tokens.weight` into the shared storage and then have the
// loader copy the same bytes into the LM head's Linear weight after
// computing the transpose. We don't expose a public "tie" primitive
// because the loader owns the HF-name → parameter mapping and is the
// only path that needs it.
class Embedding : public Module {
 public:
  Embedding(int64_t num_embeddings, int64_t embedding_dim,
            DType dtype = DType::Float32);

  Tensor forward(const Tensor& indices) override;

  const Tensor& weight() const { return weight_; }
  int64_t num_embeddings() const noexcept { return num_embeddings_; }
  int64_t embedding_dim() const noexcept { return embedding_dim_; }

 private:
  int64_t num_embeddings_;
  int64_t embedding_dim_;
  Tensor weight_;  // [num_embeddings, embedding_dim]
};

}  // namespace tesseract::nn

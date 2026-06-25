#pragma once

#include "tesseract/nn/Module.hpp"

namespace tesseract::nn {

// Wave 2b (B-020) — per-channel BatchNorm modules. Shared state layout
// matches `torch.nn.BatchNorm{1d,2d}` exactly:
//
//   parameters (when affine=true):
//     weight : [C]  — scale, init 1
//     bias   : [C]  — shift, init 0
//   buffers (always, when track_running_stats=true):
//     running_mean : [C] — init 0
//     running_var  : [C] — init 1
//
// `forward(x)` delegates to `ops::batch_norm(x, weight, bias,
// running_mean, running_var, /*training=*/is_training(), momentum, eps)`,
// which performs the in-place running-stats writeback under a
// `NoGradGuard` when the module is in training mode and broadcasts the
// running stats back out for normalization when it is in eval mode. The
// `is_training()` flag comes from `Module::train(bool)` and is
// propagated recursively by the base class so `model->eval()` flips
// every BN child at once.
//
// `Module::to(Device)` is inherited unchanged; it migrates `weight`,
// `bias`, `running_mean`, and `running_var` in lockstep via
// `Tensor::move_to_`, so calling `model->to(cuda)` puts the whole BN
// state on-device without any per-subclass override.
class BatchNorm1d : public Module {
 public:
  // BatchNorm1d accepts either `[N, C]` (fully-connected) or
  // `[N, C, L]` (conv1d feature map) inputs. The spec follows
  // `torch.nn.BatchNorm1d(num_features=C, eps, momentum, affine,
  // track_running_stats=true)`. `affine=false` drops both `weight`
  // and `bias` entirely (no learnable scale/shift) — matches PyTorch.
  explicit BatchNorm1d(int64_t num_features, double eps = 1e-5,
                       double momentum = 0.1, bool affine = true,
                       DType dtype = DType::Float32);

  Tensor forward(const Tensor& x) override;

  const Tensor& weight() const { return weight_; }
  const Tensor& bias() const { return bias_; }
  const Tensor& running_mean() const { return running_mean_; }
  const Tensor& running_var()  const { return running_var_;  }
  int64_t num_features() const noexcept { return num_features_; }
  double eps() const noexcept { return eps_; }
  double momentum() const noexcept { return momentum_; }
  bool affine() const noexcept { return affine_; }

 private:
  int64_t num_features_;
  double  eps_;
  double  momentum_;
  bool    affine_;
  Tensor  weight_;        // [C] (undefined when affine=false)
  Tensor  bias_;          // [C] (undefined when affine=false)
  Tensor  running_mean_;  // [C]
  Tensor  running_var_;   // [C]
};

// BatchNorm2d — same semantics, rank-4 `[N, C, H, W]` inputs. The
// underlying op handles both BN1d and BN2d dispatch purely off the
// input rank, so this subclass is just a convenience wrapper that
// communicates intent to the reader and validates the forward shape.
class BatchNorm2d : public Module {
 public:
  explicit BatchNorm2d(int64_t num_features, double eps = 1e-5,
                       double momentum = 0.1, bool affine = true,
                       DType dtype = DType::Float32);

  Tensor forward(const Tensor& x) override;

  const Tensor& weight() const { return weight_; }
  const Tensor& bias() const { return bias_; }
  const Tensor& running_mean() const { return running_mean_; }
  const Tensor& running_var()  const { return running_var_;  }
  int64_t num_features() const noexcept { return num_features_; }
  double eps() const noexcept { return eps_; }
  double momentum() const noexcept { return momentum_; }
  bool affine() const noexcept { return affine_; }

 private:
  int64_t num_features_;
  double  eps_;
  double  momentum_;
  bool    affine_;
  Tensor  weight_;
  Tensor  bias_;
  Tensor  running_mean_;
  Tensor  running_var_;
};

}  // namespace tesseract::nn

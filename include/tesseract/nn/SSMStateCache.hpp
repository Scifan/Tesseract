#pragma once

#include "tesseract/core/DType.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::nn {

// Decode-time recurrent state for one Mamba layer (M4 Track A2 / B-039).
//
// Where an attention layer's `nn::KVCacheBase` grows an O(t) K/V prefix, a
// selective-SSM layer keeps a *fixed-size* state, which is the whole point of
// the architecture: O(1) memory per step regardless of context length. Two
// pieces:
//   * `conv_state` [B, d_conv-1, d_inner] — the last `d_conv-1` inputs feeding
//     the causal depthwise conv1d, so a single new token can be convolved
//     without re-seeing the prefix.
//   * `ssm_state`  [B, d_inner, d_state]  — the SSM hidden state `h` threaded
//     through `ops::selective_scan`.
//
// Both start at zero, matching a prefill that begins from an empty context —
// which is what makes step-by-step decode reproduce a full-sequence forward.
class SSMStateCache {
 public:
  SSMStateCache(int64_t batch, int64_t d_inner, int64_t d_state,
                int64_t d_conv, DType dtype = DType::Float32,
                Device device = cpu_device())
      : batch_(batch), d_inner_(d_inner), d_state_(d_state), d_conv_(d_conv) {
    TESSERACT_CHECK(batch > 0 && d_inner > 0 && d_state > 0 && d_conv >= 1,
                    "SSMStateCache: batch/d_inner/d_state must be > 0 and "
                    "d_conv >= 1");
    conv_state_ = Tensor::zeros(Shape({batch, d_conv - 1, d_inner}), dtype,
                                device);
    ssm_state_  = Tensor::zeros(Shape({batch, d_inner, d_state}), dtype, device);
  }

  const Tensor& conv_state() const noexcept { return conv_state_; }
  const Tensor& ssm_state() const noexcept { return ssm_state_; }

  void set_conv_state(Tensor t) { conv_state_ = std::move(t); }
  void set_ssm_state(Tensor t) { ssm_state_ = std::move(t); }

  void reset() {
    conv_state_.fill_(0.0);
    ssm_state_.fill_(0.0);
  }

  int64_t batch() const noexcept { return batch_; }
  int64_t d_inner() const noexcept { return d_inner_; }
  int64_t d_state() const noexcept { return d_state_; }
  int64_t d_conv() const noexcept { return d_conv_; }

 private:
  int64_t batch_;
  int64_t d_inner_;
  int64_t d_state_;
  int64_t d_conv_;
  Tensor conv_state_;
  Tensor ssm_state_;
};

}  // namespace tesseract::nn

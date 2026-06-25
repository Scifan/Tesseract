#include "tesseract/models/Sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "tesseract/utils/Logging.hpp"

namespace tesseract::models {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// argmax over a logits row; ties resolve to the lowest index (matches the
// `dispatch`-based greedy path in LlamaModel::generate).
int32_t argmax(std::span<const float> v) {
  int32_t best = 0;
  float best_val = kNegInf;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i] > best_val) { best_val = v[i]; best = static_cast<int32_t>(i); }
  }
  return best;
}

}  // namespace

int32_t sample_from_logits(std::span<const float> logits,
                           const SamplingParams& params,
                           std::span<const int32_t> prev_tokens,
                           std::mt19937_64& rng) {
  const int64_t V = static_cast<int64_t>(logits.size());
  TESSERACT_CHECK(V > 0, "sample_from_logits: empty logits");

  std::vector<float> work(logits.begin(), logits.end());

  // 1) Repetition penalty (CTRL-style). Applied once per distinct prior id.
  if (params.repetition_penalty != 1.0 && !prev_tokens.empty()) {
    const float pen = static_cast<float>(params.repetition_penalty);
    for (int32_t t : prev_tokens) {
      if (t < 0 || t >= V) continue;
      float& x = work[static_cast<std::size_t>(t)];
      x = (x > 0.0f) ? (x / pen) : (x * pen);
    }
  }

  // 2) Temperature. T <= 0 ⇒ greedy, which also bypasses top-k/top-p.
  if (params.temperature <= 1e-6) {
    return argmax(work);
  }
  const float inv_t = static_cast<float>(1.0 / params.temperature);
  for (float& x : work) x *= inv_t;

  // 3) Top-k: keep the k highest logits, set the rest to -inf.
  if (params.top_k > 0 && params.top_k < V) {
    std::vector<int32_t> idx(static_cast<std::size_t>(V));
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + (params.top_k - 1), idx.end(),
                     [&](int32_t a, int32_t b) { return work[a] > work[b]; });
    const float kth = work[idx[static_cast<std::size_t>(params.top_k - 1)]];
    // Demote everything strictly below the k-th largest. (Ties at the
    // boundary are kept — matches HF's `>= kth` survivor set.)
    for (float& x : work) {
      if (x < kth) x = kNegInf;
    }
  }

  // 4) Top-p (nucleus): keep the smallest high-probability prefix whose
  //    cumulative softmax mass reaches p; demote the rest.
  if (params.top_p < 1.0) {
    std::vector<int32_t> order(static_cast<std::size_t>(V));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int32_t a, int32_t b) { return work[a] > work[b]; });

    // Softmax over the (post-top-k) logits to get cumulative mass.
    float maxv = kNegInf;
    for (int32_t i : order) maxv = std::max(maxv, work[i]);
    double denom = 0.0;
    for (int32_t i : order) {
      if (work[i] == kNegInf) continue;
      denom += std::exp(static_cast<double>(work[i] - maxv));
    }
    double cum = 0.0;
    bool reached = false;
    for (std::size_t r = 0; r < order.size(); ++r) {
      const int32_t i = order[r];
      if (work[i] == kNegInf) { break; }
      if (reached) { work[i] = kNegInf; continue; }
      cum += std::exp(static_cast<double>(work[i] - maxv)) / denom;
      // Always keep the first token even if it alone exceeds p.
      if (cum >= params.top_p) reached = true;
    }
  }

  // 5) Softmax over survivors, then multinomial draw.
  float maxv = kNegInf;
  for (float x : work) maxv = std::max(maxv, x);
  double denom = 0.0;
  for (float x : work) {
    if (x == kNegInf) continue;
    denom += std::exp(static_cast<double>(x - maxv));
  }
  TESSERACT_CHECK(denom > 0.0, "sample_from_logits: degenerate distribution");

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  double r = unif(rng) * denom;  // scale into the unnormalized cumulative space
  for (std::size_t i = 0; i < work.size(); ++i) {
    if (work[i] == kNegInf) continue;
    r -= std::exp(static_cast<double>(work[i] - maxv));
    if (r <= 0.0) return static_cast<int32_t>(i);
  }
  // Floating-point slack: fall back to the highest-logit survivor.
  return argmax(work);
}

}  // namespace tesseract::models

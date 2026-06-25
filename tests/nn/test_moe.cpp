// M4 Track A1 (B-038) — Mixture-of-Experts feed-forward.
//
// MoE correctness rests on the gating math, which we pin with three
// reference equalities that are exact by construction:
//   1. num_experts==1, top_k==1: gate is always 1.0 ⇒ MoE(x) == expert_0(x).
//   2. uniform router (zero router weight) + top_k==E: gates are 1/E each ⇒
//      MoE(x) == mean over experts of expert_e(x).
//   3. top_k==1 with a real (random) router: MoE(x)[t] == expert_{argmax_e
//      logit[t,e]}(x)[t] — proves routing + selection + gate==1.
// Plus a Llama-level MoE generate determinism + run check, and validation.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Shape.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/models/Llama.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/MoEFeedForward.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Reduction.hpp"

using namespace tesseract;

namespace {

Tensor random_input(int64_t T, int64_t D, uint64_t seed) {
  // Deterministic pseudo-random fill (no autograd needed).
  std::vector<float> v(static_cast<std::size_t>(T * D));
  uint64_t s = seed;
  for (auto& x : v) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    x = static_cast<float>((s >> 33) % 2000) / 1000.0f - 1.0f;  // [-1, 1)
  }
  return Tensor::from_vector(v, Shape({T, D}));
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
  REQUIRE(a.numel() == b.numel());
  const float* pa = a.data_ptr<float>();
  const float* pb = b.data_ptr<float>();
  double m = 0.0;
  for (int64_t i = 0; i < a.numel(); ++i)
    m = std::max(m, std::abs(static_cast<double>(pa[i]) - pb[i]));
  return m;
}

}  // namespace

TEST_CASE("MoE single-expert top1 equals the lone expert") {
  NoGradGuard nogg;
  const int64_t T = 5, D = 8, dff = 16;
  nn::MoEFeedForward moe(D, dff, /*num_experts=*/1, /*num_experts_per_tok=*/1);
  Tensor x = random_input(T, D, 0x1111u);

  Tensor y_moe = moe.forward(x);
  Tensor y_exp = moe.experts()[0]->forward(x);
  REQUIRE(y_moe.shape() == Shape({T, D}));
  REQUIRE(max_abs_diff(y_moe, y_exp) < 1e-5);
}

TEST_CASE("MoE uniform router (top_k==E) equals the mean of experts") {
  NoGradGuard nogg;
  const int64_t T = 6, D = 8, dff = 16, E = 4;
  nn::MoEFeedForward moe(D, dff, /*num_experts=*/E, /*num_experts_per_tok=*/E);

  // Zero the router weight so every logit is 0 ⇒ softmax is uniform 1/E.
  auto router = std::dynamic_pointer_cast<nn::Linear>(moe.router());
  REQUIRE(router != nullptr);
  Tensor rw = router->weight();
  rw.fill_(0.0);

  Tensor x = random_input(T, D, 0x2222u);
  Tensor y_moe = moe.forward(x);

  // Reference: mean_e expert_e(x).
  Tensor ref;
  for (int64_t e = 0; e < E; ++e) {
    Tensor y_e = moe.experts()[static_cast<std::size_t>(e)]->forward(x);
    ref = ref.defined() ? ops::add(ref, y_e) : y_e;
  }
  ref = ops::div(ref, Tensor::full(Shape({}), static_cast<double>(E)));
  REQUIRE(max_abs_diff(y_moe, ref) < 1e-5);
}

TEST_CASE("MoE top1 routes each token to its argmax expert") {
  NoGradGuard nogg;
  const int64_t T = 7, D = 8, dff = 16, E = 3;
  nn::MoEFeedForward moe(D, dff, /*num_experts=*/E, /*num_experts_per_tok=*/1);
  Tensor x = random_input(T, D, 0x3333u);

  Tensor y_moe = moe.forward(x);

  // Reference: per-token argmax over the router logits, then that expert's row.
  Tensor logits = moe.router()->forward(x);  // [T, E]
  const float* lg = logits.data_ptr<float>();
  std::vector<Tensor> ye;
  ye.reserve(static_cast<std::size_t>(E));
  for (int64_t e = 0; e < E; ++e)
    ye.push_back(moe.experts()[static_cast<std::size_t>(e)]->forward(x));

  Tensor ref = Tensor::zeros(Shape({T, D}));
  float* rp = ref.data_ptr<float>();
  for (int64_t t = 0; t < T; ++t) {
    int64_t best = 0;
    float bestv = lg[t * E + 0];
    for (int64_t e = 1; e < E; ++e) {
      if (lg[t * E + e] > bestv) { bestv = lg[t * E + e]; best = e; }
    }
    const float* src = ye[static_cast<std::size_t>(best)].data_ptr<float>();
    for (int64_t d = 0; d < D; ++d) rp[t * D + d] = src[t * D + d];
  }
  REQUIRE(max_abs_diff(y_moe, ref) < 1e-5);
}

TEST_CASE("MoE top-2 sparse dispatch matches the dense gate-weighted sum") {
  // Locks the sparse permute/combine path for k>1: the output must equal the
  // dense reference (every selected expert's output, weighted by the renormed
  // top-k softmax gate, summed in ascending-expert order).
  NoGradGuard nogg;
  const int64_t T = 9, D = 8, dff = 16, E = 5, k = 2;
  nn::MoEFeedForward moe(D, dff, E, k);
  Tensor x = random_input(T, D, 0x5151u);

  Tensor y_moe = moe.forward(x);

  // Reference, computed independently of the module's dispatch.
  Tensor logits = moe.router()->forward(x);  // [T, E]
  const float* lg = logits.data_ptr<float>();
  std::vector<Tensor> ye;
  for (int64_t e = 0; e < E; ++e)
    ye.push_back(moe.experts()[static_cast<std::size_t>(e)]->forward(x));

  Tensor ref = Tensor::zeros(Shape({T, D}));
  float* rp = ref.data_ptr<float>();
  for (int64_t t = 0; t < T; ++t) {
    // softmax over the row
    std::vector<double> p(static_cast<std::size_t>(E));
    double mx = -1e30;
    for (int64_t e = 0; e < E; ++e) mx = std::max(mx, (double)lg[t * E + e]);
    double sum = 0.0;
    for (int64_t e = 0; e < E; ++e) {
      p[e] = std::exp((double)lg[t * E + e] - mx);
      sum += p[e];
    }
    for (int64_t e = 0; e < E; ++e) p[e] /= sum;
    // top-k indices (desc prob, tie index asc)
    std::vector<int64_t> idx(static_cast<std::size_t>(E));
    for (int64_t e = 0; e < E; ++e) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int64_t a, int64_t b) {
                        if (p[a] != p[b]) return p[a] > p[b];
                        return a < b;
                      });
    double denom = 0.0;
    for (int64_t j = 0; j < k; ++j) denom += p[idx[j]];
    std::vector<int64_t> sel(idx.begin(), idx.begin() + k);
    std::sort(sel.begin(), sel.end());  // ascending-expert accumulation order
    for (int64_t e : sel) {
      const double gate = p[e] / denom;
      const float* src = ye[static_cast<std::size_t>(e)].data_ptr<float>();
      for (int64_t d = 0; d < D; ++d)
        rp[t * D + d] += static_cast<float>(gate * src[t * D + d]);
    }
  }
  REQUIRE(max_abs_diff(y_moe, ref) < 1e-5);
}

TEST_CASE("MoE sparse dispatch keeps the router trainable (gradient flows)") {
  // The combine gathers gate values from the differentiable `gates` tensor, so
  // the router must receive a finite, non-zero gradient through the sparse path.
  const int64_t T = 6, D = 8, dff = 16, E = 4, k = 2;
  nn::MoEFeedForward moe(D, dff, E, k);
  auto router = std::dynamic_pointer_cast<nn::Linear>(moe.router());
  REQUIRE(router != nullptr);

  Tensor x = random_input(T, D, 0x6262u);
  Tensor y = moe.forward(x);
  Tensor loss = ops::sum(y);
  Engine::backward(loss);

  const Tensor& g = router->weight().grad();
  REQUIRE(g.defined());
  const float* gp = g.data_ptr<float>();
  double gabs = 0.0;
  for (int64_t i = 0; i < g.numel(); ++i)
    gabs += std::fabs(static_cast<double>(gp[i]));
  REQUIRE(std::isfinite(gabs));
  REQUIRE(gabs > 0.0);
}

TEST_CASE("MoE aux loss is a finite positive scalar") {
  NoGradGuard nogg;
  nn::MoEFeedForward moe(8, 16, /*num_experts=*/4, /*num_experts_per_tok=*/2);
  Tensor x = random_input(6, 8, 0x4444u);
  (void)moe.forward(x);
  const Tensor& aux = moe.last_aux_loss();
  REQUIRE(aux.defined());
  REQUIRE(aux.numel() == 1);
  const float v = aux.data_ptr<float>()[0];
  REQUIRE(std::isfinite(v));
  REQUIRE(v > 0.0f);
}

TEST_CASE("MoE Llama builds, generates deterministically") {
  models::LlamaConfig cfg;
  cfg.vocab_size = 32;
  cfg.hidden_size = 16;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 4;
  cfg.intermediate_size = 32;
  cfg.max_position_embeddings = 64;
  cfg.num_experts = 4;
  cfg.num_experts_per_tok = 2;

  auto model = std::make_shared<models::LlamaModel>(cfg);

  // The FFN slot of every block is the MoE variant.
  for (const auto& block : model->layers()) {
    REQUIRE(block->is_moe());
    REQUIRE(block->ffn() == nullptr);
    REQUIRE(block->moe_ffn() != nullptr);
  }

  std::vector<int32_t> prompt = {1, 5, 9, 3};
  models::LlamaModel::GenerateConfig gc;
  gc.max_new_tokens = 6;
  auto a = model->generate(prompt, gc);
  auto b = model->generate(prompt, gc);
  REQUIRE(a == b);
  REQUIRE(a.size() == prompt.size() + 6);
}

TEST_CASE("MoE config.json parses Mixtral-style experts") {
  const std::string json = R"({
    "vocab_size": 32000,
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "num_local_experts": 8,
    "num_experts_per_tok": 2
  })";
  models::LlamaConfig c = models::LlamaConfig::from_json(json);
  REQUIRE(c.num_experts == 8);
  REQUIRE(c.num_experts_per_tok == 2);
}

TEST_CASE("MoE validation rejects bad config") {
  REQUIRE_THROWS(nn::MoEFeedForward(8, 16, /*num_experts=*/0, /*top_k=*/1));
}

// Phase 4: the device top-k routing kernel must produce the exact same MoE
// output as the host routing path. We build one MoE, run it on CPU, then move
// the module + input to CUDA and compare. Sweep several (E, k) so the
// multi-round argmax tie-break and renormalization are exercised on device.
TEST_CASE("MoE CUDA device-routing matches CPU forward", "[nn][moe][cuda]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  NoGradGuard nogg;
  const Device cuda0{DeviceType::CUDA, 0};

  struct Cfg { int64_t E; int64_t k; };
  for (const Cfg cfg : {Cfg{8, 2}, Cfg{4, 1}, Cfg{16, 4}, Cfg{6, 6}}) {
    const int64_t T = 64, D = 32, dff = 64;
    nn::MoEFeedForward moe(D, dff, cfg.E, cfg.k);
    Tensor x = random_input(T, D, 0x9911u + static_cast<uint64_t>(cfg.E));

    Tensor y_cpu = moe.forward(x).contiguous();

    moe.to(cuda0);
    Tensor x_cuda = x.to(cuda0);
    Tensor y_gpu = moe.forward(x_cuda).to(cpu_device()).contiguous();

    INFO("E=" << cfg.E << " k=" << cfg.k);
    REQUIRE(y_gpu.shape() == y_cpu.shape());
    REQUIRE(max_abs_diff(y_gpu, y_cpu) < 1e-4);
  }
}

// M4 Phase 8 (B-009 tail) — GpuJitEngine vs eager numerical parity.
//
// Builds a data-parallel graph, compiles it to a cubin via
// gpu.module -> NVVM -> PTX -> cubin, launches the kernel on the GPU through
// the CUDA driver (GpuJitEngine), and asserts the result matches the eager
// CPU kernels bit-for-bit-close. Skips cleanly (no failure) when no CUDA
// device is visible, so CPU-only CI stays green.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "tesseract/core/Tensor.hpp"
#include "tesseract/graph/Graph.hpp"
#include "tesseract/graph/GraphScope.hpp"
#include "tesseract/ir/GpuJitEngine.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/Arithmetic.hpp"

using namespace tesseract;

namespace {

Tensor rand_tensor(std::initializer_list<int64_t> dims, uint64_t seed) {
  auto t = Tensor::empty(dims, DType::Float32);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
  return t;
}

void require_close(const Tensor& got, const Tensor& ref, double tol = 1e-5) {
  REQUIRE(got.numel() == ref.numel());
  const Tensor a = got.contiguous();
  const Tensor b = ref.contiguous();
  const float* pa = a.data_ptr<float>();
  const float* pb = b.data_ptr<float>();
  for (int64_t i = 0; i < a.numel(); ++i) {
    const double d = std::fabs(static_cast<double>(pa[i]) - pb[i]);
    INFO("elem " << i << " gpu=" << pa[i] << " eager=" << pb[i] << " |d|=" << d);
    REQUIRE(d <= tol);
  }
}

}  // namespace

TEST_CASE("GpuJitEngine: elementwise add matches eager CUDA/CPU",
          "[ir][gpujit][parity]") {
  if (!ir::GpuJitEngine::available()) {
    SKIP("no CUDA device visible — skipping GPU JIT parity");
  }

  Tensor a = rand_tensor({64, 64}, 1);
  Tensor b = rand_tensor({64, 64}, 2);

  graph::Graph g;
  graph::ValueId a_id = 0, b_id = 0;
  {
    graph::GraphScope scope;
    a_id = graph::bind_input(a, "a");
    b_id = graph::bind_input(b, "b");
    Tensor y = ops::add(a, b);
    graph::mark_output(y);
    g = std::move(scope.graph_);
  }

  ir::GpuJitEngine engine(g);
  std::unordered_map<graph::ValueId, Tensor> bind;
  bind.emplace(a_id, a);
  bind.emplace(b_id, b);
  std::vector<Tensor> outs = engine.invoke(bind);
  REQUIRE(outs.size() == 1);

  require_close(outs[0], ops::add(a, b));
}

TEST_CASE("GpuJitEngine: fused mul+relu matches eager",
          "[ir][gpujit][parity]") {
  if (!ir::GpuJitEngine::available()) {
    SKIP("no CUDA device visible — skipping GPU JIT parity");
  }

  Tensor a = rand_tensor({128, 32}, 3);
  Tensor b = rand_tensor({128, 32}, 4);

  graph::Graph g;
  graph::ValueId a_id = 0, b_id = 0;
  {
    graph::GraphScope scope;
    a_id = graph::bind_input(a, "a");
    b_id = graph::bind_input(b, "b");
    Tensor y = ops::relu(ops::mul(a, b));
    graph::mark_output(y);
    g = std::move(scope.graph_);
  }

  ir::GpuJitEngine engine(g);
  std::unordered_map<graph::ValueId, Tensor> bind;
  bind.emplace(a_id, a);
  bind.emplace(b_id, b);
  std::vector<Tensor> outs = engine.invoke(bind);
  REQUIRE(outs.size() == 1);

  require_close(outs[0], ops::relu(ops::mul(a, b)));
}

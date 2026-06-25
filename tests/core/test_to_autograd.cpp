// M4 Phase 7 — differentiable cross-device copy (CopyBackward).
//
// `Tensor::to(device)` is now autograd-aware: the backward of a cross-device
// copy routes the gradient back to the source device. This is the primitive
// that makes tensor/data-parallel graphs differentiable across GPUs (the
// RowParallelLinear all-reduce, sharded `.to()` moves, etc.). We verify the
// gradient flows back to the CPU leaf through a cpu->cuda->(compute)->loss
// graph and matches a pure-CPU reference.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "tesseract/autograd/Engine.hpp"
#include "tesseract/core/Device.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/cuda/CudaRuntime.hpp"
#include "tesseract/ops/Arithmetic.hpp"
#include "tesseract/ops/Reduction.hpp"

using namespace tesseract;

TEST_CASE("Tensor::to is differentiable across devices (CopyBackward)",
          "[core][autograd][to][cuda]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  const Device cuda0{DeviceType::CUDA, 0};

  // x (CPU leaf) -> y = x.to(cuda) -> z = (y*y) on GPU -> loss = sum(z).
  // d loss / d x = 2x. The gradient must travel GPU -> CPU through
  // CopyBackward and land on the CPU leaf's .grad.
  std::vector<float> xv = {-2.0f, 0.5f, 1.5f, 3.0f};
  Tensor x = Tensor::from_vector(xv, Shape({4}));
  x.set_requires_grad(true);

  Tensor y = x.to(cuda0);
  Tensor z = ops::mul(y, y);
  Tensor loss = ops::sum(z);
  Engine::backward(loss);

  REQUIRE(x.grad().defined());
  REQUIRE(x.grad().device().is_cpu());
  const float* g = x.grad().data_ptr<float>();
  for (int i = 0; i < 4; ++i) {
    REQUIRE_THAT(g[i], Catch::Matchers::WithinAbs(2.0f * xv[i], 1e-5f));
  }
}

TEST_CASE("Tensor::to backward survives a round trip cpu->cuda->cpu",
          "[core][autograd][to][cuda]") {
  if (cuda::device_count() <= 0) {
    SUCCEED("CUDA not available; skipping");
    return;
  }
  const Device cuda0{DeviceType::CUDA, 0};

  std::vector<float> xv = {1.0f, -1.0f, 2.0f};
  Tensor x = Tensor::from_vector(xv, Shape({3}));
  x.set_requires_grad(true);

  // cpu -> cuda -> compute -> cuda->cpu -> loss. Gradient crosses the device
  // boundary twice; both CopyBackward nodes must compose.
  Tensor y = x.to(cuda0);
  Tensor z = ops::add(y, y);          // 2x on GPU
  Tensor back = z.to(cpu_device());   // back on CPU
  Tensor loss = ops::sum(back);
  Engine::backward(loss);

  REQUIRE(x.grad().defined());
  REQUIRE(x.grad().device().is_cpu());
  const float* g = x.grad().data_ptr<float>();
  for (int i = 0; i < 3; ++i) {
    REQUIRE_THAT(g[i], Catch::Matchers::WithinAbs(2.0f, 1e-5f));
  }
}

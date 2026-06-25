// M4 Track B1 (B-041) — native side of the Python↔C++ overhead measurement.
//
// Times the exact same ops the Python frontend exposes, in pure C++, at a few
// sizes with a controlled iteration count. The matching Python script
// (`bench/external/python_overhead.py`) calls the identical ops through the
// pybind11 bindings with the same iteration counts. The per-call delta is the
// cost the Python frontend adds (interpreter loop + arg marshaling + return
// wrapping) on top of the shared C++ kernel.
//
// Output: machine-readable `RESULT,<name>,<us_per_call>` lines so the wrapper
// can diff them against the Python run.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

#include "tesseract/core/DType.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/nn/Activation.hpp"
#include "tesseract/nn/Linear.hpp"
#include "tesseract/nn/Sequential.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/MatMul.hpp"

using namespace tesseract;
using Clock = std::chrono::steady_clock;

namespace {

double us_per_call(int iters, const std::function<void()>& fn) {
  for (int i = 0; i < 10; ++i) fn();  // warmup
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) fn();
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

}  // namespace

int main() {
  NoGradGuard nogg;

  struct MM { int64_t n; int iters; };
  const MM mms[] = {{64, 5000}, {256, 2000}, {512, 1000}, {1024, 300}};
  for (const auto& c : mms) {
    Tensor a = Tensor::ones({c.n, c.n}, DType::Float32);
    Tensor b = Tensor::ones({c.n, c.n}, DType::Float32);
    const double us = us_per_call(c.iters, [&] { (void)ops::matmul(a, b); });
    std::printf("RESULT,matmul_%ld,%.4f\n", long(c.n), us);
  }

  // MLP forward through Sequential (mirrors ts.nn.Sequential in Python).
  {
    auto seq = std::make_shared<nn::Sequential>();
    seq->add(std::make_shared<nn::Linear>(512, 512));
    seq->add(std::make_shared<nn::ReLU>());
    seq->add(std::make_shared<nn::Linear>(512, 512));
    seq->eval();
    Tensor x = Tensor::ones({64, 512}, DType::Float32);
    const double us = us_per_call(1000, [&] { (void)seq->forward(x); });
    std::printf("RESULT,mlp_fwd_b64_d512,%.4f\n", us);
  }

  std::fflush(stdout);
  return 0;
}

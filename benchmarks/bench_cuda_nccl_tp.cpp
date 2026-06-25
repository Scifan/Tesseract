// M4 Phase 7 — REAL multi-process NCCL tensor-parallel scaling + parity.
//
// One process per GPU (fork-based, no MPI): the parent generates an
// ncclUniqueId, forks `world_size` children, each of which drives its own card
// via `NcclCommBackend`. Every rank holds only its 1/N shard of a Megatron
// SwiGLU MLP (gate/up column-parallel, down row-parallel) and the forward path
// uses exactly one collective — a `ncclAllReduce(sum)` on the down output.
//
// What it proves:
//   * per-GPU weight memory = dense / N   (each card stores 1/N of the FFN)
//   * real cross-GPU all-reduce throughput (NVLink/PCIe, not a host D2D sum)
//   * forward PARITY: the all-reduced output equals the single-GPU dense MLP
//     built from the same deterministic weights (rank 0 checks, exits nonzero
//     on mismatch so ctest gates on it).
//
// Output (rank 0):
//   [bench] tesseract nccl_tp  world=<N> cfg=<...> shard_MB=<..> dense_MB=<..>
//           fwd_ms=<..> allreduce_ms=<..>
//
// Deterministic weights: shard r's gate/up columns and down rows map to dense
// global index `r*p + j`; both the shard and rank 0's dense reference fill from
// the same `det()` so the comparison is exact up to FP summation order.

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "tesseract/core/Device.hpp"
#include "tesseract/core/DType.hpp"
#include "tesseract/core/GradMode.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/distributed/NcclCommBackend.hpp"
#include "tesseract/ops/Activation.hpp"
#include "tesseract/ops/MatMul.hpp"

using namespace tesseract;
using tesseract::distributed::NcclCommBackend;
namespace ops = tesseract::ops;
using Clock = std::chrono::steady_clock;

namespace {

// Deterministic pseudo-random unit in [0,1) for global (row, col, salt). Same
// on every process so shards and the dense reference agree exactly.
inline float det01(int64_t row, int64_t col, int64_t salt) {
  uint64_t h = static_cast<uint64_t>(row) * 1000003ull +
               static_cast<uint64_t>(col) * 9176ull +
               static_cast<uint64_t>(salt) * 26699ull;
  h ^= h >> 13; h *= 0x9E3779B97F4A7C15ull; h ^= h >> 7;
  return static_cast<float>(h & 0xFFFF) / 65535.0f;
}

// Fill a CPU [rows, cols] tensor where element (i,j) = scale * det01(...).
// `scale` is POSITIVE (no sign flip): the TP-vs-dense parity is a correctness
// gate, and zero-mean weights make the FFN output a near-zero residual of huge
// cancelling sums, where TF32's reduction-order roundoff (dense [M,d_ff] vs
// sharded partials) blows up the *relative* error and masks real bugs. With
// positive, well-conditioned weights the output is large and a genuine
// sharding bug still shows as rrms ~ O(1), far above TF32 noise (~1e-3).
Tensor make_weight(int64_t rows, int64_t cols, int64_t row_off, int64_t col_off,
                   int64_t salt) {
  Tensor t = Tensor::empty(Shape({rows, cols}), DType::Float32, cpu_device());
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j)
      p[i * cols + j] = det01(row_off + i, col_off + j, salt) * 0.02f;
  return t;
}

Tensor make_input(int64_t M, int64_t d) {
  Tensor t = Tensor::empty(Shape({M, d}), DType::Float32, cpu_device());
  float* p = t.data_ptr<float>();
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < d; ++j) p[i * d + j] = det01(i, j, 777);
  return t;
}

// Worker for one rank. Returns process exit code (0 ok, 1 parity fail).
int run_rank(int rank, int world, const std::string& idhex, int64_t M,
             int64_t d, int64_t d_ff, int iters) {
  const int64_t p = d_ff / world;  // shard width
  const Device dev{DeviceType::CUDA, rank};
  NcclCommBackend comm(rank, world, rank, idhex);

  NoGradGuard nogg;

  // --- all_gather correctness (column gather, dim=-1) --------------------
  // Each rank holds [rows, p] where element(i,j) = global column (rank*p + j).
  // After gather along the last dim every rank must hold [rows, world*p] with
  // element(i,g) == g. Rank 0 verifies; exits nonzero on mismatch.
  int gather_rc = 0;
  {
    const int64_t rows = 4;
    Tensor loc = Tensor::empty(Shape({rows, p}), DType::Float32, cpu_device());
    float* lp = loc.data_ptr<float>();
    for (int64_t i = 0; i < rows; ++i)
      for (int64_t j = 0; j < p; ++j)
        lp[i * p + j] = static_cast<float>(rank * p + j);
    Tensor g = comm.all_gather(loc.to(dev), -1).to(cpu_device());
    if (rank == 0) {
      const float* gp = g.data_ptr<float>();
      double maxe = 0.0;
      for (int64_t i = 0; i < rows; ++i)
        for (int64_t col = 0; col < world * p; ++col)
          maxe = std::max(maxe, std::fabs(gp[i * (world * p) + col] -
                                          static_cast<double>(col)));
      if (g.shape()[0] == rows && g.shape()[1] == world * p && maxe == 0.0) {
        std::printf("[PASS] all_gather(dim=-1) layout correct (world=%d, "
                    "shard_w=%lld -> full_w=%lld)\n",
                    world, (long long)p, (long long)(world * p));
      } else {
        std::printf("[FAIL] all_gather(dim=-1) mismatch maxerr=%.1f shape=[%lld,%lld]\n",
                    maxe, (long long)g.shape()[0], (long long)g.shape()[1]);
        gather_rc = 1;
      }
    }
  }
  comm.barrier();

  // Shard weights: gate/up are [d, p] (columns r*p..), down is [p, d] (rows r*p..).
  Tensor gate = make_weight(d, p, 0, rank * p, 1).to(dev);
  Tensor up   = make_weight(d, p, 0, rank * p, 2).to(dev);
  Tensor down = make_weight(p, d, rank * p, 0, 3).to(dev);
  Tensor x    = make_input(M, d).to(dev);

  auto forward_local = [&]() {
    Tensor g = ops::matmul(x, gate);                 // [M, p]
    Tensor u = ops::matmul(x, up);                   // [M, p]
    Tensor h = ops::swiglu_silu_gate(g, u);          // silu(g)*u  [M, p]
    return ops::matmul(h, down);                     // [M, d] partial
  };

  // Warm-up + NCCL channel setup.
  for (int w = 0; w < 3; ++w) {
    Tensor y = forward_local();
    (void)comm.all_reduce_sum(y);
  }
  comm.barrier();

  // Timed: full forward + all-reduce.
  cudaSetDevice(rank);
  cudaDeviceSynchronize();
  const auto t0 = Clock::now();
  Tensor y_full;
  for (int it = 0; it < iters; ++it) {
    Tensor y = forward_local();
    y_full = comm.all_reduce_sum(y);
  }
  cudaDeviceSynchronize();
  const double total_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

  // Isolate all-reduce time: time the collective alone on a fixed tensor.
  Tensor probe = forward_local();
  cudaDeviceSynchronize();
  const auto ta = Clock::now();
  for (int it = 0; it < iters; ++it) (void)comm.all_reduce_sum(probe);
  cudaDeviceSynchronize();
  const double ar_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - ta).count() / iters;

  const double shard_mb =
      static_cast<double>(3 * d * p) * sizeof(float) / 1e6;
  const double dense_mb =
      static_cast<double>(3 * d * d_ff) * sizeof(float) / 1e6;

  int rc = 0;
  if (rank == 0) {
    // Parity: build the FULL dense MLP from the same det() weights and compare
    // to the all-reduced output. gate/up dense [d, d_ff], down dense [d_ff, d].
    Tensor gd = make_weight(d, d_ff, 0, 0, 1).to(dev);
    Tensor ud = make_weight(d, d_ff, 0, 0, 2).to(dev);
    Tensor dd = make_weight(d_ff, d, 0, 0, 3).to(dev);
    Tensor gg = ops::matmul(x, gd);
    Tensor uu = ops::matmul(x, ud);
    Tensor hh = ops::swiglu_silu_gate(gg, uu);
    Tensor y_dense = ops::matmul(hh, dd);

    Tensor a = y_full.to(cpu_device());
    Tensor b = y_dense.to(cpu_device());
    const float* pa = a.data_ptr<float>();
    const float* pb = b.data_ptr<float>();
    double se = 0.0, sref = 0.0, maxabs = 0.0, na = 0.0;
    for (int64_t i = 0; i < a.numel(); ++i) {
      const double diff = static_cast<double>(pa[i]) - pb[i];
      se += diff * diff; sref += static_cast<double>(pb[i]) * pb[i];
      na += static_cast<double>(pa[i]) * pa[i];
      maxabs = std::max(maxabs, std::fabs(diff));
    }
    std::printf("[dbg] ||y_full||=%.4f ||y_dense||=%.4f\n", std::sqrt(na),
                std::sqrt(sref));
    const double rrms = std::sqrt(se / (sref + 1e-12));
    std::printf(
        "[bench] tesseract nccl_tp  world=%d cfg=M%lld_d%lld_dff%lld  "
        "shard_MB=%.1f dense_MB=%.1f  fwd_ms=%.3f allreduce_ms=%.3f  "
        "parity_rrms=%.2e maxabs=%.2e\n",
        world, (long long)M, (long long)d, (long long)d_ff, shard_mb, dense_mb,
        total_ms / iters, ar_ms, rrms, maxabs);
    // TF32 tensor cores (CUBLAS_COMPUTE_32F) give ~1e-3 relative error on a
    // well-conditioned reduction; a real sharding bug is rrms ~ O(1). 1e-2
    // cleanly separates the two.
    if (rrms < 1e-2) {
      std::printf("[PASS] TP=%d all-reduce output matches dense MLP (rrms=%.2e); "
                  "per-card weight mem = dense/%d (%.1f vs %.1f MB)\n",
                  world, rrms, world, shard_mb, dense_mb);
    } else {
      std::printf("[FAIL] TP=%d parity mismatch rrms=%.2e\n", world, rrms);
      rc = 1;
    }
  }
  comm.barrier();
  return rc != 0 ? rc : gather_rc;
}

// Get the CUDA device count WITHOUT initializing a CUDA context in this
// process: `cudaGetDeviceCount` creates the runtime context, which cannot
// survive a subsequent `fork()` (children inherit a poisoned context →
// "initialization error"). We therefore probe it in a throwaway child and
// pipe the integer back, leaving the parent CUDA-free until it forks workers.
int probe_device_count() {
  int fds[2];
  if (pipe(fds) != 0) return -1;
  pid_t pid = fork();
  if (pid == 0) {
    close(fds[0]);
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
    ssize_t wr = write(fds[1], &n, sizeof(n));
    (void)wr;
    close(fds[1]);
    _exit(0);
  }
  close(fds[1]);
  int n = 0;
  ssize_t rd = read(fds[0], &n, sizeof(n));
  close(fds[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  return rd == sizeof(n) ? n : -1;
}

}  // namespace

int main(int argc, char** argv) {
  int world = 0;  // 0 => auto (min(device_count, 2))
  int64_t M = 512, d = 4096, d_ff = 11008;
  int iters = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nx = [&]() { return std::atoll(argv[++i]); };
    if (a == "--world") world = static_cast<int>(nx());
    else if (a == "--M") M = nx();
    else if (a == "--d") d = nx();
    else if (a == "--dff") d_ff = nx();
    else if (a == "--iters") iters = static_cast<int>(nx());
  }

  if (!NcclCommBackend::available()) {
    std::printf("[skip] built without NCCL (-DTESSERACT_ENABLE_NCCL=ON)\n");
    return 77;  // ctest SKIP
  }
  const int ndev = probe_device_count();
  if (ndev < 2) {
    std::printf("[skip] need >=2 CUDA devices, have %d\n", ndev);
    return 77;
  }
  if (world <= 0) world = std::min(ndev, 2);
  if (world > ndev) world = ndev;
  if (d_ff % world != 0) {
    std::printf("[skip] d_ff=%lld not divisible by world=%d\n", (long long)d_ff,
                world);
    return 77;
  }

  // Generate the bootstrap id in the parent BEFORE any CUDA init, then fork.
  const std::string idhex = NcclCommBackend::generate_unique_id_hex();

  std::vector<pid_t> pids(world);
  for (int r = 0; r < world; ++r) {
    pid_t pid = fork();
    if (pid == 0) {
      int rc = 0;
      try {
        rc = run_rank(r, world, idhex, M, d, d_ff, iters);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] exception: %s\n", r, e.what());
        rc = 2;
      }
      // _exit() skips stdio flushing; do it explicitly so rank 0's bench
      // line and PASS/FAIL aren't lost (and avoid exit() running static
      // dtors over a post-fork CUDA/NCCL state).
      std::fflush(stdout);
      std::fflush(stderr);
      _exit(rc);
    }
    pids[r] = pid;
  }

  int final_rc = 0;
  for (int r = 0; r < world; ++r) {
    int status = 0;
    waitpid(pids[r], &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
      if (code == 77 && final_rc == 0) final_rc = 77;
      else final_rc = (code == 0 ? final_rc : 1);
    }
  }
  return final_rc;
}

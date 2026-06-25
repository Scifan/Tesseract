#pragma once

namespace tesseract {

// Scoped guard disabling autograd recording. While an instance is alive, ops
// compute only the forward value and skip backward wiring. Thread-local: one
// thread's guard does not affect another's.
class NoGradGuard {
 public:
  NoGradGuard();
  ~NoGradGuard();
  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  bool prev_;
};

// Returns true when autograd recording is enabled (the default).
bool is_grad_enabled() noexcept;

}  // namespace tesseract

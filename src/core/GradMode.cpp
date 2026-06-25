#include "tesseract/core/GradMode.hpp"

namespace tesseract {

namespace {
thread_local bool g_grad_enabled = true;
}

NoGradGuard::NoGradGuard() : prev_(g_grad_enabled) { g_grad_enabled = false; }
NoGradGuard::~NoGradGuard() { g_grad_enabled = prev_; }

bool is_grad_enabled() noexcept { return g_grad_enabled; }

}  // namespace tesseract

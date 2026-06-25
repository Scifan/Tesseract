#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

#include <fmt/core.h>
#include <fmt/format.h>

namespace tesseract {

// Base exception type; all runtime errors from the framework derive from this.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class ShapeError : public Error {
 public:
  using Error::Error;
};

class DeviceError : public Error {
 public:
  using Error::Error;
};

class NotImplementedError : public Error {
 public:
  using Error::Error;
};

namespace detail {

[[noreturn]] inline void throw_error(std::string_view file, int line, std::string_view func,
                                     const std::string& msg) {
  throw Error(fmt::format("[tesseract] {} (at {}:{} in {})", msg, file, line, func));
}

}  // namespace detail

}  // namespace tesseract

// The primary runtime check. Throws tesseract::Error with formatted message.
#define TESSERACT_CHECK(cond, ...)                                                     \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      ::tesseract::detail::throw_error(__FILE__, __LINE__, __func__,                   \
                                       ::fmt::format("check failed: " #cond ". "       \
                                                     __VA_ARGS__));                    \
    }                                                                                  \
  } while (0)

// Unconditional failure.
#define TESSERACT_THROW(...)                                                           \
  ::tesseract::detail::throw_error(__FILE__, __LINE__, __func__, ::fmt::format(__VA_ARGS__))

// Debug-only assertion (compiled out in NDEBUG).
#ifdef NDEBUG
#define TESSERACT_ASSERT(cond, ...) ((void)0)
#else
#define TESSERACT_ASSERT(cond, ...) TESSERACT_CHECK(cond, __VA_ARGS__)
#endif

#include "tesseract/core/Device.hpp"

#include <array>
#include <ostream>

#include <fmt/core.h>

namespace tesseract {

namespace {
constexpr std::array<std::string_view, static_cast<size_t>(DeviceType::kNumDeviceTypes)> kNames =
    {{"cpu", "cuda", "metal", "npu"}};
}  // namespace

std::string_view device_type_name(DeviceType dt) noexcept {
  const auto idx = static_cast<size_t>(dt);
  if (idx >= kNames.size()) return "<invalid-device>";
  return kNames[idx];
}

std::ostream& operator<<(std::ostream& os, DeviceType dt) {
  return os << device_type_name(dt);
}

std::string Device::to_string() const {
  return fmt::format("{}:{}", device_type_name(type), index);
}

std::ostream& operator<<(std::ostream& os, const Device& d) {
  return os << d.to_string();
}

}  // namespace tesseract

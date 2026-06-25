#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>

namespace tesseract {

// Supported physical device backends. Only CPU is implemented at M0; the other
// enumerators exist so that APIs that take a Device argument can be written
// once and extended later without breaking ABI or call sites.
enum class DeviceType : uint8_t {
  CPU = 0,
  CUDA = 1,
  Metal = 2,
  NPU = 3,

  kNumDeviceTypes,
};

std::string_view device_type_name(DeviceType dt) noexcept;
std::ostream& operator<<(std::ostream& os, DeviceType dt);

// A Device is a (type, index) pair. The default-constructed Device is CPU:0.
struct Device {
  DeviceType type = DeviceType::CPU;
  int32_t index = 0;

  constexpr Device() = default;
  constexpr Device(DeviceType t, int32_t i = 0) : type(t), index(i) {}

  constexpr bool is_cpu() const noexcept { return type == DeviceType::CPU; }
  constexpr bool is_cuda() const noexcept { return type == DeviceType::CUDA; }

  std::string to_string() const;

  friend constexpr bool operator==(const Device& a, const Device& b) noexcept {
    return a.type == b.type && a.index == b.index;
  }
  friend constexpr bool operator!=(const Device& a, const Device& b) noexcept {
    return !(a == b);
  }
};

std::ostream& operator<<(std::ostream& os, const Device& d);

inline Device cpu_device() noexcept { return Device{DeviceType::CPU, 0}; }

}  // namespace tesseract

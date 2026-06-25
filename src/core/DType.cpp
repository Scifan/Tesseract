#include "tesseract/core/DType.hpp"

#include <array>
#include <ostream>

#include "tesseract/utils/Logging.hpp"

namespace tesseract {

namespace {

struct DTypeInfo {
  std::string_view name;
  size_t size;
  bool implemented;
  bool floating;
  bool integral;
};

constexpr std::array<DTypeInfo, static_cast<size_t>(DType::kNumDTypes)> kTable = {{
    /* Float32     */ {"f32", 4, true, true, false},
    /* Float64     */ {"f64", 8, true, true, false},
    /* Int32       */ {"i32", 4, true, false, true},
    /* Int64       */ {"i64", 8, true, false, true},
    /* Bool        */ {"bool", 1, true, false, true},
    /* Float16     */ {"f16", 2, true, true, false},
    /* BFloat16    */ {"bf16", 2, true, true, false},
    /* Float8_E4M3 */ {"f8e4m3", 1, false, true, false},
    /* Float8_E5M2 */ {"f8e5m2", 1, false, true, false},
    /* Int8        */ {"i8", 1, true, false, true},
    /* UInt8       */ {"u8", 1, false, false, true},
    /* Int4        */ {"i4", 0, false, false, true},  // sub-byte; size query unsupported
}};

const DTypeInfo& info(DType dt) {
  const auto idx = static_cast<size_t>(dt);
  TESSERACT_CHECK(idx < kTable.size(), "invalid DType value: {}", idx);
  return kTable[idx];
}

}  // namespace

size_t dtype_size(DType dt) noexcept {
  return kTable[static_cast<size_t>(dt)].size;
}

std::string_view dtype_name(DType dt) noexcept {
  return kTable[static_cast<size_t>(dt)].name;
}

bool dtype_is_implemented(DType dt) noexcept {
  return kTable[static_cast<size_t>(dt)].implemented;
}

bool dtype_is_floating(DType dt) noexcept {
  return kTable[static_cast<size_t>(dt)].floating;
}

bool dtype_is_integral(DType dt) noexcept {
  return kTable[static_cast<size_t>(dt)].integral;
}

std::ostream& operator<<(std::ostream& os, DType dt) {
  return os << info(dt).name;
}

}  // namespace tesseract

#pragma once

// Software-emulated IEEE 754 binary16 (`Half`) and Google brain-float 16
// (`BFloat16`) scalar types. Both are 16-bit POD storage classes that
// round-trip through `float` for all arithmetic, so the heavy kernels
// (matmul, broadcast, reductions) just accumulate in FP32 and pay the
// conversion cost only at the edges.
//
// The primary motivation is forward-compatibility with CUDA in M2: once
// the GPU path lands, the in-memory layout is already correct (2 bytes,
// little-endian, IEEE for `Half` / truncated-IEEE for `BFloat16`) and we
// can hand the buffers directly to cuBLAS / cuDNN without a re-pack.

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace tesseract {

namespace detail {

// --- float <-> binary16 --------------------------------------------------
// IEEE 754 binary16 (sign 1, exp 5 bias 15, mantissa 10). Round-to-nearest
// even on narrowing; preserves inf/NaN; converts subnormals correctly.
inline std::uint16_t f16_from_float(float f) noexcept {
  std::uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  const std::int32_t exp_f = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127;
  const std::uint32_t mant = x & 0x7FFFFFu;

  // Inf / NaN.
  if (exp_f == 128) {
    const std::uint32_t mant16 = (mant != 0) ? 0x200u : 0u;
    return static_cast<std::uint16_t>(sign | 0x7C00u | mant16);
  }

  std::int32_t exp16 = exp_f + 15;
  // Overflow to +/-inf.
  if (exp16 >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);

  // Subnormal or underflow.
  if (exp16 <= 0) {
    // Even smaller than the smallest f16 subnormal — flush to signed zero.
    if (exp16 < -10) return static_cast<std::uint16_t>(sign);
    const std::uint32_t mant_with_hidden = mant | 0x800000u;
    const int shift = 14 - exp16;  // in [14, 24] for the subnormal range.
    std::uint32_t m = mant_with_hidden >> shift;
    const std::uint32_t remainder_mask = (1u << shift) - 1u;
    const std::uint32_t remainder = mant_with_hidden & remainder_mask;
    const std::uint32_t halfway = 1u << (shift - 1);
    // Round to nearest, ties to even.
    if (remainder > halfway || (remainder == halfway && (m & 1u))) m += 1u;
    return static_cast<std::uint16_t>(sign | m);
  }

  // Normal range: drop the low 13 mantissa bits with RNE.
  std::uint32_t m = mant >> 13;
  const std::uint32_t remainder = mant & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (m & 1u))) {
    m += 1u;
    if (m == 0x400u) {  // mantissa overflowed into exponent.
      m = 0u;
      exp16 += 1;
      if (exp16 >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp16) << 10) | m);
}

inline float f16_to_float(std::uint16_t h) noexcept {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t mant = h & 0x3FFu;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/- zero
    } else {
      // Subnormal: normalize.
      std::uint32_t m = mant;
      int e = -1;
      do {
        m <<= 1;
        e += 1;
      } while ((m & 0x400u) == 0);
      bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    // Inf / NaN. Preserve NaN payload (lifted to the wider mantissa).
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | (static_cast<std::uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// --- float <-> bfloat16 --------------------------------------------------
// BFloat16 keeps the IEEE 754 single-precision exponent (8 bits, same bias
// as float) and truncates the mantissa to 7 bits. Dynamic range matches
// float exactly — only precision is reduced.
inline std::uint16_t bf16_from_float(float f) noexcept {
  std::uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Preserve NaNs (any non-zero mantissa remains non-zero after truncation).
  if (((bits >> 23) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu) != 0u) {
    // Force a quiet-NaN bit so downstream truncation doesn't land on inf.
    bits |= 0x00400000u;
  }
  // Round to nearest, ties to even on the dropped 16 low mantissa bits.
  const std::uint32_t rounding_bias = 0x00007FFFu + ((bits >> 16) & 1u);
  bits += rounding_bias;
  return static_cast<std::uint16_t>(bits >> 16);
}

inline float bf16_to_float(std::uint16_t b) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

}  // namespace detail

// --------------------------------------------------------------------------
// Public scalar types.
// --------------------------------------------------------------------------
//
// Design choices:
//
//  * Both types are 2-byte, trivially copyable, standard-layout so
//    `memcpy` round-trip and `Tensor::from_vector<Half>` "just work".
//  * Conversion to `float` is implicit — arithmetic and `<cmath>` calls
//    transparently lift operands into FP32. Conversion *from* `float` is
//    also implicit so literals compose naturally in tests / constants.
//  * Explicit `operator==` provides bit-equality semantics without pulling
//    in the float comparison, which would defeat the point of testing the
//    conversion pathway.

struct Half {
  std::uint16_t bits;

  constexpr Half() noexcept : bits(0) {}
  Half(float f) noexcept : bits(detail::f16_from_float(f)) {}
  Half(double d) noexcept : bits(detail::f16_from_float(static_cast<float>(d))) {}
  // Integer overload disambiguates `static_cast<Half>(some_int)` — needed
  // for `arange(..., Float16)` and the tensor's scalar-fill path, where
  // the source is `int64_t`. Without this, float/double both compete and
  // the overload set is ambiguous.
  template <typename I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  Half(I i) noexcept : bits(detail::f16_from_float(static_cast<float>(i))) {}

  // Raw-bits escape hatch. Use sparingly — mostly for serialization tests.
  struct from_bits_tag {};
  constexpr Half(from_bits_tag, std::uint16_t raw) noexcept : bits(raw) {}
  static constexpr Half from_bits(std::uint16_t raw) noexcept { return Half{from_bits_tag{}, raw}; }

  operator float() const noexcept { return detail::f16_to_float(bits); }
};

struct BFloat16 {
  std::uint16_t bits;

  constexpr BFloat16() noexcept : bits(0) {}
  BFloat16(float f) noexcept : bits(detail::bf16_from_float(f)) {}
  BFloat16(double d) noexcept : bits(detail::bf16_from_float(static_cast<float>(d))) {}
  template <typename I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  BFloat16(I i) noexcept : bits(detail::bf16_from_float(static_cast<float>(i))) {}

  struct from_bits_tag {};
  constexpr BFloat16(from_bits_tag, std::uint16_t raw) noexcept : bits(raw) {}
  static constexpr BFloat16 from_bits(std::uint16_t raw) noexcept { return BFloat16{from_bits_tag{}, raw}; }

  operator float() const noexcept { return detail::bf16_to_float(bits); }
};

static_assert(sizeof(Half) == 2, "Half must be 2 bytes to round-trip with GPU buffers");
static_assert(sizeof(BFloat16) == 2, "BFloat16 must be 2 bytes to round-trip with GPU buffers");
static_assert(std::is_trivially_copyable_v<Half>, "Half must be trivially copyable");
static_assert(std::is_trivially_copyable_v<BFloat16>, "BFloat16 must be trivially copyable");

// Arithmetic operators defined as hidden friends so they only resolve when
// at least one operand is the half/bf16 type (avoids polluting overload
// resolution for `float + float`). All arithmetic widens to FP32 — the
// down-cast happens only when the result is stored back into a tensor.
#define TESSERACT_FP16_BINOP(Type, Op) \
  inline Type operator Op (Type a, Type b) noexcept { \
    return Type(static_cast<float>(a) Op static_cast<float>(b)); \
  }

TESSERACT_FP16_BINOP(Half, +)
TESSERACT_FP16_BINOP(Half, -)
TESSERACT_FP16_BINOP(Half, *)
TESSERACT_FP16_BINOP(Half, /)
TESSERACT_FP16_BINOP(BFloat16, +)
TESSERACT_FP16_BINOP(BFloat16, -)
TESSERACT_FP16_BINOP(BFloat16, *)
TESSERACT_FP16_BINOP(BFloat16, /)
#undef TESSERACT_FP16_BINOP

inline Half operator-(Half a) noexcept { return Half(-static_cast<float>(a)); }
inline BFloat16 operator-(BFloat16 a) noexcept { return BFloat16(-static_cast<float>(a)); }

// Bit-identity equality. (Float comparison is available via the implicit
// conversion if the caller wants it; this just makes `assert(a == b)` do
// the thing tests actually want.)
inline bool operator==(Half a, Half b) noexcept { return a.bits == b.bits; }
inline bool operator!=(Half a, Half b) noexcept { return a.bits != b.bits; }
inline bool operator==(BFloat16 a, BFloat16 b) noexcept { return a.bits == b.bits; }
inline bool operator!=(BFloat16 a, BFloat16 b) noexcept { return a.bits != b.bits; }

// Type trait: `true` for every floating-point type this library knows
// about (including our software-emulated half types). Use this in kernels
// that need to distinguish "real number" from "integer" without relying on
// `std::is_floating_point_v`, which doesn't know about `Half`/`BFloat16`.
template <typename T>
struct is_tesseract_floating : std::is_floating_point<T> {};
template <> struct is_tesseract_floating<Half> : std::true_type {};
template <> struct is_tesseract_floating<BFloat16> : std::true_type {};

template <typename T>
inline constexpr bool is_tesseract_floating_v = is_tesseract_floating<T>::value;

}  // namespace tesseract

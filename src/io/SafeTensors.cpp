#include "tesseract/io/SafeTensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#include "tesseract/core/Storage.hpp"
#include "tesseract/utils/Logging.hpp"

namespace tesseract::io {

// --------------------------------------------------------------------------
// Minimal JSON parser — just enough of the grammar for safetensors headers.
//
// The header only contains:
//   - top-level object
//   - object values that are either:
//       * another object (tensor descriptor)
//       * an object mapping strings to strings (`__metadata__`)
//   - keys are strings
//   - values inside tensor descriptors:
//       * string      (dtype)
//       * array of int (shape, data_offsets)
//
// Writing a purpose-built parser (~120 LoC) is cleaner than pulling in
// nlohmann/json (25k LoC header) for this single use case.
// --------------------------------------------------------------------------

namespace {

struct JsonParser {
  std::string_view s;
  std::size_t pos{0};

  [[noreturn]] void fail(const std::string& what) const {
    // Show a small window around the failure point to aid debugging.
    const std::size_t lo = pos > 40 ? pos - 40 : 0;
    const std::size_t hi = std::min(s.size(), pos + 40);
    TESSERACT_THROW("safetensors JSON parse error at offset {}: {} (near '{}')",
                    pos, what,
                    std::string(s.substr(lo, hi - lo)));
  }

  void skip_ws() {
    while (pos < s.size()) {
      const char c = s[pos];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos;
      } else {
        break;
      }
    }
  }

  char peek() {
    skip_ws();
    if (pos >= s.size()) fail("unexpected end of input");
    return s[pos];
  }

  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos;
  }

  bool match(char c) {
    skip_ws();
    if (pos < s.size() && s[pos] == c) {
      ++pos;
      return true;
    }
    return false;
  }

  // Parses a JSON string into a C++ string. Handles the standard backslash
  // escapes we realistically see in safetensors headers: \" \\ \/ \b \f \n \r \t \uXXXX
  // (the unicode escape decodes to UTF-8).
  std::string parse_string() {
    if (peek() != '"') fail("expected string");
    ++pos;
    std::string out;
    while (pos < s.size()) {
      const char c = s[pos++];
      if (c == '"') return out;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos >= s.size()) fail("unterminated escape");
      const char esc = s[pos++];
      switch (esc) {
        case '"':  out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/':  out.push_back('/'); break;
        case 'b':  out.push_back('\b'); break;
        case 'f':  out.push_back('\f'); break;
        case 'n':  out.push_back('\n'); break;
        case 'r':  out.push_back('\r'); break;
        case 't':  out.push_back('\t'); break;
        case 'u': {
          if (pos + 4 > s.size()) fail("truncated \\u escape");
          uint32_t cp = 0;
          for (int i = 0; i < 4; ++i) {
            const char h = s[pos++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= uint32_t(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= uint32_t(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= uint32_t(h - 'A' + 10);
            else fail("invalid hex digit in \\u escape");
          }
          // UTF-16 surrogate pairs not handled — safetensors headers in
          // the wild don't use them, tokenizer JSON can, but this parser
          // is scoped to safetensors. Decode BMP codepoints to UTF-8.
          if (cp < 0x80) {
            out.push_back(char(cp));
          } else if (cp < 0x800) {
            out.push_back(char(0xC0 | (cp >> 6)));
            out.push_back(char(0x80 | (cp & 0x3F)));
          } else {
            out.push_back(char(0xE0 | (cp >> 12)));
            out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(char(0x80 | (cp & 0x3F)));
          }
          break;
        }
        default: fail("unknown escape");
      }
    }
    fail("unterminated string");
  }

  int64_t parse_int() {
    skip_ws();
    const std::size_t start = pos;
    if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) ++pos;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    if (pos == start) fail("expected integer");
    try {
      return std::stoll(std::string(s.substr(start, pos - start)));
    } catch (const std::exception&) {
      fail("integer out of range");
    }
  }

  std::vector<int64_t> parse_int_array() {
    expect('[');
    std::vector<int64_t> out;
    if (match(']')) return out;
    while (true) {
      out.push_back(parse_int());
      if (match(']')) return out;
      expect(',');
    }
  }

  // Skip one JSON value of any kind (used to ignore fields we don't care about).
  void skip_value() {
    const char c = peek();
    if (c == '"') { parse_string(); return; }
    if (c == '[') {
      ++pos;
      if (match(']')) return;
      while (true) {
        skip_value();
        if (match(']')) return;
        expect(',');
      }
    }
    if (c == '{') {
      ++pos;
      if (match('}')) return;
      while (true) {
        (void)parse_string();
        expect(':');
        skip_value();
        if (match('}')) return;
        expect(',');
      }
    }
    if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
      // number (possibly with fraction/exponent — we tolerate it here for
      // skip even though parse_int doesn't). Eat everything number-ish.
      while (pos < s.size()) {
        const char ch = s[pos];
        const bool num = (ch >= '0' && ch <= '9') || ch == '-' || ch == '+' ||
                         ch == '.' || ch == 'e' || ch == 'E';
        if (!num) break;
        ++pos;
      }
      return;
    }
    // true/false/null
    if (s.compare(pos, 4, "true") == 0)  { pos += 4; return; }
    if (s.compare(pos, 5, "false") == 0) { pos += 5; return; }
    if (s.compare(pos, 4, "null") == 0)  { pos += 4; return; }
    fail("unexpected value");
  }
};

// --------------------------------------------------------------------------
// dtype string <-> DType mapping
// --------------------------------------------------------------------------

struct DTypePair {
  std::string_view name;
  DType dtype;
};

constexpr std::array<DTypePair, 8> kDTypeTable = {{
    {"F32",  DType::Float32},
    {"F64",  DType::Float64},
    {"F16",  DType::Float16},
    {"BF16", DType::BFloat16},
    {"I32",  DType::Int32},
    {"I64",  DType::Int64},
    {"I8",   DType::Int8},
    {"BOOL", DType::Bool},
}};

}  // namespace

DType safetensors_dtype_from_string(std::string_view s) {
  for (const auto& p : kDTypeTable) {
    if (p.name == s) return p.dtype;
  }
  TESSERACT_THROW("safetensors: unsupported dtype string '{}'", std::string(s));
}

std::string_view safetensors_dtype_to_string(DType dt) {
  for (const auto& p : kDTypeTable) {
    if (p.dtype == dt) return p.name;
  }
  TESSERACT_THROW("safetensors: no string mapping for DType '{}'", dtype_name(dt));
}

// --------------------------------------------------------------------------
// SafeTensors
// --------------------------------------------------------------------------

SafeTensors::SafeTensors() = default;

SafeTensors::~SafeTensors() {
  close_impl();
}

SafeTensors::SafeTensors(SafeTensors&& other) noexcept
    : fd_(other.fd_),
      mapped_(other.mapped_),
      mapped_size_(other.mapped_size_),
      header_size_(other.header_size_),
      views_(std::move(other.views_)),
      keys_(std::move(other.keys_)),
      metadata_(std::move(other.metadata_)) {
  other.fd_ = -1;
  other.mapped_ = nullptr;
  other.mapped_size_ = 0;
  other.header_size_ = 0;
}

SafeTensors& SafeTensors::operator=(SafeTensors&& other) noexcept {
  if (this != &other) {
    close_impl();
    fd_           = other.fd_;
    mapped_       = other.mapped_;
    mapped_size_  = other.mapped_size_;
    header_size_  = other.header_size_;
    views_        = std::move(other.views_);
    keys_         = std::move(other.keys_);
    metadata_     = std::move(other.metadata_);
    other.fd_ = -1;
    other.mapped_ = nullptr;
    other.mapped_size_ = 0;
    other.header_size_ = 0;
  }
  return *this;
}

SafeTensors SafeTensors::open(const std::string& path) {
  SafeTensors st;
  st.open_impl(path);
  return st;
}

void SafeTensors::open_impl(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0) {
    const int err = errno;
    TESSERACT_THROW("safetensors: open('{}') failed: {}", path, std::strerror(err));
  }

  struct stat st;
  if (::fstat(fd_, &st) < 0) {
    const int err = errno;
    ::close(fd_);
    fd_ = -1;
    TESSERACT_THROW("safetensors: fstat('{}') failed: {}", path, std::strerror(err));
  }
  const std::size_t file_size = static_cast<std::size_t>(st.st_size);
  if (file_size < 8) {
    ::close(fd_);
    fd_ = -1;
    TESSERACT_THROW("safetensors: file '{}' too small ({} bytes, need >= 8)",
                    path, file_size);
  }

  void* mp = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mp == MAP_FAILED) {
    const int err = errno;
    ::close(fd_);
    fd_ = -1;
    TESSERACT_THROW("safetensors: mmap('{}') failed: {}", path, std::strerror(err));
  }
  mapped_      = static_cast<const std::byte*>(mp);
  mapped_size_ = file_size;

  // 1) Read the 8-byte little-endian JSON header length.
  uint64_t n = 0;
  std::memcpy(&n, mapped_, 8);
  // (Host is assumed little-endian; all target platforms — x86_64, aarch64,
  //  CUDA hosts — are LE. If we ever target a BE platform we'd swap here.)
  if (n == 0 || n > file_size - 8) {
    TESSERACT_THROW("safetensors: malformed header length {} (file size {})",
                    n, file_size);
  }
  header_size_ = static_cast<std::size_t>(n);

  // 2) Parse the JSON header.
  const std::string_view json(reinterpret_cast<const char*>(mapped_) + 8,
                              header_size_);
  JsonParser jp{json, 0};

  if (jp.peek() != '{') {
    TESSERACT_THROW("safetensors: header root is not an object");
  }
  ++jp.pos;

  // Data region starts right after the header.
  const std::size_t data_base = 8 + header_size_;
  const std::size_t data_size = file_size - data_base;

  auto finalize_error = [&](const std::string& msg) {
    // Unmap so destructor doesn't try to double-unmap after we throw.
    ::munmap(const_cast<std::byte*>(mapped_), mapped_size_);
    mapped_ = nullptr;
    mapped_size_ = 0;
    ::close(fd_);
    fd_ = -1;
    TESSERACT_THROW("{}", msg);
  };

  bool first = true;
  while (!jp.match('}')) {
    if (!first) jp.expect(',');
    first = false;
    const std::string key = jp.parse_string();
    jp.expect(':');

    if (key == "__metadata__") {
      // Object of strings.
      jp.expect('{');
      bool meta_first = true;
      while (!jp.match('}')) {
        if (!meta_first) jp.expect(',');
        meta_first = false;
        const std::string mk = jp.parse_string();
        jp.expect(':');
        const std::string mv = jp.parse_string();
        metadata_.emplace(std::move(mk), std::move(mv));
      }
      continue;
    }

    // Tensor descriptor.
    jp.expect('{');
    View v{};
    std::int64_t off_begin = -1;
    std::int64_t off_end   = -1;
    bool have_dtype = false;
    bool have_shape = false;
    bool have_offsets = false;

    bool inner_first = true;
    while (!jp.match('}')) {
      if (!inner_first) jp.expect(',');
      inner_first = false;
      const std::string field = jp.parse_string();
      jp.expect(':');
      if (field == "dtype") {
        const std::string dt = jp.parse_string();
        v.dtype = safetensors_dtype_from_string(dt);
        have_dtype = true;
      } else if (field == "shape") {
        auto dims = jp.parse_int_array();
        for (int64_t d : dims) {
          if (d < 0) {
            finalize_error(fmt::format("safetensors: tensor '{}' has negative dim {}",
                                        key, d));
          }
        }
        v.shape = Shape(std::move(dims));
        have_shape = true;
      } else if (field == "data_offsets") {
        auto offs = jp.parse_int_array();
        if (offs.size() != 2) {
          finalize_error(fmt::format("safetensors: tensor '{}' data_offsets must have 2 elements, got {}",
                                      key, offs.size()));
        }
        off_begin = offs[0];
        off_end   = offs[1];
        have_offsets = true;
      } else {
        // Unknown field — ignore for forward compatibility.
        jp.skip_value();
      }
    }

    if (!have_dtype || !have_shape || !have_offsets) {
      finalize_error(fmt::format(
          "safetensors: tensor '{}' missing required fields (dtype/shape/data_offsets)",
          key));
    }
    if (off_begin < 0 || off_end < off_begin ||
        static_cast<std::size_t>(off_end) > data_size) {
      finalize_error(fmt::format(
          "safetensors: tensor '{}' data_offsets [{}, {}] out of range (data region {} bytes)",
          key, off_begin, off_end, data_size));
    }
    const std::size_t nbytes = static_cast<std::size_t>(off_end - off_begin);

    // Validate nbytes matches shape × dtype_size for dtypes where we know
    // the element size. (FP8/INT4 have dtype_size==0 or 1 with sub-byte
    // packing — we don't policy-check those here; they're unsupported in
    // safetensors_dtype_from_string anyway.)
    int64_t numel = 1;
    for (int64_t d : v.shape) numel *= d;
    const std::size_t expect_bytes =
        static_cast<std::size_t>(numel) * dtype_size(v.dtype);
    if (expect_bytes != nbytes) {
      finalize_error(fmt::format(
          "safetensors: tensor '{}' byte count mismatch: header says {} "
          "bytes but shape {} × {} = {} bytes",
          key, nbytes, v.shape.to_string(), dtype_name(v.dtype), expect_bytes));
    }

    v.byte_offset = data_base + static_cast<std::size_t>(off_begin);
    v.nbytes      = nbytes;
    views_.emplace(key, v);
    keys_.push_back(key);
  }

  jp.skip_ws();
  if (jp.pos != json.size()) {
    // Extra trailing junk — tolerated (mirrors what HF's parser does).
  }
}

void SafeTensors::close_impl() noexcept {
  if (mapped_ != nullptr) {
    ::munmap(const_cast<std::byte*>(mapped_), mapped_size_);
    mapped_ = nullptr;
    mapped_size_ = 0;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  views_.clear();
  keys_.clear();
  metadata_.clear();
  header_size_ = 0;
}

bool SafeTensors::contains(std::string_view name) const noexcept {
  return views_.find(std::string(name)) != views_.end();
}

const SafeTensors::View& SafeTensors::view(std::string_view name) const {
  const auto it = views_.find(std::string(name));
  if (it == views_.end()) {
    TESSERACT_THROW("safetensors: tensor '{}' not found", std::string(name));
  }
  return it->second;
}

const std::byte* SafeTensors::raw(std::string_view name) const {
  const View& v = view(name);
  return mapped_ + v.byte_offset;
}

Tensor SafeTensors::load(std::string_view name, Device device) const {
  const View& v = view(name);
  // Always allocate fresh storage and copy — keeps the mmap independent of
  // any Tensor lifetime (user can `close` the reader freely after load).
  Tensor out = Tensor::empty(v.shape, v.dtype, device);
  TESSERACT_CHECK(out.is_contiguous(),
                  "safetensors::load: expected contiguous destination");
  TESSERACT_CHECK(out.nbytes() == v.nbytes,
                  "safetensors::load: size mismatch ({} vs {})",
                  out.nbytes(), v.nbytes);
  if (v.nbytes > 0) {
    Storage::copy_device_bytes(out.raw_data(), device,
                               mapped_ + v.byte_offset, cpu_device(),
                               v.nbytes);
  }
  return out;
}

}  // namespace tesseract::io

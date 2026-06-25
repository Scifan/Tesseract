// Synthetic round-trip tests for `tesseract::io::SafeTensors`.
//
// We don't ship an HF checkpoint in the repo; instead, every test builds a
// byte-for-byte-valid safetensors file in a tmp directory, reads it back
// through the loader, and asserts the tensor contents match.
//
// The synthetic writer implemented here is intentionally minimal (no JSON
// escaping beyond quotes, no `__metadata__` unless the test asks for it)
// so that the parser we're exercising is the only thing under test.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tesseract/core/Float16.hpp"
#include "tesseract/core/Tensor.hpp"
#include "tesseract/io/SafeTensors.hpp"
#include "tesseract/utils/Logging.hpp"

namespace fs = std::filesystem;
using tesseract::BFloat16;
using tesseract::DType;
using tesseract::Half;
using tesseract::Shape;
using tesseract::Tensor;
using tesseract::io::SafeTensors;

namespace {

struct RawTensor {
  std::string name;
  DType dtype;
  std::vector<int64_t> shape;
  std::vector<std::byte> bytes;  // row-major contents
};

// Build a safetensors byte blob from a list of tensors and an optional
// metadata map. Returns the full file payload (8-byte header length + JSON
// header + concatenated raw bytes).
std::vector<std::byte> build_safetensors(const std::vector<RawTensor>& tensors,
                                         const std::vector<std::pair<std::string, std::string>>& metadata = {}) {
  auto dtype_str = [](DType dt) -> std::string_view {
    switch (dt) {
      case DType::Float32:  return "F32";
      case DType::Float64:  return "F64";
      case DType::Float16:  return "F16";
      case DType::BFloat16: return "BF16";
      case DType::Int32:    return "I32";
      case DType::Int64:    return "I64";
      case DType::Int8:     return "I8";
      case DType::Bool:     return "BOOL";
      default: TESSERACT_THROW("test helper: unsupported dtype");
    }
  };

  std::string json = "{";
  bool first = true;
  if (!metadata.empty()) {
    json += "\"__metadata__\":{";
    bool mfirst = true;
    for (const auto& [k, v] : metadata) {
      if (!mfirst) json += ',';
      mfirst = false;
      json += '"'; json += k; json += "\":\""; json += v; json += '"';
    }
    json += '}';
    first = false;
  }

  std::size_t running = 0;
  for (const auto& t : tensors) {
    if (!first) json += ',';
    first = false;
    json += '"'; json += t.name; json += "\":{\"dtype\":\"";
    json += dtype_str(t.dtype);
    json += "\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
      if (i) json += ',';
      json += std::to_string(t.shape[i]);
    }
    json += "],\"data_offsets\":[";
    json += std::to_string(running);
    json += ',';
    json += std::to_string(running + t.bytes.size());
    json += "]}";
    running += t.bytes.size();
  }
  json += '}';

  const uint64_t n = static_cast<uint64_t>(json.size());

  std::vector<std::byte> out;
  out.reserve(8 + json.size() + running);
  out.resize(8);
  std::memcpy(out.data(), &n, 8);
  for (char c : json) out.push_back(static_cast<std::byte>(c));
  for (const auto& t : tensors) {
    out.insert(out.end(), t.bytes.begin(), t.bytes.end());
  }
  return out;
}

// Write a byte blob to a tmp path (uses test-local unique name to avoid
// collisions when Catch2 runs tests in parallel).
fs::path write_blob(const std::string& tag, const std::vector<std::byte>& blob) {
  auto dir = fs::temp_directory_path() / "tesseract_safetensors_tests";
  fs::create_directories(dir);
  auto path = dir / (tag + "_" + std::to_string(::getpid()) + ".safetensors");
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  REQUIRE(f.good());
  f.write(reinterpret_cast<const char*>(blob.data()),
          static_cast<std::streamsize>(blob.size()));
  REQUIRE(f.good());
  return path;
}

template <typename T>
std::vector<std::byte> as_bytes(const std::vector<T>& v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

}  // namespace

TEST_CASE("SafeTensors: empty file rejected cleanly") {
  auto path = write_blob("empty", {});
  REQUIRE_THROWS(SafeTensors::open(path.string()));
  fs::remove(path);
}

TEST_CASE("SafeTensors: round-trip one F32 tensor") {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  RawTensor t{"weight", DType::Float32, {2, 3}, as_bytes(data)};
  auto path = write_blob("f32_simple", build_safetensors({t}));

  auto st = SafeTensors::open(path.string());
  REQUIRE(st.keys().size() == 1);
  REQUIRE(st.keys()[0] == "weight");
  REQUIRE(st.contains("weight"));
  REQUIRE_FALSE(st.contains("missing"));

  const auto& v = st.view("weight");
  REQUIRE(v.dtype == DType::Float32);
  REQUIRE(v.shape == Shape({2, 3}));
  REQUIRE(v.nbytes == data.size() * sizeof(float));

  Tensor loaded = st.load("weight");
  REQUIRE(loaded.shape() == Shape({2, 3}));
  REQUIRE(loaded.dtype() == DType::Float32);
  const float* p = loaded.data_ptr<float>();
  for (std::size_t i = 0; i < data.size(); ++i) {
    REQUIRE(p[i] == data[i]);
  }
  fs::remove(path);
}

TEST_CASE("SafeTensors: round-trip multiple dtypes and shapes") {
  std::vector<RawTensor> tensors;

  // F64 rank-1
  std::vector<double> d64 = {-1.5, 2.5, 3.25, 4.125};
  tensors.push_back({"a_f64", DType::Float64, {4}, as_bytes(d64)});

  // I64 rank-2
  std::vector<int64_t> d_i64 = {7, 8, 9, -1, -2, -3};
  tensors.push_back({"b_i64", DType::Int64, {2, 3}, as_bytes(d_i64)});

  // I32 rank-1
  std::vector<int32_t> d_i32 = {100, 200, 300};
  tensors.push_back({"c_i32", DType::Int32, {3}, as_bytes(d_i32)});

  // Bool rank-1 (stored as bytes 0 / 1)
  std::vector<uint8_t> d_bool = {1, 0, 1, 1, 0};
  std::vector<std::byte> bool_bytes(d_bool.size());
  for (std::size_t i = 0; i < d_bool.size(); ++i)
    bool_bytes[i] = static_cast<std::byte>(d_bool[i]);
  tensors.push_back({"d_bool", DType::Bool, {5}, bool_bytes});

  // Zero-size tensor: legal in safetensors (data_offsets=[x, x])
  tensors.push_back({"e_empty", DType::Float32, {0, 4}, {}});

  auto path = write_blob("multi_dtype", build_safetensors(tensors));
  auto st = SafeTensors::open(path.string());
  REQUIRE(st.keys().size() == 5);

  {
    Tensor t = st.load("a_f64");
    REQUIRE(t.dtype() == DType::Float64);
    REQUIRE(t.shape() == Shape({4}));
    const double* p = t.data_ptr<double>();
    for (std::size_t i = 0; i < d64.size(); ++i) REQUIRE(p[i] == d64[i]);
  }
  {
    Tensor t = st.load("b_i64");
    REQUIRE(t.dtype() == DType::Int64);
    REQUIRE(t.shape() == Shape({2, 3}));
    const int64_t* p = t.data_ptr<int64_t>();
    for (std::size_t i = 0; i < d_i64.size(); ++i) REQUIRE(p[i] == d_i64[i]);
  }
  {
    Tensor t = st.load("c_i32");
    REQUIRE(t.dtype() == DType::Int32);
    const int32_t* p = t.data_ptr<int32_t>();
    for (std::size_t i = 0; i < d_i32.size(); ++i) REQUIRE(p[i] == d_i32[i]);
  }
  {
    Tensor t = st.load("d_bool");
    REQUIRE(t.dtype() == DType::Bool);
    REQUIRE(t.shape() == Shape({5}));
  }
  {
    Tensor t = st.load("e_empty");
    REQUIRE(t.shape() == Shape({0, 4}));
    REQUIRE(t.numel() == 0);
  }
  fs::remove(path);
}

TEST_CASE("SafeTensors: half-precision round-trip (F16 + BF16)") {
  std::vector<Half> d_f16;
  for (float x : {0.0f, 0.5f, -1.25f, 3.75f, 100.0f}) d_f16.emplace_back(x);
  std::vector<BFloat16> d_bf16;
  for (float x : {0.0f, 0.5f, -1.25f, 3.75f, 100.0f}) d_bf16.emplace_back(x);

  std::vector<RawTensor> tensors = {
      {"f16", DType::Float16,  {5}, as_bytes(d_f16)},
      {"bf16", DType::BFloat16, {5}, as_bytes(d_bf16)},
  };
  auto path = write_blob("half", build_safetensors(tensors));
  auto st = SafeTensors::open(path.string());

  Tensor f16 = st.load("f16");
  REQUIRE(f16.dtype() == DType::Float16);
  const Half* p_f16 = f16.data_ptr<Half>();
  for (std::size_t i = 0; i < d_f16.size(); ++i) {
    REQUIRE(static_cast<float>(p_f16[i]) == static_cast<float>(d_f16[i]));
  }
  Tensor bf16 = st.load("bf16");
  REQUIRE(bf16.dtype() == DType::BFloat16);
  const BFloat16* p_bf16 = bf16.data_ptr<BFloat16>();
  for (std::size_t i = 0; i < d_bf16.size(); ++i) {
    REQUIRE(static_cast<float>(p_bf16[i]) == static_cast<float>(d_bf16[i]));
  }
  fs::remove(path);
}

TEST_CASE("SafeTensors: __metadata__ round-trip") {
  std::vector<float> data = {1, 2, 3, 4};
  RawTensor t{"w", DType::Float32, {4}, as_bytes(data)};
  auto blob = build_safetensors({t},
                                 {{"format", "pt"},
                                  {"model_name", "test-stub"}});
  auto path = write_blob("meta", blob);
  auto st = SafeTensors::open(path.string());
  REQUIRE(st.metadata().size() == 2);
  REQUIRE(st.metadata().at("format") == "pt");
  REQUIRE(st.metadata().at("model_name") == "test-stub");
  REQUIRE(st.load("w").numel() == 4);
  fs::remove(path);
}

TEST_CASE("SafeTensors: move construct + move assign preserve state") {
  std::vector<float> d = {1.0f, 2.0f};
  auto path = write_blob("move",
                          build_safetensors({{"x", DType::Float32, {2}, as_bytes(d)}}));

  SafeTensors a = SafeTensors::open(path.string());
  REQUIRE(a.keys().size() == 1);

  SafeTensors b = std::move(a);
  REQUIRE(b.keys().size() == 1);
  REQUIRE(b.load("x").data_ptr<float>()[1] == 2.0f);

  SafeTensors c = SafeTensors::open(path.string());
  c = std::move(b);
  REQUIRE(c.load("x").data_ptr<float>()[0] == 1.0f);
  fs::remove(path);
}

TEST_CASE("SafeTensors: malformed headers rejected") {
  // Header length > file
  {
    std::vector<std::byte> blob(16, std::byte{0});
    uint64_t n = 9999;
    std::memcpy(blob.data(), &n, 8);
    auto path = write_blob("badlen", blob);
    REQUIRE_THROWS(SafeTensors::open(path.string()));
    fs::remove(path);
  }

  // Non-object root
  {
    std::string junk = "[1,2,3]";
    std::vector<std::byte> blob(8);
    uint64_t n = junk.size();
    std::memcpy(blob.data(), &n, 8);
    for (char c : junk) blob.push_back(static_cast<std::byte>(c));
    auto path = write_blob("badroot", blob);
    REQUIRE_THROWS(SafeTensors::open(path.string()));
    fs::remove(path);
  }

  // Tensor byte count mismatch (shape says 2 floats = 8B, offsets say 4B)
  {
    std::string json = R"({"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})";
    std::vector<std::byte> blob(8);
    uint64_t n = json.size();
    std::memcpy(blob.data(), &n, 8);
    for (char c : json) blob.push_back(static_cast<std::byte>(c));
    blob.resize(blob.size() + 4);  // only 4 data bytes
    auto path = write_blob("sizemismatch", blob);
    REQUIRE_THROWS(SafeTensors::open(path.string()));
    fs::remove(path);
  }

  // data_offsets out of range (runs past file end)
  {
    std::string json = R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    std::vector<std::byte> blob(8);
    uint64_t n = json.size();
    std::memcpy(blob.data(), &n, 8);
    for (char c : json) blob.push_back(static_cast<std::byte>(c));
    // No data region at all — off_end=4 > data_size=0.
    auto path = write_blob("overshoot", blob);
    REQUIRE_THROWS(SafeTensors::open(path.string()));
    fs::remove(path);
  }

  // Unknown dtype
  {
    std::string json = R"({"w":{"dtype":"Q42","shape":[1],"data_offsets":[0,4]}})";
    std::vector<std::byte> blob(8);
    uint64_t n = json.size();
    std::memcpy(blob.data(), &n, 8);
    for (char c : json) blob.push_back(static_cast<std::byte>(c));
    blob.resize(blob.size() + 4);
    auto path = write_blob("baddtype", blob);
    REQUIRE_THROWS(SafeTensors::open(path.string()));
    fs::remove(path);
  }
}

TEST_CASE("SafeTensors: dtype string mapping is complete") {
  using namespace tesseract::io;
  REQUIRE(safetensors_dtype_from_string("F32")  == DType::Float32);
  REQUIRE(safetensors_dtype_from_string("F64")  == DType::Float64);
  REQUIRE(safetensors_dtype_from_string("F16")  == DType::Float16);
  REQUIRE(safetensors_dtype_from_string("BF16") == DType::BFloat16);
  REQUIRE(safetensors_dtype_from_string("I64")  == DType::Int64);
  REQUIRE(safetensors_dtype_from_string("I32")  == DType::Int32);
  REQUIRE(safetensors_dtype_from_string("I8")   == DType::Int8);
  REQUIRE(safetensors_dtype_from_string("BOOL") == DType::Bool);

  REQUIRE(safetensors_dtype_to_string(DType::Float32) == "F32");
  REQUIRE(safetensors_dtype_to_string(DType::BFloat16) == "BF16");

  REQUIRE_THROWS(safetensors_dtype_from_string("UNKNOWN"));
}

// Tesseract Studio — minimal dependency-free JSON (M5 / B-047).
//
// A compact JSON value type with a recursive-descent parser and a stable
// (insertion-ordered) serializer. Studio deliberately avoids any third-party
// dependency (the dev box has no reliable network for FetchContent), mirroring
// the framework's existing "dependency-free parser" convention (see the HF
// config.json reader in src/models/Llama.cpp). Scope is exactly what the
// .tsb block-graph schema and the REST control plane need: null / bool /
// number / string / array / object, with order-preserving objects so emitted
// files are byte-stable.

#ifndef TESSERACT_STUDIO_JSON_HPP
#define TESSERACT_STUDIO_JSON_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tesseract::studio {

class Json {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  using Array = std::vector<Json>;
  using Member = std::pair<std::string, Json>;
  using Object = std::vector<Member>;  // insertion-ordered

  Json() : type_(Type::Null) {}
  Json(std::nullptr_t) : type_(Type::Null) {}
  Json(bool b) : type_(Type::Bool), bool_(b) {}
  Json(int v) : type_(Type::Number), num_(static_cast<double>(v)) {}
  Json(int64_t v) : type_(Type::Number), num_(static_cast<double>(v)) {}
  Json(double v) : type_(Type::Number), num_(v) {}
  Json(const char* s) : type_(Type::String), str_(s) {}
  Json(std::string s) : type_(Type::String), str_(std::move(s)) {}

  static Json array() { Json j; j.type_ = Type::Array; return j; }
  static Json object() { Json j; j.type_ = Type::Object; return j; }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  // Typed accessors with defaults (forgiving — Studio config is user-authored).
  bool as_bool(bool d = false) const { return is_bool() ? bool_ : d; }
  double as_number(double d = 0.0) const { return is_number() ? num_ : d; }
  int64_t as_int(int64_t d = 0) const {
    return is_number() ? static_cast<int64_t>(num_) : d;
  }
  const std::string& as_string(const std::string& d = empty_string()) const {
    return is_string() ? str_ : d;
  }

  const Array& items() const { return arr_; }
  Array& items() { return arr_; }
  const Object& members() const { return obj_; }

  // Object access. `contains` / `at` are read-only; `operator[]` inserts.
  bool contains(const std::string& key) const { return find(key) != nullptr; }
  const Json& at(const std::string& key) const {
    const Json* p = find(key);
    if (!p) throw std::out_of_range("Json: missing key '" + key + "'");
    return *p;
  }
  const Json& value(const std::string& key) const {
    const Json* p = find(key);
    return p ? *p : null_singleton();
  }
  Json& operator[](const std::string& key);

  // Array building.
  void push_back(Json v) {
    if (type_ != Type::Array) { type_ = Type::Array; arr_.clear(); }
    arr_.push_back(std::move(v));
  }
  // Object building (keeps insertion order; overwrites an existing key).
  void set(const std::string& key, Json v);

  // Serialize. `indent >= 0` pretty-prints with that many spaces per level.
  std::string dump(int indent = -1) const;

  // Parse. Throws std::runtime_error with a 1-based offset on malformed input.
  static Json parse(const std::string& text);

 private:
  static const std::string& empty_string();
  static const Json& null_singleton();
  const Json* find(const std::string& key) const;

  void dump_to(std::string& out, int indent, int depth) const;

  Type type_;
  bool bool_ = false;
  double num_ = 0.0;
  std::string str_;
  Array arr_;
  Object obj_;
};

}  // namespace tesseract::studio

#endif  // TESSERACT_STUDIO_JSON_HPP

#include "tesseract/studio/Json.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>

namespace tesseract::studio {

const std::string& Json::empty_string() {
  static const std::string kEmpty;
  return kEmpty;
}

const Json& Json::null_singleton() {
  static const Json kNull;
  return kNull;
}

const Json* Json::find(const std::string& key) const {
  if (type_ != Type::Object) return nullptr;
  for (const auto& m : obj_)
    if (m.first == key) return &m.second;
  return nullptr;
}

Json& Json::operator[](const std::string& key) {
  if (type_ != Type::Object) { type_ = Type::Object; obj_.clear(); }
  for (auto& m : obj_)
    if (m.first == key) return m.second;
  obj_.emplace_back(key, Json());
  return obj_.back().second;
}

void Json::set(const std::string& key, Json v) {
  (*this)[key] = std::move(v);
}

// --------------------------------------------------------------------------- //
// Serialization                                                               //
// --------------------------------------------------------------------------- //
namespace {

void escape_string(const std::string& s, std::string& out) {
  out.push_back('"');
  for (char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

void format_number(double v, std::string& out) {
  if (std::isfinite(v) && v == static_cast<double>(static_cast<int64_t>(v))) {
    out += std::to_string(static_cast<int64_t>(v));
    return;
  }
  std::ostringstream ss;
  ss.precision(17);
  ss << v;
  out += ss.str();
}

}  // namespace

void Json::dump_to(std::string& out, int indent, int depth) const {
  const bool pretty = indent >= 0;
  const std::string nl = pretty ? "\n" : "";
  auto pad = [&](int d) {
    if (pretty) out.append(static_cast<size_t>(indent) * d, ' ');
  };
  switch (type_) {
    case Type::Null:   out += "null"; break;
    case Type::Bool:   out += bool_ ? "true" : "false"; break;
    case Type::Number: format_number(num_, out); break;
    case Type::String: escape_string(str_, out); break;
    case Type::Array: {
      if (arr_.empty()) { out += "[]"; break; }
      out += "[";
      out += nl;
      for (size_t i = 0; i < arr_.size(); ++i) {
        pad(depth + 1);
        arr_[i].dump_to(out, indent, depth + 1);
        if (i + 1 < arr_.size()) out += ",";
        out += nl;
      }
      pad(depth);
      out += "]";
      break;
    }
    case Type::Object: {
      if (obj_.empty()) { out += "{}"; break; }
      out += "{";
      out += nl;
      for (size_t i = 0; i < obj_.size(); ++i) {
        pad(depth + 1);
        escape_string(obj_[i].first, out);
        out += pretty ? ": " : ":";
        obj_[i].second.dump_to(out, indent, depth + 1);
        if (i + 1 < obj_.size()) out += ",";
        out += nl;
      }
      pad(depth);
      out += "}";
      break;
    }
  }
}

std::string Json::dump(int indent) const {
  std::string out;
  dump_to(out, indent, 0);
  return out;
}

// --------------------------------------------------------------------------- //
// Parsing                                                                     //
// --------------------------------------------------------------------------- //
namespace {

struct Parser {
  const std::string& s;
  size_t i = 0;

  explicit Parser(const std::string& text) : s(text) {}

  [[noreturn]] void fail(const std::string& msg) const {
    throw std::runtime_error("JSON parse error at offset " +
                             std::to_string(i + 1) + ": " + msg);
  }

  void skip_ws() {
    while (i < s.size()) {
      char c = s[i];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i;
      else break;
    }
  }

  char peek() {
    if (i >= s.size()) fail("unexpected end of input");
    return s[i];
  }

  Json parse_value() {
    skip_ws();
    char c = peek();
    switch (c) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Json(parse_string());
      case 't': case 'f': return parse_bool();
      case 'n': return parse_null();
      default:  return parse_number();
    }
  }

  std::string parse_string() {
    if (peek() != '"') fail("expected string");
    ++i;
    std::string out;
    while (true) {
      if (i >= s.size()) fail("unterminated string");
      char c = s[i++];
      if (c == '"') break;
      if (c == '\\') {
        if (i >= s.size()) fail("unterminated escape");
        char e = s[i++];
        switch (e) {
          case '"':  out.push_back('"');  break;
          case '\\': out.push_back('\\'); break;
          case '/':  out.push_back('/');  break;
          case 'n':  out.push_back('\n'); break;
          case 't':  out.push_back('\t'); break;
          case 'r':  out.push_back('\r'); break;
          case 'b':  out.push_back('\b'); break;
          case 'f':  out.push_back('\f'); break;
          case 'u': {
            if (i + 4 > s.size()) fail("bad \\u escape");
            unsigned cp = 0;
            for (int k = 0; k < 4; ++k) {
              char h = s[i++];
              cp <<= 4;
              if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
              else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
              else fail("bad hex in \\u escape");
            }
            // Encode the BMP code point as UTF-8 (surrogate pairs not needed
            // for the ASCII-centric .tsb schema).
            if (cp < 0x80) {
              out.push_back(static_cast<char>(cp));
            } else if (cp < 0x800) {
              out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
              out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
              out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            break;
          }
          default: fail("invalid escape");
        }
      } else {
        out.push_back(c);
      }
    }
    return out;
  }

  Json parse_number() {
    size_t start = i;
    if (peek() == '-') ++i;
    while (i < s.size()) {
      char c = s[i];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
          c == '+' || c == '-') {
        ++i;
      } else {
        break;
      }
    }
    if (i == start) fail("expected value");
    try {
      return Json(std::stod(s.substr(start, i - start)));
    } catch (...) {
      fail("invalid number");
    }
  }

  Json parse_bool() {
    if (s.compare(i, 4, "true") == 0) { i += 4; return Json(true); }
    if (s.compare(i, 5, "false") == 0) { i += 5; return Json(false); }
    fail("invalid literal");
  }

  Json parse_null() {
    if (s.compare(i, 4, "null") == 0) { i += 4; return Json(); }
    fail("invalid literal");
  }

  Json parse_array() {
    ++i;  // consume '['
    Json arr = Json::array();
    skip_ws();
    if (peek() == ']') { ++i; return arr; }
    while (true) {
      arr.push_back(parse_value());
      skip_ws();
      char c = peek();
      if (c == ',') { ++i; continue; }
      if (c == ']') { ++i; break; }
      fail("expected ',' or ']'");
    }
    return arr;
  }

  Json parse_object() {
    ++i;  // consume '{'
    Json obj = Json::object();
    skip_ws();
    if (peek() == '}') { ++i; return obj; }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      if (peek() != ':') fail("expected ':'");
      ++i;
      obj.set(key, parse_value());
      skip_ws();
      char c = peek();
      if (c == ',') { ++i; continue; }
      if (c == '}') { ++i; break; }
      fail("expected ',' or '}'");
    }
    return obj;
  }
};

}  // namespace

Json Json::parse(const std::string& text) {
  Parser p(text);
  Json v = p.parse_value();
  p.skip_ws();
  if (p.i != text.size())
    p.fail("trailing characters after value");
  return v;
}

}  // namespace tesseract::studio

#include "tesseract/io/BpeTokenizer.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#include "tesseract/utils/Logging.hpp"

namespace tesseract::io {

namespace {

// ===========================================================================
// Minimal generic JSON parser (recursive descent). tokenizer.json is far
// richer than a safetensors header — nested objects, arrays of arrays,
// 128k-entry vocab — so this is a full value-tree parser rather than the
// targeted scanner in SafeTensors.cpp. Objects preserve insertion order
// in a vector<pair> (we never random-access into the vocab object; we
// iterate it once to build the reverse index, so a hash map over 128k
// keys would be wasted work).
// ===========================================================================

struct JValue {
  enum class T { Null, Bool, Num, Str, Arr, Obj };
  T t = T::Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<JValue> arr;
  std::vector<std::pair<std::string, JValue>> obj;

  const JValue* find(std::string_view key) const {
    for (const auto& kv : obj) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }
};

struct JsonParser {
  std::string_view s;
  std::size_t pos{0};

  [[noreturn]] void fail(const std::string& what) const {
    const std::size_t lo = pos > 40 ? pos - 40 : 0;
    const std::size_t hi = std::min(s.size(), pos + 40);
    TESSERACT_THROW("tokenizer.json parse error at offset {}: {} (near '{}')",
                    pos, what, std::string(s.substr(lo, hi - lo)));
  }

  void skip_ws() {
    while (pos < s.size()) {
      const char c = s[pos];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos;
      else break;
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
    if (pos < s.size() && s[pos] == c) { ++pos; return true; }
    return false;
  }

  void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out.push_back(char(cp));
    } else if (cp < 0x800) {
      out.push_back(char(0xC0 | (cp >> 6)));
      out.push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(char(0xE0 | (cp >> 12)));
      out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(char(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(char(0xF0 | (cp >> 18)));
      out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(char(0x80 | (cp & 0x3F)));
    }
  }

  uint32_t parse_hex4() {
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
    return cp;
  }

  std::string parse_string() {
    if (peek() != '"') fail("expected string");
    ++pos;
    std::string out;
    while (pos < s.size()) {
      const char c = s[pos++];
      if (c == '"') return out;
      if (c != '\\') { out.push_back(c); continue; }
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
          uint32_t cp = parse_hex4();
          // UTF-16 surrogate pair: high surrogate followed by \uXXXX low.
          if (cp >= 0xD800 && cp <= 0xDBFF && pos + 1 < s.size() &&
              s[pos] == '\\' && s[pos + 1] == 'u') {
            pos += 2;
            const uint32_t lo = parse_hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("unknown escape");
      }
    }
    fail("unterminated string");
  }

  double parse_number() {
    skip_ws();
    const std::size_t start = pos;
    if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) ++pos;
    while (pos < s.size()) {
      const char ch = s[pos];
      const bool num = (ch >= '0' && ch <= '9') || ch == '.' || ch == 'e' ||
                       ch == 'E' || ch == '+' || ch == '-';
      if (!num) break;
      ++pos;
    }
    if (pos == start) fail("expected number");
    try {
      return std::stod(std::string(s.substr(start, pos - start)));
    } catch (const std::exception&) {
      fail("number out of range");
    }
  }

  JValue parse_value() {
    const char c = peek();
    if (c == '"') { JValue v; v.t = JValue::T::Str; v.str = parse_string(); return v; }
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == 't') { if (s.compare(pos, 4, "true") == 0)  { pos += 4; JValue v; v.t = JValue::T::Bool; v.b = true;  return v; } fail("bad literal"); }
    if (c == 'f') { if (s.compare(pos, 5, "false") == 0) { pos += 5; JValue v; v.t = JValue::T::Bool; v.b = false; return v; } fail("bad literal"); }
    if (c == 'n') { if (s.compare(pos, 4, "null") == 0)  { pos += 4; return JValue{}; } fail("bad literal"); }
    JValue v; v.t = JValue::T::Num; v.num = parse_number(); return v;
  }

  JValue parse_object() {
    expect('{');
    JValue v; v.t = JValue::T::Obj;
    if (match('}')) return v;
    while (true) {
      std::string key = parse_string();
      expect(':');
      v.obj.emplace_back(std::move(key), parse_value());
      if (match('}')) return v;
      expect(',');
    }
  }

  JValue parse_array() {
    expect('[');
    JValue v; v.t = JValue::T::Arr;
    if (match(']')) return v;
    while (true) {
      v.arr.push_back(parse_value());
      if (match(']')) return v;
      expect(',');
    }
  }
};

// ===========================================================================
// GPT-2 byte <-> unicode reversible map.
// ===========================================================================

std::string utf8_of_codepoint(uint32_t cp) {
  std::string out;
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
  return out;
}

struct ByteUnicode {
  std::array<std::string, 256> byte_to_sym;     // byte -> UTF-8 of mapped cp
  std::unordered_map<uint32_t, int> cp_to_byte; // mapped cp -> original byte

  ByteUnicode() {
    std::array<bool, 256> in_printable{};
    std::array<uint32_t, 256> cp{};
    auto mark = [&](int lo, int hi) {
      for (int b = lo; b <= hi; ++b) { in_printable[b] = true; cp[b] = uint32_t(b); }
    };
    // Same printable ranges as GPT-2's bytes_to_unicode().
    mark('!', '~');      // 33..126
    mark(0xA1, 0xAC);    // ¡..¬
    mark(0xAE, 0xFF);    // ®..ÿ
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (!in_printable[b]) { cp[b] = uint32_t(256 + n); ++n; }
    }
    for (int b = 0; b < 256; ++b) {
      byte_to_sym[b] = utf8_of_codepoint(cp[b]);
      cp_to_byte[cp[b]] = b;
    }
  }
};

const ByteUnicode& byte_unicode() {
  static const ByteUnicode bu;
  return bu;
}

// Decode a UTF-8 string into codepoints.
std::vector<uint32_t> utf8_to_codepoints(std::string_view s) {
  std::vector<uint32_t> out;
  std::size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp = 0;
    int extra = 0;
    if (c < 0x80) { cp = c; extra = 0; }
    else if ((c >> 5) == 0x6) { cp = c & 0x1F; extra = 1; }
    else if ((c >> 4) == 0xE) { cp = c & 0x0F; extra = 2; }
    else if ((c >> 3) == 0x1E) { cp = c & 0x07; extra = 3; }
    else { cp = c; extra = 0; }  // invalid lead — pass through
    ++i;
    for (int k = 0; k < extra && i < s.size(); ++k, ++i) {
      cp = (cp << 6) | (static_cast<unsigned char>(s[i]) & 0x3F);
    }
    out.push_back(cp);
  }
  return out;
}

// ---- Byte classification for the GPT-2 pre-tokenizer (ASCII-exact). ----
bool is_space(unsigned char b) {
  return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\f' || b == '\v';
}
bool is_letter(unsigned char b) {
  return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b >= 0x80;
}
bool is_digit(unsigned char b) { return b >= '0' && b <= '9'; }
bool is_other(unsigned char b) {
  return !is_space(b) && !is_letter(b) && !is_digit(b);
}

// GPT-2 pre-tokenization, implemented as a hand-rolled scanner over the
// raw bytes so we control byte classification exactly (std::regex can't
// express `\p{L}` and mishandles high bytes). Reproduces:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::string> pre_tokenize(std::string_view text) {
  std::vector<std::string> pieces;
  const std::size_t n = text.size();
  std::size_t i = 0;
  auto at = [&](std::size_t k) { return static_cast<unsigned char>(text[k]); };

  while (i < n) {
    const unsigned char c = at(i);

    // 1) Contractions (lowercase only, matching the GPT-2 pattern).
    if (c == '\'') {
      if (i + 2 < n) {
        const unsigned char a = at(i + 1), b = at(i + 2);
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
            (a == 'l' && b == 'l')) {
          pieces.emplace_back(text.substr(i, 3)); i += 3; continue;
        }
      }
      if (i + 1 < n) {
        const unsigned char a = at(i + 1);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') {
          pieces.emplace_back(text.substr(i, 2)); i += 2; continue;
        }
      }
    }

    // 2-4) Optional single leading space, then a run of letters / digits /
    //      "other" (non-space non-letter non-digit).
    const bool has_space = (c == ' ');
    const std::size_t k = has_space ? i + 1 : i;
    if (k < n) {
      const unsigned char ck = at(k);
      if (is_letter(ck)) {
        std::size_t m = k; while (m < n && is_letter(at(m))) ++m;
        pieces.emplace_back(text.substr(i, m - i)); i = m; continue;
      }
      if (is_digit(ck)) {
        std::size_t m = k; while (m < n && is_digit(at(m))) ++m;
        pieces.emplace_back(text.substr(i, m - i)); i = m; continue;
      }
      if (is_other(ck)) {
        std::size_t m = k; while (m < n && is_other(at(m))) ++m;
        pieces.emplace_back(text.substr(i, m - i)); i = m; continue;
      }
    }

    // 5) Whitespace run: `\s+(?!\S)|\s+`. Match the maximal run; if it is
    //    followed by a non-whitespace char, give back the last space so it
    //    becomes the leading space of the next token (handled by 2-4 above
    //    on the next iteration). A run that reaches end-of-string is kept
    //    whole.
    if (is_space(c)) {
      std::size_t m = i; while (m < n && is_space(at(m))) ++m;
      if (m < n && m - 1 > i) {
        pieces.emplace_back(text.substr(i, (m - 1) - i)); i = m - 1; continue;
      }
      if (m >= n) { pieces.emplace_back(text.substr(i, m - i)); i = m; continue; }
      // Single trailing space before a non-ws char: it is the next token's
      // leading space — let 2-4 consume it next iteration. But 2-4 only ran
      // for position i (this same space) and fell through because the char
      // after the space was itself whitespace... which can't happen here
      // (m-1 == i means run length 1, followed by non-ws). Defensive: emit
      // it standalone to guarantee forward progress.
      pieces.emplace_back(text.substr(i, 1)); i += 1; continue;
    }

    // Should be unreachable (every byte is space/letter/digit/other), but
    // guarantee progress.
    pieces.emplace_back(text.substr(i, 1)); ++i;
  }
  return pieces;
}

constexpr int kNoRank = std::numeric_limits<int>::max();

}  // namespace

// ===========================================================================
// BpeTokenizer
// ===========================================================================

BpeTokenizer::BpeTokenizer(Config cfg) { build(std::move(cfg)); }

void BpeTokenizer::build(Config&& cfg) {
  TESSERACT_CHECK(!cfg.vocab.empty(), "BpeTokenizer: vocab must be non-empty");

  token_to_id_.reserve(cfg.vocab.size());
  id_to_token_.reserve(cfg.vocab.size());
  for (auto& [tok, id] : cfg.vocab) {
    token_to_id_.try_emplace(tok, id);
    id_to_token_.try_emplace(id, tok);
  }
  // Added tokens extend the id space; record them in both maps so
  // vocab_size + decode see them.
  for (auto& [content, id] : cfg.added_tokens) {
    token_to_id_.try_emplace(content, id);
    id_to_token_.try_emplace(id, content);
    added_id_set_[id] = true;
  }
  added_tokens_ = std::move(cfg.added_tokens);

  merge_rank_.reserve(cfg.merges.size());
  for (std::size_t r = 0; r < cfg.merges.size(); ++r) {
    const auto& [a, b] = cfg.merges[r];
    merge_rank_.try_emplace(a + " " + b, static_cast<int>(r));
  }

  add_prefix_space_ = cfg.add_prefix_space;
  bos_id_ = cfg.bos_id;
  eos_id_ = cfg.eos_id;
  pad_id_ = cfg.pad_id;
  unk_id_ = cfg.unk_id;
}

bool BpeTokenizer::is_added_id(int32_t id) const noexcept {
  return added_id_set_.find(id) != added_id_set_.end();
}

std::vector<std::string> BpeTokenizer::bpe(
    const std::vector<std::string>& symbols) const {
  if (symbols.size() < 2) return symbols;
  std::vector<std::string> word = symbols;

  while (word.size() >= 2) {
    // Find the adjacent pair with the lowest merge rank.
    int best_rank = kNoRank;
    std::size_t best_i = 0;
    for (std::size_t i = 0; i + 1 < word.size(); ++i) {
      const auto it = merge_rank_.find(word[i] + " " + word[i + 1]);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_rank == kNoRank) break;

    const std::string first = word[best_i];
    const std::string second = word[best_i + 1];
    // Merge every non-overlapping occurrence of (first, second) — the
    // canonical GPT-2 `bpe()` behavior.
    std::vector<std::string> merged;
    merged.reserve(word.size());
    std::size_t i = 0;
    while (i < word.size()) {
      if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
        merged.push_back(first + second);
        i += 2;
      } else {
        merged.push_back(word[i]);
        i += 1;
      }
    }
    word.swap(merged);
  }
  return word;
}

void BpeTokenizer::encode_piece(const std::string& piece,
                                std::vector<int32_t>& out) const {
  if (piece.empty()) return;
  const auto& bu = byte_unicode();

  // Initial symbols: each raw byte mapped through bytes_to_unicode.
  std::vector<std::string> symbols;
  symbols.reserve(piece.size());
  for (unsigned char b : piece) symbols.push_back(bu.byte_to_sym[b]);

  const std::vector<std::string> merged = bpe(symbols);
  for (const auto& sym : merged) {
    const auto it = token_to_id_.find(sym);
    if (it != token_to_id_.end()) {
      out.push_back(it->second);
    } else if (unk_id_ >= 0) {
      out.push_back(unk_id_);
    } else {
      TESSERACT_THROW("BpeTokenizer: symbol '{}' missing from vocab and no "
                      "unk token configured", sym);
    }
  }
}

std::vector<int32_t> BpeTokenizer::encode(std::string_view text,
                                          bool add_special_tokens) const {
  std::vector<int32_t> out;
  if (add_special_tokens && bos_id_ >= 0) out.push_back(bos_id_);

  // Split off any added/special tokens (matched as whole units, longest
  // match wins) so BPE never sees inside them.
  std::string buffered;  // accumulates ordinary text between specials
  auto flush = [&](std::string_view chunk) {
    if (chunk.empty()) return;
    std::string staged;
    if (add_prefix_space_ && (chunk.empty() || chunk.front() != ' ')) {
      staged.push_back(' ');
    }
    staged.append(chunk);
    for (const auto& piece : pre_tokenize(staged)) encode_piece(piece, out);
  };

  std::size_t i = 0;
  const std::size_t n = text.size();
  std::size_t seg_start = 0;
  while (i < n) {
    // Longest-match against the added-token contents at position i.
    int32_t hit_id = -1;
    std::size_t hit_len = 0;
    for (const auto& [content, id] : added_tokens_) {
      if (!content.empty() && content.size() > hit_len &&
          i + content.size() <= n &&
          text.compare(i, content.size(), content) == 0) {
        hit_len = content.size();
        hit_id = id;
      }
    }
    if (hit_id >= 0) {
      flush(text.substr(seg_start, i - seg_start));
      out.push_back(hit_id);
      i += hit_len;
      seg_start = i;
    } else {
      ++i;
    }
  }
  flush(text.substr(seg_start, n - seg_start));

  if (add_special_tokens && eos_id_ >= 0) out.push_back(eos_id_);
  return out;
}

std::string BpeTokenizer::decode(std::span<const int32_t> ids,
                                 bool skip_special_tokens) const {
  // Concatenate the byte-unicode token strings, then invert the
  // bytes_to_unicode map codepoint-by-codepoint back to raw bytes.
  std::string acc;
  for (int32_t id : ids) {
    if (skip_special_tokens && is_added_id(id)) continue;
    const auto it = id_to_token_.find(id);
    if (it == id_to_token_.end()) continue;  // OOB / unknown id — skip
    acc.append(it->second);
  }

  const auto& bu = byte_unicode();
  std::string out;
  out.reserve(acc.size());
  for (uint32_t cp : utf8_to_codepoints(acc)) {
    const auto it = bu.cp_to_byte.find(cp);
    if (it != bu.cp_to_byte.end()) {
      out.push_back(static_cast<char>(static_cast<unsigned char>(it->second)));
    }
    // Codepoints outside the byte-unicode map (e.g. raw special-token
    // text that wasn't byte-encoded) are dropped — they have no byte
    // preimage. Special tokens are normally skipped above anyway.
  }
  return out;
}

// ---- tokenizer.json loading ----------------------------------------------

namespace {

// Pull an integer out of a JSON number value.
int32_t as_int(const JValue& v) {
  TESSERACT_CHECK(v.t == JValue::T::Num, "tokenizer.json: expected number");
  return static_cast<int32_t>(v.num);
}

// Find the `pre_tokenizer`'s add_prefix_space, scanning a possible
// `Sequence` of pre-tokenizers for a `ByteLevel` entry.
bool extract_add_prefix_space(const JValue& root) {
  const JValue* pt = root.find("pre_tokenizer");
  if (!pt || pt->t != JValue::T::Obj) return false;
  auto read = [](const JValue& o) -> int {
    const JValue* aps = o.find("add_prefix_space");
    if (aps && aps->t == JValue::T::Bool) return aps->b ? 1 : 0;
    return -1;
  };
  const JValue* type = pt->find("type");
  if (type && type->t == JValue::T::Str && type->str == "ByteLevel") {
    const int r = read(*pt); if (r >= 0) return r != 0;
  }
  const JValue* seq = pt->find("pretokenizers");
  if (seq && seq->t == JValue::T::Arr) {
    for (const auto& e : seq->arr) {
      const JValue* t = e.find("type");
      if (t && t->t == JValue::T::Str && t->str == "ByteLevel") {
        const int r = read(e); if (r >= 0) return r != 0;
      }
    }
  }
  return false;
}

}  // namespace

BpeTokenizer BpeTokenizer::from_json(std::string_view json) {
  JsonParser jp{json, 0};
  const JValue root = jp.parse_value();
  TESSERACT_CHECK(root.t == JValue::T::Obj,
                  "tokenizer.json: top-level value must be an object");

  const JValue* model = root.find("model");
  TESSERACT_CHECK(model && model->t == JValue::T::Obj,
                  "tokenizer.json: missing 'model' object");

  Config cfg;

  // model.vocab : { token_string : id }
  const JValue* vocab = model->find("vocab");
  TESSERACT_CHECK(vocab && vocab->t == JValue::T::Obj,
                  "tokenizer.json: missing 'model.vocab' object");
  cfg.vocab.reserve(vocab->obj.size());
  for (const auto& [tok, id] : vocab->obj) {
    cfg.vocab.emplace_back(tok, as_int(id));
  }

  // model.merges : either ["a b", ...] or [["a","b"], ...].
  const JValue* merges = model->find("merges");
  if (merges && merges->t == JValue::T::Arr) {
    cfg.merges.reserve(merges->arr.size());
    for (const auto& m : merges->arr) {
      if (m.t == JValue::T::Str) {
        const std::string& ms = m.str;
        const auto sp = ms.find(' ');
        TESSERACT_CHECK(sp != std::string::npos,
                        "tokenizer.json: malformed merge rule '{}'", ms);
        cfg.merges.emplace_back(ms.substr(0, sp), ms.substr(sp + 1));
      } else if (m.t == JValue::T::Arr && m.arr.size() == 2 &&
                 m.arr[0].t == JValue::T::Str && m.arr[1].t == JValue::T::Str) {
        cfg.merges.emplace_back(m.arr[0].str, m.arr[1].str);
      } else {
        TESSERACT_THROW("tokenizer.json: malformed merge entry");
      }
    }
  }

  // added_tokens : [{ id, content, special, ... }]
  const JValue* added = root.find("added_tokens");
  if (added && added->t == JValue::T::Arr) {
    for (const auto& a : added->arr) {
      const JValue* id = a.find("id");
      const JValue* content = a.find("content");
      if (id && content && content->t == JValue::T::Str) {
        cfg.added_tokens.emplace_back(content->str, as_int(*id));
      }
    }
  }

  cfg.add_prefix_space = extract_add_prefix_space(root);

  // Resolve BOS/EOS/PAD/UNK by the usual content conventions. The first
  // match wins; callers can re-set ids after construction if a checkpoint
  // uses non-standard specials.
  auto find_special = [&](std::initializer_list<const char*> names) -> int32_t {
    for (const char* nm : names) {
      for (const auto& [content, id] : cfg.added_tokens) {
        if (content == nm) return id;
      }
    }
    return -1;
  };
  cfg.bos_id = find_special({"<|begin_of_text|>", "<s>", "<|startoftext|>"});
  cfg.eos_id = find_special({"<|end_of_text|>", "<|eot_id|>", "</s>",
                             "<|endoftext|>"});
  cfg.pad_id = find_special({"<|pad|>", "<pad>", "[PAD]"});
  cfg.unk_id = find_special({"<unk>", "[UNK]"});

  return BpeTokenizer(std::move(cfg));
}

BpeTokenizer BpeTokenizer::from_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  TESSERACT_CHECK(f.good(), "BpeTokenizer: cannot open '{}'", path);
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string json = ss.str();
  return from_json(json);
}

}  // namespace tesseract::io

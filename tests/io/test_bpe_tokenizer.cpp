// Unit + golden-parity tests for `tesseract::io::BpeTokenizer`.
//
// Three layers of coverage:
//   1. In-memory algorithm tests — a hand-built vocab + merge table whose
//      output we can verify by hand, exercising the rank-priority
//      merge-all-occurrences loop and the bytes_to_unicode round-trip.
//   2. tokenizer.json loading — parse a real HF `tokenizers` ByteLevel BPE
//      checkpoint (committed under fixtures/) and sanity-check the parsed
//      vocab/merges/specials.
//   3. Golden parity — re-encode every fixture sentence and assert the ids
//      match the `tokenizers.Tokenizer.encode` output byte-for-byte. The
//      fixtures are produced by fixtures/generate_bpe_fixture.py against
//      the real HF `tokenizers` library.

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/io/BpeTokenizer.hpp"

using tesseract::io::BpeTokenizer;
using tesseract::io::Tokenizer;

#ifndef TESSERACT_BPE_FIXTURE_DIR
#error "TESSERACT_BPE_FIXTURE_DIR must be defined by the build"
#endif

namespace {

std::string fixture_path(const char* name) {
  return std::string(TESSERACT_BPE_FIXTURE_DIR) + "/" + name;
}

// Decode a JSON-encoded string literal (the right column of the golden
// file) back to raw bytes. Handles the escapes the generator emits:
// \t \n \r \" \\ \/ and \uXXXX (BMP only — fixtures are ASCII).
std::string json_unescape(std::string_view s) {
  // Strip surrounding quotes.
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
    s = s.substr(1, s.size() - 2);
  }
  std::string out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '\\') { out.push_back(s[i]); continue; }
    ++i;
    if (i >= s.size()) break;
    switch (s[i]) {
      case 't': out.push_back('\t'); break;
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case '/': out.push_back('/'); break;
      case 'u': {
        if (i + 4 < s.size()) {
          const std::string hex(s.substr(i + 1, 4));
          const auto cp = static_cast<unsigned>(std::stoul(hex, nullptr, 16));
          if (cp < 0x80) out.push_back(static_cast<char>(cp));
          i += 4;
        }
        break;
      }
      default: out.push_back(s[i]); break;
    }
  }
  return out;
}

struct GoldenLine {
  std::vector<int32_t> ids;
  std::string text;
};

std::vector<GoldenLine> load_golden() {
  std::ifstream f(fixture_path("bpe_golden.txt"));
  REQUIRE(f.good());
  std::vector<GoldenLine> out;
  std::string line;
  while (std::getline(f, line)) {
    const auto tab = line.find('\t');
    REQUIRE(tab != std::string::npos);
    GoldenLine g;
    {
      std::istringstream ids_ss(line.substr(0, tab));
      int32_t id = 0;
      while (ids_ss >> id) g.ids.push_back(id);
    }
    g.text = json_unescape(line.substr(tab + 1));
    out.push_back(std::move(g));
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Layer 1 — in-memory algorithm
// ---------------------------------------------------------------------------

TEST_CASE("BpeTokenizer: rank-priority merge-all-occurrences loop") {
  // vocab: a=0, aa=1, aaa=2 ; merges: (a,a) rank0, (aa,a) rank1.
  // Encoding "aaa": [a,a,a] --(a,a)--> [aa,a] --(aa,a)--> [aaa] => [2].
  BpeTokenizer::Config cfg;
  cfg.vocab = {{"a", 0}, {"aa", 1}, {"aaa", 2}};
  cfg.merges = {{"a", "a"}, {"aa", "a"}};
  cfg.unk_id = -1;
  BpeTokenizer tok(cfg);

  REQUIRE(tok.encode("aaa", /*add_special=*/false) == std::vector<int32_t>{2});
  // "aa" stops after one merge -> [aa] => [1].
  REQUIRE(tok.encode("aa", /*add_special=*/false) == std::vector<int32_t>{1});
  // "a" is a lone symbol => [0].
  REQUIRE(tok.encode("a", /*add_special=*/false) == std::vector<int32_t>{0});
}

TEST_CASE("BpeTokenizer: lower-rank pair wins regardless of position") {
  // vocab: b=0,c=1,d=2,bc=3,cd=4 ; merges: (c,d) rank0, (b,c) rank1.
  // Encoding "bcd": [b,c,d]. Lowest rank pair is (c,d) -> [b,cd]. No more
  // applicable merges (b,cd not a rule) => [b=0, cd=4].
  BpeTokenizer::Config cfg;
  cfg.vocab = {{"b", 0}, {"c", 1}, {"d", 2}, {"bc", 3}, {"cd", 4}};
  cfg.merges = {{"c", "d"}, {"b", "c"}};
  BpeTokenizer tok(cfg);
  REQUIRE(tok.encode("bcd", /*add_special=*/false) ==
          std::vector<int32_t>{0, 4});
}

TEST_CASE("BpeTokenizer: HF byte-level golden parity") {
  const BpeTokenizer tok = BpeTokenizer::from_file(fixture_path("bpe_tokenizer.json"));
  const auto golden = load_golden();
  REQUIRE(!golden.empty());
  for (const auto& g : golden) {
    INFO("text=[" << g.text << "]");
    const auto ids = tok.encode(g.text, /*add_special=*/false);
    REQUIRE(ids == g.ids);
  }
}

TEST_CASE("BpeTokenizer: loaded vocab size and specials") {
  const BpeTokenizer tok = BpeTokenizer::from_file(fixture_path("bpe_tokenizer.json"));
  // Trainer was asked for vocab_size=400 incl. the three specials.
  REQUIRE(tok.vocab_size() == 400);
  // <|begin_of_text|> / <|end_of_text|> resolved as BOS/EOS, <unk> as UNK.
  REQUIRE(tok.bos_token_id() >= 0);
  REQUIRE(tok.eos_token_id() >= 0);
  REQUIRE(tok.unk_token_id() >= 0);
  REQUIRE(tok.bos_token_id() != tok.eos_token_id());
}

TEST_CASE("BpeTokenizer: byte-level decode is the inverse of encode") {
  const BpeTokenizer tok = BpeTokenizer::from_file(fixture_path("bpe_tokenizer.json"));
  // Byte-level BPE is lossless: decode(encode(text)) == text for any ASCII
  // input (the byte alphabet covers all 256 bytes).
  const std::vector<std::string> texts = {
      "the quick brown fox",
      "punctuation, like; colons: periods.",
      "numbers 12 34 567 here",
      "  leading spaces",
      "CamelCase and snake_case",
  };
  for (const auto& text : texts) {
    INFO("text=[" << text << "]");
    const auto ids = tok.encode(text, /*add_special=*/false);
    REQUIRE(tok.decode(ids, /*skip_special=*/true) == text);
  }
}

TEST_CASE("BpeTokenizer: add_special prepends BOS and appends EOS") {
  const BpeTokenizer tok = BpeTokenizer::from_file(fixture_path("bpe_tokenizer.json"));
  const auto plain = tok.encode("the quick brown fox", /*add_special=*/false);
  const auto with_special = tok.encode("the quick brown fox", /*add_special=*/true);
  REQUIRE(with_special.size() == plain.size() + 2);
  REQUIRE(with_special.front() == tok.bos_token_id());
  REQUIRE(with_special.back() == tok.eos_token_id());
  // Interior ids unchanged.
  const std::vector<int32_t> interior(with_special.begin() + 1,
                                      with_special.end() - 1);
  REQUIRE(interior == plain);
}

TEST_CASE("BpeTokenizer: special-token strings are isolated, not BPE'd") {
  const BpeTokenizer tok = BpeTokenizer::from_file(fixture_path("bpe_tokenizer.json"));
  const std::string text = "<|begin_of_text|>the quick brown fox<|end_of_text|>";
  const auto ids = tok.encode(text, /*add_special=*/false);
  REQUIRE(ids.front() == tok.bos_token_id());
  REQUIRE(ids.back() == tok.eos_token_id());
  // The interior equals the plain encoding of the inner text.
  const auto inner = tok.encode("the quick brown fox", /*add_special=*/false);
  const std::vector<int32_t> interior(ids.begin() + 1, ids.end() - 1);
  REQUIRE(interior == inner);
}

TEST_CASE("BpeTokenizer: virtual dispatch via Tokenizer*") {
  const BpeTokenizer impl =
      BpeTokenizer::from_file(fixture_path("bpe_tokenizer.json"));
  const Tokenizer* base = &impl;
  const auto ids = base->encode("the quick brown fox", false);
  REQUIRE(!ids.empty());
  REQUIRE(base->decode(ids, true) == "the quick brown fox");
}

TEST_CASE("BpeTokenizer: rejects empty vocab") {
  BpeTokenizer::Config cfg;
  REQUIRE_THROWS(BpeTokenizer{cfg});
}

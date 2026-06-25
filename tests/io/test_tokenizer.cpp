// Unit tests for `tesseract::io::WhitespaceTokenizer`.
//
// Covers the `Tokenizer` contract that model loaders target:
//   - Round-trip: decode(encode(text)) is whitespace-normalized but
//     token-by-token identical.
//   - Special tokens (BOS/EOS) are prepended/appended when requested
//     and dropped by `decode(skip_special_tokens=true)`.
//   - UNK replacement when a token is not in vocab.
//   - vocab_size / *_token_id accessors reflect configured state.
//   - Config validation: missing specials rejected; empty vocab rejected.

#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/io/Tokenizer.hpp"

using tesseract::io::Tokenizer;
using tesseract::io::WhitespaceTokenizer;

namespace {

// Build a reusable test vocab: [<pad>=0, <bos>=1, <eos>=2, <unk>=3,
//   hello=4, world=5, foo=6, bar=7, tesseract=8].
WhitespaceTokenizer::Config make_config() {
  WhitespaceTokenizer::Config cfg;
  cfg.vocab = {
      "<pad>", "<bos>", "<eos>", "<unk>",
      "hello", "world", "foo", "bar", "tesseract",
  };
  cfg.bos = "<bos>";
  cfg.eos = "<eos>";
  cfg.pad = "<pad>";
  cfg.unk = "<unk>";
  return cfg;
}

}  // namespace

TEST_CASE("WhitespaceTokenizer: vocab_size and specials") {
  WhitespaceTokenizer tok(make_config());
  REQUIRE(tok.vocab_size() == 9);
  REQUIRE(tok.bos_token_id() == 1);
  REQUIRE(tok.eos_token_id() == 2);
  REQUIRE(tok.pad_token_id() == 0);
  REQUIRE(tok.unk_token_id() == 3);
}

TEST_CASE("WhitespaceTokenizer: encode adds bos/eos and maps tokens") {
  WhitespaceTokenizer tok(make_config());
  const auto ids = tok.encode("hello world tesseract", /*add_special=*/true);
  // Expected: <bos>=1, hello=4, world=5, tesseract=8, <eos>=2.
  REQUIRE(ids == std::vector<int32_t>{1, 4, 5, 8, 2});
}

TEST_CASE("WhitespaceTokenizer: encode(add_special=false) skips bos/eos") {
  WhitespaceTokenizer tok(make_config());
  const auto ids = tok.encode("foo bar", /*add_special=*/false);
  REQUIRE(ids == std::vector<int32_t>{6, 7});
}

TEST_CASE("WhitespaceTokenizer: unknown tokens map to UNK") {
  WhitespaceTokenizer tok(make_config());
  const auto ids = tok.encode("hello quuz world", /*add_special=*/false);
  // quuz -> <unk>=3.
  REQUIRE(ids == std::vector<int32_t>{4, 3, 5});
}

TEST_CASE("WhitespaceTokenizer: decode drops specials by default") {
  WhitespaceTokenizer tok(make_config());
  // <bos> hello world <eos>
  const std::vector<int32_t> ids = {1, 4, 5, 2};
  const auto out = tok.decode(ids, /*skip_special=*/true);
  REQUIRE(out == "hello world");
}

TEST_CASE("WhitespaceTokenizer: decode(skip_special=false) keeps specials") {
  WhitespaceTokenizer tok(make_config());
  const std::vector<int32_t> ids = {1, 4, 5, 2};
  const auto out = tok.decode(ids, /*skip_special=*/false);
  REQUIRE(out == "<bos> hello world <eos>");
}

TEST_CASE("WhitespaceTokenizer: round-trip is whitespace-normalized") {
  WhitespaceTokenizer tok(make_config());
  // Input has multiple whitespace kinds; decoded output normalizes to
  // single spaces and drops specials.
  const std::string text = "  hello\tworld\nfoo  \r\n bar tesseract ";
  const auto ids = tok.encode(text, /*add_special=*/true);
  const auto back = tok.decode(ids, /*skip_special=*/true);
  REQUIRE(back == "hello world foo bar tesseract");
}

TEST_CASE("WhitespaceTokenizer: decode ignores out-of-range ids") {
  WhitespaceTokenizer tok(make_config());
  // 999 is OOB; should be silently skipped.
  const std::vector<int32_t> ids = {4, 999, 5};
  const auto out = tok.decode(ids, /*skip_special=*/true);
  REQUIRE(out == "hello world");
}

TEST_CASE("WhitespaceTokenizer: rejects missing special in vocab") {
  WhitespaceTokenizer::Config cfg;
  cfg.vocab = {"hello", "world"};
  cfg.bos = "<bos>";  // not in vocab -> must throw.
  REQUIRE_THROWS(WhitespaceTokenizer{cfg});
}

TEST_CASE("WhitespaceTokenizer: rejects empty vocab") {
  WhitespaceTokenizer::Config cfg;
  REQUIRE_THROWS(WhitespaceTokenizer{cfg});
}

TEST_CASE("WhitespaceTokenizer: virtual dispatch via Tokenizer*") {
  auto tok = std::make_unique<WhitespaceTokenizer>(make_config());
  Tokenizer* base = tok.get();
  const auto ids = base->encode("hello tesseract", true);
  REQUIRE(ids == std::vector<int32_t>{1, 4, 8, 2});
  const auto back = base->decode(ids, true);
  REQUIRE(back == "hello tesseract");
  REQUIRE(base->vocab_size() == 9);
}

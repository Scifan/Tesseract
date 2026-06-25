// Wave 17 (B-034) — structured / grammar-constrained generation.
//
// Two layers under test:
//   * RegexAutomaton: a regex compiled to a byte-level automaton. The
//     `matches` contract is whole-string (anchored) and must agree with a
//     reference understanding of the supported subset.
//   * GrammarConstraint: masks a logits row so only tokens that keep the
//     automaton live survive (EOS only when accepting). A model-free greedy
//     loop driven purely by the constraint must emit a string the regex
//     accepts — that is the structured-generation guarantee.

#include <limits>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "tesseract/models/StructuredDecoding.hpp"

using tesseract::models::ByteAutomaton;
using tesseract::models::GrammarConstraint;
using tesseract::models::RegexAutomaton;

TEST_CASE("RegexAutomaton: literals, classes, anchoring", "[models][structured]") {
  auto a = RegexAutomaton::compile("abc");
  REQUIRE(a.matches("abc"));
  REQUIRE_FALSE(a.matches("ab"));     // not accepting yet
  REQUIRE_FALSE(a.matches("abcd"));   // trailing input ⇒ dead
  REQUIRE_FALSE(a.matches(""));

  auto d = RegexAutomaton::compile("[0-9]");
  REQUIRE(d.matches("5"));
  REQUIRE_FALSE(d.matches("a"));
  REQUIRE_FALSE(d.matches("55"));
}

TEST_CASE("RegexAutomaton: quantifiers and alternation", "[models][structured]") {
  auto a = RegexAutomaton::compile("a*b+");
  REQUIRE(a.matches("b"));
  REQUIRE(a.matches("aaabbb"));
  REQUIRE(a.matches("bb"));
  REQUIRE_FALSE(a.matches("a"));     // needs ≥1 b
  REQUIRE_FALSE(a.matches("ba"));

  auto alt = RegexAutomaton::compile("cat|dog|fish");
  REQUIRE(alt.matches("cat"));
  REQUIRE(alt.matches("dog"));
  REQUIRE(alt.matches("fish"));
  REQUIRE_FALSE(alt.matches("cot"));
  REQUIRE_FALSE(alt.matches("catdog"));

  auto opt = RegexAutomaton::compile("colou?r");
  REQUIRE(opt.matches("color"));
  REQUIRE(opt.matches("colour"));
  REQUIRE_FALSE(opt.matches("colouur"));
}

TEST_CASE("RegexAutomaton: counted repetition and escapes", "[models][structured]") {
  auto phone = RegexAutomaton::compile("\\d{3}-\\d{4}");
  REQUIRE(phone.matches("123-4567"));
  REQUIRE_FALSE(phone.matches("12-4567"));
  REQUIRE_FALSE(phone.matches("123-456"));
  REQUIRE_FALSE(phone.matches("1234567"));

  auto rng = RegexAutomaton::compile("a{2,4}");
  REQUIRE_FALSE(rng.matches("a"));
  REQUIRE(rng.matches("aa"));
  REQUIRE(rng.matches("aaa"));
  REQUIRE(rng.matches("aaaa"));
  REQUIRE_FALSE(rng.matches("aaaaa"));

  auto open = RegexAutomaton::compile("x{2,}");
  REQUIRE_FALSE(open.matches("x"));
  REQUIRE(open.matches("xx"));
  REQUIRE(open.matches("xxxxxx"));

  auto esc = RegexAutomaton::compile("3\\.14");  // literal dot
  REQUIRE(esc.matches("3.14"));
  REQUIRE_FALSE(esc.matches("3x14"));            // '.' is escaped, not wildcard
}

TEST_CASE("RegexAutomaton: negated class and dot", "[models][structured]") {
  auto nc = RegexAutomaton::compile("[^0-9]+");
  REQUIRE(nc.matches("abc"));
  REQUIRE_FALSE(nc.matches("ab1"));

  auto dot = RegexAutomaton::compile("a.c");
  REQUIRE(dot.matches("axc"));
  REQUIRE(dot.matches("a c"));
  REQUIRE_FALSE(dot.matches("ac"));
}

namespace {

// Greedy decode loop driven ONLY by the constraint (no model): all logits
// equal, so argmax picks the lowest-index allowed token each step. Returns
// the concatenated bytes. Stops when EOS is the chosen (allowed) token or
// after `max_steps`.
std::string constrained_greedy(const GrammarConstraint& base,
                               const std::vector<std::string>& vocab,
                               int32_t eos_id, int max_steps = 64) {
  GrammarConstraint c = base;  // copy ⇒ fresh state
  c.reset();
  std::string out;
  for (int step = 0; step < max_steps; ++step) {
    std::vector<float> logits(vocab.size(), 0.0f);
    c.apply(std::span<float>(logits.data(), logits.size()));
    // argmax (first non -inf).
    int32_t pick = -1;
    for (std::size_t i = 0; i < logits.size(); ++i) {
      if (logits[i] != -std::numeric_limits<float>::infinity()) {
        pick = static_cast<int32_t>(i);
        break;
      }
    }
    REQUIRE(pick >= 0);
    if (pick == eos_id) break;
    out += vocab[static_cast<std::size_t>(pick)];
    c.accept(pick);
  }
  return out;
}

}  // namespace

TEST_CASE("GrammarConstraint: masks to a valid token set", "[models][structured]") {
  // Vocab with an explicit EOS at index 0.
  const std::vector<std::string> vocab = {"<eos>", "1", "2", "3", "-", "x"};
  const int32_t eos = 0;
  auto automaton = RegexAutomaton::compile("\\d{3}-\\d{4}");
  GrammarConstraint c(automaton, vocab, eos);

  // At the start, only digit tokens 1/2/3 are legal; '-', 'x', and EOS are not.
  std::vector<float> logits(vocab.size(), 0.0f);
  const int64_t allowed = c.apply(std::span<float>(logits.data(), logits.size()));
  REQUIRE(allowed == 3);
  const float ninf = -std::numeric_limits<float>::infinity();
  REQUIRE(logits[0] == ninf);   // EOS not accepting at start
  REQUIRE(logits[1] != ninf);   // "1"
  REQUIRE(logits[4] == ninf);   // "-" illegal first
  REQUIRE(logits[5] == ninf);   // "x" never legal
}

TEST_CASE("GrammarConstraint: constrained greedy always matches the grammar", "[models][structured]") {
  const std::vector<std::string> vocab = {"<eos>", "0", "1", "2", "3", "4",
                                          "5", "-", "9"};
  const int32_t eos = 0;

  auto phone = RegexAutomaton::compile("\\d{3}-\\d{4}");
  const std::string out1 = constrained_greedy(GrammarConstraint(phone, vocab, eos),
                                              vocab, eos);
  REQUIRE(phone.matches(out1));               // guaranteed valid
  REQUIRE(out1.size() == 8);                  // ddd-dddd

  auto word = RegexAutomaton::compile("(cat|dog)s?");
  const std::vector<std::string> wv = {"<eos>", "cat", "dog", "s", "x"};
  const std::string out2 = constrained_greedy(GrammarConstraint(word, wv, 0),
                                              wv, 0);
  REQUIRE(word.matches(out2));

  // Multi-byte tokens that straddle automaton transitions still compose.
  // Vocab ordered so this greedy (lowest-index) policy reaches the required
  // '.' before looping on a repeatable digit token — the constraint only
  // *masks*; reaching an accepting state is the decoder policy's job.
  auto num = RegexAutomaton::compile("-?[0-9]+\\.[0-9]+");
  const std::vector<std::string> nv = {"<eos>", "-", ".", "3", "45", "12", "z"};
  const std::string out3 = constrained_greedy(GrammarConstraint(num, nv, 0), nv, 0);
  REQUIRE(num.matches(out3));
}

TEST_CASE("GrammarConstraint: accept advances state, EOS gated on acceptance",
          "[models][structured]") {
  const std::vector<std::string> vocab = {"<eos>", "a", "b"};
  auto a = RegexAutomaton::compile("ab");
  GrammarConstraint c(a, vocab, 0);

  REQUIRE_FALSE(c.at_accepting());
  c.accept(1);  // "a"
  REQUIRE_FALSE(c.at_accepting());
  c.accept(2);  // "b" ⇒ now at accepting
  REQUIRE(c.at_accepting());

  // After acceptance EOS is allowed; "a"/"b" would dead-end the grammar.
  std::vector<float> logits(vocab.size(), 0.0f);
  const int64_t allowed = c.apply(std::span<float>(logits.data(), logits.size()));
  REQUIRE(allowed == 1);
  REQUIRE(logits[0] != -std::numeric_limits<float>::infinity());  // EOS ok
}

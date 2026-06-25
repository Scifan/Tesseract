#pragma once

// Wave 17 (B-034) — structured / grammar-constrained generation.
//
// Structured generation forces a model's output to conform to a formal
// language (a regex, a JSON schema, an enum of allowed strings) by MASKING
// the logits at every decode step: any token whose bytes would take the
// grammar into a dead state is set to -inf before argmax / sampling, so the
// sampled sequence is guaranteed valid by construction. This is the
// mechanism behind Outlines / llama.cpp GBNF / vLLM guided decoding.
//
// Two layers:
//
//   * `ByteAutomaton` — a byte-level acceptor (a DFA-like interface).
//     `RegexAutomaton` compiles a practical regex subset into a Thompson
//     NFA and exposes it through lazy subset construction (a DFA state is
//     an epsilon-closed set of NFA states, memoized).
//
//   * `GrammarConstraint` — maps the automaton's byte language onto a token
//     vocabulary. It tracks the current automaton state, builds the
//     per-step allow-mask over the vocab (a token is allowed iff feeding
//     its decoded bytes keeps the automaton live; EOS is allowed iff the
//     state is accepting), and advances on the committed token.
//
// The constraint is decode-strategy agnostic: it operates on a host FP32
// logits row, so it composes with both greedy argmax and the full Sampler
// (temperature / top-k / top-p) — the mask is applied first, sampling runs
// on the survivors.

#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tesseract/io/Tokenizer.hpp"

namespace tesseract::models {

// Byte-level automaton. States are >= 0; `kDead` (-1) is the absorbing
// reject state. `step()` returns `kDead` for any byte with no transition,
// and `kDead` stays `kDead`. Whole-string (anchored) acceptance:
// `accepting(state)` is true iff the language accepts the bytes consumed so
// far to reach `state`.
class ByteAutomaton {
 public:
  static constexpr int64_t kDead = -1;
  virtual ~ByteAutomaton() = default;
  virtual int64_t start() const = 0;
  virtual int64_t step(int64_t state, std::uint8_t byte) const = 0;
  virtual bool accepting(int64_t state) const = 0;
};

// Regex → byte-level automaton. Supported syntax (enough for JSON values,
// numbers, dates, enums, identifiers):
//   * literals and '.' (any byte except none — matches every byte 0..255);
//   * escapes: \d \D \w \W \s \S \n \t \r \f \v \\ \. \( \) \[ \] \{ \} \| \+ \* \? \-;
//   * character classes [abc], ranges [a-z0-9], negation [^...];
//   * grouping ( ... ), alternation a|b;
//   * quantifiers * + ? and counted {m} {m,} {m,n}.
// Matching is anchored at both ends (the entire string must match).
class RegexAutomaton final : public ByteAutomaton {
 public:
  static RegexAutomaton compile(const std::string& pattern);

  int64_t start() const override;
  int64_t step(int64_t state, std::uint8_t byte) const override;
  bool accepting(int64_t state) const override;

  // Convenience: does the whole of `s` match the pattern?
  bool matches(std::string_view s) const;

 private:
  struct NfaState {
    bool has_byte = false;          // a single labeled (consuming) edge…
    std::array<bool, 256> set{};    // …accepting any byte in `set`…
    int out = -1;                   // …to `out`.
    std::vector<int> eps;           // epsilon edges.
  };

  // Epsilon-closure of an NFA-state set (returns sorted, unique).
  std::vector<int> closure(std::vector<int> set) const;
  // Lazily realize the DFA id for a (closed) NFA-state set.
  int64_t intern(std::vector<int> closed_set) const;

  std::vector<NfaState> nfa_;
  int nfa_start_ = 0;
  int nfa_accept_ = 0;

  // DFA memo (interior mutability: step()/start() are logically const but
  // grow the cache).
  mutable std::vector<std::vector<int>> dfa_sets_;         // id → closed set
  mutable std::map<std::vector<int>, int64_t> set_to_id_;  // set → id
  mutable std::vector<std::array<int64_t, 256>> trans_;    // id,byte → id (-2 = unknown)
  mutable std::vector<char> accepts_;                      // id → accepting?
};

// Maps a `ByteAutomaton`'s byte language onto a token vocabulary for
// constrained decoding. Build once per (automaton, tokenizer); call
// `reset()` at the start of each sequence.
class GrammarConstraint {
 public:
  // `token_bytes[id]` is the byte string committing token `id` appends to
  // the output (i.e. `tokenizer.decode({id})`). `eos_id` may be -1 (no EOS).
  GrammarConstraint(const ByteAutomaton& automaton,
                    std::vector<std::string> token_bytes, int32_t eos_id);

  // Build `token_bytes` by decoding every id in `[0, tk.vocab_size())`.
  static GrammarConstraint from_tokenizer(const ByteAutomaton& automaton,
                                          const io::Tokenizer& tk);

  void reset();

  // Mask `logits` (length = vocab) in place: disallowed tokens → -inf. A
  // token is allowed iff feeding its bytes from the current state keeps the
  // automaton live; EOS is allowed iff the current state is accepting.
  // Returns the count of still-allowed tokens. Throws if that count is 0
  // (an unsatisfiable dead-end — the caller hit a state with no legal
  // continuation and that isn't accepting).
  int64_t apply(std::span<float> logits) const;

  // Advance the tracked state by committing `token_id` (no-op for EOS).
  void accept(int32_t token_id);

  bool at_accepting() const;
  int64_t state() const noexcept { return state_; }
  int32_t eos_id() const noexcept { return eos_id_; }

 private:
  int64_t step_bytes(int64_t state, const std::string& bytes) const;

  const ByteAutomaton& a_;
  std::vector<std::string> token_bytes_;
  int32_t eos_id_;
  int64_t state_;
};

}  // namespace tesseract::models

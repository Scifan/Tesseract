#pragma once

// Tokenizer API — Wave 1a (Tesseract).
//
// Narrow, dependency-free abstract interface that model loaders can
// target without knowing the concrete algorithm (whitespace for tests,
// byte-level BPE for HF Llama/GPT-2, SentencePiece for T5). The two
// required entry points mirror `transformers.PreTrainedTokenizer.encode`
// / `.decode` with identical semantics:
//
//   encode(text, add_special=true)  -> token ids, prepending/appending
//                                      BOS/EOS when the impl has them
//                                      and `add_special` is true.
//   decode(ids, skip_special=true)  -> string; special ids are dropped
//                                      when `skip_special` is true.
//
// Special tokens are reported as `int32_t` ids; `-1` means the impl
// does not define that special. `vocab_size()` is the full table size
// (including specials).
//
// Concrete implementations shipped so far:
//   * `WhitespaceTokenizer` — reference, zero-dep, splits on ASCII
//     whitespace and looks up tokens in an in-memory vocab. Useful
//     for test fixtures and synthetic data; not byte-identical with
//     any HF tokenizer.
//
// Follow-ups (tracked in docs/backlog.md):
//   * `BpeTokenizer` — byte-level BPE, loads `tokenizer.json` from a
//     HF checkpoint, byte-identical with `transformers.GPT2Tokenizer`
//     / `LlamaTokenizerFast`.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tesseract::io {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  // Encode a UTF-8 string to a flat vector of token ids. When
  // `add_special_tokens=true`, the implementation may prepend `bos` and
  // append `eos` if it defines them.
  virtual std::vector<int32_t> encode(std::string_view text,
                                      bool add_special_tokens = true) const = 0;

  // Decode ids back to UTF-8. When `skip_special_tokens=true`, the
  // implementation drops BOS/EOS/PAD/UNK from the output if they are
  // encountered.
  virtual std::string decode(std::span<const int32_t> ids,
                             bool skip_special_tokens = true) const = 0;

  // Full vocabulary size (includes any special tokens).
  virtual std::size_t vocab_size() const noexcept = 0;

  // Special-token ids. Return `-1` for specials the impl does not
  // define. Defaults give implementations that don't use a given
  // special a no-op entry without having to override.
  virtual int32_t bos_token_id() const noexcept { return -1; }
  virtual int32_t eos_token_id() const noexcept { return -1; }
  virtual int32_t pad_token_id() const noexcept { return -1; }
  virtual int32_t unk_token_id() const noexcept { return -1; }

 protected:
  Tokenizer() = default;
  Tokenizer(const Tokenizer&) = default;
  Tokenizer(Tokenizer&&) noexcept = default;
  Tokenizer& operator=(const Tokenizer&) = default;
  Tokenizer& operator=(Tokenizer&&) noexcept = default;
};

// Deterministic whitespace tokenizer: splits `text` on ASCII whitespace
// (space, tab, \n, \r, \v, \f) and maps each non-empty span through
// `vocab`. Tokens not present in the vocab map to `unk_token_id` (or
// are silently dropped if no UNK is configured). BOS/EOS behave like
// the HF convention when configured.
//
// Thread-safety: all methods are const and internal state is
// read-only after construction, so a single instance is safe to share
// across threads.
class WhitespaceTokenizer final : public Tokenizer {
 public:
  struct Config {
    // Ordered vocabulary — `vocab[i]` is the surface form of id `i`.
    std::vector<std::string> vocab;
    // Optional special token strings. Empty string ⇒ "this special
    // does not exist" (the corresponding `*_token_id()` returns -1).
    // The specials are expected to already be present in `vocab` at
    // the id that `*_token_id()` will return.
    std::string bos;
    std::string eos;
    std::string pad;
    std::string unk;
  };

  explicit WhitespaceTokenizer(Config cfg);

  std::vector<int32_t> encode(std::string_view text,
                              bool add_special_tokens = true) const override;
  std::string decode(std::span<const int32_t> ids,
                     bool skip_special_tokens = true) const override;

  std::size_t vocab_size() const noexcept override { return vocab_.size(); }
  int32_t bos_token_id() const noexcept override { return bos_id_; }
  int32_t eos_token_id() const noexcept override { return eos_id_; }
  int32_t pad_token_id() const noexcept override { return pad_id_; }
  int32_t unk_token_id() const noexcept override { return unk_id_; }

  // Introspection helpers — tests can walk the vocab directly.
  const std::vector<std::string>& vocab() const noexcept { return vocab_; }

 private:
  int32_t lookup_or_unk(std::string_view tok) const;
  bool is_special_id(int32_t id) const noexcept;

  std::vector<std::string> vocab_;
  std::unordered_map<std::string, int32_t> index_;
  int32_t bos_id_ = -1;
  int32_t eos_id_ = -1;
  int32_t pad_id_ = -1;
  int32_t unk_id_ = -1;
};

}  // namespace tesseract::io

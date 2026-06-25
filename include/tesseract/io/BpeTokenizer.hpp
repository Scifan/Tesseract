#pragma once

// Byte-level BPE tokenizer — B-018 (Tesseract).
//
// Byte-identical drop-in for HuggingFace `tokenizers`' byte-level BPE
// (GPT-2 / Llama-3 `LlamaTokenizerFast`, `GPT2TokenizerFast`). Loads a
// `tokenizer.json` checkpoint and reproduces `tokenizers.Tokenizer.encode`
// on ASCII input byte-for-byte (validated against a `tokenizers`-produced
// golden fixture in `tests/io/test_bpe_tokenizer.cpp`).
//
// Pipeline (matches the `ByteLevel` pre-tokenizer + `BPE` model):
//
//   1. Isolate any configured special tokens (added_tokens) — they are
//      matched as whole units and never split by BPE.
//   2. Pre-tokenize each non-special span with the GPT-2 regex
//      (`'s|'t|...| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`).
//      Unicode classes are ASCII-scoped (`\p{L}` → `[A-Za-z]`, `\p{N}` →
//      `[0-9]`); bytes ≥ 0x80 group with letters so multi-byte UTF-8
//      stays together. Exact on ASCII, best-effort on non-ASCII.
//   3. Byte-encode each piece through GPT-2's reversible `bytes_to_unicode`
//      map, then run rank-priority BPE merges and look the final symbols
//      up in the vocab.
//
// `decode` inverts the process: ids → vocab strings → reverse
// byte-unicode map → original bytes.
//
// Special tokens (BOS/EOS/PAD/UNK) follow the `Tokenizer` contract:
// `encode(add_special_tokens=true)` prepends BOS / appends EOS when the
// loaded config defines them; `decode(skip_special_tokens=true)` drops
// any added-token id.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tesseract/io/Tokenizer.hpp"

namespace tesseract::io {

class BpeTokenizer final : public Tokenizer {
 public:
  struct Config {
    // Ordered vocabulary entries `(token_string, id)` in the
    // byte-unicode symbol space (i.e. the keys exactly as they appear
    // in `tokenizer.json`'s `model.vocab`).
    std::vector<std::pair<std::string, int32_t>> vocab;
    // Ordered merge rules `(left, right)` in priority order — earlier
    // entries have lower rank (applied first). Each side is a
    // byte-unicode symbol string.
    std::vector<std::pair<std::string, std::string>> merges;
    // Special / added tokens `(content, id)` matched as whole units.
    std::vector<std::pair<std::string, int32_t>> added_tokens;
    // GPT-2 ByteLevel: prepend a space to the input before
    // pre-tokenization. Default false (Llama-3 / modern GPT-2).
    bool add_prefix_space = false;
    // Special-token ids (resolved by content elsewhere); -1 ⇒ unset.
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t pad_id = -1;
    int32_t unk_id = -1;
  };

  explicit BpeTokenizer(Config cfg);

  // Load from a HF `tokenizer.json` on disk / from an in-memory JSON
  // string. Parses `model.vocab`, `model.merges`, `added_tokens`, and
  // the `ByteLevel` pre-tokenizer's `add_prefix_space`. BOS/EOS are
  // resolved from `added_tokens` by the usual content conventions
  // (`<|begin_of_text|>` / `<s>` for BOS, `<|end_of_text|>` / `</s>`
  // / `<|eot_id|>` for EOS) unless overridden.
  static BpeTokenizer from_file(const std::string& path);
  static BpeTokenizer from_json(std::string_view json);

  std::vector<int32_t> encode(std::string_view text,
                              bool add_special_tokens = true) const override;
  std::string decode(std::span<const int32_t> ids,
                     bool skip_special_tokens = true) const override;

  std::size_t vocab_size() const noexcept override { return id_to_token_.size(); }
  int32_t bos_token_id() const noexcept override { return bos_id_; }
  int32_t eos_token_id() const noexcept override { return eos_id_; }
  int32_t pad_token_id() const noexcept override { return pad_id_; }
  int32_t unk_token_id() const noexcept override { return unk_id_; }

 private:
  void build(Config&& cfg);

  // Apply rank-priority BPE merges to a single pre-token already split
  // into per-byte symbol strings. Returns the merged symbol sequence.
  std::vector<std::string> bpe(const std::vector<std::string>& symbols) const;

  // Encode one already-byte-unicode-mapped piece into ids (appends).
  void encode_piece(const std::string& mapped, std::vector<int32_t>& out) const;

  bool is_added_id(int32_t id) const noexcept;

  // token string (byte-unicode space) -> id, and the reverse.
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::unordered_map<int32_t, std::string> id_to_token_;
  // (left, right) merge -> rank (lower applied first).
  std::unordered_map<std::string, int32_t> merge_rank_;
  // added/special content -> id, plus the id set for fast skip-on-decode.
  std::vector<std::pair<std::string, int32_t>> added_tokens_;
  std::unordered_map<int32_t, bool> added_id_set_;

  bool add_prefix_space_ = false;
  int32_t bos_id_ = -1;
  int32_t eos_id_ = -1;
  int32_t pad_id_ = -1;
  int32_t unk_id_ = -1;
};

}  // namespace tesseract::io

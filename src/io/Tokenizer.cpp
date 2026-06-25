#include "tesseract/io/Tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

#include "tesseract/utils/Logging.hpp"

namespace tesseract::io {

namespace {

bool is_ascii_whitespace(char c) {
  const unsigned char u = static_cast<unsigned char>(c);
  return u == ' ' || u == '\t' || u == '\n' || u == '\r' || u == '\v' || u == '\f';
}

// Returns the first id matching `tok` in the vocab, or -1 if absent.
// The Config::bos/eos/pad/unk strings are expected to already be in
// `vocab`, so we resolve them by searching — duplicates (e.g. empty
// token appearing at multiple indices) resolve to the lowest index.
int32_t resolve_special(const std::vector<std::string>& vocab, const std::string& tok) {
  if (tok.empty()) return -1;
  for (std::size_t i = 0; i < vocab.size(); ++i) {
    if (vocab[i] == tok) return static_cast<int32_t>(i);
  }
  return -1;
}

}  // namespace

WhitespaceTokenizer::WhitespaceTokenizer(Config cfg)
    : vocab_(std::move(cfg.vocab)) {
  TESSERACT_CHECK(!vocab_.empty(), "WhitespaceTokenizer: vocab must be non-empty");

  // Build the reverse index. If the same surface form appears twice we
  // keep the first id — matches HF behaviour where the canonical entry
  // wins and duplicates are added tokens.
  index_.reserve(vocab_.size());
  for (std::size_t i = 0; i < vocab_.size(); ++i) {
    index_.try_emplace(vocab_[i], static_cast<int32_t>(i));
  }

  bos_id_ = resolve_special(vocab_, cfg.bos);
  eos_id_ = resolve_special(vocab_, cfg.eos);
  pad_id_ = resolve_special(vocab_, cfg.pad);
  unk_id_ = resolve_special(vocab_, cfg.unk);

  // Whenever a special was requested but not found, treat it as a
  // config error — the caller shouldn't silently lose specials.
  TESSERACT_CHECK(cfg.bos.empty() || bos_id_ >= 0,
                  "WhitespaceTokenizer: bos token '{}' not in vocab", cfg.bos);
  TESSERACT_CHECK(cfg.eos.empty() || eos_id_ >= 0,
                  "WhitespaceTokenizer: eos token '{}' not in vocab", cfg.eos);
  TESSERACT_CHECK(cfg.pad.empty() || pad_id_ >= 0,
                  "WhitespaceTokenizer: pad token '{}' not in vocab", cfg.pad);
  TESSERACT_CHECK(cfg.unk.empty() || unk_id_ >= 0,
                  "WhitespaceTokenizer: unk token '{}' not in vocab", cfg.unk);
}

int32_t WhitespaceTokenizer::lookup_or_unk(std::string_view tok) const {
  const auto it = index_.find(std::string(tok));
  if (it != index_.end()) return it->second;
  return unk_id_;
}

bool WhitespaceTokenizer::is_special_id(int32_t id) const noexcept {
  return id >= 0 && (id == bos_id_ || id == eos_id_ || id == pad_id_ || id == unk_id_);
}

std::vector<int32_t> WhitespaceTokenizer::encode(std::string_view text,
                                                 bool add_special_tokens) const {
  std::vector<int32_t> out;
  out.reserve(text.size() / 4 + 4);  // rough guess; tokens >= 1 char
  if (add_special_tokens && bos_id_ >= 0) {
    out.push_back(bos_id_);
  }

  // Single linear scan — no allocation per token, we slice views into
  // `text` and hash them via the lookup. Unknown tokens emit `unk_id_`
  // when configured, otherwise they're silently dropped (caller's
  // responsibility to configure UNK if they need it).
  std::size_t i = 0;
  const std::size_t n = text.size();
  while (i < n) {
    while (i < n && is_ascii_whitespace(text[i])) ++i;
    const std::size_t start = i;
    while (i < n && !is_ascii_whitespace(text[i])) ++i;
    if (start == i) break;
    const int32_t id = lookup_or_unk(text.substr(start, i - start));
    if (id >= 0) out.push_back(id);
  }

  if (add_special_tokens && eos_id_ >= 0) {
    out.push_back(eos_id_);
  }
  return out;
}

std::string WhitespaceTokenizer::decode(std::span<const int32_t> ids,
                                        bool skip_special_tokens) const {
  std::string out;
  out.reserve(ids.size() * 4);
  bool first = true;
  for (int32_t id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= vocab_.size()) {
      // Out-of-range ids are an error class we want visible without
      // throwing — decoders often see garbage from probing code.
      continue;
    }
    if (skip_special_tokens && is_special_id(id)) continue;
    if (!first) out.push_back(' ');
    out.append(vocab_[static_cast<std::size_t>(id)]);
    first = false;
  }
  return out;
}

}  // namespace tesseract::io

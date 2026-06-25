#!/usr/bin/env python3
"""Generate the byte-level BPE golden fixture for `test_bpe_tokenizer.cpp`.

Builds a *real* HuggingFace `tokenizers` ByteLevel BPE tokenizer (same
pre-tokenizer regex + bytes_to_unicode map + rank-priority merge loop that
GPT-2 / Llama-3 `tokenizer.json` checkpoints use), trains a small vocab on
an ASCII corpus so the merge table is non-trivial, then dumps:

  * bpe_tokenizer.json  — the HF checkpoint our C++ `BpeTokenizer::from_file`
                          loads.
  * bpe_golden.txt      — one line per fixture sentence:
                            <space-separated ids>\\t<original text repr>
                          produced by `tokenizers.Tokenizer.encode`.

The C++ test loads the same `tokenizer.json`, re-encodes each fixture line,
and asserts the ids match byte-for-byte. Re-run after changing the corpus or
fixtures:

    python3 tests/io/fixtures/generate_bpe_fixture.py

Requires: pip install tokenizers
"""

import json
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

HERE = os.path.dirname(os.path.abspath(__file__))

# Small ASCII training corpus — enough repetition that the trainer learns
# real multi-character merges (e.g. "th", "the", "ing", "tion").
CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the dog runs and the fox jumps and the cat sleeps",
    "tokenization is the action of turning text into tokens",
    "byte level encoding maps every byte to a printable character",
    "the transformer reads tokens and predicts the next token",
    "merging the most frequent pair repeatedly builds the vocabulary",
    "she said she'll be there and he said he'd come too",
    "numbers like 12 34 567 and 8901 are split into digit runs",
    "punctuation, like commas; colons: and periods. matters!",
    "running jumping reading writing thinking learning coding",
    "the the the the the quick quick brown brown fox fox",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
] * 8


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        min_frequency=1,
        special_tokens=["<unk>", "<|begin_of_text|>", "<|end_of_text|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer=trainer)
    return tok


# Sentences exercising: contractions, digit runs, punctuation runs, leading
# / interior / trailing whitespace (multiple spaces, tabs, newlines), mixed
# case, and tokens that fall back to single-byte symbols.
FIXTURES = [
    "the quick brown fox",
    "she'll be there and he'd come",
    "numbers 12 34 567 and 8901 here",
    "punctuation, like; colons: periods. and bangs!",
    "  leading and  interior   spaces  ",
    "tabs\tand\nnewlines\there",
    "the the the quick quick fox",
    "TOKENIZATION in UPPERCASE and lowercase",
    "zzz qqq unseen-ish gibberish wxyz",
    "trailing whitespace   ",
    "",
    "a",
    "running jumping reading writing",
]


def main() -> None:
    tok = build_tokenizer()
    json_path = os.path.join(HERE, "bpe_tokenizer.json")
    tok.save(json_path)

    # Re-load from disk to guarantee the golden ids come from exactly the
    # serialized checkpoint the C++ side reads.
    tok = Tokenizer.from_file(json_path)

    golden_path = os.path.join(HERE, "bpe_golden.txt")
    with open(golden_path, "w", encoding="utf-8") as f:
        for text in FIXTURES:
            ids = tok.encode(text, add_special_tokens=False).ids
            # JSON-encode the text so tabs / newlines / quotes survive a
            # single golden line; the C++ side json-decodes it.
            f.write(" ".join(str(i) for i in ids))
            f.write("\t")
            f.write(json.dumps(text))
            f.write("\n")

    with open(json_path, "r", encoding="utf-8") as f:
        model = json.load(f)["model"]
    print(f"wrote {json_path} (vocab={len(model['vocab'])}, "
          f"merges={len(model['merges'])})")
    print(f"wrote {golden_path} ({len(FIXTURES)} fixture lines)")


if __name__ == "__main__":
    main()

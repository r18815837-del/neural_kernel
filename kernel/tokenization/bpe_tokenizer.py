"""Byte-Pair Encoding (BPE) tokenizer — trains from raw text.

Zero external dependencies. Implements the classic BPE algorithm:
1. Start with byte-level vocabulary (256 tokens)
2. Count adjacent pairs in the corpus
3. Merge the most frequent pair → new token
4. Repeat until desired vocab size

Usage:
    tok = BPETokenizer()
    tok.train("path/to/corpus.txt", vocab_size=4096)
    tok.save("tokenizer.json")

    tok2 = BPETokenizer.load("tokenizer.json")
    ids = tok2.encode("Hello world")
    text = tok2.decode(ids)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Sequence

from .base import BaseTokenizer
from .types import TokenizerInfo

log = logging.getLogger(__name__)

# Special tokens.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# Pre-tokenization: split on whitespace boundaries, keeping the space
# attached to the following word (GPT-2 style).
_PRETOK_RE = re.compile(r"""\s?\S+|\s+""")


class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding tokenizer trained from scratch."""

    def __init__(self) -> None:
        # token → id  and  id → token
        self._token2id: dict[str, int] = {}
        self._id2token: dict[int, str] = {}

        # Learned merge rules: (tokenA, tokenB) → merged_token
        self._merges: list[tuple[str, str]] = []

        self._vocab_size: int = 0
        self._trained: bool = False

        # Cache: word → list of token strings (avoids re-running merges).
        self._word_cache: dict[str, list[str]] = {}

        # Pre-populate special tokens.
        for i, tok in enumerate(SPECIAL_TOKENS):
            self._token2id[tok] = i
            self._id2token[i] = tok

    # ==============================================================
    # BaseTokenizer interface
    # ==============================================================

    def info(self) -> TokenizerInfo:
        return TokenizerInfo(
            name="BPETokenizer",
            vocab_size=self._vocab_size,
            bos_token_id=self._token2id.get(BOS_TOKEN),
            eos_token_id=self._token2id.get(EOS_TOKEN),
            pad_token_id=self._token2id.get(PAD_TOKEN),
            unk_token_id=self._token2id.get(UNK_TOKEN),
        )

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode text → list of token IDs."""
        if not self._trained:
            raise RuntimeError("Tokenizer not trained yet. Call train() first.")

        ids: list[int] = []
        if add_special_tokens:
            ids.append(self._token2id[BOS_TOKEN])

        # Pre-tokenize into words.
        words = _PRETOK_RE.findall(text)

        unk_id = self._token2id[UNK_TOKEN]
        for word in words:
            # Check cache first — huge speedup for repeated words.
            if word in self._word_cache:
                tokens = self._word_cache[word]
            else:
                # Start with individual bytes.
                tokens = [ch for ch in word]
                # Apply merges in order.
                tokens = self._apply_merges(tokens)
                self._word_cache[word] = tokens
            # Convert to IDs.
            for t in tokens:
                ids.append(self._token2id.get(t, unk_id))

        if add_special_tokens:
            ids.append(self._token2id[EOS_TOKEN])

        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs → string."""
        parts: list[str] = []
        special_ids = {
            self._token2id.get(t) for t in SPECIAL_TOKENS
        }

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            token = self._id2token.get(tid, UNK_TOKEN)
            if token not in SPECIAL_TOKENS:
                parts.append(token)

        return "".join(parts)

    # ==============================================================
    # Training
    # ==============================================================

    def train(
        self,
        corpus_path: str,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        max_lines: int | None = None,
    ) -> None:
        """Train BPE on a text file.

        Args:
            corpus_path: Path to a UTF-8 text file (one doc per line).
            vocab_size:  Target vocabulary size (including specials).
            min_frequency: Minimum pair count to merge.
            max_lines: Cap on lines to read (for quick experiments).
        """
        log.info("bpe: loading corpus from %s", corpus_path)
        word_freqs = self._count_words(corpus_path, max_lines)

        # Initial vocab = specials + all unique bytes in the corpus.
        all_chars: set[str] = set()
        for word in word_freqs:
            all_chars.update(word)

        # Build initial vocabulary.
        self._token2id = {}
        self._id2token = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self._token2id[tok] = i
            self._id2token[i] = tok

        idx = len(SPECIAL_TOKENS)
        for ch in sorted(all_chars):
            if ch not in self._token2id:
                self._token2id[ch] = idx
                self._id2token[idx] = ch
                idx += 1

        # Split each word into character lists.
        # word_splits: { ("h","e","l","l","o"): 42, ... }
        word_splits: dict[tuple[str, ...], int] = {}
        for word, freq in word_freqs.items():
            key = tuple(word)
            word_splits[key] = freq

        num_merges = vocab_size - len(self._token2id)
        self._merges = []

        log.info(
            "bpe: initial vocab=%d, target=%d, merges needed=%d",
            len(self._token2id),
            vocab_size,
            num_merges,
        )

        for step in range(num_merges):
            # Count all adjacent pairs.
            pair_counts: Counter[tuple[str, str]] = Counter()
            for tokens, freq in word_splits.items():
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i + 1])] += freq

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0]
            if best_pair[1] < min_frequency:
                log.info("bpe: stopping — best pair freq=%d < min=%d", best_pair[1], min_frequency)
                break

            pair = best_pair[0]
            merged = pair[0] + pair[1]

            # Register new token.
            if merged not in self._token2id:
                new_id = len(self._token2id)
                self._token2id[merged] = new_id
                self._id2token[new_id] = merged

            self._merges.append(pair)

            # Apply merge to all word splits.
            new_splits: dict[tuple[str, ...], int] = {}
            for tokens, freq in word_splits.items():
                new_tokens = self._merge_pair(tokens, pair, merged)
                new_splits[new_tokens] = freq
            word_splits = new_splits

            if (step + 1) % 500 == 0:
                log.info("bpe: merge %d/%d — '%s'+'%s' → '%s' (freq=%d)",
                         step + 1, num_merges, pair[0], pair[1], merged, best_pair[1])

        self._vocab_size = len(self._token2id)
        self._trained = True
        self._word_cache = {}  # Reset cache after training.

        log.info("bpe: done — vocab_size=%d, merges=%d", self._vocab_size, len(self._merges))

    # ==============================================================
    # Save / Load
    # ==============================================================

    def save(self, path: str) -> None:
        """Save tokenizer to JSON."""
        data = {
            "vocab": self._token2id,
            "merges": [list(m) for m in self._merges],
            "vocab_size": self._vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("bpe: saved to %s", path)

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        """Load tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls()
        tok._token2id = data["vocab"]
        tok._id2token = {int(v): k for k, v in data["vocab"].items()}
        tok._merges = [tuple(m) for m in data["merges"]]
        tok._vocab_size = data["vocab_size"]
        tok._trained = True
        tok._word_cache = {}

        log.info("bpe: loaded from %s — vocab_size=%d", path, tok._vocab_size)
        return tok

    # ==============================================================
    # Internals
    # ==============================================================

    @staticmethod
    def _count_words(path: str, max_lines: int | None) -> dict[str, int]:
        """Pre-tokenize corpus and count word frequencies."""
        word_freqs: Counter[str] = Counter()
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                words = _PRETOK_RE.findall(line.strip())
                word_freqs.update(words)
        return dict(word_freqs)

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        """Apply all learned merges in order."""
        for pair in self._merges:
            merged = pair[0] + pair[1]
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [merged] + tokens[i + 2:]
                else:
                    i += 1
        return tokens

    @staticmethod
    def _merge_pair(
        tokens: tuple[str, ...],
        pair: tuple[str, str],
        merged: str,
    ) -> tuple[str, ...]:
        """Merge all occurrences of `pair` in `tokens`."""
        new: list[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new.append(merged)
                i += 2
            else:
                new.append(tokens[i])
                i += 1
        return tuple(new)

"""
02_train_tokenizer.py
=====================

Тренирует BPE-токенизатор на всех .txt файлах из data/corpus/
(tinystories, wiki_en, wiki_ru, или что угодно, что ты туда положишь).

Объединяет их в один временный файл, чтобы BPETokenizer прочитал
как единый корпус. Сохраняет в data/tokenizer/tokenizer.json.

Usage:
    python training/llm/02_train_tokenizer.py
    python training/llm/02_train_tokenizer.py --vocab-size 4096
    python training/llm/02_train_tokenizer.py --max-lines 50000   # для quick iteration
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kernel.tokenization.bpe_tokenizer import BPETokenizer  # noqa: E402
from training.llm.config import CONFIG, CORPUS_DIR, TOKENIZER_DIR, TOKENIZER_PATH  # noqa: E402


def concat_corpus(glob_pattern: str, out_path: Path) -> tuple[int, int]:
    """Склеивает все файлы, подходящие под glob, в один. Возвращает (#файлов, bytes)."""
    files = sorted(CORPUS_DIR.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {glob_pattern!r} in {CORPUS_DIR}. "
            f"Run 01_download_data.py first or drop your .txt files there."
        )

    total_bytes = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for fp in files:
            size = fp.stat().st_size
            print(f"  + {fp.name:40s}  ({size / 1024 / 1024:6.2f} MB)")
            with open(fp, "r", encoding="utf-8", errors="replace") as src:
                for line in src:
                    line = line.rstrip("\n")
                    if line:
                        out.write(line + "\n")
            total_bytes += size

    return len(files), total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=CONFIG.tokenizer.vocab_size,
        help="Target vocabulary size.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=CONFIG.tokenizer.min_frequency,
        help="Stop merging pairs below this frequency.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=CONFIG.tokenizer.max_lines,
        help="Cap the number of lines used for training (None = all).",
    )
    parser.add_argument(
        "--corpus-glob",
        default=CONFIG.dataset.corpus_glob,
        help="Glob pattern inside data/corpus to use (default: *.txt).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[step 1/3] Concatenating corpus files from {CORPUS_DIR}")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        n_files, total_bytes = concat_corpus(args.corpus_glob, tmp_path)
        print(f"  combined {n_files} file(s), {total_bytes / 1024 / 1024:.2f} MB")

        print(f"\n[step 2/3] Training BPE (vocab={args.vocab_size})")
        tok = BPETokenizer()
        tok.train(
            str(tmp_path),
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            max_lines=args.max_lines,
        )

        print(f"\n[step 3/3] Saving → {TOKENIZER_PATH}")
        tok.save(str(TOKENIZER_PATH))

        info = tok.info()
        print()
        print(f"  vocab_size: {info.vocab_size}")
        print(f"  BOS: {info.bos_token_id}  EOS: {info.eos_token_id}  "
              f"PAD: {info.pad_token_id}  UNK: {info.unk_token_id}")

        # Быстрый sanity check.
        sample = "Once upon a time, there was a little dragon who loved books."
        ids = tok.encode(sample, add_special_tokens=False)
        round_trip = tok.decode(ids, skip_special_tokens=True)
        print()
        print(f"  sample   : {sample!r}")
        print(f"  → tokens : {len(ids)} ids")
        print(f"  → decoded: {round_trip!r}")

    finally:
        tmp_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
03_pack_dataset.py
==================

Токенизирует весь корпус и сохраняет как «упакованный» бинарный массив
uint16 (так делает nanoGPT и большинство современных тренировочных
пайплайнов). Это даёт два важных свойства:

1. Тренировка читает любой случайный кусок данных мгновенно через
   memory-mapped numpy — никакой повторной токенизации каждый эпох.
2. Один файл = один кусок памяти, легко считать perplexity и т.п.

Делит корпус на train / val (по умолчанию 95 / 5).

Output:
    data/packed/train.bin   — uint16 array (T_train,)
    data/packed/val.bin     — uint16 array (T_val,)
    data/packed/meta.json   — vocab_size, специальные токены, размеры

Usage:
    python training/llm/03_pack_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kernel.tokenization.bpe_tokenizer import BPETokenizer  # noqa: E402
from training.llm.config import (  # noqa: E402
    CONFIG,
    CORPUS_DIR,
    META_PATH,
    PACKED_DIR,
    TOKENIZER_PATH,
    TRAIN_BIN,
    VAL_BIN,
)


def encode_corpus(tok: BPETokenizer, corpus_glob: str) -> np.ndarray:
    files = sorted(CORPUS_DIR.glob(corpus_glob))
    if not files:
        raise FileNotFoundError(
            f"No files matching {corpus_glob!r} in {CORPUS_DIR}."
        )

    eos_id = tok._token2id.get("<eos>")  # type: ignore[attr-defined]
    if eos_id is None:
        raise RuntimeError("Tokenizer is missing <eos> token.")

    all_ids: list[int] = []
    started = time.time()
    last_log = started

    for fp in files:
        size_mb = fp.stat().st_size / 1024 / 1024
        print(f"  encoding {fp.name}  ({size_mb:.2f} MB)")
        n_lines = 0
        with open(fp, "r", encoding="utf-8", errors="replace") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue

                ids = tok.encode(line, add_special_tokens=False)
                all_ids.extend(ids)
                # Каждая строка завершается <eos> — даёт модели сигнал «конец фразы».
                all_ids.append(eos_id)
                n_lines += 1

                now = time.time()
                if now - last_log > 5.0:
                    elapsed = now - started
                    rate = len(all_ids) / max(elapsed, 1e-6)
                    print(
                        f"    ... {n_lines:>8d} lines, "
                        f"{len(all_ids):>10d} tokens, "
                        f"{rate:>8.0f} tok/s"
                    )
                    last_log = now

        print(f"  → {n_lines} lines done ({len(all_ids)} cumulative tokens)")

    return np.asarray(all_ids, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize and pack corpus.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=CONFIG.dataset.val_fraction,
    )
    parser.add_argument(
        "--corpus-glob",
        default=CONFIG.dataset.corpus_glob,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.train.seed,
        help="Seed for the contiguous train/val split point.",
    )
    args = parser.parse_args()

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}. "
            f"Run 02_train_tokenizer.py first."
        )

    PACKED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[step 1/3] Loading tokenizer from {TOKENIZER_PATH}")
    tok = BPETokenizer.load(str(TOKENIZER_PATH))
    info = tok.info()
    vocab_size = info.vocab_size
    print(f"  vocab_size = {vocab_size}")

    if vocab_size > 65535:
        raise ValueError(
            f"vocab_size={vocab_size} doesn't fit in uint16. "
            f"Either shrink vocab or change dtype."
        )

    print(f"\n[step 2/3] Encoding corpus from {CORPUS_DIR}")
    ids = encode_corpus(tok, args.corpus_glob)
    print(f"  total tokens: {len(ids):,}")
    print(f"  bytes/token : {ids.nbytes / max(len(ids), 1):.2f}")

    # 95/5 split — берём непрерывный «хвост» как val (так делают в nanoGPT,
    # чтобы val-выборка не перетекала в train-окна).
    n = len(ids)
    n_val = max(1, int(n * args.val_fraction))
    n_train = n - n_val

    train_ids = ids[:n_train].astype(np.uint16)
    val_ids = ids[n_train:].astype(np.uint16)

    print(f"\n[step 3/3] Saving packed files to {PACKED_DIR}")
    train_ids.tofile(TRAIN_BIN)
    val_ids.tofile(VAL_BIN)
    print(f"  {TRAIN_BIN.name}: {train_ids.nbytes / 1024 / 1024:.2f} MB ({len(train_ids):,} tokens)")
    print(f"  {VAL_BIN.name}:   {val_ids.nbytes / 1024 / 1024:.2f} MB ({len(val_ids):,} tokens)")

    meta = {
        "vocab_size": vocab_size,
        "bos_token_id": info.bos_token_id,
        "eos_token_id": info.eos_token_id,
        "pad_token_id": info.pad_token_id,
        "unk_token_id": info.unk_token_id,
        "n_train_tokens": int(len(train_ids)),
        "n_val_tokens": int(len(val_ids)),
        "dtype": "uint16",
        "tokenizer_path": str(TOKENIZER_PATH.relative_to(_ROOT)),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  {META_PATH.name}: {json.dumps(meta, indent=2)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

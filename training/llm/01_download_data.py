"""
01_download_data.py
===================

Скачивает небольшой открытый корпус для тренировки LLM с нуля.

По умолчанию — TinyStories validation split (~22 MB, ~21k коротких историй
на простом английском). Это идеальный датасет для tiny-GPT: модель на
5–30M параметров действительно учится писать связный текст.

Альтернативы, которые можно подключить сменой --dataset:
  - tinystories-valid  (~22 MB, быстрее всего)
  - tinystories-train  (~1.6 GB, нужно для настоящего качества)
  - wikitext-2         (~12 MB, стандартный бенчмарк)

Usage:
    python training/llm/01_download_data.py
    python training/llm/01_download_data.py --dataset wikitext-2
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы импортировать training.llm.config
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.llm.config import CORPUS_DIR  # noqa: E402


DATASETS = {
    # TinyStories — syntactically clean English micro-stories.
    # Авторы: Ronen Eldan, Yuanzhi Li (Microsoft Research).
    "tinystories-valid": {
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
        "filename": "tinystories_valid.txt",
        "size_mb": 22,
    },
    "tinystories-train": {
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        "filename": "tinystories_train.txt",
        "size_mb": 1600,
    },
    # WikiText-2 raw — стандартный benchmark для LM.
    "wikitext-2": {
        "url": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
        "filename": "wikitext_2_raw.zip",
        "size_mb": 12,
        "zip_member": "wikitext-2-raw/wiki.train.raw",
        "extract_to": "wikitext_2_train.txt",
    },
}


class _Progress:
    def __init__(self, total_bytes: int | None):
        self.total = total_bytes
        self.last_pct = -1
        self.started = time.time()

    def update(self, block_num: int, block_size: int, total_size: int):
        total = total_size if total_size > 0 else self.total or 0
        downloaded = block_num * block_size
        if total > 0:
            pct = int(100 * downloaded / total)
            if pct != self.last_pct and pct % 5 == 0:
                mb = downloaded / 1024 / 1024
                total_mb = total / 1024 / 1024
                elapsed = max(time.time() - self.started, 1e-6)
                speed = mb / elapsed
                print(
                    f"  {pct:3d}%  {mb:7.1f} / {total_mb:7.1f} MB  "
                    f"({speed:5.2f} MB/s)",
                    flush=True,
                )
                self.last_pct = pct


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"→ {url}")
    print(f"  saving to {dst}")

    progress = _Progress(None)
    urllib.request.urlretrieve(url, dst, reporthook=progress.update)

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"  done — {size_mb:.1f} MB")


def extract_zip_member(zip_path: Path, member: str, dst: Path) -> None:
    import zipfile

    print(f"  extracting '{member}' from zip → {dst}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member) as src, open(dst, "wb") as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"  extracted — {size_mb:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download training corpus.")
    parser.add_argument(
        "--dataset",
        default="tinystories-valid",
        choices=sorted(DATASETS.keys()),
        help="Dataset to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    args = parser.parse_args()

    spec = DATASETS[args.dataset]

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    dst = CORPUS_DIR / spec["filename"]

    if dst.exists() and not args.force:
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"[skip] {dst.name} already exists ({size_mb:.1f} MB). Use --force to overwrite.")
    else:
        print(f"[download] {args.dataset} (~{spec['size_mb']} MB)")
        download(spec["url"], dst)

    # Для wikitext нужно распаковать конкретный файл из zip.
    if "zip_member" in spec:
        extracted = CORPUS_DIR / spec["extract_to"]
        if extracted.exists() and not args.force:
            size_mb = extracted.stat().st_size / 1024 / 1024
            print(f"[skip] {extracted.name} already extracted ({size_mb:.1f} MB).")
        else:
            extract_zip_member(dst, spec["zip_member"], extracted)

    print()
    print("Corpus directory now contains:")
    for path in sorted(CORPUS_DIR.glob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {path.name:40s}  {size_mb:8.2f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

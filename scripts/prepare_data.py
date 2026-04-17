"""Prepare training data for the Neural Kernel LM.

Downloads a small Wikipedia dump, cleans it, and saves as a plain
text corpus ready for BPE tokenizer training and LM training.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --lang ru --max-articles 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data/corpus")

# Wikipedia API — get random articles in bulk.
_USER_AGENT = "NeuralKernel/1.0"


def fetch_random_articles(
    lang: str = "en",
    count: int = 500,
    batch_size: int = 50,
) -> list[str]:
    """Fetch `count` random Wikipedia article extracts."""
    base = f"https://{lang}.wikipedia.org/w/api.php"
    articles: list[str] = []

    while len(articles) < count:
        n = min(batch_size, count - len(articles))
        params = urllib.parse.urlencode({
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": "0",
            "grnlimit": str(n),
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "exlimit": str(n),
        })
        url = f"{base}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "").strip()
                if len(extract) > 100:  # Skip tiny stubs.
                    articles.append(clean_text(extract))

            log.info("Fetched %d / %d articles", len(articles), count)

        except Exception as e:
            log.error("Fetch failed: %s", e)
            break

    return articles


def clean_text(text: str) -> str:
    """Basic text cleanup."""
    # Remove multiple newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove Wikipedia artifacts.
    text = re.sub(r"\[.*?\]", "", text)
    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def save_corpus(articles: list[str], path: Path) -> None:
    """Save articles as one-per-line text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for article in articles:
            # One article per line (newlines within replaced).
            line = article.replace("\n", " ").strip()
            if line:
                f.write(line + "\n")

    total_chars = sum(len(a) for a in articles)
    log.info("Saved %d articles (%d chars) to %s", len(articles), total_chars, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training corpus")
    parser.add_argument("--lang", default="en", help="Wikipedia language (en/ru)")
    parser.add_argument("--max-articles", type=int, default=2000, help="Number of articles")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    output = Path(args.output) if args.output else DATA_DIR / f"wiki_{args.lang}.txt"

    log.info("Fetching %d articles from %s.wikipedia.org...", args.max_articles, args.lang)
    articles = fetch_random_articles(lang=args.lang, count=args.max_articles)

    if not articles:
        log.error("No articles fetched! Check your network connection.")
        return

    save_corpus(articles, output)
    log.info("Done! Corpus ready at: %s", output)


if __name__ == "__main__":
    main()

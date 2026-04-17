"""Поиск знаний — Wikipedia API (en + ru, потом добавим Google/arxiv).

Uses only stdlib (urllib + json) — zero external dependencies.
Wikipedia REST API: https://en.wikipedia.org/api/rest_v1/
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Optional

log = logging.getLogger(__name__)

_USER_AGENT = "NeuralKernel/1.0 (https://github.com/neural-kernel)"
_TIMEOUT = 8  # seconds


class WebSearcher:
    """Searches Wikipedia for knowledge (auto-detects language).

    Two-step approach per wiki:
    1. Search API — find the best matching page title
    2. Summary API — grab the extract (first paragraphs)

    Falls back gracefully if the network is down or nothing is found.
    """

    def __init__(self) -> None:
        self._wikis = {
            "en": "https://en.wikipedia.org",
            "ru": "https://ru.wikipedia.org",
        }

    def search(self, query: str) -> Optional[str]:
        """Search Wikipedia — return a summary or None.

        Auto-picks Russian wiki if query has Cyrillic characters,
        then falls back to English if nothing found.
        """
        langs = self._detect_langs(query)

        for lang in langs:
            base = self._wikis[lang]
            try:
                title = self._find_title(base, query)
                if not title:
                    continue

                summary = self._get_summary(base, title)
                if summary:
                    log.info(
                        "searcher: got %d chars for '%s' [%s]",
                        len(summary),
                        title,
                        lang,
                    )
                    return summary

            except Exception as exc:
                log.error("searcher: %s wiki failed — %s", lang, exc)

        log.info("searcher: no results for '%s'", query[:60])
        return None

    # ----------------------------------------------------------
    # Language detection
    # ----------------------------------------------------------

    @staticmethod
    def _has_cyrillic(text: str) -> bool:
        return any("\u0400" <= c <= "\u04ff" for c in text)

    def _detect_langs(self, query: str) -> list[str]:
        """Return language priority list: primary first, fallback second."""
        if self._has_cyrillic(query):
            return ["ru", "en"]
        return ["en", "ru"]

    # ----------------------------------------------------------
    # Wikipedia API calls
    # ----------------------------------------------------------

    def _find_title(self, base: str, query: str) -> Optional[str]:
        """Use Wikipedia opensearch to find the best matching title."""
        params = urllib.parse.urlencode({
            "action": "opensearch",
            "search": query,
            "limit": "1",
            "namespace": "0",
            "format": "json",
        })
        url = f"{base}/w/api.php?{params}"
        data = self._get_json(url)

        # opensearch returns [query, [titles], [descriptions], [urls]]
        if data and len(data) >= 2 and data[1]:
            return data[1][0]
        return None

    def _get_summary(self, base: str, title: str) -> Optional[str]:
        """Fetch page summary via the REST API."""
        safe_title = urllib.parse.quote(title.replace(" ", "_"))
        url = f"{base}/api/rest_v1/page/summary/{safe_title}"
        data = self._get_json(url)

        if data and "extract" in data:
            extract: str = data["extract"]
            # Cap at ~500 chars for the cognition pipeline.
            if len(extract) > 500:
                cut = extract[:500].rfind(". ")
                if cut > 100:
                    return extract[: cut + 1]
                return extract[:500] + "..."
            return extract
        return None

    def _get_json(self, url: str) -> Optional[object]:
        """Fetch a URL and parse as JSON.  Returns None on any failure."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            log.debug("searcher: HTTP error — %s", exc)
            return None

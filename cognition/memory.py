"""Two-tier memory store (short-term / long-term) with optional SQLite persistence."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from .models import Query

if TYPE_CHECKING:
    from persistence.base import BaseStore

log = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    query_text: str
    answer: str
    hits: int = 0
    source_query_id: str = ""


class Memory:
    PROMOTE_THRESHOLD = 3

    def __init__(self, store: Optional[BaseStore] = None) -> None:
        self._store = store
        self._short: Dict[str, MemoryEntry] = {}
        self._long: Dict[str, MemoryEntry] = {}

    def remember(self, query: Query) -> None:
        if not query.succeeded:
            return
        key = self._normalise(query.text)
        entry = MemoryEntry(
            query_text=query.text,
            answer=query.answer,
            source_query_id=query.id,
        )
        if self._store:
            self._store.save_memory_entry(
                key=key, answer=entry.answer, hits=0, tier="short",
                query_text=entry.query_text, source_query_id=entry.source_query_id,
            )
        else:
            self._short[key] = entry
        log.debug("memory: remembered '%s'", key[:60])

    def recall(self, text: str) -> Optional[str]:
        key = self._normalise(text)

        if self._store:
            row = self._store.get_memory_entry(key)
            if row:
                new_hits = row["hits"] + 1
                tier = row["tier"]
                if new_hits >= self.PROMOTE_THRESHOLD and tier == "short":
                    tier = "long"
                    log.info("memory: promoted '%s' to long-term", key[:60])
                self._store.save_memory_entry(
                    key=key, answer=row["answer"], hits=new_hits, tier=tier,
                    query_text=row.get("query_text", key),
                    source_query_id=row.get("source_query_id", ""),
                )
                return row["answer"]
            return None

        entry = self._short.get(key) or self._long.get(key)
        if entry is not None:
            entry.hits += 1
            return entry.answer
        return None

    def promote(self, text: str) -> bool:
        key = self._normalise(text)

        if self._store:
            row = self._store.get_memory_entry(key)
            if row and row["tier"] == "short":
                new_hits = row["hits"] + 1
                if new_hits >= self.PROMOTE_THRESHOLD:
                    self._store.save_memory_entry(
                        key=key, answer=row["answer"], hits=new_hits, tier="long",
                        query_text=row.get("query_text", key),
                        source_query_id=row.get("source_query_id", ""),
                    )
                    return True
                self._store.save_memory_entry(
                    key=key, answer=row["answer"], hits=new_hits, tier="short",
                    query_text=row.get("query_text", key),
                    source_query_id=row.get("source_query_id", ""),
                )
            return False

        entry = self._short.get(key)
        if entry:
            entry.hits += 1
            if entry.hits >= self.PROMOTE_THRESHOLD:
                self._long[key] = entry
                del self._short[key]
                return True
        return False

    def forget(self, text: str) -> bool:
        key = self._normalise(text)
        if self._store:
            return self._store.delete_memory_entry(key)
        removed = False
        if key in self._short:
            del self._short[key]
            removed = True
        if key in self._long:
            del self._long[key]
            removed = True
        return removed

    def dump(self) -> Dict[str, List[Dict]]:
        if self._store:
            short = self._store.list_memory(tier="short")
            long = self._store.list_memory(tier="long")
            return {
                "short_term": [{"key": e["key"], "answer": e["answer"][:80], "hits": e["hits"]} for e in short],
                "long_term": [{"key": e["key"], "answer": e["answer"][:80], "hits": e["hits"]} for e in long],
            }
        return {
            "short_term": [{"key": k, "answer": v.answer[:80], "hits": v.hits} for k, v in self._short.items()],
            "long_term": [{"key": k, "answer": v.answer[:80], "hits": v.hits} for k, v in self._long.items()],
        }

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(text.lower().split())

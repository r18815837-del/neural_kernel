"""Проверяет факты в ответе по источникам."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FactCheckResult:
    trusted: bool
    confidence: float  # 0.0 → 1.0
    warnings: List[str] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)


class FactChecker:
    """Оценивает достоверность ответа на основе его источников.

    Позже сюда подключится RAG cross-check, web verification,
    и LLM-судья для детекции галлюцинаций.
    """

    def __init__(self) -> None:
        # Источники которым доверяем (потом сюда RAG/web)
        self._trusted_sources = {"builtin_knowledge", "memory", "wikipedia"}
        self._untrusted_sources = {"fallback"}

    def check(
        self,
        answer: str,
        sources: List[str],
    ) -> FactCheckResult:
        warnings: List[str] = []

        # 1. Если источник — fallback → не доверяем
        if any(s in self._untrusted_sources for s in sources):
            warnings.append("answer came from fallback — needs verification")
            return FactCheckResult(
                trusted=False,
                confidence=0.1,
                warnings=warnings,
                sources_used=sources,
            )

        # 2. Если источник надёжный → доверяем
        #    (exact match or prefix match for "specialist:*")
        if any(
            s in self._trusted_sources or s.startswith("specialist:")
            for s in sources
        ):
            return FactCheckResult(
                trusted=True,
                confidence=0.9,
                sources_used=sources,
            )

        # 3. Неизвестный источник → осторожно
        warnings.append("unknown source — treat with caution")
        return FactCheckResult(
            trusted=False,
            confidence=0.4,
            warnings=warnings,
            sources_used=sources,
        )

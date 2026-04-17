"""Базовый специалист — эксперт по одной теме."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SpecialistResult:
    topic: str
    answer: str
    confidence: float  # 0.0 → 1.0
    tips: List[str] = field(default_factory=list)  # советы для пользователя


class BaseSpecialist:
    """Abstract base — subclass and override `handle()`."""

    topic: str = "general"
    keywords: List[str] = []

    def can_handle(self, question: str) -> bool:
        """Могу ли я ответить на этот вопрос?"""
        q = question.lower()
        return any(kw in q for kw in self.keywords)

    def handle(self, question: str) -> SpecialistResult:
        raise NotImplementedError

"""Учится на ошибках — запоминает что было плохо."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

log = logging.getLogger(__name__)


@dataclass
class Mistake:
    question: str
    bad_answer: str
    reason: str
    timestamp: float = field(default_factory=time.time)


class FeedbackLearner:
    """Собирает ошибки и успехи, чтобы не повторять плохое
    и усиливать хорошее.

    Позже сюда подключится персистентное хранение и
    авто-тюнинг промптов на основе накопленной статистики.
    """

    def __init__(self) -> None:
        self._mistakes: List[Mistake] = []
        self._good_patterns: Dict[str, int] = {}  # что сработало → счётчик

    def record_mistake(
        self,
        question: str,
        bad_answer: str,
        reason: str,
    ) -> None:
        """Запомнить ошибку — больше не повторять."""
        mistake = Mistake(question, bad_answer, reason)
        self._mistakes.append(mistake)
        log.warning("feedback: mistake recorded — %s", reason)

    def record_success(self, pattern: str) -> None:
        """Запомнить что сработало хорошо."""
        self._good_patterns[pattern] = (
            self._good_patterns.get(pattern, 0) + 1
        )

    def was_mistake(self, question: str) -> bool:
        """Проверить — был ли такой вопрос провальным раньше?"""
        q = question.lower().strip()
        return any(q in m.question.lower() for m in self._mistakes)

    def get_mistakes(self) -> List[Mistake]:
        return list(self._mistakes)

    def summary(self) -> Dict:
        return {
            "total_mistakes": len(self._mistakes),
            "good_patterns": self._good_patterns,
        }

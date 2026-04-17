"""Проверяет ответ на базовые проблемы перед отдачей пользователю."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CheckResult:
    passed: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class LogicChecker:
    """Набор быстрых эвристик для валидации ответа.

    Позже сюда подключится LLM-судья, перекрёстная проверка
    по источникам и детектор галлюцинаций.
    """

    def check(self, question: str, answer: str) -> CheckResult:
        issues: List[str] = []
        suggestions: List[str] = []

        # 1. Ответ не пустой
        if not answer or not answer.strip():
            issues.append("answer is empty")

        # 2. Ответ не слишком короткий
        if answer and len(answer.strip()) < 10:
            issues.append("answer too short")
            suggestions.append("provide more detail")

        # 3. Ответ подозрительно длинный
        if len(answer) > 5000:
            issues.append("answer suspiciously long")
            suggestions.append("consider summarising")

        # 4. Нет явного "не знаю" если вопрос простой
        if "i don't" in answer.lower() and "?" in question:
            suggestions.append("consider searching for this")

        # 5. Ответ не повторяет вопрос дословно
        if answer.strip().lower() == question.strip().lower():
            issues.append("answer is just the question repeated")

        # TODO: LLM-судья для проверки противоречий
        # TODO: перекрёстная проверка по источникам

        return CheckResult(
            passed=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
        )

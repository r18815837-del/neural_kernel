"""Физик — отвечает на вопросы по физике."""

from .base_specialist import BaseSpecialist, SpecialistResult


class PhysicsSpecialist(BaseSpecialist):
    topic = "physics"
    keywords = [
        "physics", "force", "energy", "quantum",
        "gravity", "velocity", "atom", "electron",
    ]

    def handle(self, question: str) -> SpecialistResult:
        return SpecialistResult(
            topic=self.topic,
            answer=f"[Physics expert] Searching: {question}",
            confidence=0.8,
            tips=["Check formula units", "Draw a diagram"],
        )

"""Data models for the cognition pipeline.

Every query flows through a fixed sequence of stages.  These models
capture the state at each stage so any part of the system can inspect
what happened, why, and what was decided.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class Confidence(Enum):
    """How sure the system is that it can answer without external help."""

    HIGH = "high"          # answer from local knowledge / memory
    MEDIUM = "medium"      # probably knows, but should double-check
    LOW = "low"            # needs external lookup
    NONE = "none"          # no idea — must search


class QueryIntent(Enum):
    """Coarse classification of what the user wants."""

    QUESTION = "question"          # wants information
    COMMAND = "command"             # wants an action executed
    CLARIFICATION = "clarification"  # follow-up on a prior exchange
    CREATIVE = "creative"          # open-ended generation
    UNKNOWN = "unknown"


class StepKind(Enum):
    """Labels for individual reasoning steps."""

    CLASSIFY = "classify"        # intent + confidence assessment
    RECALL = "recall"            # memory lookup
    SEARCH = "search"            # external retrieval (stub)
    REASON = "reason"            # synthesise an answer
    VALIDATE = "validate"        # sanity-check the draft answer
    PERSIST = "persist"          # save to memory


# ------------------------------------------------------------------
# Step trace — one entry per pipeline stage
# ------------------------------------------------------------------

@dataclass
class StepTrace:
    """Record of a single reasoning step."""

    kind: StepKind
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<Step {self.kind.value}: {self.output_summary[:60]}>"


# ------------------------------------------------------------------
# Query — the full lifecycle of one user question
# ------------------------------------------------------------------

@dataclass
class Query:
    """Immutable snapshot of a user query and the system's response."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str = ""
    intent: QueryIntent = QueryIntent.UNKNOWN
    confidence: Confidence = Confidence.NONE
    needs_search: bool = False
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    steps: List[StepTrace] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    error: Optional[str] = None

    # -- helpers --

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.answer)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "intent": self.intent.value,
            "confidence": self.confidence.value,
            "needs_search": self.needs_search,
            "answer": self.answer,
            "sources": self.sources,
            "steps": [
                {
                    "kind": s.kind.value,
                    "input": s.input_summary,
                    "output": s.output_summary,
                    "ms": s.duration_ms,
                }
                for s in self.steps
            ],
            "created_at": self.created_at.isoformat(),
            "error": self.error,
        }

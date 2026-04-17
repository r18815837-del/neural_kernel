"""Cognition routes — ask questions to the thinking engine."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from cognition import Orchestrator

router = APIRouter(prefix="/api/v1", tags=["cognition"])

_orch = Orchestrator()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class StepInfo(BaseModel):
    kind: str
    output: str
    duration_ms: float | None = None


class AskResponse(BaseModel):
    id: str
    question: str
    answer: str
    confidence: str
    sources: list[str]
    steps: list[StepInfo]
    error: str | None = None


class MemoryDump(BaseModel):
    short_term: list[dict]
    long_term: list[dict]


class LearnerSummary(BaseModel):
    total_mistakes: int
    good_patterns: dict[str, int]


@router.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest) -> AskResponse:
    result = await _orch.ask(body.question)
    return AskResponse(
        id=result.id,
        question=result.text,
        answer=result.answer,
        confidence=result.confidence.value,
        sources=result.sources,
        steps=[
            StepInfo(
                kind=s.kind.value,
                output=s.output_summary[:120],
                duration_ms=s.duration_ms,
            )
            for s in result.steps
        ],
        error=result.error,
    )


@router.get("/memory", response_model=MemoryDump)
async def get_memory() -> MemoryDump:
    return MemoryDump(**_orch.memory.dump())


@router.get("/learner", response_model=LearnerSummary)
async def get_learner() -> LearnerSummary:
    return LearnerSummary(**_orch.learner.summary())

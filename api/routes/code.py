"""Code assistant routes — analyze, explain, and run code."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from cognition.specialists.coding_specialist import (
    CodingSpecialist, analyze_python, detect_language,
)
from cognition.specialists.code_executor import CodeExecutor

router = APIRouter(prefix="/api/v1", tags=["code"])

_specialist = CodingSpecialist()
_executor = CodeExecutor(timeout=5)


class CodeAnalyzeRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10000)
    question: str = ""


class CodeAnalyzeResponse(BaseModel):
    language: str | None
    syntax_valid: bool
    functions: list[dict]
    classes: list[dict]
    imports: list[str]
    issues: list[str]
    suggestions: list[str]
    explanation: str


class CodeRunRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=5000)


class CodeRunResponse(BaseModel):
    stdout: str
    stderr: str
    success: bool
    exit_code: int
    timed_out: bool
    error_summary: str | None = None


class CodeAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    code: str = ""


class CodeAskResponse(BaseModel):
    answer: str
    confidence: float
    language: str | None
    tips: list[str]


@router.post("/code/analyze", response_model=CodeAnalyzeResponse)
async def analyze_code(body: CodeAnalyzeRequest) -> CodeAnalyzeResponse:
    lang = detect_language(body.code)
    analysis = analyze_python(body.code) if lang == "python" else {
        "syntax_valid": True, "syntax_error": None,
        "functions": [], "classes": [], "imports": [],
        "issues": [], "suggestions": [], "complexity": "unknown",
    }
    result = _specialist.handle(body.code if not body.question else body.question)
    return CodeAnalyzeResponse(
        language=lang,
        syntax_valid=analysis.get("syntax_valid", True),
        functions=analysis.get("functions", []),
        classes=analysis.get("classes", []),
        imports=analysis.get("imports", []),
        issues=analysis.get("issues", []),
        suggestions=analysis.get("suggestions", []),
        explanation=result.answer,
    )


@router.post("/code/run", response_model=CodeRunResponse)
async def run_code(body: CodeRunRequest) -> CodeRunResponse:
    result = _executor.run(body.code)
    return CodeRunResponse(**result.to_dict())


@router.post("/code/ask", response_model=CodeAskResponse)
async def ask_code(body: CodeAskRequest) -> CodeAskResponse:
    question = body.question
    if body.code:
        question = f"{body.question}\n\n```\n{body.code}\n```"
    result = _specialist.handle(question)
    lang = detect_language(body.code) if body.code else detect_language(body.question)
    return CodeAskResponse(
        answer=result.answer,
        confidence=result.confidence,
        language=lang,
        tips=result.tips,
    )

"""Orchestrator — the central thinking loop.

Pipeline: CLASSIFY → RECALL → TRY_SPECIALISTS → SEARCH → REASON → VALIDATE → PERSIST
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .critic.logic_checker import LogicChecker
from .learning.feedback import FeedbackLearner
from .lm_backend import LMBackend
from .memory import Memory
from .rag.web_searcher import WebSearcher
from .specialists.coding_specialist import CodingSpecialist
from .specialists.physics_specialist import PhysicsSpecialist
from .truth.fact_checker import FactChecker
from .models import Confidence, Query, QueryIntent, StepKind, StepTrace

log = logging.getLogger(__name__)

_BUILTIN_KNOWLEDGE: Dict[str, str] = {
    "neural kernel": (
        "Neural Kernel — AI-native backend platform that turns a text "
        "prompt into a fully generated software project, including code, "
        "tests, and deployment artifacts."
    ),
    "what is neural kernel": (
        "Neural Kernel is an AI-powered backend that accepts a natural-"
        "language project description and produces a complete codebase "
        "with build scripts, tests, and packaging."
    ),
}


class Orchestrator:
    def __init__(
        self,
        memory: Optional[Memory] = None,
        knowledge: Optional[Dict[str, str]] = None,
        plugins: Optional[List[Callable]] = None,
        lm_backend: Optional[LMBackend] = None,
        load_lm: bool = True,
        store: Optional[Any] = None,
    ) -> None:
        self._store = store
        self._memory = memory or Memory(store=store)
        self._knowledge: Dict[str, str] = {**_BUILTIN_KNOWLEDGE}
        if knowledge:
            self._knowledge.update({k.lower().strip(): v for k, v in knowledge.items()})
        self._plugins = plugins or []
        self._checker = LogicChecker()
        self._fact_checker = FactChecker()
        self._learner = FeedbackLearner()
        self._searcher = WebSearcher()
        self._specialists = [PhysicsSpecialist(), CodingSpecialist()]

        if lm_backend is not None:
            self._lm = lm_backend
        elif load_lm:
            self._lm = LMBackend.from_checkpoint()
            if self._lm:
                log.info("orchestrator: Neural Kernel LM loaded!")
            else:
                log.info("orchestrator: No trained LM found — using templates only")
        else:
            self._lm = None

    async def ask(self, text: str) -> Query:
        query = Query(text=text)
        log.info("orchestrator: new query id=%s text='%s'", query.id, text[:80])

        try:
            self._classify(query)
            self._recall(query)

            if not query.answer:
                self._try_specialists(query)

            if query.needs_search and not query.answer:
                self._search(query)

            self._reason(query)
            self._validate(query)
            self._persist(query)
        except Exception as exc:
            query.error = str(exc)
            log.exception("orchestrator: pipeline failed for id=%s", query.id)

        log.info(
            "orchestrator: done id=%s ok=%s steps=%d",
            query.id, query.succeeded, len(query.steps),
        )
        return query

    def ask_sync(self, text: str) -> Query:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.ask(text)).result()
        return asyncio.run(self.ask(text))

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def learner(self) -> FeedbackLearner:
        return self._learner

    def _classify(self, query: Query) -> None:
        t0 = time.perf_counter()
        lower = query.text.lower().strip()

        if lower.endswith("?") or lower.startswith(
            ("what", "who", "how", "why", "when", "where", "is", "are", "can", "do", "does")
        ):
            query.intent = QueryIntent.QUESTION
        elif lower.startswith(("create", "make", "build", "run", "delete", "stop", "start")):
            query.intent = QueryIntent.COMMAND
        else:
            query.intent = QueryIntent.QUESTION

        normalised = " ".join(lower.split())
        if normalised in self._knowledge:
            query.confidence = Confidence.HIGH
        elif any(k in normalised for k in self._knowledge):
            query.confidence = Confidence.MEDIUM
        else:
            query.confidence = Confidence.LOW

        query.needs_search = query.confidence in (Confidence.LOW, Confidence.NONE)
        query.steps.append(StepTrace(
            kind=StepKind.CLASSIFY,
            input_summary=query.text[:100],
            output_summary=f"intent={query.intent.value} confidence={query.confidence.value} needs_search={query.needs_search}",
            duration_ms=_elapsed(t0),
        ))

    def _recall(self, query: Query) -> None:
        t0 = time.perf_counter()
        remembered = self._memory.recall(query.text)
        if remembered is not None:
            query.answer = remembered
            query.confidence = Confidence.HIGH
            query.needs_search = False
            query.sources.append("memory")
        query.steps.append(StepTrace(
            kind=StepKind.RECALL,
            input_summary=query.text[:100],
            output_summary="hit" if remembered else "miss",
            duration_ms=_elapsed(t0),
        ))

    def _try_specialists(self, query: Query) -> None:
        t0 = time.perf_counter()
        for specialist in self._specialists:
            if specialist.can_handle(query.text):
                spec_result = specialist.handle(query.text)
                if spec_result.confidence >= 0.6:
                    query.answer = spec_result.answer
                    query.confidence = Confidence.MEDIUM
                    query.needs_search = False
                    query.sources.append(f"specialist:{spec_result.topic}")
                    log.info(
                        "orchestrator: specialist '%s' answered with confidence %.2f",
                        spec_result.topic, spec_result.confidence,
                    )
                    break

        query.steps.append(StepTrace(
            kind=StepKind.REASON,
            input_summary=f"specialists ({len(self._specialists)} available)",
            output_summary=(
                f"answered by {query.sources[-1]}" if query.answer
                else "no specialist match"
            ),
            duration_ms=_elapsed(t0),
        ))

    def _search(self, query: Query) -> None:
        t0 = time.perf_counter()
        result = self._searcher.search(query.text)

        if result:
            query.answer = result
            query.confidence = Confidence.MEDIUM
            query.sources.append("wikipedia")
            query.needs_search = False
            log.info("orchestrator: search hit — %d chars from wikipedia", len(result))
        else:
            log.info("orchestrator: search miss for '%s'", query.text[:60])

        query.steps.append(StepTrace(
            kind=StepKind.SEARCH,
            input_summary=query.text[:100],
            output_summary=f"wikipedia: {len(result)} chars" if result else "no results",
            duration_ms=_elapsed(t0),
            metadata={"source": "wikipedia", "found": result is not None},
        ))

    def _reason(self, query: Query) -> None:
        t0 = time.perf_counter()

        if query.answer:
            query.steps.append(StepTrace(
                kind=StepKind.REASON,
                input_summary="answer already set by earlier stage",
                output_summary=query.answer[:100],
                duration_ms=_elapsed(t0),
            ))
            return

        normalised = " ".join(query.text.lower().split())
        answer = self._knowledge.get(normalised)
        if answer is None:
            for key, value in self._knowledge.items():
                if key in normalised:
                    answer = value
                    break

        if answer:
            query.answer = answer
            query.confidence = Confidence.HIGH
            query.sources.append("builtin_knowledge")

        if not query.answer and self._lm:
            try:
                context = ""
                for s in query.steps:
                    if s.kind == StepKind.SEARCH and "chars" in s.output_summary:
                        context = s.output_summary
                        break
                lm_answer = self._lm.complete(context, query.text, max_tokens=60)
                if lm_answer and len(lm_answer) > 5:
                    query.answer = lm_answer
                    query.confidence = Confidence.LOW
                    query.sources.append("neural_kernel_lm")
                    log.info("orchestrator: LM generated answer (%d chars)", len(lm_answer))
            except Exception as exc:
                log.warning("orchestrator: LM generation failed: %s", exc)

        if not query.answer:
            query.answer = (
                f"I don't have enough information to answer: "
                f"'{query.text[:80]}'. This would require more training data "
                f"or a larger model."
            )
            query.confidence = Confidence.NONE
            query.sources.append("fallback")

        if query.confidence == Confidence.HIGH:
            self._learner.record_success(query.text[:50])

        for plugin in self._plugins:
            try:
                plugin(query)
            except Exception:
                log.exception("orchestrator: plugin %s failed", plugin)

        query.steps.append(StepTrace(
            kind=StepKind.REASON,
            input_summary=query.text[:100],
            output_summary=query.answer[:100],
            duration_ms=_elapsed(t0),
        ))

    def _validate(self, query: Query) -> None:
        t0 = time.perf_counter()

        logic = self._checker.check(query.text, query.answer)
        if not logic.passed:
            log.warning("critic: issues=%s suggestions=%s", logic.issues, logic.suggestions)

        facts = self._fact_checker.check(query.answer, query.sources)
        if not facts.trusted:
            log.warning("truth: warnings=%s", facts.warnings)
            self._learner.record_mistake(
                question=query.text,
                bad_answer=query.answer,
                reason=str(facts.warnings),
            )

        query.steps.append(StepTrace(
            kind=StepKind.VALIDATE,
            input_summary=query.answer[:100],
            output_summary=(
                f"logic={'ok' if logic.passed else logic.issues} "
                f"facts={'trusted' if facts.trusted else facts.warnings}"
            ),
            duration_ms=_elapsed(t0),
            metadata={
                "logic_passed": logic.passed,
                "logic_issues": logic.issues,
                "fact_trusted": facts.trusted,
                "fact_confidence": facts.confidence,
                "fact_warnings": facts.warnings,
            },
        ))

    def _persist(self, query: Query) -> None:
        t0 = time.perf_counter()
        self._memory.remember(query)
        query.steps.append(StepTrace(
            kind=StepKind.PERSIST,
            input_summary=query.id,
            output_summary="stored" if query.succeeded else "skipped (failed)",
            duration_ms=_elapsed(t0),
        ))


def _elapsed(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)

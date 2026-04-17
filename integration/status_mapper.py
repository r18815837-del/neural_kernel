"""Derive client-friendly status fields from internal data.

Centralizes all the logic that a Flutter client should NOT have
to compute itself: progress percent, human-readable labels,
boolean flags, agent counts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Machine status → human-readable label
_STATUS_LABELS: Dict[str, str] = {
    "pending": "Queued",
    "in_progress": "Generating…",
    "completed": "Ready",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "archived": "Archived",
    "regenerating": "Regenerating…",
}


class StatusMapper:
    """Pure-function helpers to derive client-friendly status fields."""

    # ------------------------------------------------------------------
    # Status label
    # ------------------------------------------------------------------

    @staticmethod
    def status_label(status: str) -> str:
        """Return a human-readable label for a machine status."""
        return _STATUS_LABELS.get(status, status.replace("_", " ").title())

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    @staticmethod
    def progress_percent(
        status: str,
        raw_progress: float | None = None,
    ) -> int:
        """Convert internal 0.0-1.0 progress to 0-100 integer.

        If no raw_progress is given, derive from status.
        """
        if raw_progress is not None:
            return max(0, min(100, int(raw_progress * 100)))

        mapping = {
            "pending": 0,
            "in_progress": 50,
            "completed": 100,
            "failed": 100,
            "cancelled": 100,
            "archived": 100,
            "regenerating": 10,
        }
        return mapping.get(status, 0)

    # ------------------------------------------------------------------
    # Artifact availability
    # ------------------------------------------------------------------

    @staticmethod
    def artifact_available(
        status: str,
        artifact_path: str | None = None,
    ) -> bool:
        return status == "completed" and bool(artifact_path)

    # ------------------------------------------------------------------
    # Execution readiness
    # ------------------------------------------------------------------

    @staticmethod
    def execution_ready(payload: Dict[str, Any]) -> bool | None:
        """Derive execution_ready from execution_validation metadata.

        Returns None if no execution validation data is present.
        """
        ev = payload.get("execution_validation")
        if not ev:
            return None
        if isinstance(ev, dict):
            return ev.get("success", None)
        return None

    # ------------------------------------------------------------------
    # Quality score
    # ------------------------------------------------------------------

    @staticmethod
    def quality_score(payload: Dict[str, Any]) -> dict[str, Any] | None:
        """Build a flat quality summary from internal metadata."""
        scaffold = payload.get("scaffold_validation", {})
        consistency = payload.get("consistency", {})
        execution = payload.get("execution_validation", {})

        has_any = bool(scaffold or consistency or execution)
        if not has_any:
            return None

        scaffold_valid = None
        if isinstance(scaffold, dict) and scaffold:
            missing = scaffold.get("missing_files", [])
            empty = scaffold.get("empty_files", [])
            scaffold_valid = not missing and not empty

        exec_ready = None
        if isinstance(execution, dict) and execution:
            exec_ready = execution.get("success")

        consistency_ok = None
        if isinstance(consistency, dict) and consistency:
            consistency_ok = consistency.get("is_consistent")

        # Overall: ratio of passing signals
        signals = [v for v in (scaffold_valid, exec_ready, consistency_ok) if v is not None]
        overall = sum(signals) / len(signals) if signals else None

        return {
            "scaffold_valid": scaffold_valid,
            "execution_ready": exec_ready,
            "consistency_ok": consistency_ok,
            "overall_score": round(overall, 2) if overall is not None else None,
        }

    # ------------------------------------------------------------------
    # LLM used
    # ------------------------------------------------------------------

    @staticmethod
    def llm_used(agent_details: List[Dict[str, Any]]) -> bool:
        """True if any agent in the pipeline used LLM (not fallback)."""
        for a in agent_details:
            if a.get("llm_powered"):
                return True
        return False

    # ------------------------------------------------------------------
    # Agent counts
    # ------------------------------------------------------------------

    @staticmethod
    def agent_counts(agent_details: List[Dict[str, Any]]) -> tuple[int, int, int]:
        """Return (total, successful, failed) agent counts."""
        total = len(agent_details)
        successful = sum(1 for a in agent_details if a.get("success"))
        failed = total - successful
        return total, successful, failed

    # ------------------------------------------------------------------
    # Human message
    # ------------------------------------------------------------------

    @staticmethod
    def derive_message(
        status: str,
        internal_message: str | None = None,
        error: str | None = None,
    ) -> str:
        """Produce a clean human-readable message."""
        if error and status == "failed":
            # Don't expose internal stack traces
            if len(error) > 200:
                return "Generation failed — see error details"
            return f"Failed: {error}"
        if internal_message:
            return internal_message
        return _STATUS_LABELS.get(status, f"Status: {status}")

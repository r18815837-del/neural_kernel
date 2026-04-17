"""Tests for ClientContractService facade + lifecycle/list mapping."""
from __future__ import annotations
import os, sys
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.client_contract import ClientContractService
from integration.response_mapper import ResponseMapper


# Minimal stub for StoredProject
@dataclass
class FakeStoredProject:
    project_id: str = "p1"
    user_id: str = "u1"
    session_id: str = "s1"
    raw_text: str = "Build me an app"
    project_name: str = "myapp"
    summary: str = "test"
    project_type: str = "application"
    features_json: str = '[{"name": "auth", "description": "Auth"}]'
    tech_stack_json: str = '{"backend": "FastAPI", "database": "PostgreSQL"}'
    status: str = "completed"
    artifact_path: Optional[str] = "/tmp/myapp.zip"
    error_message: Optional[str] = None
    created_at: datetime = datetime(2025, 1, 1)
    updated_at: datetime = datetime(2025, 1, 1, 0, 10)


def test_contract_create_project():
    c = ClientContractService()
    dto = c.create_project_response("p1", "pending", "Queued", "myapp")
    assert dto.project_id == "p1"
    assert dto.status == "pending"


def test_contract_project_status():
    c = ClientContractService()
    data = {
        "status": "in_progress",
        "progress": 0.6,
        "message": "Working...",
        "project_name": "myapp",
        "created_at": datetime(2025, 1, 1),
        "features_detected": ["auth"],
        "tech_stack": {"backend": "FastAPI"},
        "agent_details": [{"success": True, "llm_powered": True}],
    }
    dto = c.project_status("p1", data)
    assert dto.progress_percent == 60
    assert dto.status_label == "Generating…"
    assert dto.llm_used is True
    assert dto.agent_count == 1


def test_contract_project_status_from_stored():
    c = ClientContractService()
    stored = FakeStoredProject()
    dto = c.project_status_from_stored(stored)
    assert dto.project_id == "p1"
    assert dto.status == "completed"
    assert dto.status_label == "Ready"
    assert dto.artifact_available is True
    assert len(dto.features) == 1
    assert dto.features[0].name == "auth"
    assert dto.tech_stack is not None
    assert dto.tech_stack.backend == "FastAPI"
    assert dto.download_url == "/api/v1/download/p1"


def test_contract_project_list():
    c = ClientContractService()
    stored_list = [
        FakeStoredProject(project_id="p1", project_name="app1", status="completed"),
        FakeStoredProject(project_id="p2", project_name="app2", status="failed", artifact_path=None),
    ]
    dto = c.project_list(stored_list, limit=50, offset=0)
    assert dto.total == 2
    assert dto.projects[0].status == "completed"
    assert dto.projects[0].artifact_available is True
    assert dto.projects[1].artifact_available is False
    assert dto.projects[0].status_label == "Ready"


def test_contract_transition():
    c = ClientContractService()
    dto = c.transition_response("p1", "completed", "archived")
    assert dto.old_status == "completed"
    assert dto.new_status == "archived"
    assert "Transitioned" in dto.message


def test_contract_lifecycle_state():
    c = ClientContractService()
    dto = c.lifecycle_state("p1", "completed", ["archived", "regenerating"])
    assert dto.current_status == "completed"
    assert "archived" in dto.allowed_transitions


def test_contract_not_found():
    c = ClientContractService()
    dto = c.not_found("project", "p1")
    assert dto.code == "project_not_found"
    assert dto.resource_id == "p1"


def test_contract_validation_error():
    c = ClientContractService()
    dto = c.validation_error("Bad input")
    assert dto.code == "validation_error"


def test_contract_error_retryable():
    c = ClientContractService()
    dto = c.error("rate_limit", "Too many requests", retryable=True)
    assert dto.retryable is True


def test_status_dto_to_dict_completeness():
    """to_dict() must include all required fields for Flutter rendering."""
    c = ClientContractService()
    data = {
        "status": "completed",
        "progress": 1.0,
        "message": "Done",
        "project_name": "myapp",
        "created_at": datetime(2025, 1, 1),
        "completed_at": datetime(2025, 1, 1, 0, 5),
        "artifact_path": "/tmp/myapp.zip",
        "artifact_size_bytes": 1024,
        "features_detected": [{"name": "auth"}],
        "tech_stack": {"backend": "FastAPI"},
        "agent_details": [{"success": True, "llm_powered": True}],
        "error": None,
    }
    dto = c.project_status("p1", data)
    d = dto.to_dict()
    required = [
        "project_id", "project_name", "status", "status_label", "message",
        "created_at", "progress_percent", "artifact_available",
        "llm_used", "agent_count",
    ]
    for key in required:
        assert key in d, f"Missing key: {key}"


def test_list_dto_to_dict():
    c = ClientContractService()
    stored_list = [FakeStoredProject()]
    dto = c.project_list(stored_list)
    d = dto.to_dict()
    assert "projects" in d
    assert "total" in d
    assert d["projects"][0]["status_label"] == "Ready"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")

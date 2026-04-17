"""Tests for ProjectStatusDTO mapping — progress, llm_used, execution_ready."""
from __future__ import annotations
import os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.response_mapper import ResponseMapper
from integration.status_mapper import StatusMapper
from integration.dto.project_status import ProjectStatusDTO, FeatureItemDTO, TechStackDTO


def test_progress_percent_from_raw():
    assert StatusMapper.progress_percent("in_progress", 0.5) == 50
    assert StatusMapper.progress_percent("in_progress", 0.0) == 0
    assert StatusMapper.progress_percent("completed", 1.0) == 100


def test_progress_percent_from_status():
    assert StatusMapper.progress_percent("pending") == 0
    assert StatusMapper.progress_percent("in_progress") == 50
    assert StatusMapper.progress_percent("completed") == 100
    assert StatusMapper.progress_percent("failed") == 100


def test_status_label():
    assert StatusMapper.status_label("pending") == "Queued"
    assert StatusMapper.status_label("completed") == "Ready"
    assert StatusMapper.status_label("in_progress") == "Generating…"
    assert StatusMapper.status_label("unknown_thing") == "Unknown Thing"


def test_llm_used_true():
    agents = [
        {"agent_name": "Architect", "success": True, "llm_powered": True},
        {"agent_name": "Backend", "success": True, "llm_powered": False},
    ]
    assert StatusMapper.llm_used(agents) is True


def test_llm_used_false():
    agents = [
        {"agent_name": "Backend", "success": True, "llm_powered": False},
    ]
    assert StatusMapper.llm_used(agents) is False


def test_execution_ready_from_payload():
    assert StatusMapper.execution_ready({"execution_validation": {"success": True}}) is True
    assert StatusMapper.execution_ready({"execution_validation": {"success": False}}) is False
    assert StatusMapper.execution_ready({}) is None


def test_agent_counts():
    agents = [
        {"success": True}, {"success": True}, {"success": False},
    ]
    total, ok, fail = StatusMapper.agent_counts(agents)
    assert total == 3
    assert ok == 2
    assert fail == 1


def test_derive_message_error():
    msg = StatusMapper.derive_message("failed", error="Connection refused")
    assert "Connection refused" in msg


def test_derive_message_long_error():
    msg = StatusMapper.derive_message("failed", error="x" * 300)
    assert "see error details" in msg


def test_mapper_to_project_status_dto():
    mapper = ResponseMapper(base_url="/api/v1")
    data = {
        "status": "completed",
        "progress": 1.0,
        "message": "Done",
        "project_name": "myapp",
        "created_at": datetime(2025, 1, 1),
        "completed_at": datetime(2025, 1, 1, 0, 5),
        "artifact_path": "/tmp/fake.zip",  # doesn't need to exist for mapping
        "artifact_size_bytes": 12345,
        "features_detected": [
            {"name": "auth", "description": "Authentication", "priority": "high"},
            "crud",
        ],
        "tech_stack": {"backend": "FastAPI", "database": "PostgreSQL"},
        "agent_details": [
            {"agent_name": "Arch", "success": True, "llm_powered": True},
            {"agent_name": "BE", "success": False, "llm_powered": False},
        ],
        "error": None,
    }
    dto = mapper.to_project_status_dto("p1", data)
    assert dto.project_id == "p1"
    assert dto.progress_percent == 100
    assert dto.artifact_available is True
    assert dto.llm_used is True
    assert dto.agent_count == 2
    assert dto.successful_agent_count == 1
    assert dto.failed_agent_count == 1
    assert len(dto.features) == 2
    assert dto.features[0].name == "auth"
    assert dto.features[1].name == "crud"
    assert dto.tech_stack is not None
    assert dto.tech_stack.backend == "FastAPI"
    assert dto.download_url == "/api/v1/download/p1"


def test_dto_to_dict_no_internal_blobs():
    """Ensure to_dict() doesn't expose huge internal structures."""
    mapper = ResponseMapper()
    data = {
        "status": "completed",
        "progress": 1.0,
        "message": "OK",
        "project_name": "x",
        "created_at": datetime(2025, 1, 1),
        "artifact_path": "/tmp/fake.zip",
        "features_detected": [],
        "tech_stack": {},
        "agent_details": [],
    }
    dto = mapper.to_project_status_dto("p1", data)
    d = dto.to_dict()
    # Must not contain internal keys
    assert "agent_details" not in d
    assert "scaffold_validation" not in d
    assert "pipeline" not in d
    assert "artifact_path" not in d
    # Must contain flat client fields
    assert "progress_percent" in d
    assert "artifact_available" in d
    assert "llm_used" in d


def test_quality_score():
    qs = StatusMapper.quality_score({
        "scaffold_validation": {"missing_files": [], "empty_files": []},
        "execution_validation": {"success": True},
        "consistency": {"is_consistent": True},
    })
    assert qs is not None
    assert qs["scaffold_valid"] is True
    assert qs["execution_ready"] is True
    assert qs["overall_score"] == 1.0


def test_quality_score_partial():
    qs = StatusMapper.quality_score({
        "scaffold_validation": {"missing_files": ["x.py"], "empty_files": []},
    })
    assert qs is not None
    assert qs["scaffold_valid"] is False
    assert qs["overall_score"] == 0.0


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")

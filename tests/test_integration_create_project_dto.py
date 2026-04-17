"""Tests for CreateProjectResponseDTO and its mapping."""
from __future__ import annotations
import os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.dto.create_project import CreateProjectResponseDTO
from integration.response_mapper import ResponseMapper


def test_dto_fields():
    dto = CreateProjectResponseDTO(
        project_id="p1",
        project_name="myapp",
        status="pending",
        message="Queued",
        created_at="2025-01-01T00:00:00",
    )
    assert dto.project_id == "p1"
    assert dto.artifact_available is False
    assert dto.artifact_name is None


def test_to_dict():
    dto = CreateProjectResponseDTO(
        project_id="p1",
        project_name="myapp",
        status="completed",
        message="Done",
        created_at="2025-01-01T00:00:00",
        artifact_available=True,
        artifact_name="myapp.zip",
    )
    d = dto.to_dict()
    assert d["artifact_available"] is True
    assert d["artifact_name"] == "myapp.zip"
    assert "project_id" in d


def test_mapper_create_project():
    mapper = ResponseMapper()
    dto = mapper.to_create_project_dto(
        project_id="p1",
        status="pending",
        message="Queued",
        project_name="myapp",
        created_at=datetime(2025, 1, 1),
    )
    assert dto.project_id == "p1"
    assert dto.status == "pending"
    assert dto.artifact_available is False
    assert dto.created_at == "2025-01-01T00:00:00"


def test_mapper_create_project_with_artifact():
    mapper = ResponseMapper()
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        f.write(b"fake")
        f.flush()
        dto = mapper.to_create_project_dto(
            project_id="p1",
            status="completed",
            artifact_path=f.name,
        )
        assert dto.artifact_available is True
        assert dto.artifact_name == Path(f.name).name
        os.unlink(f.name)


def test_frozen_dto():
    dto = CreateProjectResponseDTO(
        project_id="p1", project_name="x", status="pending",
        message="q", created_at="2025-01-01T00:00:00",
    )
    try:
        dto.project_id = "p2"  # type: ignore
        assert False, "Should raise"
    except AttributeError:
        pass  # Frozen dataclass


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")

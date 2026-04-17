"""Tests for error DTO mapping."""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.dto.errors import ErrorDTO, NotFoundErrorDTO, ValidationErrorDTO
from integration.response_mapper import ResponseMapper


def test_error_dto():
    e = ErrorDTO(code="internal", message="Something broke", retryable=True)
    d = e.to_dict()
    assert d["code"] == "internal"
    assert d["retryable"] is True
    assert "details" not in d  # empty dict → omitted


def test_error_dto_with_details():
    e = ErrorDTO(code="limit", message="Too many", details={"max": 100})
    d = e.to_dict()
    assert d["details"] == {"max": 100}


def test_not_found_dto():
    e = NotFoundErrorDTO(
        code="project_not_found",
        message="Project not found",
        resource_type="project",
        resource_id="p1",
    )
    d = e.to_dict()
    assert d["resource_type"] == "project"
    assert d["resource_id"] == "p1"
    assert d["code"] == "project_not_found"


def test_validation_error_dto():
    e = ValidationErrorDTO(
        code="validation_error",
        message="Bad input",
        field_errors=[{"field": "text", "error": "required"}],
    )
    d = e.to_dict()
    assert d["field_errors"][0]["field"] == "text"


def test_mapper_to_error_dto():
    mapper = ResponseMapper()
    dto = mapper.to_error_dto("timeout", "Request timed out", retryable=True)
    assert dto.code == "timeout"
    assert dto.retryable is True


def test_mapper_to_not_found_dto():
    mapper = ResponseMapper()
    dto = mapper.to_not_found_dto("artifact", "a1")
    assert dto.code == "artifact_not_found"
    assert dto.resource_id == "a1"


def test_mapper_to_validation_error_dto():
    mapper = ResponseMapper()
    dto = mapper.to_validation_error_dto("Missing field", [{"field": "name", "error": "required"}])
    assert dto.code == "validation_error"
    assert len(dto.field_errors) == 1


def test_error_retryable_default():
    e = ErrorDTO(code="x", message="y")
    assert e.retryable is False


def test_lifecycle_error():
    """Lifecycle transition errors should map cleanly."""
    mapper = ResponseMapper()
    dto = mapper.to_error_dto(
        "invalid_transition",
        "Cannot transition from completed to in_progress",
        details={"current": "completed", "target": "in_progress"},
        retryable=False,
    )
    assert dto.code == "invalid_transition"
    assert dto.details["current"] == "completed"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")

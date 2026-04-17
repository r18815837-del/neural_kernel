"""Tests for ArtifactMetadataDTO mapping."""
from __future__ import annotations
import os, sys, tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.response_mapper import ResponseMapper
from integration.artifact_service import ArtifactService
from integration.dto.artifact_metadata import ArtifactMetadataDTO


def test_artifact_metadata_mapping():
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        f.write(b"A" * 1024)
        f.flush()
        mapper = ResponseMapper(base_url="/api/v1")
        dto = mapper.to_artifact_metadata_dto(
            project_id="p1",
            project_name="myapp",
            artifact_path=f.name,
            features=[{"name": "auth", "description": "Auth module"}],
            tech_stack={"backend": "FastAPI"},
            created_at=datetime(2025, 6, 1),
            version=2,
        )
        assert dto.artifact_size_bytes == 1024
        assert dto.artifact_name == Path(f.name).name
        assert dto.download_url == "/api/v1/download/p1"
        assert dto.version == 2
        assert len(dto.features) == 1
        assert dto.tech_stack is not None
        assert dto.tech_stack.backend == "FastAPI"
        os.unlink(f.name)


def test_artifact_metadata_to_dict():
    dto = ArtifactMetadataDTO(
        project_id="p1",
        project_name="myapp",
        artifact_name="myapp.zip",
        artifact_size_bytes=500,
        packaging_format="zip",
        download_url="/api/v1/download/p1",
        created_at="2025-01-01T00:00:00",
    )
    d = dto.to_dict()
    assert d["packaging_format"] == "zip"
    assert d["download_url"] == "/api/v1/download/p1"
    assert "version" not in d  # None → omitted


def test_artifact_service_available():
    svc = ArtifactService()
    assert svc.is_available(None) is False
    assert svc.is_available("/nonexistent/path") is False
    with tempfile.NamedTemporaryFile() as f:
        assert svc.is_available(f.name) is True


def test_artifact_service_file_size():
    svc = ArtifactService()
    assert svc.file_size(None) is None
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"hello")
        f.flush()
        assert svc.file_size(f.name) == 5


def test_artifact_service_download_url():
    svc = ArtifactService(base_url="/api/v1")
    assert svc.download_url("p1") == "/api/v1/download/p1"


def test_artifact_service_get_metadata():
    svc = ArtifactService(base_url="/api/v1")
    assert svc.get_metadata("p1", "myapp", "/nonexistent") is None

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        f.write(b"data")
        f.flush()
        dto = svc.get_metadata("p1", "myapp", f.name)
        assert dto is not None
        assert dto.artifact_size_bytes == 4
        os.unlink(f.name)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")

"""Tests for DockerFilesValidator."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.docker_files import DockerFilesValidator
from artifacts.file_tree import ProjectFileTree
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


def _ctx(files: dict[str, str], deployment: str | None = None) -> ExecutionValidationContext:
    tree = ProjectFileTree()
    spec = ProjectSpec(
        project_name="myapp",
        summary="test",
        tech_stack=TechStackSpec(deployment=deployment) if deployment else None,
    )
    for path, content in files.items():
        tree.add_file(f"myapp/{path}", content)
    return ExecutionValidationContext(project_spec=spec, tree=tree)


def test_no_docker_no_deployment():
    ctx = _ctx({"README.md": "hi"})
    r = DockerFilesValidator().validate(ctx)
    assert r.success is True


def test_dockerfile_present_nonempty():
    ctx = _ctx({"Dockerfile": "FROM python:3.11\nCMD uvicorn main:app"})
    r = DockerFilesValidator().validate(ctx)
    assert r.success is True
    assert "Dockerfile" in r.details["dockerfiles"]


def test_dockerfile_empty():
    ctx = _ctx({"Dockerfile": ""})
    r = DockerFilesValidator().validate(ctx)
    assert r.success is False
    assert any("empty" in e for e in r.errors)


def test_compose_missing_services():
    ctx = _ctx({"docker-compose.yml": "version: '3'\n"})
    r = DockerFilesValidator().validate(ctx)
    assert r.success is True  # not empty → OK
    assert any("services" in w for w in r.warnings)


def test_docker_expected_but_missing():
    ctx = _ctx({"README.md": "hi"}, deployment="Docker")
    r = DockerFilesValidator().validate(ctx)
    assert r.success is False
    assert any("no Dockerfile" in e for e in r.errors)


def test_docker_expected_compose_missing():
    ctx = _ctx({"Dockerfile": "FROM python:3.11\nCMD true"}, deployment="Docker Compose")
    r = DockerFilesValidator().validate(ctx)
    assert r.success is True  # Dockerfile present
    assert any("docker-compose" in w.lower() for w in r.warnings)


def test_compose_with_services():
    ctx = _ctx({
        "docker-compose.yml": "services:\n  app:\n    build: .\n",
        "Dockerfile": "FROM python:3.11\nCMD true",
    }, deployment="Docker")
    r = DockerFilesValidator().validate(ctx)
    assert r.success is True
    assert "docker compose" in r.details.get("suggested_docker_command", "")


def test_no_from_in_dockerfile():
    ctx = _ctx({"Dockerfile": "RUN pip install fastapi\n"})
    r = DockerFilesValidator().validate(ctx)
    assert r.success is True  # not empty → pass
    assert any("FROM" in w for w in r.warnings)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

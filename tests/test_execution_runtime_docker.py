"""Tests for DockerComposeRuntimeValidator — docker compose config check."""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.docker_compose_runtime import DockerComposeRuntimeValidator
from artifacts.execution_validation.sandbox import command_available
from runtime.specs.project_spec import ProjectSpec


def _write_project(tmpdir: str, files: dict[str, str]) -> Path:
    root = Path(tmpdir) / "myapp"
    for rel, content in files.items():
        fp = root / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
    return root


def _ctx(root: Path) -> ExecutionValidationContext:
    spec = ProjectSpec(project_name="myapp", summary="test")
    return ExecutionValidationContext(project_spec=spec, project_root=root)


def test_no_compose_file_skipped():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {"README.md": "hi"})
        r = DockerComposeRuntimeValidator().validate(_ctx(root))
        assert r.success is True
        assert "skipped" in r.message.lower() or "no compose" in r.message.lower()


def test_skipped_without_project_root():
    spec = ProjectSpec(project_name="myapp", summary="test")
    ctx = ExecutionValidationContext(project_spec=spec, project_root=None)
    r = DockerComposeRuntimeValidator().validate(ctx)
    assert r.success is True
    assert r.details.get("skipped") is True


def test_docker_not_available_graceful():
    """If docker is not on PATH, should succeed with warning."""
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "docker-compose.yml": "services:\n  app:\n    image: python:3.11\n",
        })
        r = DockerComposeRuntimeValidator().validate(_ctx(root))
        if not command_available("docker"):
            # Docker not installed — should succeed with warning
            assert r.success is True
            assert any("not available" in w.lower() or "not installed" in w.lower() for w in r.warnings)
        else:
            # Docker is installed — should actually validate
            assert isinstance(r.success, bool)
            assert "exit_code" in r.details


def test_valid_compose_if_docker_available():
    """If docker is available, valid compose should pass."""
    if not command_available("docker"):
        return  # Skip on machines without docker

    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "docker-compose.yml": "services:\n  app:\n    image: python:3.11\n",
        })
        r = DockerComposeRuntimeValidator().validate(_ctx(root))
        assert r.success is True
        assert r.details["exit_code"] == 0


def test_invalid_compose_if_docker_available():
    """If docker is available, invalid compose should fail."""
    if not command_available("docker"):
        return  # Skip on machines without docker

    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "docker-compose.yml": "this: is: not: valid: compose: {{{",
        })
        r = DockerComposeRuntimeValidator().validate(_ctx(root))
        assert r.success is False


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

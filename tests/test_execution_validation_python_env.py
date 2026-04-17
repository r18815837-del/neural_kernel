"""Tests for PythonEnvValidator."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.python_env import PythonEnvValidator
from artifacts.file_tree import ProjectFileTree
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


def _ctx(files: dict[str, str], backend: str | None = None) -> ExecutionValidationContext:
    tree = ProjectFileTree()
    spec = ProjectSpec(
        project_name="myapp",
        summary="test",
        tech_stack=TechStackSpec(backend=backend) if backend else None,
    )
    for path, content in files.items():
        tree.add_file(f"myapp/{path}", content)
    return ExecutionValidationContext(project_spec=spec, tree=tree)


def test_with_requirements_txt():
    ctx = _ctx({"requirements.txt": "fastapi\nuvicorn\n"})
    r = PythonEnvValidator().validate(ctx)
    assert r.success is True
    assert r.details["requirements_count"] == 2


def test_with_pyproject_toml():
    ctx = _ctx({"pyproject.toml": "[project]\nname = 'x'\n"})
    r = PythonEnvValidator().validate(ctx)
    assert r.success is True


def test_no_deps_at_all():
    ctx = _ctx({"main.py": "print(1)"})
    r = PythonEnvValidator().validate(ctx)
    assert r.success is False
    assert any("No dependency declaration" in e for e in r.errors)


def test_empty_requirements():
    ctx = _ctx({"requirements.txt": ""})
    r = PythonEnvValidator().validate(ctx)
    assert r.success is False
    assert any("empty" in e for e in r.errors)


def test_comments_only_requirements():
    ctx = _ctx({"requirements.txt": "# comment\n# another\n"})
    r = PythonEnvValidator().validate(ctx)
    assert r.success is True  # file not empty
    assert any("comments" in w for w in r.warnings)


def test_python_backend_without_deps():
    ctx = _ctx({"main.py": "print(1)"}, backend="FastAPI")
    r = PythonEnvValidator().validate(ctx)
    assert r.success is False
    assert r.details.get("python_backend_detected") is True


def test_python_backend_with_deps():
    ctx = _ctx({"requirements.txt": "fastapi\n"}, backend="FastAPI")
    r = PythonEnvValidator().validate(ctx)
    assert r.success is True


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

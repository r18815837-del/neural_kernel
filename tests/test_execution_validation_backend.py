"""Tests for BackendStartValidator."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.backend_start import BackendStartValidator
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


def test_no_backend_declared_skip():
    ctx = _ctx({"README.md": "hi"})
    r = BackendStartValidator().validate(ctx)
    assert r.success is True
    assert "skipped" in r.message.lower()


def test_missing_entrypoint():
    ctx = _ctx({"README.md": "hi"}, backend="FastAPI")
    r = BackendStartValidator().validate(ctx)
    assert r.success is False
    assert any("entrypoint" in e.lower() for e in r.errors)


def test_fastapi_detected():
    code = "from fastapi import FastAPI\napp = FastAPI()\n"
    ctx = _ctx({"backend/main.py": code}, backend="FastAPI")
    r = BackendStartValidator().validate(ctx)
    assert r.success is True
    assert r.details["detected_framework"] == "fastapi"
    assert "uvicorn" in r.details.get("suggested_start_command", "")


def test_flask_detected():
    code = "from flask import Flask\napp = Flask(__name__)\n"
    ctx = _ctx({"backend/main.py": code}, backend="Flask")
    r = BackendStartValidator().validate(ctx)
    assert r.success is True
    assert r.details["detected_framework"] == "flask"


def test_django_detected():
    code = "import django\nos.environ['DJANGO_SETTINGS_MODULE'] = 'myapp.settings'\n"
    ctx = _ctx({"manage.py": code}, backend="Django")
    r = BackendStartValidator().validate(ctx)
    assert r.success is True
    assert r.details["detected_framework"] == "django"


def test_framework_mismatch_warning():
    code = "from flask import Flask\napp = Flask(__name__)\n"
    ctx = _ctx({"backend/main.py": code}, backend="FastAPI")
    r = BackendStartValidator().validate(ctx)
    assert r.success is True  # entrypoint exists — pass
    assert any("Declared backend" in w for w in r.warnings)


def test_unknown_framework_warning():
    code = "print('hello')\n"
    ctx = _ctx({"backend/main.py": code}, backend="FastAPI")
    r = BackendStartValidator().validate(ctx)
    assert r.success is True
    assert any("Could not detect" in w for w in r.warnings)


def test_app_py_as_entrypoint():
    code = "from fastapi import FastAPI\napp = FastAPI()\n"
    ctx = _ctx({"backend/app.py": code}, backend="FastAPI")
    r = BackendStartValidator().validate(ctx)
    assert r.success is True


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

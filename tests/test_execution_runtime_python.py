"""Tests for PythonRuntimeValidator — py_compile smoke check."""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.python_runtime import PythonRuntimeValidator
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


def test_valid_python_files():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "backend/main.py": "from pathlib import Path\nprint('ok')\n",
            "backend/__init__.py": "",
        })
        r = PythonRuntimeValidator().validate(_ctx(root))
        assert r.success is True
        assert r.details["compiled_ok"] >= 2


def test_syntax_error_fails():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "backend/main.py": "def broken(\n",
        })
        r = PythonRuntimeValidator().validate(_ctx(root))
        assert r.success is False
        assert len(r.details["compile_failed"]) == 1
        assert any("py_compile" in e for e in r.errors)


def test_no_py_files():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "README.md": "# hello",
        })
        r = PythonRuntimeValidator().validate(_ctx(root))
        assert r.success is True


def test_skipped_without_project_root():
    spec = ProjectSpec(project_name="myapp", summary="test")
    ctx = ExecutionValidationContext(project_spec=spec, project_root=None)
    r = PythonRuntimeValidator().validate(ctx)
    assert r.success is True
    assert r.details.get("skipped") is True


def test_ignores_pycache():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "backend/main.py": "x = 1\n",
            "__pycache__/bad.py": "this is not valid python {{{\n",
        })
        r = PythonRuntimeValidator().validate(_ctx(root))
        assert r.success is True


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

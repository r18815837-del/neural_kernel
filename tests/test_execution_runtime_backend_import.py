"""Tests for BackendImportValidator — module import smoke check."""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.backend_import import BackendImportValidator
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


def test_no_entrypoint_skipped():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {"README.md": "hi"})
        r = BackendImportValidator().validate(_ctx(root))
        assert r.success is True
        assert "skipped" in r.message.lower()


def test_import_simple_module():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "backend/__init__.py": "",
            "backend/main.py": "x = 42\n",
        })
        r = BackendImportValidator().validate(_ctx(root))
        assert r.success is True
        assert r.details["module_path"] == "backend.main"


def test_import_syntax_error():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "backend/__init__.py": "",
            "backend/main.py": "def broken(\n",
        })
        r = BackendImportValidator().validate(_ctx(root))
        assert r.success is False
        assert len(r.errors) > 0


def test_import_missing_third_party():
    """Missing third-party dep is warning, not failure."""
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "backend/__init__.py": "",
            "backend/main.py": "import some_nonexistent_package_xyz\n",
        })
        r = BackendImportValidator().validate(_ctx(root))
        # Should succeed with warning about missing dep
        assert r.success is True
        assert r.details.get("third_party_missing") is True


def test_skipped_without_root():
    spec = ProjectSpec(project_name="myapp", summary="test")
    ctx = ExecutionValidationContext(project_spec=spec, project_root=None)
    r = BackendImportValidator().validate(ctx)
    assert r.success is True
    assert r.details.get("skipped") is True


def test_derive_module():
    dm = BackendImportValidator._derive_module
    assert dm("backend/main.py") == "backend.main"
    assert dm("app.py") == "app"
    assert dm("src/app.py") == "src.app"
    assert dm("not_python.txt") is None


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

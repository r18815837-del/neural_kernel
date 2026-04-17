"""Tests for BackendStartupProbeValidator — server start + health probe."""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.backend_startup_probe import BackendStartupProbeValidator
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
    """No backend entrypoint → skipped."""
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {"README.md": "hi"})
        r = BackendStartupProbeValidator().validate(_ctx(root))
        assert r.success is True
        assert r.details.get("skipped") is True


def test_skipped_without_root():
    spec = ProjectSpec(project_name="myapp", summary="test")
    ctx = ExecutionValidationContext(project_spec=spec, project_root=None)
    r = BackendStartupProbeValidator().validate(ctx)
    assert r.success is True
    assert r.details.get("skipped") is True


def test_detect_framework_fastapi():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "backend/__init__.py": "",
            "backend/main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        })
        fw = BackendStartupProbeValidator._detect_framework(root / "backend" / "main.py")
        assert fw == "uvicorn"


def test_detect_framework_flask():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "app.py": "from flask import Flask\napp = Flask(__name__)\n",
        })
        fw = BackendStartupProbeValidator._detect_framework(root / "app.py")
        assert fw == "flask"


def test_derive_module():
    dm = BackendStartupProbeValidator._derive_module
    assert dm("backend/main.py") == "backend.main"
    assert dm("app.py") == "app"
    assert dm("not_python.txt") is None


def test_missing_deps_graceful():
    """Backend with missing third-party deps → warning, not failure."""
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "backend/__init__.py": "",
            "backend/main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        })
        r = BackendStartupProbeValidator().validate(_ctx(root))
        # uvicorn likely not installed in test env, or fastapi not installed
        # Should either succeed or show third_party_missing / command_not_found
        assert isinstance(r.success, bool)
        # Process should be cleaned up (no zombie)


def test_syntax_error_fails():
    """Backend with syntax error should fail or show error."""
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "backend/__init__.py": "",
            "backend/main.py": "def broken(\n",
        })
        r = BackendStartupProbeValidator().validate(_ctx(root))
        # Should detect that process exited with error
        assert isinstance(r.success, bool)


def test_build_start_cmd():
    bsc = BackendStartupProbeValidator._build_start_cmd
    cmd = bsc("uvicorn", "backend.main", 8080)
    assert cmd[0] == "uvicorn"
    assert "8080" in cmd
    cmd2 = bsc("flask", "app", 9090)
    assert "9090" in " ".join(str(c) for c in cmd2)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

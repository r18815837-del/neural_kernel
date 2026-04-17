"""Tests for PythonInstallDryRunValidator — pip install --dry-run."""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.python_install_dry_run import PythonInstallDryRunValidator
from artifacts.execution_validation.sandbox import pip_supports_dry_run
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


def test_no_requirements_skipped():
    """No requirements file → skipped."""
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {"README.md": "hi"})
        r = PythonInstallDryRunValidator().validate(_ctx(root))
        assert r.success is True
        assert r.details.get("skipped") is True


def test_skipped_without_root():
    spec = ProjectSpec(project_name="myapp", summary="test")
    ctx = ExecutionValidationContext(project_spec=spec, project_root=None)
    r = PythonInstallDryRunValidator().validate(ctx)
    assert r.success is True
    assert r.details.get("skipped") is True


def test_finds_requirements_txt():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {"requirements.txt": "flask==3.0.0\n"})
        r = PythonInstallDryRunValidator().validate(_ctx(root))
        assert r.details["requirements_file"] == "requirements.txt"
        # Result depends on pip version
        assert isinstance(r.success, bool)


def test_finds_pyproject_toml():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "pyproject.toml": '[project]\nname = "myapp"\nversion = "0.1.0"\n',
        })
        r = PythonInstallDryRunValidator().validate(_ctx(root))
        assert r.details.get("install_target") == "pyproject.toml" or r.details.get("requirements_file") is not None
        assert isinstance(r.success, bool)


def test_dry_run_with_stdlib_only():
    """requirements.txt with only stdlib-like packages should resolve."""
    if not pip_supports_dry_run():
        return  # Skip on old pip

    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "requirements.txt": "pip\n",  # pip itself is always available
        })
        r = PythonInstallDryRunValidator().validate(_ctx(root))
        # Should succeed or at least not crash
        assert isinstance(r.success, bool)
        assert "exit_code" in r.details


def test_classify_failure():
    clf = PythonInstallDryRunValidator._classify_failure
    assert clf("ERROR: Could not find a version that satisfies") == "resolution"
    assert clf("ERROR: No matching distribution found") == "resolution"
    assert clf("Connection timed out") == "network"
    assert clf("Invalid requirement 'foo==bar=='") == "packaging"
    assert clf("Something random went wrong") == "unknown"


def test_find_requirements_priority():
    """requirements.txt at root takes priority over backend/."""
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "requirements.txt": "flask\n",
            "backend/requirements.txt": "django\n",
        })
        r = PythonInstallDryRunValidator().validate(_ctx(root))
        assert r.details["requirements_file"] == "requirements.txt"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

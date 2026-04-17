"""Tests for PytestCollectValidator — pytest --collect-only smoke."""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.pytest_collect import PytestCollectValidator
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


def test_no_tests_dir_skipped():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {"README.md": "hi"})
        r = PytestCollectValidator().validate(_ctx(root))
        assert r.success is True
        assert "skipped" in r.message.lower() or "no test" in r.message.lower()


def test_collect_valid_tests():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "tests/test_smoke.py": "def test_ok(): assert True\n",
        })
        r = PytestCollectValidator().validate(_ctx(root))
        # If pytest is installed, should collect successfully
        if r.details.get("pytest_available") is True:
            assert r.success is True
            assert r.details.get("tests_collected", 0) >= 1
        else:
            # pytest not available — should skip
            assert r.success is True


def test_collect_syntax_error():
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "tests/test_broken.py": "def test_ok(\n",  # syntax error
        })
        r = PytestCollectValidator().validate(_ctx(root))
        if r.details.get("pytest_available") is True:
            assert r.success is False
            assert r.details.get("exit_code", 0) != 0
        else:
            assert r.success is True  # skipped


def test_skipped_without_project_root():
    spec = ProjectSpec(project_name="myapp", summary="test")
    ctx = ExecutionValidationContext(project_spec=spec, project_root=None)
    r = PytestCollectValidator().validate(ctx)
    assert r.success is True
    assert r.details.get("skipped") is True


def test_parse_collected_count():
    assert PytestCollectValidator._parse_collected_count("3 tests collected\n") == 3
    assert PytestCollectValidator._parse_collected_count("no tests ran\n") == 0
    assert PytestCollectValidator._parse_collected_count("12 items\n") == 12
    assert PytestCollectValidator._parse_collected_count("") == 0


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

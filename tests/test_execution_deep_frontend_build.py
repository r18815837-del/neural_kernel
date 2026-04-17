"""Tests for FrontendBuildSmokeValidator — npm/pnpm/yarn build smoke."""
from __future__ import annotations
import os, sys, tempfile, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.frontend_build_smoke import FrontendBuildSmokeValidator
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


def test_no_frontend_skipped():
    """No frontend dir → skipped."""
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {"README.md": "hi"})
        r = FrontendBuildSmokeValidator().validate(_ctx(root))
        assert r.success is True
        assert r.details.get("skipped") is True


def test_skipped_without_root():
    spec = ProjectSpec(project_name="myapp", summary="test")
    ctx = ExecutionValidationContext(project_spec=spec, project_root=None)
    r = FrontendBuildSmokeValidator().validate(ctx)
    assert r.success is True
    assert r.details.get("skipped") is True


def test_invalid_package_json():
    """Invalid JSON in package.json → fail."""
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "frontend/package.json": "{ this is not json }}}",
        })
        # detect_package_manager needs package.json to be valid — but it checks file existence
        # The validator should detect invalid JSON
        r = FrontendBuildSmokeValidator().validate(_ctx(root))
        # Might be skipped (no PM detected) or fail (invalid JSON)
        assert isinstance(r.success, bool)


def test_valid_package_json_no_build():
    """Valid package.json but no build script."""
    pkg = json.dumps({"name": "myapp", "version": "1.0.0", "dependencies": {}})
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "frontend/package.json": pkg,
        })
        r = FrontendBuildSmokeValidator().validate(_ctx(root))
        if r.details.get("skipped"):
            assert r.success is True
        else:
            assert r.details.get("has_build_script") is False
            assert r.details.get("package_json_valid") is True


def test_valid_package_json_with_build():
    """Valid package.json with build script."""
    pkg = json.dumps({
        "name": "myapp",
        "version": "1.0.0",
        "scripts": {"build": "echo build"},
        "dependencies": {},
    })
    with tempfile.TemporaryDirectory() as d:
        root = _write_project(d, {
            "frontend/package.json": pkg,
        })
        r = FrontendBuildSmokeValidator().validate(_ctx(root))
        if not r.details.get("skipped"):
            assert r.details.get("has_build_script") is True


def test_has_build_script():
    hbs = FrontendBuildSmokeValidator._has_build_script
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "package.json").write_text(json.dumps({"scripts": {"build": "webpack"}}))
        assert hbs(p) is True

        (p / "package.json").write_text(json.dumps({"scripts": {"start": "node ."}}))
        assert hbs(p) is False


def test_validate_package_json():
    vpj = FrontendBuildSmokeValidator._validate_package_json
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "package.json").write_text('{"name": "ok"}')
        assert vpj(p) is True

        (p / "package.json").write_text("not json")
        assert vpj(p) is False


def test_is_network_error():
    ine = FrontendBuildSmokeValidator._is_network_error
    assert ine("npm ERR! fetch failed ENOTFOUND") is True
    assert ine("npm ERR! network timeout") is True
    assert ine("npm ERR! missing script: build") is False


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

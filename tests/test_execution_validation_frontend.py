"""Tests for FrontendPresenceValidator."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.validators.frontend_presence import FrontendPresenceValidator
from artifacts.file_tree import ProjectFileTree
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


def _ctx(files: dict[str, str], frontend: str | None = None) -> ExecutionValidationContext:
    tree = ProjectFileTree()
    spec = ProjectSpec(
        project_name="myapp",
        summary="test",
        tech_stack=TechStackSpec(frontend=frontend) if frontend else None,
    )
    for path, content in files.items():
        tree.add_file(f"myapp/{path}", content)
    return ExecutionValidationContext(project_spec=spec, tree=tree)


def test_no_frontend_declared_skip():
    ctx = _ctx({"README.md": "hi"})
    r = FrontendPresenceValidator().validate(ctx)
    assert r.success is True
    assert "skipped" in r.message.lower()


def test_react_present():
    ctx = _ctx({
        "frontend/src/App.tsx": "export default function App() {}",
        "frontend/package.json": '{"name":"frontend"}',
    }, frontend="React")
    r = FrontendPresenceValidator().validate(ctx)
    assert r.success is True
    assert r.details["frontend_category"] == "react"


def test_react_missing():
    ctx = _ctx({"README.md": "hi"}, frontend="React")
    r = FrontendPresenceValidator().validate(ctx)
    assert r.success is False
    assert any("no entry files" in e.lower() for e in r.errors)


def test_vue_present():
    ctx = _ctx({
        "frontend/src/App.vue": "<template></template>",
        "frontend/package.json": '{"name":"frontend"}',
    }, frontend="Vue")
    r = FrontendPresenceValidator().validate(ctx)
    assert r.success is True


def test_generic_html():
    ctx = _ctx({
        "frontend/index.html": "<html></html>",
    }, frontend="Vanilla JS")
    r = FrontendPresenceValidator().validate(ctx)
    assert r.success is True
    assert r.details["frontend_category"] == "generic"


def test_empty_package_json_warning():
    ctx = _ctx({
        "frontend/index.html": "<html></html>",
        "frontend/package.json": "",
    }, frontend="React")
    r = FrontendPresenceValidator().validate(ctx)
    assert any("empty" in w for w in r.warnings)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

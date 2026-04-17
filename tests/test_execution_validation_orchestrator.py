"""Tests for ExecutionValidationOrchestrator."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.execution_validation.base import BaseExecutionValidator
from artifacts.execution_validation.context import ExecutionValidationContext
from artifacts.execution_validation.orchestrator import (
    ExecutionValidationOrchestrator,
    default_validators,
)
from artifacts.execution_validation.result import ExecutionValidationResult
from artifacts.file_tree import ProjectFileTree
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


def _full_ctx() -> ExecutionValidationContext:
    """Build a realistic project context with backend + tests + docker."""
    tree = ProjectFileTree()
    files = {
        "myapp/requirements.txt": "fastapi\nuvicorn\n",
        "myapp/backend/main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "myapp/tests/test_smoke.py": "def test_ok(): pass\n",
        "myapp/Dockerfile": "FROM python:3.11\nCOPY . .\nCMD uvicorn backend.main:app",
        "myapp/docker-compose.yml": "services:\n  app:\n    build: .\n",
        "myapp/README.md": "# MyApp",
    }
    for path, content in files.items():
        tree.add_file(path, content)
    spec = ProjectSpec(
        project_name="myapp",
        summary="test",
        tech_stack=TechStackSpec(backend="FastAPI", deployment="Docker"),
    )
    return ExecutionValidationContext(project_spec=spec, tree=tree)


def test_default_validators_count():
    assert len(default_validators()) == 6


def test_orchestrator_runs_all():
    ctx = _full_ctx()
    orch = ExecutionValidationOrchestrator()
    report = orch.run(ctx)
    assert report.checks_run == 6
    assert report.project_name == "myapp"


def test_orchestrator_all_pass():
    ctx = _full_ctx()
    report = ExecutionValidationOrchestrator().run(ctx)
    assert report.success is True
    assert report.checks_failed == 0


def test_orchestrator_aggregates_failures():
    """Force one validator to fail by removing requirements.txt."""
    tree = ProjectFileTree()
    tree.add_file("myapp/backend/main.py", "from fastapi import FastAPI\napp = FastAPI()")
    tree.add_file("myapp/README.md", "hi")
    spec = ProjectSpec(
        project_name="myapp",
        summary="test",
        tech_stack=TechStackSpec(backend="FastAPI"),
    )
    ctx = ExecutionValidationContext(project_spec=spec, tree=tree)
    report = ExecutionValidationOrchestrator().run(ctx)
    assert report.checks_failed >= 1
    assert report.success is False


def test_orchestrator_custom_validators():
    class AlwaysFail(BaseExecutionValidator):
        name = "always_fail"
        def validate(self, ctx):
            return ExecutionValidationResult(name=self.name, success=False, message="nope", errors=["boom"])

    ctx = _full_ctx()
    orch = ExecutionValidationOrchestrator(validators=[AlwaysFail()])
    report = orch.run(ctx)
    assert report.checks_run == 1
    assert report.checks_failed == 1


def test_orchestrator_exception_safety():
    class Exploder(BaseExecutionValidator):
        name = "exploder"
        def validate(self, ctx):
            raise RuntimeError("kaboom")

    ctx = _full_ctx()
    orch = ExecutionValidationOrchestrator(validators=[Exploder()])
    report = orch.run(ctx)
    assert report.checks_run == 1
    assert report.results[0].success is False
    assert "kaboom" in report.results[0].message


def test_report_to_dict():
    ctx = _full_ctx()
    report = ExecutionValidationOrchestrator().run(ctx)
    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["project_name"] == "myapp"
    assert isinstance(d["results"], list)
    assert len(d["results"]) == 6


def test_summary_contains_commands():
    ctx = _full_ctx()
    report = ExecutionValidationOrchestrator().run(ctx)
    assert "install_command" in report.summary
    assert "backend_run_command" in report.summary


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

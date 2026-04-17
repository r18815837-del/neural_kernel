"""Integration tests: runtime execution validation in builder & orchestrator."""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.builder import ArtifactBuilder
from artifacts.execution_validation import (
    ExecutionValidationOrchestrator,
    ExecutionValidationContext,
    ExecutionValidationReport,
)
from artifacts.execution_validation.orchestrator import (
    default_runtime_validators,
    default_static_validators,
)
from artifacts.execution_validation.validators.python_runtime import PythonRuntimeValidator
from artifacts.execution_validation.validators.backend_import import BackendImportValidator
from artifacts.consistency_validator import ConsistencyValidator
from artifacts.generators.docs_generator import DocsGenerator
from artifacts.generators.env_generator import EnvGenerator
from artifacts.generators.structure_generator import StructureGenerator
from artifacts.generators.tests_generator import TestsGenerator
from artifacts.manifest import ArtifactManifest
from artifacts.validators import ArtifactValidator
from artifacts.writer import ArtifactWriter
from artifacts.zip_packager import ZipPackager
from runtime.specs.artifact_spec import ArtifactSpec
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


# ------------------------------------------------------------------
# Orchestrator integration
# ------------------------------------------------------------------

def test_orchestrator_runtime_validators_default():
    rv = default_runtime_validators()
    names = {v.name for v in rv}
    assert "python_runtime" in names
    assert "pytest_collect" in names
    assert "backend_import" in names
    assert "docker_compose_runtime" in names


def test_orchestrator_run_runtime_on_disk():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "myapp"
        root.mkdir()
        (root / "backend").mkdir()
        (root / "backend" / "__init__.py").write_text("")
        (root / "backend" / "main.py").write_text("x = 1\n")
        (root / "tests").mkdir()
        (root / "tests" / "test_smoke.py").write_text("def test_ok(): assert True\n")

        spec = ProjectSpec(project_name="myapp", summary="test")
        ctx = ExecutionValidationContext(project_spec=spec, project_root=root)

        orch = ExecutionValidationOrchestrator()
        report = orch.run_runtime(ctx)
        assert report.checks_run >= 3  # python_runtime, pytest_collect, backend_import, docker
        assert report.project_name == "myapp"


def test_merge_reports():
    static = ExecutionValidationReport(
        project_name="myapp",
        results=[],
        summary={"install_command": "pip install -r requirements.txt"},
    )
    static.results.append(
        __import__("artifacts.execution_validation.result", fromlist=["ExecutionValidationResult"])
        .ExecutionValidationResult(name="python_env", success=True, message="ok")
    )

    runtime = ExecutionValidationReport(project_name="myapp", results=[])
    runtime.results.append(
        __import__("artifacts.execution_validation.result", fromlist=["ExecutionValidationResult"])
        .ExecutionValidationResult(name="python_runtime", success=True, message="ok")
    )

    merged = ExecutionValidationOrchestrator.merge_reports(static, runtime)
    assert merged.checks_run == 2
    assert merged.summary.get("runtime_checks_run") == 1
    assert merged.summary.get("static_checks_run") == 1
    assert merged.summary.get("install_command") == "pip install -r requirements.txt"


def test_merge_reports_no_runtime():
    static = ExecutionValidationReport(project_name="myapp", results=[], summary={"x": 1})
    merged = ExecutionValidationOrchestrator.merge_reports(static, None)
    assert merged is static


# ------------------------------------------------------------------
# Builder integration
# ------------------------------------------------------------------

def _make_builder() -> ArtifactBuilder:
    return ArtifactBuilder(
        structure_generator=StructureGenerator(),
        docs_generator=DocsGenerator(),
        env_generator=EnvGenerator(),
        tests_generator=TestsGenerator(),
        writer=ArtifactWriter(),
        validator=ArtifactValidator(),
        zip_packager=ZipPackager(),
        execution_validator=ExecutionValidationOrchestrator(),
    )


def test_builder_build_includes_runtime_results():
    builder = _make_builder()
    spec = ProjectSpec(
        project_name="testapp",
        summary="Integration test",
        tech_stack=TechStackSpec(backend="FastAPI"),
    )
    aspec = ArtifactSpec(artifact_name="testapp", packaging="folder", include_tests=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path, manifest = builder.build(tmpdir, spec, aspec)

        ev = manifest.metadata.get("execution_validation")
        assert ev is not None
        assert isinstance(ev, dict)

        # Should have both static and runtime results
        names = {r["name"] for r in ev.get("results", [])}
        # Static validators
        assert "python_env" in names
        assert "commands" in names
        # Runtime validators
        assert "python_runtime" in names


def test_builder_runtime_nonblocking():
    """Even if runtime checks fail, build() should not raise."""
    builder = _make_builder()
    spec = ProjectSpec(
        project_name="emptyapp",
        summary="minimal",
        tech_stack=TechStackSpec(backend="FastAPI"),
    )
    aspec = ArtifactSpec(artifact_name="emptyapp", packaging="folder", include_tests=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not raise
        path, manifest = builder.build(tmpdir, spec, aspec)
        ev = manifest.metadata.get("execution_validation")
        assert ev is not None
        assert "checks_run" in ev


def test_report_summary_has_runtime_stats():
    builder = _make_builder()
    spec = ProjectSpec(
        project_name="testapp",
        summary="Integration test",
        tech_stack=TechStackSpec(backend="FastAPI"),
    )
    aspec = ArtifactSpec(artifact_name="testapp", packaging="folder", include_tests=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        _, manifest = builder.build(tmpdir, spec, aspec)
        ev = manifest.metadata["execution_validation"]
        summary = ev.get("summary", {})
        assert "runtime_checks_run" in summary
        assert "static_checks_run" in summary


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

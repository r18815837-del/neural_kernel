"""Integration tests: deep validation in orchestrator & builder."""
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
    default_deep_validators,
    default_runtime_validators,
    default_static_validators,
)
from artifacts.execution_validation.result import ExecutionValidationResult
from artifacts.execution_validation.validators.python_install_dry_run import PythonInstallDryRunValidator
from artifacts.execution_validation.validators.backend_startup_probe import BackendStartupProbeValidator
from artifacts.execution_validation.validators.frontend_build_smoke import FrontendBuildSmokeValidator
from artifacts.execution_validation.validators.docker_build_probe import DockerBuildProbeValidator
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
# Orchestrator deep phase
# ------------------------------------------------------------------

def test_default_deep_validators():
    dv = default_deep_validators()
    names = {v.name for v in dv}
    assert "python_install_dry_run" in names
    assert "backend_startup_probe" in names
    assert "frontend_build_smoke" in names
    assert "docker_build_probe" in names


def test_orchestrator_run_deep():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "myapp"
        root.mkdir()
        (root / "backend").mkdir()
        (root / "backend" / "__init__.py").write_text("")
        (root / "backend" / "main.py").write_text("x = 1\n")
        (root / "requirements.txt").write_text("pip\n")

        spec = ProjectSpec(project_name="myapp", summary="test")
        ctx = ExecutionValidationContext(project_spec=spec, project_root=root)

        orch = ExecutionValidationOrchestrator()
        report = orch.run_deep(ctx)
        assert report.checks_run >= 4
        assert report.project_name == "myapp"


def test_merge_three_reports():
    """merge_reports with static + runtime + deep."""
    static = ExecutionValidationReport(
        project_name="myapp",
        results=[ExecutionValidationResult(name="python_env", success=True, message="ok")],
        summary={"install_command": "pip install -r requirements.txt"},
    )
    runtime = ExecutionValidationReport(
        project_name="myapp",
        results=[ExecutionValidationResult(name="python_runtime", success=True, message="ok")],
    )
    deep = ExecutionValidationReport(
        project_name="myapp",
        results=[
            ExecutionValidationResult(name="python_install_dry_run", success=True, message="ok"),
            ExecutionValidationResult(name="backend_startup_probe", success=True, message="ok"),
        ],
    )

    merged = ExecutionValidationOrchestrator.merge_reports(static, runtime, deep)
    assert merged.checks_run == 4
    assert merged.summary.get("static_checks_run") == 1
    assert merged.summary.get("runtime_checks_run") == 1
    assert merged.summary.get("deep_checks_run") == 2
    assert merged.summary.get("install_command") == "pip install -r requirements.txt"


def test_merge_no_deep():
    """merge_reports with deep=None preserves old behavior."""
    static = ExecutionValidationReport(
        project_name="myapp",
        results=[ExecutionValidationResult(name="python_env", success=True, message="ok")],
        summary={"x": 1},
    )
    runtime = ExecutionValidationReport(
        project_name="myapp",
        results=[ExecutionValidationResult(name="python_runtime", success=True, message="ok")],
    )

    merged = ExecutionValidationOrchestrator.merge_reports(static, runtime, None)
    assert merged.checks_run == 2
    assert "deep_checks_run" not in merged.summary


def test_merge_deep_only():
    """merge_reports with runtime=None but deep present."""
    static = ExecutionValidationReport(
        project_name="myapp",
        results=[ExecutionValidationResult(name="python_env", success=True, message="ok")],
    )
    deep = ExecutionValidationReport(
        project_name="myapp",
        results=[ExecutionValidationResult(name="docker_build_probe", success=True, message="ok")],
    )

    merged = ExecutionValidationOrchestrator.merge_reports(static, None, deep)
    assert merged.checks_run == 2
    assert merged.summary.get("deep_checks_run") == 1
    assert "runtime_checks_run" not in merged.summary


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


def test_builder_build_includes_deep_results():
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

        names = {r["name"] for r in ev.get("results", [])}
        # Deep validators should appear
        assert "python_install_dry_run" in names or "backend_startup_probe" in names or "docker_build_probe" in names

        summary = ev.get("summary", {})
        assert "deep_checks_run" in summary


def test_builder_deep_nonblocking():
    """Even if deep checks fail, build() should not raise."""
    builder = _make_builder()
    spec = ProjectSpec(
        project_name="emptyapp",
        summary="minimal",
        tech_stack=TechStackSpec(backend="FastAPI"),
    )
    aspec = ArtifactSpec(artifact_name="emptyapp", packaging="folder", include_tests=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path, manifest = builder.build(tmpdir, spec, aspec)
        ev = manifest.metadata.get("execution_validation")
        assert ev is not None
        assert "deep_checks_run" in ev.get("summary", {})


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

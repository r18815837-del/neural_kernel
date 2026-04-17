"""Integration tests: execution validation in ArtifactBuilder."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.builder import ArtifactBuilder
from artifacts.execution_validation import ExecutionValidationOrchestrator
from artifacts.consistency_validator import ConsistencyValidator
from artifacts.file_tree import ProjectFileTree
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
from runtime.specs.feature_spec import FeatureSpec


def _make_builder(
    with_execution: bool = True,
    with_consistency: bool = False,
) -> ArtifactBuilder:
    return ArtifactBuilder(
        structure_generator=StructureGenerator(),
        docs_generator=DocsGenerator(),
        env_generator=EnvGenerator(),
        tests_generator=TestsGenerator(),
        writer=ArtifactWriter(),
        validator=ArtifactValidator(),
        zip_packager=ZipPackager(),
        consistency_validator=ConsistencyValidator() if with_consistency else None,
        execution_validator=ExecutionValidationOrchestrator() if with_execution else None,
    )


def _make_spec(
    backend: str = "FastAPI",
    features: list[str] | None = None,
) -> ProjectSpec:
    return ProjectSpec(
        project_name="testapp",
        summary="Integration test project",
        tech_stack=TechStackSpec(backend=backend, database="PostgreSQL"),
        features=[FeatureSpec(name=f, description=f) for f in (features or [])],
    )


def _make_artifact_spec() -> ArtifactSpec:
    return ArtifactSpec(artifact_name="testapp", packaging="folder", include_tests=True)


# ------------------------------------------------------------------
# build_tree includes execution_validation in manifest
# ------------------------------------------------------------------

def test_build_tree_has_execution_validation():
    builder = _make_builder(with_execution=True)
    spec = _make_spec()
    aspec = _make_artifact_spec()
    tree, manifest = builder.build_tree(spec, aspec)
    assert "execution_validation" in manifest.metadata
    ev = manifest.metadata["execution_validation"]
    assert isinstance(ev, dict)
    assert "checks_run" in ev
    assert ev["checks_run"] >= 5


def test_build_tree_without_execution_validation():
    builder = _make_builder(with_execution=False)
    spec = _make_spec()
    aspec = _make_artifact_spec()
    tree, manifest = builder.build_tree(spec, aspec)
    assert "execution_validation" not in manifest.metadata


def test_execution_validation_report_shape():
    builder = _make_builder()
    spec = _make_spec()
    aspec = _make_artifact_spec()
    _, manifest = builder.build_tree(spec, aspec)
    ev = manifest.metadata["execution_validation"]
    assert "results" in ev
    assert isinstance(ev["results"], list)
    for r in ev["results"]:
        assert "name" in r
        assert "success" in r
        assert "message" in r


def test_execution_validation_checks_names():
    builder = _make_builder()
    spec = _make_spec()
    aspec = _make_artifact_spec()
    _, manifest = builder.build_tree(spec, aspec)
    ev = manifest.metadata["execution_validation"]
    names = {r["name"] for r in ev["results"]}
    expected = {"python_env", "backend_start", "frontend_presence", "tests_runner", "docker_files", "commands"}
    assert expected == names


def test_execution_and_consistency_both():
    builder = _make_builder(with_execution=True, with_consistency=True)
    spec = _make_spec()
    aspec = _make_artifact_spec()
    _, manifest = builder.build_tree(spec, aspec)
    assert "execution_validation" in manifest.metadata
    assert "consistency" in manifest.metadata


def test_build_returns_tuple():
    """build() now returns (path, manifest) tuple."""
    import tempfile
    builder = _make_builder()
    spec = _make_spec()
    aspec = ArtifactSpec(artifact_name="testapp", packaging="folder", include_tests=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = builder.build(tmpdir, spec, aspec)
        assert isinstance(result, tuple)
        path, manifest = result
        assert isinstance(path, str)
        assert isinstance(manifest, ArtifactManifest)
        assert "execution_validation" in manifest.metadata


def test_execution_validation_nonblocking():
    """Execution failures must not block generation."""
    builder = _make_builder()
    # No backend files → backend_start may warn but generation proceeds
    spec = ProjectSpec(
        project_name="emptyapp",
        summary="minimal",
        tech_stack=TechStackSpec(backend="FastAPI"),
    )
    aspec = _make_artifact_spec()
    tree, manifest = builder.build_tree(spec, aspec)
    ev = manifest.metadata["execution_validation"]
    # Some checks may fail, but build_tree didn't raise
    assert "checks_run" in ev
    assert ev["checks_run"] >= 5


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

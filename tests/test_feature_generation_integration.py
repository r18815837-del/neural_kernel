"""Integration tests — FeatureOrchestrator inside ArtifactBuilder."""

from __future__ import annotations

import pytest

from artifacts.builder import ArtifactBuilder
from artifacts.feature_orchestrator import FeatureOrchestrator
from artifacts.features.defaults import build_default_registry
from artifacts.features.registry import FeatureGeneratorRegistry
from artifacts.file_tree import ProjectFileTree
from artifacts.generators.docs_generator import DocsGenerator
from artifacts.generators.env_generator import EnvGenerator
from artifacts.generators.structure_generator import StructureGenerator
from artifacts.generators.tests_generator import TestsGenerator
from artifacts.validators import ArtifactValidator
from artifacts.writer import ArtifactWriter
from artifacts.zip_packager import ZipPackager
from runtime.specs.artifact_spec import ArtifactSpec
from runtime.specs.feature_spec import FeatureSpec
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


def _make_builder(with_features: bool = True) -> ArtifactBuilder:
    registry = build_default_registry() if with_features else None
    orchestrator = (
        FeatureOrchestrator(registry=registry)
        if registry is not None
        else None
    )

    return ArtifactBuilder(
        structure_generator=StructureGenerator(),
        docs_generator=DocsGenerator(),
        env_generator=EnvGenerator(),
        tests_generator=TestsGenerator(),
        writer=ArtifactWriter(),
        validator=ArtifactValidator(),
        zip_packager=ZipPackager(),
        feature_orchestrator=orchestrator,
    )


def _make_specs(
    features: list[str],
) -> tuple[ProjectSpec, ArtifactSpec]:
    project_spec = ProjectSpec(
        project_name="beauty_crm",
        summary="CRM for beauty salon",
        features=[FeatureSpec(name=f, description=f) for f in features],
        tech_stack=TechStackSpec(
            backend="FastAPI",
            frontend="React",
            database="PostgreSQL",
        ),
    )
    artifact_spec = ArtifactSpec(
        artifact_name="beauty_crm",
        include_tests=True,
        packaging="folder",
    )
    return project_spec, artifact_spec


class TestBuilderWithFeatureOrchestrator:
    """Feature orchestrator integrated into ArtifactBuilder.build_tree()."""

    def test_build_tree_includes_feature_files(self):
        builder = _make_builder(with_features=True)
        p_spec, a_spec = _make_specs(["client_database", "auth", "roles"])

        tree, manifest = builder.build_tree(p_spec, a_spec)

        paths = tree.list_file_paths()
        # Backend feature files exist
        assert any("backend/clients/routes.py" in p for p in paths)
        assert any("backend/auth/routes.py" in p for p in paths)
        assert any("backend/roles/routes.py" in p for p in paths)

    def test_build_tree_without_orchestrator_still_works(self):
        builder = _make_builder(with_features=False)
        p_spec, a_spec = _make_specs(["client_database"])

        tree, manifest = builder.build_tree(p_spec, a_spec)

        # Should still generate base structure
        paths = tree.list_file_paths()
        assert any("README.md" in p for p in paths)
        assert any("requirements.txt" in p for p in paths)

    def test_build_tree_schema_sql_enriched(self):
        builder = _make_builder(with_features=True)
        p_spec, a_spec = _make_specs(["client_database"])

        tree, manifest = builder.build_tree(p_spec, a_spec)

        schema = tree.get_file_content("beauty_crm/database/schema.sql")
        assert "CREATE TABLE IF NOT EXISTS clients" in schema

    def test_readme_enriched_with_feature_docs(self):
        builder = _make_builder(with_features=True)
        p_spec, a_spec = _make_specs(["client_database", "export_functionality"])

        tree, manifest = builder.build_tree(p_spec, a_spec)

        readme = tree.get_file_content("beauty_crm/README.md")
        assert "Client Management" in readme
        assert "Data Export" in readme

    def test_full_feature_set_no_errors(self):
        """Smoke test: all supported features at once."""
        builder = _make_builder(with_features=True)
        p_spec, a_spec = _make_specs([
            "auth",
            "roles",
            "admin_panel",
            "client_database",
            "export",
        ])

        tree, manifest = builder.build_tree(p_spec, a_spec)

        paths = tree.list_file_paths()
        # Spot-check one file from each feature
        assert any("auth/models.py" in p for p in paths)
        assert any("roles/models.py" in p for p in paths)
        assert any("admin/routes.py" in p for p in paths)
        assert any("clients/models.py" in p for p in paths)
        assert any("export/service.py" in p for p in paths)

    def test_database_migrations_created(self):
        builder = _make_builder(with_features=True)
        p_spec, a_spec = _make_specs(["client_database", "auth", "roles"])

        tree, manifest = builder.build_tree(p_spec, a_spec)

        paths = tree.list_file_paths()
        assert any("001_clients.sql" in p for p in paths)
        assert any("003_roles.sql" in p for p in paths)

    def test_manifest_includes_feature_files(self):
        builder = _make_builder(with_features=True)
        p_spec, a_spec = _make_specs(["client_database"])

        tree, manifest = builder.build_tree(p_spec, a_spec)

        assert any("clients/routes.py" in f for f in manifest.files)

    def test_generated_python_files_compile(self):
        """All generated .py files should be valid Python."""
        builder = _make_builder(with_features=True)
        p_spec, a_spec = _make_specs(["auth", "client_database", "roles", "admin_panel", "export"])

        tree, manifest = builder.build_tree(p_spec, a_spec)

        for f in tree.files:
            if f.path.endswith(".py") and f.content:
                try:
                    compile(f.content, f.path, "exec")
                except SyntaxError as e:
                    pytest.fail(f"SyntaxError in {f.path}: {e}")


class TestRegistryLookup:
    """FeatureGeneratorRegistry tests."""

    def test_supported_features(self):
        registry = build_default_registry()
        supported = registry.supported_features()

        assert "auth" in supported
        assert "roles" in supported
        assert "client_database" in supported
        assert "admin_panel" in supported
        assert "export" in supported
        assert "appointments" in supported

    def test_get_generators_returns_all_layers(self):
        registry = build_default_registry()
        gens = registry.get_generators("client_database")

        layers = {g.layer for g in gens}
        assert "backend" in layers
        assert "database" in layers
        assert "docs" in layers

    def test_get_generator_specific_layer(self):
        registry = build_default_registry()
        gen = registry.get_generator("auth", "backend")
        assert gen is not None
        assert gen.feature_name == "auth"

    def test_get_generator_missing_returns_none(self):
        registry = build_default_registry()
        gen = registry.get_generator("nonexistent", "backend")
        assert gen is None

    def test_has_feature(self):
        registry = build_default_registry()
        assert registry.has_feature("auth")
        assert not registry.has_feature("nonexistent")

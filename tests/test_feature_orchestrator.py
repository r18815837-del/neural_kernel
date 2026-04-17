"""Tests for FeatureOrchestrator — apply generators to tree."""

from __future__ import annotations

import pytest

from artifacts.feature_orchestrator import FeatureOrchestrator
from artifacts.features.defaults import build_default_registry
from artifacts.features.planner import FeatureGenerationPlanner
from artifacts.features.registry import FeatureGeneratorRegistry
from artifacts.file_tree import ProjectFileTree
from runtime.specs.feature_spec import FeatureSpec
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


def _make_spec(features: list[str]) -> ProjectSpec:
    return ProjectSpec(
        project_name="testproj",
        summary="Test project",
        features=[FeatureSpec(name=f, description=f"{f} feature") for f in features],
        tech_stack=TechStackSpec(backend="FastAPI", database="PostgreSQL"),
    )


def _make_tree(project_name: str = "testproj") -> ProjectFileTree:
    tree = ProjectFileTree()
    tree.add_directory(project_name)
    tree.add_file(f"{project_name}/README.md", content=f"# {project_name}\n")
    tree.add_directory(f"{project_name}/backend")
    tree.add_directory(f"{project_name}/database")
    tree.add_file(f"{project_name}/database/schema.sql", content="-- base schema\n")
    return tree


class TestApplyFeatures:
    """FeatureOrchestrator.apply() integration."""

    def test_no_features_returns_unchanged_tree(self):
        registry = build_default_registry()
        orch = FeatureOrchestrator(registry=registry)
        spec = ProjectSpec(project_name="empty", summary="No features")
        tree = _make_tree("empty")

        result_tree, results = orch.apply(tree, spec)
        assert results == []
        assert result_tree is tree

    def test_client_database_creates_files(self):
        registry = build_default_registry()
        orch = FeatureOrchestrator(registry=registry)
        spec = _make_spec(["client_database"])
        tree = _make_tree()

        tree, results = orch.apply(tree, spec)

        paths = tree.list_file_paths()
        assert any("backend/clients/models.py" in p for p in paths)
        assert any("backend/clients/routes.py" in p for p in paths)
        assert any("backend/clients/service.py" in p for p in paths)
        # Database migration
        assert any("migrations/001_clients.sql" in p for p in paths)
        # Docs appended to README
        readme = tree.get_file_content("testproj/README.md")
        assert "Client Management" in readme

    def test_roles_creates_files(self):
        registry = build_default_registry()
        orch = FeatureOrchestrator(registry=registry)
        spec = _make_spec(["auth", "roles"])
        tree = _make_tree()

        tree, results = orch.apply(tree, spec)

        paths = tree.list_file_paths()
        assert any("backend/roles/models.py" in p for p in paths)
        assert any("backend/roles/routes.py" in p for p in paths)
        assert any("migrations/003_roles.sql" in p for p in paths)

    def test_appends_to_schema_sql(self):
        registry = build_default_registry()
        orch = FeatureOrchestrator(registry=registry)
        spec = _make_spec(["client_database"])
        tree = _make_tree()

        tree, _ = orch.apply(tree, spec)

        schema = tree.get_file_content("testproj/database/schema.sql")
        assert "-- base schema" in schema
        assert "CREATE TABLE IF NOT EXISTS clients" in schema

    def test_multiple_features_all_applied(self):
        registry = build_default_registry()
        orch = FeatureOrchestrator(registry=registry)
        spec = _make_spec(["auth", "roles", "client_database", "admin_panel", "export"])
        tree = _make_tree()

        tree, results = orch.apply(tree, spec)

        # All results should be successful
        assert all(r.success for r in results)

        paths = tree.list_file_paths()
        # Auth
        assert any("backend/auth/routes.py" in p for p in paths)
        # Roles
        assert any("backend/roles/routes.py" in p for p in paths)
        # Client DB
        assert any("backend/clients/routes.py" in p for p in paths)
        # Admin
        assert any("backend/admin/routes.py" in p for p in paths)
        # Export
        assert any("backend/export/routes.py" in p for p in paths)

    def test_unknown_features_gracefully_skipped(self):
        registry = build_default_registry()
        orch = FeatureOrchestrator(registry=registry)
        spec = _make_spec(["some_unknown_feature"])
        tree = _make_tree()

        tree, results = orch.apply(tree, spec)
        assert results == []

    def test_export_functionality_normalised(self):
        """export_functionality should be normalised to export."""
        registry = build_default_registry()
        orch = FeatureOrchestrator(registry=registry)
        spec = _make_spec(["client_database", "export_functionality"])
        tree = _make_tree()

        tree, results = orch.apply(tree, spec)

        paths = tree.list_file_paths()
        assert any("backend/export/routes.py" in p for p in paths)

    def test_completed_features_tracked(self):
        """Context's completed_features list grows as features run."""
        registry = build_default_registry()
        orch = FeatureOrchestrator(registry=registry)
        spec = _make_spec(["auth", "client_database"])
        tree = _make_tree()

        tree, results = orch.apply(tree, spec)
        # All generators succeeded
        assert len(results) > 0
        assert all(r.success for r in results)

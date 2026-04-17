"""Tests for backend feature generators."""

from __future__ import annotations

import pytest

from artifacts.features.backend.auth import AuthBackendGenerator
from artifacts.features.backend.client_database import ClientDatabaseBackendGenerator
from artifacts.features.backend.roles import RolesBackendGenerator
from artifacts.features.backend.admin_panel import AdminPanelBackendGenerator
from artifacts.features.backend.export import ExportBackendGenerator
from artifacts.features.context import FeatureGenerationContext
from artifacts.file_tree import ProjectFileTree
from runtime.specs.feature_spec import FeatureSpec
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


def _ctx(features: list[str] | None = None) -> FeatureGenerationContext:
    spec = ProjectSpec(
        project_name="testproj",
        summary="Test",
        features=[FeatureSpec(name=f, description=f) for f in (features or [])],
        tech_stack=TechStackSpec(backend="FastAPI", database="PostgreSQL"),
    )
    tree = ProjectFileTree()
    tree.add_directory("testproj")
    return FeatureGenerationContext.from_spec(tree, spec)


class TestAuthBackendGenerator:
    def test_creates_expected_files(self):
        gen = AuthBackendGenerator()
        result = gen.generate(_ctx(["auth"]))

        assert result.success
        assert result.feature_name == "auth"
        assert any("auth/models.py" in p for p in result.created_files)
        assert any("auth/routes.py" in p for p in result.created_files)
        assert any("auth/dependencies.py" in p for p in result.created_files)

    def test_appends_schema_sql(self):
        gen = AuthBackendGenerator()
        result = gen.generate(_ctx(["auth"]))

        assert "testproj/database/schema.sql" in result.appended_fragments
        assert "CREATE TABLE" in result.appended_fragments["testproj/database/schema.sql"]

    def test_generated_python_compiles(self):
        gen = AuthBackendGenerator()
        result = gen.generate(_ctx(["auth"]))

        for path, content in result.created_files.items():
            if path.endswith(".py") and content:
                compile(content, path, "exec")


class TestClientDatabaseBackendGenerator:
    def test_creates_expected_files(self):
        gen = ClientDatabaseBackendGenerator()
        result = gen.generate(_ctx(["client_database"]))

        assert result.success
        assert any("clients/models.py" in p for p in result.created_files)
        assert any("clients/routes.py" in p for p in result.created_files)
        assert any("clients/service.py" in p for p in result.created_files)

    def test_appends_schema_sql(self):
        gen = ClientDatabaseBackendGenerator()
        result = gen.generate(_ctx(["client_database"]))

        frag = result.appended_fragments.get("testproj/database/schema.sql", "")
        assert "CREATE TABLE IF NOT EXISTS clients" in frag
        assert "CREATE INDEX" in frag

    def test_generated_python_compiles(self):
        gen = ClientDatabaseBackendGenerator()
        result = gen.generate(_ctx(["client_database"]))

        for path, content in result.created_files.items():
            if path.endswith(".py") and content:
                compile(content, path, "exec")


class TestRolesBackendGenerator:
    def test_creates_expected_files(self):
        gen = RolesBackendGenerator()
        result = gen.generate(_ctx(["roles"]))

        assert result.success
        assert any("roles/models.py" in p for p in result.created_files)
        assert any("roles/routes.py" in p for p in result.created_files)
        assert any("roles/dependencies.py" in p for p in result.created_files)

    def test_appends_rbac_tables(self):
        gen = RolesBackendGenerator()
        result = gen.generate(_ctx(["roles"]))

        frag = result.appended_fragments.get("testproj/database/schema.sql", "")
        assert "roles" in frag
        assert "role_permissions" in frag
        assert "user_roles" in frag


class TestAdminPanelBackendGenerator:
    def test_creates_expected_files(self):
        gen = AdminPanelBackendGenerator()
        result = gen.generate(_ctx(["admin_panel"]))

        assert result.success
        assert any("admin/routes.py" in p for p in result.created_files)
        assert any("admin/dependencies.py" in p for p in result.created_files)

    def test_no_schema_append(self):
        gen = AdminPanelBackendGenerator()
        result = gen.generate(_ctx(["admin_panel"]))
        assert not result.appended_fragments


class TestExportBackendGenerator:
    def test_creates_expected_files(self):
        gen = ExportBackendGenerator()
        result = gen.generate(_ctx(["export"]))

        assert result.success
        assert any("export/routes.py" in p for p in result.created_files)
        assert any("export/service.py" in p for p in result.created_files)

    def test_generated_python_compiles(self):
        gen = ExportBackendGenerator()
        result = gen.generate(_ctx(["export"]))

        for path, content in result.created_files.items():
            if path.endswith(".py") and content:
                compile(content, path, "exec")

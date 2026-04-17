"""Tests for database feature generators."""

from __future__ import annotations

import pytest

from artifacts.features.database.client_database import ClientDatabaseDatabaseGenerator
from artifacts.features.database.roles import RolesDatabaseGenerator
from artifacts.features.database.appointments import AppointmentsDatabaseGenerator
from artifacts.features.context import FeatureGenerationContext
from artifacts.file_tree import ProjectFileTree
from runtime.specs.feature_spec import FeatureSpec
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


def _ctx() -> FeatureGenerationContext:
    spec = ProjectSpec(
        project_name="testproj",
        summary="Test",
        features=[],
        tech_stack=TechStackSpec(backend="FastAPI", database="PostgreSQL"),
    )
    tree = ProjectFileTree()
    tree.add_directory("testproj")
    return FeatureGenerationContext.from_spec(tree, spec)


class TestClientDatabaseDatabaseGenerator:
    def test_creates_migrations(self):
        gen = ClientDatabaseDatabaseGenerator()
        result = gen.generate(_ctx())

        assert result.success
        assert result.feature_name == "client_database"
        assert any("001_clients.sql" in p for p in result.created_files)
        assert any("002_client_history.sql" in p for p in result.created_files)

    def test_migrations_have_create_table(self):
        gen = ClientDatabaseDatabaseGenerator()
        result = gen.generate(_ctx())

        for path, content in result.created_files.items():
            assert "CREATE TABLE" in content
            assert "-- Migration:" in content

    def test_clients_migration_has_indexes(self):
        gen = ClientDatabaseDatabaseGenerator()
        result = gen.generate(_ctx())

        clients_sql = next(
            c for p, c in result.created_files.items() if "001_clients" in p
        )
        assert "CREATE INDEX" in clients_sql
        assert "idx_clients_email" in clients_sql


class TestRolesDatabaseGenerator:
    def test_creates_migration(self):
        gen = RolesDatabaseGenerator()
        result = gen.generate(_ctx())

        assert result.success
        assert any("003_roles.sql" in p for p in result.created_files)

    def test_migration_has_all_tables(self):
        gen = RolesDatabaseGenerator()
        result = gen.generate(_ctx())

        sql = next(c for p, c in result.created_files.items() if "003_roles" in p)
        assert "CREATE TABLE IF NOT EXISTS roles" in sql
        assert "role_permissions" in sql
        assert "user_roles" in sql

    def test_seeds_default_roles(self):
        gen = RolesDatabaseGenerator()
        result = gen.generate(_ctx())

        sql = next(c for p, c in result.created_files.items() if "003_roles" in p)
        assert "INSERT INTO roles" in sql
        assert "admin" in sql
        assert "manager" in sql
        assert "staff" in sql


class TestAppointmentsDatabaseGenerator:
    def test_creates_migration(self):
        gen = AppointmentsDatabaseGenerator()
        result = gen.generate(_ctx())

        assert result.success
        assert any("004_appointments.sql" in p for p in result.created_files)

    def test_migration_has_services_and_appointments(self):
        gen = AppointmentsDatabaseGenerator()
        result = gen.generate(_ctx())

        sql = next(c for p, c in result.created_files.items() if "004_appointments" in p)
        assert "CREATE TABLE IF NOT EXISTS services" in sql
        assert "CREATE TABLE IF NOT EXISTS appointments" in sql
        assert "client_id" in sql

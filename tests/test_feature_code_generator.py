from __future__ import annotations

import pytest

from artifacts.file_tree import ProjectFileTree
from artifacts.generators.feature_code_generator import FeatureCodeGenerator
from runtime.specs.feature_spec import FeatureSpec
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


@pytest.fixture
def generator():
    """Create a FeatureCodeGenerator instance."""
    return FeatureCodeGenerator()


@pytest.fixture
def base_project_spec():
    """Create a basic ProjectSpec."""
    return ProjectSpec(
        project_name="test_project",
        summary="A test project",
        project_type="application",
    )


@pytest.fixture
def empty_tree():
    """Create an empty ProjectFileTree."""
    return ProjectFileTree()


class TestFeatureCodeGeneratorBasics:
    """Test basic functionality of FeatureCodeGenerator."""

    def test_generator_initialization(self, generator):
        """Test that generator is initialized properly."""
        assert generator is not None
        assert hasattr(generator, "generate")

    def test_generate_with_empty_features_returns_same_tree(
        self, generator, base_project_spec, empty_tree
    ):
        """Test that tree is unchanged when project has no features."""
        result = generator.generate(empty_tree, base_project_spec)
        assert result == empty_tree
        assert len(result.files) == 0

    def test_generate_returns_tree(self, generator, empty_tree):
        """Test that generate returns a ProjectFileTree."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="auth", description="Auth feature")],
        )
        result = generator.generate(empty_tree, spec)
        assert isinstance(result, ProjectFileTree)


class TestAuthFeatureGeneration:
    """Test auth feature generation."""

    def test_generate_auth_feature_creates_files(self, generator, empty_tree):
        """Test that auth feature generates required files."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="auth", description="Auth feature")],
        )
        result = generator.generate(empty_tree, spec)

        file_paths = [f.path for f in result.files]
        assert any("backend/auth/__init__.py" in path for path in file_paths)
        assert any("backend/auth/models.py" in path for path in file_paths)
        assert any("backend/auth/routes.py" in path for path in file_paths)
        assert any("backend/auth/dependencies.py" in path for path in file_paths)
        assert any("001_auth.sql" in path for path in file_paths)

    def test_auth_models_has_content(self, generator, empty_tree):
        """Test that auth models file has non-empty content."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="auth", description="Auth feature")],
        )
        result = generator.generate(empty_tree, spec)

        auth_models = next(
            (f for f in result.files if "auth/models.py" in f.path), None
        )
        assert auth_models is not None
        assert len(auth_models.content) > 0
        assert "UserCreate" in auth_models.content
        assert "TokenResponse" in auth_models.content

    def test_auth_routes_has_content(self, generator, empty_tree):
        """Test that auth routes file has non-empty content."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="auth", description="Auth feature")],
        )
        result = generator.generate(empty_tree, spec)

        auth_routes = next(
            (f for f in result.files if "auth/routes.py" in f.path), None
        )
        assert auth_routes is not None
        assert len(auth_routes.content) > 0
        assert "/register" in auth_routes.content
        assert "/login" in auth_routes.content


class TestAdminPanelFeatureGeneration:
    """Test admin panel feature generation."""

    def test_generate_admin_panel_feature(self, generator, empty_tree):
        """Test that admin_panel feature generates required files."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="admin_panel", description="Admin feature")],
        )
        result = generator.generate(empty_tree, spec)

        file_paths = [f.path for f in result.files]
        assert any("backend/admin/__init__.py" in path for path in file_paths)
        assert any("backend/admin/routes.py" in path for path in file_paths)
        assert any("backend/admin/dependencies.py" in path for path in file_paths)

    def test_admin_routes_has_content(self, generator, empty_tree):
        """Test that admin routes has non-empty content."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="admin_panel", description="Admin feature")],
        )
        result = generator.generate(empty_tree, spec)

        admin_routes = next(
            (f for f in result.files if "admin/routes.py" in f.path), None
        )
        assert admin_routes is not None
        assert len(admin_routes.content) > 0
        assert "list_users" in admin_routes.content


class TestClientDatabaseFeatureGeneration:
    """Test client database feature generation."""

    def test_generate_client_database_feature(self, generator, empty_tree):
        """Test that client_database feature generates required files."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[
                FeatureSpec(
                    name="client_database", description="Client database feature"
                )
            ],
        )
        result = generator.generate(empty_tree, spec)

        file_paths = [f.path for f in result.files]
        assert any("backend/clients/__init__.py" in path for path in file_paths)
        assert any("backend/clients/models.py" in path for path in file_paths)
        assert any("backend/clients/routes.py" in path for path in file_paths)
        assert any("002_clients.sql" in path for path in file_paths)

    def test_client_models_has_content(self, generator, empty_tree):
        """Test that client models has non-empty content."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[
                FeatureSpec(
                    name="client_database", description="Client database feature"
                )
            ],
        )
        result = generator.generate(empty_tree, spec)

        client_models = next(
            (f for f in result.files if "clients/models.py" in f.path), None
        )
        assert client_models is not None
        assert len(client_models.content) > 0
        assert "ClientCreate" in client_models.content


class TestExportFeatureGeneration:
    """Test export feature generation."""

    def test_generate_export_feature(self, generator, empty_tree):
        """Test that export feature generates required files."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="export", description="Export feature")],
        )
        result = generator.generate(empty_tree, spec)

        file_paths = [f.path for f in result.files]
        assert any("backend/export/__init__.py" in path for path in file_paths)
        assert any("backend/export/routes.py" in path for path in file_paths)
        assert any("backend/export/service.py" in path for path in file_paths)

    def test_export_service_has_content(self, generator, empty_tree):
        """Test that export service has non-empty content."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="export", description="Export feature")],
        )
        result = generator.generate(empty_tree, spec)

        export_service = next(
            (f for f in result.files if "export/service.py" in f.path), None
        )
        assert export_service is not None
        assert len(export_service.content) > 0
        assert "ExportService" in export_service.content


class TestRolesFeatureGeneration:
    """Test roles feature generation."""

    def test_generate_roles_feature(self, generator, empty_tree):
        """Test that roles feature generates required files."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="roles", description="Roles feature")],
        )
        result = generator.generate(empty_tree, spec)

        file_paths = [f.path for f in result.files]
        assert any("backend/roles/__init__.py" in path for path in file_paths)
        assert any("backend/roles/models.py" in path for path in file_paths)
        assert any("backend/roles/routes.py" in path for path in file_paths)
        assert any("003_roles.sql" in path for path in file_paths)

    def test_roles_migration_has_content(self, generator, empty_tree):
        """Test that roles migration has valid SQL content."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="roles", description="Roles feature")],
        )
        result = generator.generate(empty_tree, spec)

        roles_migration = next(
            (f for f in result.files if "003_roles.sql" in f.path), None
        )
        assert roles_migration is not None
        assert len(roles_migration.content) > 0
        assert "CREATE TABLE" in roles_migration.content


class TestNotificationsFeatureGeneration:
    """Test notifications feature generation."""

    def test_generate_notifications_feature(self, generator, empty_tree):
        """Test that notifications feature generates required files."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="notifications", description="Notifications feature")],
        )
        result = generator.generate(empty_tree, spec)

        file_paths = [f.path for f in result.files]
        assert any("backend/notifications/__init__.py" in path for path in file_paths)
        assert any("backend/notifications/routes.py" in path for path in file_paths)
        assert any("backend/notifications/service.py" in path for path in file_paths)

    def test_notifications_service_has_content(self, generator, empty_tree):
        """Test that notifications service has non-empty content."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="notifications", description="Notifications feature")],
        )
        result = generator.generate(empty_tree, spec)

        notif_service = next(
            (f for f in result.files if "notifications/service.py" in f.path), None
        )
        assert notif_service is not None
        assert len(notif_service.content) > 0
        assert "NotificationService" in notif_service.content


class TestAPIFeatureGeneration:
    """Test API feature generation."""

    def test_generate_api_feature(self, generator, empty_tree):
        """Test that api feature generates required files."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="api", description="API feature")],
        )
        result = generator.generate(empty_tree, spec)

        file_paths = [f.path for f in result.files]
        assert any("backend/api/__init__.py" in path for path in file_paths)
        assert any("backend/api/routes.py" in path for path in file_paths)
        assert any("backend/api/schemas.py" in path for path in file_paths)

    def test_api_routes_has_content(self, generator, empty_tree):
        """Test that API routes has non-empty content."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="api", description="API feature")],
        )
        result = generator.generate(empty_tree, spec)

        api_routes = next(
            (f for f in result.files if "api/routes.py" in f.path), None
        )
        assert api_routes is not None
        assert len(api_routes.content) > 0
        assert "v1" in api_routes.content


class TestMultipleFeaturesGeneration:
    """Test generation with multiple features."""

    def test_generate_multiple_features(self, generator, empty_tree):
        """Test that multiple features generate all their files."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[
                FeatureSpec(name="auth", description="Auth"),
                FeatureSpec(name="admin_panel", description="Admin"),
                FeatureSpec(name="export", description="Export"),
            ],
        )
        result = generator.generate(empty_tree, spec)

        file_paths = [f.path for f in result.files]
        # Auth files
        assert any("backend/auth/models.py" in path for path in file_paths)
        # Admin files
        assert any("backend/admin/routes.py" in path for path in file_paths)
        # Export files
        assert any("backend/export/service.py" in path for path in file_paths)

    def test_all_features_together(self, generator, empty_tree):
        """Test generation of all supported features."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[
                FeatureSpec(name="auth", description="Auth"),
                FeatureSpec(name="admin_panel", description="Admin"),
                FeatureSpec(name="client_database", description="Clients"),
                FeatureSpec(name="export", description="Export"),
                FeatureSpec(name="roles", description="Roles"),
                FeatureSpec(name="notifications", description="Notifications"),
                FeatureSpec(name="api", description="API"),
            ],
        )
        result = generator.generate(empty_tree, spec)

        assert len(result.files) > 0
        # Verify we have files from multiple features
        file_paths = [f.path for f in result.files]
        assert any("auth" in path for path in file_paths)
        assert any("admin" in path for path in file_paths)
        assert any("clients" in path for path in file_paths)
        assert any("export" in path for path in file_paths)


class TestUnknownFeatureHandling:
    """Test handling of unknown features."""

    def test_unknown_feature_is_ignored(self, generator, empty_tree):
        """Test that unknown features don't cause errors but are ignored."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[
                FeatureSpec(name="unknown_feature", description="Unknown"),
                FeatureSpec(name="auth", description="Auth"),
            ],
        )
        result = generator.generate(empty_tree, spec)

        # Should still generate auth files
        file_paths = [f.path for f in result.files]
        assert any("backend/auth/models.py" in path for path in file_paths)

    def test_case_insensitive_feature_names(self, generator, empty_tree):
        """Test that feature names are case-insensitive."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[
                FeatureSpec(name="AUTH", description="Auth"),
                FeatureSpec(name="Admin_Panel", description="Admin"),
            ],
        )
        result = generator.generate(empty_tree, spec)

        file_paths = [f.path for f in result.files]
        assert any("backend/auth/models.py" in path for path in file_paths)
        assert any("backend/admin/routes.py" in path for path in file_paths)


class TestGeneratedCodeQuality:
    """Test quality of generated code."""

    def test_generated_code_is_valid_python(self, generator, empty_tree):
        """Test that generated code can be compiled as valid Python."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="auth", description="Auth")],
        )
        result = generator.generate(empty_tree, spec)

        for file in result.files:
            if file.path.endswith(".py"):
                try:
                    compile(file.content, file.path, "exec")
                except SyntaxError as e:
                    pytest.fail(f"Invalid Python in {file.path}: {e}")

    def test_generated_migrations_contain_sql(self, generator, empty_tree):
        """Test that migration files contain valid SQL keywords."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="auth", description="Auth")],
        )
        result = generator.generate(empty_tree, spec)

        migration_files = [f for f in result.files if f.path.endswith(".sql")]
        assert len(migration_files) > 0

        for file in migration_files:
            assert any(
                keyword in file.content.upper()
                for keyword in ["CREATE", "TABLE", "INSERT"]
            )

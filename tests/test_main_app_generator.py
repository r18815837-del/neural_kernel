from __future__ import annotations

import pytest

from artifacts.generators.main_app_generator import MainAppGenerator
from runtime.specs.feature_spec import FeatureSpec
from runtime.specs.project_spec import ProjectSpec


@pytest.fixture
def generator():
    """Create a MainAppGenerator instance."""
    return MainAppGenerator()


@pytest.fixture
def base_project_spec():
    """Create a basic ProjectSpec."""
    return ProjectSpec(
        project_name="test_app",
        summary="Test API application",
    )


class TestMainAppGeneratorBasics:
    """Test basic functionality of MainAppGenerator."""

    def test_generator_initialization(self, generator):
        """Test that generator is initialized properly."""
        assert generator is not None
        assert hasattr(generator, "generate")

    def test_generate_returns_string(self, generator, base_project_spec):
        """Test that generate returns a string."""
        result = generator.generate(base_project_spec)
        assert isinstance(result, str)

    def test_generate_includes_fastapi_import(self, generator, base_project_spec):
        """Test that generated code imports FastAPI."""
        result = generator.generate(base_project_spec)
        assert "from fastapi import FastAPI" in result
        assert "from __future__ import annotations" in result


class TestMainAppGeneratorWithoutFeatures:
    """Test main app generation without features."""

    def test_generate_without_features(self, generator, base_project_spec):
        """Test that main.py is generated even without features."""
        result = generator.generate(base_project_spec)
        assert len(result) > 0
        assert "app = FastAPI" in result
        assert "base_router" in result

    def test_zero_features_includes_base_router_only(self, generator, base_project_spec):
        """Test that zero features only includes base router."""
        result = generator.generate(base_project_spec)
        assert "app.include_router(base_router)" in result
        assert "app.include_router(auth_router)" not in result
        assert "app.include_router(admin_router)" not in result


class TestMainAppGeneratorWithFeatures:
    """Test main app generation with various features."""

    def test_generate_with_auth_feature(self, generator, base_project_spec):
        """Test that auth feature includes auth router."""
        base_project_spec.features = [
            FeatureSpec(name="auth", description="Authentication")
        ]
        result = generator.generate(base_project_spec)
        assert "from backend.auth.routes import router as auth_router" in result
        assert "app.include_router(auth_router)" in result

    def test_generate_with_admin_feature(self, generator, base_project_spec):
        """Test that admin_panel feature includes admin router."""
        base_project_spec.features = [
            FeatureSpec(name="admin_panel", description="Admin dashboard")
        ]
        result = generator.generate(base_project_spec)
        assert "from backend.admin.routes import router as admin_router" in result
        assert "app.include_router(admin_router)" in result

    def test_generate_with_clients_feature(self, generator, base_project_spec):
        """Test that client_database feature includes clients router."""
        base_project_spec.features = [
            FeatureSpec(name="client_database", description="Client database")
        ]
        result = generator.generate(base_project_spec)
        assert "from backend.clients.routes import router as clients_router" in result
        assert "app.include_router(clients_router)" in result

    def test_generate_with_export_feature(self, generator, base_project_spec):
        """Test that export feature includes export router."""
        base_project_spec.features = [
            FeatureSpec(name="export", description="Data export")
        ]
        result = generator.generate(base_project_spec)
        assert "from backend.export.routes import router as export_router" in result
        assert "app.include_router(export_router)" in result

    def test_generate_with_roles_feature(self, generator, base_project_spec):
        """Test that roles feature includes roles router."""
        base_project_spec.features = [
            FeatureSpec(name="roles", description="Role management")
        ]
        result = generator.generate(base_project_spec)
        assert "from backend.roles.routes import router as roles_router" in result
        assert "app.include_router(roles_router)" in result

    def test_generate_with_notifications_feature(self, generator, base_project_spec):
        """Test that notifications feature includes notifications router."""
        base_project_spec.features = [
            FeatureSpec(name="notifications", description="Notifications")
        ]
        result = generator.generate(base_project_spec)
        assert (
            "from backend.notifications.routes import router as notifications_router"
            in result
        )
        assert "app.include_router(notifications_router)" in result

    def test_generate_with_api_feature(self, generator, base_project_spec):
        """Test that api feature includes api router."""
        base_project_spec.features = [
            FeatureSpec(name="api", description="API endpoints")
        ]
        result = generator.generate(base_project_spec)
        assert "from backend.api.routes import router as api_router" in result
        assert "app.include_router(api_router)" in result


class TestMainAppGeneratorMultipleFeatures:
    """Test main app generation with multiple features."""

    def test_generate_with_multiple_features(self, generator, base_project_spec):
        """Test that multiple features generate all routers."""
        base_project_spec.features = [
            FeatureSpec(name="auth", description="Auth"),
            FeatureSpec(name="admin_panel", description="Admin"),
            FeatureSpec(name="export", description="Export"),
        ]
        result = generator.generate(base_project_spec)

        assert "from backend.auth.routes import router as auth_router" in result
        assert "from backend.admin.routes import router as admin_router" in result
        assert "from backend.export.routes import router as export_router" in result

        assert "app.include_router(auth_router)" in result
        assert "app.include_router(admin_router)" in result
        assert "app.include_router(export_router)" in result

    def test_generate_with_all_features(self, generator, base_project_spec):
        """Test that all features can be included together."""
        base_project_spec.features = [
            FeatureSpec(name="auth", description="Auth"),
            FeatureSpec(name="admin_panel", description="Admin"),
            FeatureSpec(name="client_database", description="Clients"),
            FeatureSpec(name="export", description="Export"),
            FeatureSpec(name="roles", description="Roles"),
            FeatureSpec(name="notifications", description="Notifications"),
            FeatureSpec(name="api", description="API"),
        ]
        result = generator.generate(base_project_spec)

        # Check all imports
        assert "from backend.auth.routes import router as auth_router" in result
        assert "from backend.admin.routes import router as admin_router" in result
        assert "from backend.clients.routes import router as clients_router" in result
        assert "from backend.export.routes import router as export_router" in result
        assert "from backend.roles.routes import router as roles_router" in result
        assert (
            "from backend.notifications.routes import router as notifications_router"
            in result
        )
        assert "from backend.api.routes import router as api_router" in result

        # Check all includes
        assert "app.include_router(auth_router)" in result
        assert "app.include_router(admin_router)" in result
        assert "app.include_router(clients_router)" in result
        assert "app.include_router(export_router)" in result
        assert "app.include_router(roles_router)" in result
        assert "app.include_router(notifications_router)" in result
        assert "app.include_router(api_router)" in result


class TestMainAppGeneratorCORSMiddleware:
    """Test CORS middleware configuration."""

    def test_generated_code_includes_cors_middleware(self, generator, base_project_spec):
        """Test that CORS middleware is configured."""
        result = generator.generate(base_project_spec)
        assert "CORSMiddleware" in result
        assert "from fastapi.middleware.cors import CORSMiddleware" in result

    def test_cors_middleware_configuration(self, generator, base_project_spec):
        """Test that CORS is properly configured."""
        result = generator.generate(base_project_spec)
        assert "app.add_middleware" in result
        assert "allow_origins=[\"*\"]" in result
        assert "allow_credentials=True" in result
        assert "allow_methods=[\"*\"]" in result
        assert "allow_headers=[\"*\"]" in result

    def test_cors_configuration_present_in_all_variants(
        self, generator, base_project_spec
    ):
        """Test that CORS is present regardless of features."""
        base_project_spec.features = [
            FeatureSpec(name="auth", description="Auth")
        ]
        result = generator.generate(base_project_spec)
        assert "CORSMiddleware" in result


class TestMainAppGeneratorProjectMetadata:
    """Test that project metadata is included."""

    def test_app_title_from_project_name(self, generator, base_project_spec):
        """Test that app title is set to project name."""
        result = generator.generate(base_project_spec)
        assert 'title="test_app"' in result

    def test_app_description_from_summary(self, generator, base_project_spec):
        """Test that app description is set to project summary."""
        result = generator.generate(base_project_spec)
        assert "description=" in result

    def test_app_version(self, generator, base_project_spec):
        """Test that app version is set."""
        result = generator.generate(base_project_spec)
        assert 'version="0.1.0"' in result

    def test_different_project_names(self, generator):
        """Test that project names are correctly used."""
        spec1 = ProjectSpec(project_name="api_v1", summary="API V1")
        spec2 = ProjectSpec(project_name="mobile_backend", summary="Mobile Backend")

        result1 = generator.generate(spec1)
        result2 = generator.generate(spec2)

        assert 'title="api_v1"' in result1
        assert 'title="mobile_backend"' in result2


class TestMainAppGeneratorCodeStructure:
    """Test the structure of generated code."""

    def test_generated_code_is_valid_python(self, generator, base_project_spec):
        """Test that generated code is valid Python."""
        base_project_spec.features = [
            FeatureSpec(name="auth", description="Auth"),
            FeatureSpec(name="admin_panel", description="Admin"),
        ]
        result = generator.generate(base_project_spec)

        try:
            compile(result, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax error: {e}")

    def test_generated_code_has_proper_structure(self, generator, base_project_spec):
        """Test that generated code has proper structure."""
        result = generator.generate(base_project_spec)

        lines = result.strip().split("\n")
        # Check that imports come first
        assert "from __future__ import annotations" in lines[0]

        # Check that app initialization comes after imports
        app_init_index = next(
            (i for i, line in enumerate(lines) if "app = FastAPI" in line), None
        )
        assert app_init_index is not None

        # Check that router includes come after app initialization
        router_include_index = next(
            (i for i, line in enumerate(lines) if "include_router" in line), None
        )
        assert router_include_index is not None
        assert router_include_index > app_init_index

    def test_base_router_always_included_first(self, generator, base_project_spec):
        """Test that base router is included first."""
        base_project_spec.features = [
            FeatureSpec(name="auth", description="Auth")
        ]
        result = generator.generate(base_project_spec)

        # Find positions of includes
        base_router_pos = result.find("app.include_router(base_router)")
        auth_router_pos = result.find("app.include_router(auth_router)")

        assert base_router_pos != -1
        assert auth_router_pos != -1
        assert base_router_pos < auth_router_pos


class TestMainAppGeneratorWithExplicitFeatures:
    """Test generation with explicit feature lists."""

    def test_generate_with_explicit_feature_list(self, generator, base_project_spec):
        """Test that explicit feature list parameter works."""
        features = ["auth", "admin_panel"]
        result = generator.generate(base_project_spec, features)

        assert "from backend.auth.routes import router as auth_router" in result
        assert "from backend.admin.routes import router as admin_router" in result

    def test_explicit_features_override_spec_features(self, generator, base_project_spec):
        """Test that explicit features override project spec features."""
        base_project_spec.features = [
            FeatureSpec(name="export", description="Export")
        ]
        explicit_features = ["auth", "admin_panel"]
        result = generator.generate(base_project_spec, explicit_features)

        # Should include explicit features
        assert "from backend.auth.routes import router as auth_router" in result
        assert "from backend.admin.routes import router as admin_router" in result
        # Should NOT include export since it was overridden
        assert "export_router" not in result

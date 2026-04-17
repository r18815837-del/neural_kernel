from __future__ import annotations

import pytest

from artifacts.generators.templates.registry import TemplateRegistry
from runtime.specs.feature_spec import FeatureSpec
from runtime.specs.project_spec import ProjectSpec


@pytest.fixture
def template_registry():
    """Get the default template registry."""
    return TemplateRegistry.get_default_registry()


class TestTemplateRegistry:
    """Test TemplateRegistry functionality."""

    def test_registry_initialization(self):
        """Test that registry initializes properly."""
        registry = TemplateRegistry()
        assert registry is not None
        assert registry.templates is not None

    def test_get_default_registry(self):
        """Test getting default registry."""
        registry = TemplateRegistry.get_default_registry()
        assert registry is not None
        assert len(registry.templates) > 0

    def test_default_registry_has_all_templates(self, template_registry):
        """Test that default registry has all 5 templates."""
        template_names = template_registry.list_templates()
        assert len(template_names) == 5
        assert "crm" in template_names
        assert "bot" in template_names
        assert "landing" in template_names
        assert "saas" in template_names
        assert "ecommerce" in template_names

    def test_register_template(self, template_registry):
        """Test registering a template."""
        templates = template_registry.list_templates()
        assert len(templates) >= 5

    def test_get_template_by_name(self, template_registry):
        """Test getting template by name."""
        template = template_registry.get("crm")
        assert template is not None
        assert template.template_name == "crm"

    def test_get_nonexistent_template_returns_none(self, template_registry):
        """Test that getting nonexistent template returns None."""
        template = template_registry.get("nonexistent")
        assert template is None

    def test_list_templates(self, template_registry):
        """Test listing all templates."""
        templates = template_registry.list_templates()
        assert isinstance(templates, list)
        assert len(templates) >= 5


class TestCRMTemplate:
    """Test CRM template."""

    def test_crm_template_from_registry(self, template_registry):
        """Test getting CRM template from registry."""
        template = template_registry.get("crm")
        assert template is not None
        assert template.template_name == "crm"

    def test_crm_has_default_features(self, template_registry):
        """Test that CRM template has default features."""
        template = template_registry.get("crm")
        assert template.default_features is not None
        assert len(template.default_features) > 0

    def test_crm_features_include_auth(self, template_registry):
        """Test that CRM includes auth feature."""
        template = template_registry.get("crm")
        feature_names = [f.name for f in template.default_features]
        assert "auth" in feature_names

    def test_crm_features_include_admin_panel(self, template_registry):
        """Test that CRM includes admin_panel feature."""
        template = template_registry.get("crm")
        feature_names = [f.name for f in template.default_features]
        assert "admin_panel" in feature_names

    def test_crm_features_include_client_database(self, template_registry):
        """Test that CRM includes client_database feature."""
        template = template_registry.get("crm")
        feature_names = [f.name for f in template.default_features]
        assert "client_database" in feature_names

    def test_crm_has_tech_stack(self, template_registry):
        """Test that CRM template has tech stack."""
        template = template_registry.get("crm")
        assert template.default_tech_stack is not None
        assert template.default_tech_stack.backend is not None

    def test_crm_tech_stack_backend_is_fastapi(self, template_registry):
        """Test that CRM tech stack uses FastAPI."""
        template = template_registry.get("crm")
        assert template.default_tech_stack.backend == "FastAPI"

    def test_crm_get_extra_directories(self, template_registry):
        """Test CRM extra directories."""
        template = template_registry.get("crm")
        directories = template.get_extra_directories("test_project")
        assert len(directories) > 0

    def test_crm_extra_directories_contains_crm_folder(self, template_registry):
        """Test that CRM directories include crm folder."""
        template = template_registry.get("crm")
        directories = template.get_extra_directories("test_project")
        assert any("crm" in d for d in directories)

    def test_crm_get_extra_files(self, template_registry):
        """Test CRM extra files."""
        template = template_registry.get("crm")
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test CRM",
        )
        files = template.get_extra_files("test_project", spec)
        assert isinstance(files, dict)
        assert len(files) > 0

    def test_crm_extra_files_have_content(self, template_registry):
        """Test that CRM extra files have non-empty content (except __init__.py)."""
        template = template_registry.get("crm")
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test CRM",
        )
        files = template.get_extra_files("test_project", spec)

        for path, content in files.items():
            # Skip __init__.py files which are intentionally empty
            if not path.endswith("__init__.py"):
                assert len(content) > 0, f"File {path} has empty content"

    def test_crm_get_database_schema(self, template_registry):
        """Test CRM database schema."""
        template = template_registry.get("crm")
        schema = template.get_database_schema("test_project")
        assert len(schema) > 0
        assert "CREATE TABLE" in schema


class TestBotTemplate:
    """Test Bot template."""

    def test_bot_template_from_registry(self, template_registry):
        """Test getting Bot template from registry."""
        template = template_registry.get("bot")
        assert template is not None
        assert template.template_name == "bot"

    def test_bot_has_default_features(self, template_registry):
        """Test that Bot template has default features."""
        template = template_registry.get("bot")
        assert template.default_features is not None
        assert len(template.default_features) > 0

    def test_bot_has_tech_stack(self, template_registry):
        """Test that Bot template has tech stack."""
        template = template_registry.get("bot")
        assert template.default_tech_stack is not None

    def test_bot_get_extra_directories(self, template_registry):
        """Test Bot extra directories."""
        template = template_registry.get("bot")
        directories = template.get_extra_directories("test_project")
        assert len(directories) > 0

    def test_bot_get_extra_files(self, template_registry):
        """Test Bot extra files."""
        template = template_registry.get("bot")
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test Bot",
        )
        files = template.get_extra_files("test_project", spec)
        assert isinstance(files, dict)
        assert len(files) > 0

    def test_bot_extra_files_have_content(self, template_registry):
        """Test that Bot extra files have non-empty content (except __init__.py)."""
        template = template_registry.get("bot")
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test Bot",
        )
        files = template.get_extra_files("test_project", spec)

        for path, content in files.items():
            # Skip __init__.py files which are intentionally empty
            if not path.endswith("__init__.py"):
                assert len(content) > 0


class TestLandingTemplate:
    """Test Landing Page template."""

    def test_landing_template_from_registry(self, template_registry):
        """Test getting Landing template from registry."""
        template = template_registry.get("landing")
        assert template is not None
        assert template.template_name == "landing"

    def test_landing_has_default_features(self, template_registry):
        """Test that Landing template has default features."""
        template = template_registry.get("landing")
        assert template.default_features is not None
        assert len(template.default_features) > 0

    def test_landing_has_tech_stack(self, template_registry):
        """Test that Landing template has tech stack."""
        template = template_registry.get("landing")
        assert template.default_tech_stack is not None

    def test_landing_get_extra_directories(self, template_registry):
        """Test Landing extra directories."""
        template = template_registry.get("landing")
        directories = template.get_extra_directories("test_project")
        assert len(directories) > 0

    def test_landing_get_extra_files(self, template_registry):
        """Test Landing extra files."""
        template = template_registry.get("landing")
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test Landing",
        )
        files = template.get_extra_files("test_project", spec)
        assert isinstance(files, dict)


class TestSaaSTemplate:
    """Test SaaS template."""

    def test_saas_template_from_registry(self, template_registry):
        """Test getting SaaS template from registry."""
        template = template_registry.get("saas")
        assert template is not None
        assert template.template_name == "saas"

    def test_saas_has_default_features(self, template_registry):
        """Test that SaaS template has default features."""
        template = template_registry.get("saas")
        assert template.default_features is not None
        assert len(template.default_features) > 0

    def test_saas_has_tech_stack(self, template_registry):
        """Test that SaaS template has tech stack."""
        template = template_registry.get("saas")
        assert template.default_tech_stack is not None

    def test_saas_get_extra_directories(self, template_registry):
        """Test SaaS extra directories."""
        template = template_registry.get("saas")
        directories = template.get_extra_directories("test_project")
        assert len(directories) > 0

    def test_saas_get_extra_files(self, template_registry):
        """Test SaaS extra files."""
        template = template_registry.get("saas")
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test SaaS",
        )
        files = template.get_extra_files("test_project", spec)
        assert isinstance(files, dict)


class TestEcommerceTemplate:
    """Test E-commerce template."""

    def test_ecommerce_template_from_registry(self, template_registry):
        """Test getting E-commerce template from registry."""
        template = template_registry.get("ecommerce")
        assert template is not None
        assert template.template_name == "ecommerce"

    def test_ecommerce_has_default_features(self, template_registry):
        """Test that E-commerce template has default features."""
        template = template_registry.get("ecommerce")
        assert template.default_features is not None
        assert len(template.default_features) > 0

    def test_ecommerce_has_tech_stack(self, template_registry):
        """Test that E-commerce template has tech stack."""
        template = template_registry.get("ecommerce")
        assert template.default_tech_stack is not None

    def test_ecommerce_get_extra_directories(self, template_registry):
        """Test E-commerce extra directories."""
        template = template_registry.get("ecommerce")
        directories = template.get_extra_directories("test_project")
        assert len(directories) > 0

    def test_ecommerce_get_extra_files(self, template_registry):
        """Test E-commerce extra files."""
        template = template_registry.get("ecommerce")
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test E-commerce",
        )
        files = template.get_extra_files("test_project", spec)
        assert isinstance(files, dict)


class TestTemplateFeatures:
    """Test template feature handling."""

    def test_template_features_are_feature_specs(self, template_registry):
        """Test that template features are FeatureSpec instances."""
        template = template_registry.get("crm")
        for feature in template.default_features:
            assert isinstance(feature, FeatureSpec)

    def test_template_features_have_required_fields(self, template_registry):
        """Test that template features have required fields."""
        template = template_registry.get("crm")
        for feature in template.default_features:
            assert feature.name is not None
            assert feature.description is not None
            assert feature.priority is not None

    def test_all_templates_have_features(self, template_registry):
        """Test that all templates have default features."""
        for template_name in ["crm", "bot", "landing", "saas", "ecommerce"]:
            template = template_registry.get(template_name)
            assert len(template.default_features) > 0

    def test_all_templates_have_tech_stack(self, template_registry):
        """Test that all templates have tech stacks."""
        for template_name in ["crm", "bot", "landing", "saas", "ecommerce"]:
            template = template_registry.get(template_name)
            assert template.default_tech_stack is not None
            # Landing template is HTML/CSS only (frontend-only) so it won't have backend
            if template_name == "landing":
                assert template.default_tech_stack.frontend is not None
            else:
                assert template.default_tech_stack.backend is not None


class TestTemplateIntegration:
    """Test template integration patterns."""

    def test_create_project_from_crm_template(self, template_registry):
        """Test creating a project using CRM template."""
        template = template_registry.get("crm")
        spec = ProjectSpec(
            project_name="my_crm",
            summary="My CRM",
            features=template.default_features,
            tech_stack=template.default_tech_stack,
        )

        assert spec.project_name == "my_crm"
        assert len(spec.features) > 0
        assert spec.tech_stack is not None

    def test_all_templates_generate_files(self, template_registry):
        """Test that all templates can generate files."""
        for template_name in ["crm", "bot", "landing", "saas", "ecommerce"]:
            template = template_registry.get(template_name)
            spec = ProjectSpec(
                project_name="test_project",
                summary="Test",
            )
            files = template.get_extra_files("test_project", spec)
            assert len(files) > 0

    def test_template_directories_are_properly_formatted(self, template_registry):
        """Test that template directories are properly formatted."""
        template = template_registry.get("crm")
        directories = template.get_extra_directories("my_project")

        for directory in directories:
            assert directory.startswith("my_project")
            assert "/" in directory  # Should have proper path structure

    def test_template_files_paths_contain_project_name(self, template_registry):
        """Test that file paths contain project name."""
        template = template_registry.get("crm")
        spec = ProjectSpec(
            project_name="my_app",
            summary="Test",
        )
        files = template.get_extra_files("my_app", spec)

        for path in files.keys():
            assert "my_app" in path


class TestTemplateConsistency:
    """Test consistency across templates."""

    def test_all_templates_have_descriptions(self, template_registry):
        """Test that all templates have descriptions."""
        for template_name in template_registry.list_templates():
            template = template_registry.get(template_name)
            assert template.description is not None
            assert len(template.description) > 0

    def test_templates_handle_project_name_correctly(self, template_registry):
        """Test that templates handle different project names."""
        template = template_registry.get("crm")

        for project_name in ["my_app", "test_123", "ProjectName"]:
            files = template.get_extra_files(project_name, ProjectSpec(
                project_name=project_name,
                summary="Test",
            ))
            assert all(project_name in path for path in files.keys())

    def test_template_files_are_python_or_sql(self, template_registry):
        """Test that template files have correct extensions."""
        template = template_registry.get("crm")
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
        )
        files = template.get_extra_files("test_project", spec)

        for path in files.keys():
            assert path.endswith(".py") or path.endswith(".sql"), \
                f"Unexpected file type: {path}"

    def test_template_python_files_are_valid_python(self, template_registry):
        """Test that template Python files are valid Python."""
        template = template_registry.get("crm")
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
        )
        files = template.get_extra_files("test_project", spec)

        for path, content in files.items():
            if path.endswith(".py"):
                try:
                    compile(content, path, "exec")
                except SyntaxError as e:
                    pytest.fail(f"Invalid Python in {path}: {e}")

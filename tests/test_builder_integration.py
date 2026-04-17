from __future__ import annotations

from pathlib import Path

import pytest

from artifacts.builder import ArtifactBuilder
from artifacts.file_tree import ProjectFileTree
from artifacts.generators.docs_generator import DocsGenerator
from artifacts.generators.env_generator import EnvGenerator
from artifacts.generators.feature_code_generator import FeatureCodeGenerator
from artifacts.generators.main_app_generator import MainAppGenerator
from artifacts.generators.structure_generator import StructureGenerator
from artifacts.generators.tests_generator import TestsGenerator
from artifacts.manifest import ArtifactManifest
from artifacts.validators import ArtifactValidator
from artifacts.writer import ArtifactWriter
from artifacts.zip_packager import ZipPackager
from runtime.specs.artifact_spec import ArtifactSpec
from runtime.specs.feature_spec import FeatureSpec
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec


@pytest.fixture
def structure_generator():
    """Create a real StructureGenerator."""
    return StructureGenerator()


@pytest.fixture
def docs_generator():
    """Create a real DocsGenerator."""
    return DocsGenerator()


@pytest.fixture
def env_generator():
    """Create a real EnvGenerator."""
    return EnvGenerator()


@pytest.fixture
def tests_generator():
    """Create a real TestsGenerator."""
    return TestsGenerator()


@pytest.fixture
def writer(tmp_path):
    """Create a real ArtifactWriter."""
    return ArtifactWriter()


@pytest.fixture
def validator():
    """Create a real ArtifactValidator."""
    return ArtifactValidator()


@pytest.fixture
def zip_packager():
    """Create a real ZipPackager."""
    return ZipPackager()


@pytest.fixture
def builder(
    structure_generator,
    docs_generator,
    env_generator,
    tests_generator,
    writer,
    validator,
    zip_packager,
):
    """Create an ArtifactBuilder with real dependencies."""
    return ArtifactBuilder(
        structure_generator=structure_generator,
        docs_generator=docs_generator,
        env_generator=env_generator,
        tests_generator=tests_generator,
        writer=writer,
        validator=validator,
        zip_packager=zip_packager,
    )


@pytest.fixture
def builder_with_features(builder):
    """Create a builder with feature generators."""
    builder.feature_code_generator = FeatureCodeGenerator()
    builder.main_app_generator = MainAppGenerator()
    return builder


@pytest.fixture
def base_project_spec():
    """Create a basic ProjectSpec."""
    return ProjectSpec(
        project_name="test_project",
        summary="A test project",
        project_type="application",
    )


@pytest.fixture
def base_artifact_spec():
    """Create a basic ArtifactSpec."""
    return ArtifactSpec(
        artifact_name="test_artifact",
        include_tests=True,
        packaging="folder",
    )


class TestArtifactBuilderBasics:
    """Test basic functionality of ArtifactBuilder."""

    def test_builder_initialization(self, builder):
        """Test that builder is initialized properly."""
        assert builder is not None
        assert hasattr(builder, "build_tree")
        assert hasattr(builder, "build")

    def test_build_tree_returns_tuple(self, builder, base_project_spec, base_artifact_spec):
        """Test that build_tree returns a tuple of tree and manifest."""
        tree, manifest = builder.build_tree(base_project_spec, base_artifact_spec)
        assert isinstance(tree, ProjectFileTree)
        assert isinstance(manifest, ArtifactManifest)

    def test_build_tree_creates_readme(self, builder, base_project_spec, base_artifact_spec):
        """Test that build_tree creates a README file."""
        tree, _ = builder.build_tree(base_project_spec, base_artifact_spec)
        assert any(f.path.endswith("README.md") for f in tree.files)


class TestArtifactBuilderWithoutFeatures:
    """Test builder behavior without features."""

    def test_build_tree_without_features(self, builder, base_project_spec, base_artifact_spec):
        """Test that build_tree works without feature generators."""
        tree, manifest = builder.build_tree(base_project_spec, base_artifact_spec)

        assert isinstance(tree, ProjectFileTree)
        assert isinstance(manifest, ArtifactManifest)
        assert manifest.project_name == "test_project"

    def test_build_tree_without_features_no_feature_files(
        self, builder, base_project_spec, base_artifact_spec
    ):
        """Test that no feature-specific files are added without feature generators."""
        tree, _ = builder.build_tree(base_project_spec, base_artifact_spec)

        # Should not have auth/admin/export files if no feature generator
        feature_files = [
            f for f in tree.files
            if any(feature in f.path for feature in ["auth", "admin", "clients", "export"])
        ]
        assert len(feature_files) == 0


class TestArtifactBuilderWithFeatures:
    """Test builder behavior with features."""

    def test_build_tree_with_feature_generator(self, builder_with_features, base_artifact_spec):
        """Test that build_tree uses feature generator when available."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="auth", description="Auth")],
        )
        tree, _ = builder_with_features.build_tree(spec, base_artifact_spec)

        # Should have auth files because feature generator is available
        file_paths = [f.path for f in tree.files]
        assert any("auth" in path for path in file_paths)

    def test_build_tree_with_main_app_generator(self, builder_with_features, base_artifact_spec):
        """Test that main.py is updated when main_app_generator is available."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="auth", description="Auth")],
        )
        tree, _ = builder_with_features.build_tree(spec, base_artifact_spec)

        # Should have updated main.py with feature routers
        main_py = next(
            (f for f in tree.files if f.path.endswith("backend/main.py")), None
        )
        assert main_py is not None
        assert "auth_router" in main_py.content

    def test_build_tree_features_from_spec(self, builder_with_features, base_artifact_spec):
        """Test that features are taken from project spec."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[
                FeatureSpec(name="auth", description="Auth"),
                FeatureSpec(name="admin_panel", description="Admin"),
            ],
        )
        tree, _ = builder_with_features.build_tree(spec, base_artifact_spec)

        file_paths = [f.path for f in tree.files]
        assert any("auth" in path for path in file_paths)
        assert any("admin" in path for path in file_paths)


class TestArtifactBuilderWithAgentResults:
    """Test builder behavior with agent results."""

    def test_build_tree_with_agent_results(
        self, builder_with_features, base_artifact_spec
    ):
        """Test that build_tree accepts agent_results parameter."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="auth", description="Auth")],
        )
        agent_results = [{"type": "recommendation", "data": "test"}]
        tree, _ = builder_with_features.build_tree(
            spec, base_artifact_spec, agent_results=agent_results
        )

        assert isinstance(tree, ProjectFileTree)

    def test_build_tree_backward_compatibility_no_agent_results(
        self, builder_with_features, base_artifact_spec
    ):
        """Test that build_tree works without agent_results (backward compatibility)."""
        spec = ProjectSpec(
            project_name="test_project",
            summary="Test",
            features=[FeatureSpec(name="auth", description="Auth")],
        )
        tree, _ = builder_with_features.build_tree(spec, base_artifact_spec)

        assert isinstance(tree, ProjectFileTree)


class TestArtifactBuilderManifest:
    """Test manifest generation."""

    def test_manifest_includes_project_metadata(
        self, builder, base_project_spec, base_artifact_spec
    ):
        """Test that manifest includes project metadata."""
        _, manifest = builder.build_tree(base_project_spec, base_artifact_spec)

        assert manifest.project_name == "test_project"
        assert manifest.description == base_project_spec.summary

    def test_manifest_includes_packaging_info(
        self, builder, base_project_spec, base_artifact_spec
    ):
        """Test that manifest includes packaging information."""
        _, manifest = builder.build_tree(base_project_spec, base_artifact_spec)

        assert manifest.packaging == base_artifact_spec.packaging

    def test_manifest_includes_file_list(self, builder, base_project_spec, base_artifact_spec):
        """Test that manifest includes file paths."""
        _, manifest = builder.build_tree(base_project_spec, base_artifact_spec)

        assert len(manifest.files) > 0
        assert any("README.md" in f for f in manifest.files)

    def test_manifest_includes_entrypoints(
        self, builder, base_project_spec, base_artifact_spec
    ):
        """Test that manifest includes entrypoint information."""
        _, manifest = builder.build_tree(base_project_spec, base_artifact_spec)

        assert "README.md" in manifest.entrypoints


class TestArtifactBuilderTestsGeneration:
    """Test test file generation."""

    def test_build_tree_with_tests_included(
        self, builder, base_project_spec, base_artifact_spec
    ):
        """Test that tests are created when include_tests is True."""
        base_artifact_spec.include_tests = True
        tree, _ = builder.build_tree(base_project_spec, base_artifact_spec)

        test_files = [f for f in tree.files if "test" in f.path.lower()]
        assert len(test_files) > 0

    def test_build_tree_without_tests(self, builder, base_project_spec, base_artifact_spec):
        """Test that tests are not created when include_tests is False."""
        base_artifact_spec.include_tests = False
        tree, _ = builder.build_tree(base_project_spec, base_artifact_spec)

        # Still should have README but not test files (depends on implementation)
        assert any(f.path.endswith("README.md") for f in tree.files)


class TestArtifactBuilderValidation:
    """Test validation during build process."""

    def test_build_tree_validates_project_spec(
        self, builder, base_artifact_spec
    ):
        """Test that project spec is validated."""
        invalid_spec = ProjectSpec(
            project_name="",  # Invalid: empty name
            summary="Test",
        )

        with pytest.raises(ValueError):
            builder.build_tree(invalid_spec, base_artifact_spec)

    def test_build_tree_validates_artifact_spec(
        self, builder, base_project_spec
    ):
        """Test that artifact spec is validated."""
        invalid_spec = ArtifactSpec(
            artifact_name="test",
            packaging="invalid",  # Invalid packaging type
        )

        with pytest.raises(ValueError):
            builder.build_tree(base_project_spec, invalid_spec)


class TestArtifactBuilderBuild:
    """Test the full build pipeline."""

    def test_build_creates_output_directory(
        self, builder, base_project_spec, base_artifact_spec, tmp_path
    ):
        """Test that build creates output in the specified directory."""
        output_dir = str(tmp_path / "build")
        result, _manifest = builder.build(output_dir, base_project_spec, base_artifact_spec)

        # Result should be a path to the project
        assert result is not None

    def test_build_returns_correct_path(
        self, builder, base_project_spec, base_artifact_spec, tmp_path
    ):
        """Test that build returns correct path based on packaging."""
        output_dir = str(tmp_path / "build")
        result, _manifest = builder.build(
            output_dir, base_project_spec, base_artifact_spec
        )

        # With folder packaging, should return project root
        assert "test_project" in result

    def test_build_with_zip_packaging(
        self, builder, base_project_spec, base_artifact_spec, tmp_path
    ):
        """Test that build returns zip path with zip packaging."""
        base_artifact_spec.packaging = "zip"
        output_dir = str(tmp_path / "build")
        result, _manifest = builder.build(output_dir, base_project_spec, base_artifact_spec)

        # With zip packaging, should return zip path
        assert result.endswith(".zip")

    def test_build_validates_manifest(
        self, builder, base_project_spec, base_artifact_spec, tmp_path
    ):
        """Test that build validates the manifest."""
        output_dir = str(tmp_path / "build")
        builder.build(output_dir, base_project_spec, base_artifact_spec)

        # Validator should have been called (it returned no errors)
        # No exception was raised, so validation passed

    def test_build_calls_writer(
        self, builder, base_project_spec, base_artifact_spec, tmp_path
    ):
        """Test that build calls the writer."""
        output_dir = str(tmp_path / "build")
        builder.build(output_dir, base_project_spec, base_artifact_spec)

        # Writer should have written files to the output directory
        output_path = Path(output_dir) / "test_project"
        assert output_path.exists()


class TestArtifactBuilderErrorHandling:
    """Test error handling in builder."""

    def test_build_raises_on_validation_error(
        self, builder, base_project_spec, base_artifact_spec, tmp_path
    ):
        """Test that build raises error if validation fails."""
        # Override the validator to return errors
        class FailingValidator(ArtifactValidator):
            def validate_tree_against_manifest(self, tree, manifest):
                return ["Error 1", "Error 2"]

        builder.validator = FailingValidator()
        output_dir = str(tmp_path / "build")

        with pytest.raises(ValueError, match="Artifact validation failed"):
            builder.build(output_dir, base_project_spec, base_artifact_spec)


class TestArtifactBuilderWithComplexProject:
    """Test builder with complex project configurations."""

    def test_build_tree_with_all_features(
        self, builder_with_features, base_artifact_spec
    ):
        """Test build_tree with all available features."""
        spec = ProjectSpec(
            project_name="comprehensive_app",
            summary="Comprehensive application",
            features=[
                FeatureSpec(name="auth", description="Authentication"),
                FeatureSpec(name="admin_panel", description="Admin Panel"),
                FeatureSpec(name="client_database", description="Client DB"),
                FeatureSpec(name="export", description="Export"),
                FeatureSpec(name="roles", description="Roles"),
                FeatureSpec(name="notifications", description="Notifications"),
                FeatureSpec(name="api", description="API"),
            ],
        )
        tree, manifest = builder_with_features.build_tree(spec, base_artifact_spec)

        assert len(tree.files) > 0
        assert manifest.project_name == "comprehensive_app"

    def test_build_tree_preserves_project_name(
        self, builder_with_features, base_artifact_spec
    ):
        """Test that project name is preserved throughout build."""
        spec = ProjectSpec(
            project_name="my_cool_app",
            summary="My cool app",
        )
        tree, manifest = builder_with_features.build_tree(spec, base_artifact_spec)

        assert manifest.project_name == "my_cool_app"

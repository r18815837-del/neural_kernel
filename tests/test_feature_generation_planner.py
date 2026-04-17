"""Tests for FeatureGenerationPlanner — ordering and normalisation."""

from __future__ import annotations

import pytest

from artifacts.features.planner import FeatureGenerationPlanner
from runtime.specs.feature_spec import FeatureSpec


def _fs(name: str) -> FeatureSpec:
    """Shorthand to create a FeatureSpec."""
    return FeatureSpec(name=name, description=f"{name} feature")


class TestNormalisation:
    """Feature name normalisation."""

    def test_export_functionality_alias(self):
        planner = FeatureGenerationPlanner()
        assert planner.normalise_name("export_functionality") == "export"

    def test_data_export_alias(self):
        planner = FeatureGenerationPlanner()
        assert planner.normalise_name("data_export") == "export"

    def test_user_roles_alias(self):
        planner = FeatureGenerationPlanner()
        assert planner.normalise_name("user_roles") == "roles"

    def test_authentication_alias(self):
        planner = FeatureGenerationPlanner()
        assert planner.normalise_name("authentication") == "auth"

    def test_unknown_name_passes_through(self):
        planner = FeatureGenerationPlanner()
        assert planner.normalise_name("my_custom_feature") == "my_custom_feature"

    def test_strips_whitespace_and_lowercases(self):
        planner = FeatureGenerationPlanner()
        assert planner.normalise_name("  Export_Functionality  ") == "export"

    def test_hyphen_normalisation(self):
        planner = FeatureGenerationPlanner()
        assert planner.normalise_name("client-database") == "client_database"


class TestOrdering:
    """Topological ordering with dependency constraints."""

    def test_auth_before_roles(self):
        planner = FeatureGenerationPlanner()
        result = planner.plan([_fs("roles"), _fs("auth")])
        assert result.index("auth") < result.index("roles")

    def test_roles_before_admin_panel(self):
        planner = FeatureGenerationPlanner()
        result = planner.plan([_fs("admin_panel"), _fs("roles"), _fs("auth")])
        assert result.index("roles") < result.index("admin_panel")

    def test_client_database_before_export(self):
        planner = FeatureGenerationPlanner()
        result = planner.plan([_fs("export"), _fs("client_database")])
        assert result.index("client_database") < result.index("export")

    def test_full_chain(self):
        """auth → roles → admin_panel, client_database → export."""
        planner = FeatureGenerationPlanner()
        features = [
            _fs("export"),
            _fs("admin_panel"),
            _fs("roles"),
            _fs("client_database"),
            _fs("auth"),
        ]
        result = planner.plan(features)

        assert result.index("auth") < result.index("roles")
        assert result.index("roles") < result.index("admin_panel")
        assert result.index("client_database") < result.index("export")

    def test_normalises_during_plan(self):
        planner = FeatureGenerationPlanner()
        result = planner.plan([_fs("export_functionality"), _fs("client_database")])
        assert "export" in result
        assert "export_functionality" not in result

    def test_deduplicates(self):
        planner = FeatureGenerationPlanner()
        result = planner.plan([_fs("auth"), _fs("authentication")])
        assert result == ["auth"]

    def test_single_feature(self):
        planner = FeatureGenerationPlanner()
        result = planner.plan([_fs("client_database")])
        assert result == ["client_database"]

    def test_empty_features(self):
        planner = FeatureGenerationPlanner()
        result = planner.plan([])
        assert result == []

    def test_unrelated_features_keep_order(self):
        planner = FeatureGenerationPlanner()
        result = planner.plan([_fs("notifications"), _fs("api")])
        assert result == ["notifications", "api"]

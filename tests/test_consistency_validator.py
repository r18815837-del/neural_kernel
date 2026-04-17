"""Tests for artifacts.consistency_validator."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from artifacts.consistency_validator import (
    ConsistencyIssue,
    ConsistencyReport,
    ConsistencyValidator,
)
from artifacts.file_tree import ProjectFileTree
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.tech_stack_spec import TechStackSpec
from runtime.specs.feature_spec import FeatureSpec


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_spec(
    name: str = "myapp",
    backend: str | None = None,
    database: str | None = None,
    features: list[str] | None = None,
) -> ProjectSpec:
    tech = None
    if backend or database:
        tech = TechStackSpec(backend=backend, database=database)
    feat_list = [FeatureSpec(name=f, description=f) for f in (features or [])]
    return ProjectSpec(
        project_name=name,
        summary="test",
        tech_stack=tech,
        features=feat_list,
    )


def _make_tree_with_readme(root: str, content: str) -> ProjectFileTree:
    tree = ProjectFileTree()
    tree.add_file(f"{root}/README.md", content)
    return tree


# ------------------------------------------------------------------
# ConsistencyReport
# ------------------------------------------------------------------

def test_report_empty_is_consistent():
    r = ConsistencyReport()
    assert r.is_consistent is True
    assert r.error_count == 0
    assert r.warning_count == 0


def test_report_warning_still_consistent():
    r = ConsistencyReport(issues=[
        ConsistencyIssue(check="x", severity="warning", message="w"),
    ])
    assert r.is_consistent is True
    assert r.warning_count == 1


def test_report_error_not_consistent():
    r = ConsistencyReport(issues=[
        ConsistencyIssue(check="x", severity="error", message="e"),
    ])
    assert r.is_consistent is False
    assert r.error_count == 1


def test_report_to_dict():
    r = ConsistencyReport(checks_run=2, issues=[
        ConsistencyIssue(check="a", severity="warning", message="w1"),
        ConsistencyIssue(check="b", severity="error", message="e1"),
    ])
    d = r.to_dict()
    assert d["is_consistent"] is False
    assert d["checks_run"] == 2
    assert d["errors"] == 1
    assert d["warnings"] == 1
    assert len(d["issues"]) == 2


# ------------------------------------------------------------------
# Check 1: README ↔ tech_stack
# ------------------------------------------------------------------

def test_check_readme_missing():
    v = ConsistencyValidator()
    spec = _make_spec("myapp", backend="FastAPI")
    tree = ProjectFileTree()  # no README
    report = v.validate(tree, spec)
    errors = [i for i in report.issues if i.check == "readme_stack" and i.severity == "error"]
    assert len(errors) == 1
    assert "missing" in errors[0].message.lower()


def test_check_readme_mentions_backend():
    v = ConsistencyValidator()
    spec = _make_spec("myapp", backend="FastAPI")
    tree = _make_tree_with_readme("myapp", "# MyApp\nBuilt with FastAPI and love.")
    report = v.validate(tree, spec)
    readme_warnings = [i for i in report.issues if i.check == "readme_stack"]
    assert len(readme_warnings) == 0


def test_check_readme_missing_backend():
    v = ConsistencyValidator()
    spec = _make_spec("myapp", backend="FastAPI")
    tree = _make_tree_with_readme("myapp", "# MyApp\nA great app.")
    report = v.validate(tree, spec)
    warnings = [i for i in report.issues if i.check == "readme_stack" and "FastAPI" in i.message]
    assert len(warnings) == 1


def test_check_readme_missing_database():
    v = ConsistencyValidator()
    spec = _make_spec("myapp", backend="FastAPI", database="PostgreSQL")
    tree = _make_tree_with_readme("myapp", "# MyApp\nBuilt with FastAPI.")
    report = v.validate(tree, spec)
    db_warns = [i for i in report.issues if "PostgreSQL" in i.message]
    assert len(db_warns) == 1


def test_check_readme_missing_feature():
    v = ConsistencyValidator()
    spec = _make_spec("myapp", features=["auth", "export"])
    tree = _make_tree_with_readme("myapp", "# MyApp\nHas auth support.")
    report = v.validate(tree, spec)
    feat_warns = [i for i in report.issues if "export" in i.message]
    assert len(feat_warns) == 1


# ------------------------------------------------------------------
# Check 2: Dockerfile ↔ backend
# ------------------------------------------------------------------

def test_dockerfile_fastapi_no_uvicorn():
    v = ConsistencyValidator()
    spec = _make_spec("myapp", backend="FastAPI")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "FastAPI app")
    tree.add_file("myapp/Dockerfile", "FROM python:3.11\nCMD python main.py")
    report = v.validate(tree, spec)
    errs = [i for i in report.issues if i.check == "dockerfile_backend" and i.severity == "error"]
    assert any("uvicorn" in e.message for e in errs)


def test_dockerfile_fastapi_with_uvicorn():
    v = ConsistencyValidator()
    spec = _make_spec("myapp", backend="FastAPI")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "FastAPI app")
    tree.add_file("myapp/Dockerfile", "FROM python:3.11\nCOPY requirements.txt .\nCMD uvicorn main:app")
    report = v.validate(tree, spec)
    errs = [i for i in report.issues if i.check == "dockerfile_backend" and i.severity == "error"]
    assert len(errs) == 0


def test_dockerfile_missing_requirements():
    v = ConsistencyValidator()
    spec = _make_spec("myapp", backend="Django")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "Django app")
    tree.add_file("myapp/requirements.txt", "django\ngunicorn\n")
    tree.add_file("myapp/Dockerfile", "FROM python:3.11\nCMD gunicorn")
    report = v.validate(tree, spec)
    warns = [i for i in report.issues if i.check == "dockerfile_backend" and "requirements.txt" in i.message]
    assert len(warns) == 1


# ------------------------------------------------------------------
# Check 3: CI ↔ tests
# ------------------------------------------------------------------

def test_ci_missing_pytest():
    v = ConsistencyValidator()
    spec = _make_spec("myapp")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "app")
    tree.add_file("myapp/requirements.txt", "fastapi")
    tree.add_file("myapp/tests/test_smoke.py", "def test(): pass")
    tree.add_file("myapp/.github/workflows/ci.yml", "name: CI\nsteps:\n  - run: echo ok")
    report = v.validate(tree, spec)
    errs = [i for i in report.issues if i.check == "ci_tests" and "pytest" in i.message]
    assert len(errs) == 1


def test_ci_with_pytest_ok():
    v = ConsistencyValidator()
    spec = _make_spec("myapp")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "app")
    tree.add_file("myapp/requirements.txt", "fastapi")
    tree.add_file("myapp/tests/test_smoke.py", "def test(): pass")
    tree.add_file("myapp/.github/workflows/ci.yml", "name: CI\nsteps:\n  - run: pip install -r requirements.txt\n  - run: pytest")
    report = v.validate(tree, spec)
    errs = [i for i in report.issues if i.check == "ci_tests" and i.severity == "error"]
    assert len(errs) == 0


# ------------------------------------------------------------------
# Check 4: deps ↔ code
# ------------------------------------------------------------------

def test_deps_missing_requirements_file():
    v = ConsistencyValidator()
    spec = _make_spec("myapp", backend="FastAPI")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "FastAPI app")
    report = v.validate(tree, spec)
    errs = [i for i in report.issues if i.check == "deps_code" and "missing" in i.message.lower()]
    assert len(errs) == 1


def test_deps_code_import_not_in_requirements():
    v = ConsistencyValidator()
    spec = _make_spec("myapp")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "app")
    tree.add_file("myapp/requirements.txt", "pydantic\n")
    tree.add_file("myapp/backend/main.py", "from fastapi import FastAPI\napp = FastAPI()")
    report = v.validate(tree, spec)
    warns = [i for i in report.issues if i.check == "deps_code" and "fastapi" in i.message]
    assert len(warns) == 1


def test_deps_code_all_present():
    v = ConsistencyValidator()
    spec = _make_spec("myapp")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "app")
    tree.add_file("myapp/requirements.txt", "fastapi\nuvicorn\npydantic\n")
    tree.add_file("myapp/backend/main.py", "from fastapi import FastAPI\nimport uvicorn")
    report = v.validate(tree, spec)
    warns = [i for i in report.issues if i.check == "deps_code" and i.severity == "warning"]
    assert len(warns) == 0


# ------------------------------------------------------------------
# Check 5: routes ↔ main.py
# ------------------------------------------------------------------

def test_routes_not_included_in_main():
    v = ConsistencyValidator()
    spec = _make_spec("myapp")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "app")
    tree.add_file("myapp/requirements.txt", "fastapi\n")
    tree.add_file("myapp/backend/main.py", "from fastapi import FastAPI\napp = FastAPI()")
    tree.add_file("myapp/backend/auth/routes.py", "router = ...")
    tree.add_file("myapp/backend/export/routes.py", "router = ...")
    report = v.validate(tree, spec)
    warns = [i for i in report.issues if i.check == "routes_consistency"]
    modules = {w.message.split("'")[1].split("/")[1] for w in warns}
    assert "auth" in modules
    assert "export" in modules


def test_routes_included_ok():
    v = ConsistencyValidator()
    spec = _make_spec("myapp")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "app")
    tree.add_file("myapp/requirements.txt", "fastapi\n")
    tree.add_file("myapp/backend/main.py", "from backend.auth.routes import router as auth_router\napp.include_router(auth_router)")
    tree.add_file("myapp/backend/auth/routes.py", "router = ...")
    report = v.validate(tree, spec)
    warns = [i for i in report.issues if i.check == "routes_consistency"]
    assert len(warns) == 0


# ------------------------------------------------------------------
# Full validate — all checks run
# ------------------------------------------------------------------

def test_validate_runs_all_checks():
    v = ConsistencyValidator()
    spec = _make_spec("myapp", backend="FastAPI", database="PostgreSQL")
    tree = ProjectFileTree()
    tree.add_file("myapp/README.md", "# MyApp\nFastAPI + PostgreSQL")
    tree.add_file("myapp/requirements.txt", "fastapi\nuvicorn\npsycopg2\n")
    report = v.validate(tree, spec)
    assert report.checks_run == 5


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS: {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL: {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")

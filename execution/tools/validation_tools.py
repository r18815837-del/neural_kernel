from __future__ import annotations

from pathlib import Path
from typing import List
from runtime.specs.project_spec import ProjectSpec
from .base import BaseTool, ToolResult


class ValidatePathExistsTool(BaseTool):
    name = "validate_path_exists"
    description = "Validate that a file or directory exists"

    def run(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(
                success=False,
                message="Missing 'path'",
                error="missing_path",
            )

        target = Path(path)
        if not target.exists():
            return ToolResult(
                success=False,
                message=f"Path does not exist: {target}",
                error="path_not_found",
            )

        return ToolResult(
            success=True,
            message=f"Path exists: {target}",
            outputs={
                "path": str(target),
                "is_file": target.is_file(),
                "is_dir": target.is_dir(),
            },
        )


class ValidateRequiredFilesTool(BaseTool):
    name = "validate_required_files"
    description = "Validate that all required files exist inside a project root"

    def run(self, **kwargs) -> ToolResult:
        root_path = kwargs.get("root_path")
        required_files = kwargs.get("required_files", [])

        if not root_path:
            return ToolResult(
                success=False,
                message="Missing 'root_path'",
                error="missing_root_path",
            )

        root = Path(root_path)
        if not root.exists():
            return ToolResult(
                success=False,
                message=f"Root path does not exist: {root}",
                error="root_not_found",
            )

        missing: List[str] = []
        existing: List[str] = []

        for rel_path in required_files:
            candidate = root / rel_path
            if candidate.exists():
                existing.append(str(candidate))
            else:
                missing.append(rel_path)

        if missing:
            return ToolResult(
                success=False,
                message="Some required files are missing",
                outputs={
                    "root_path": str(root),
                    "missing": missing,
                    "existing": existing,
                },
                error="missing_required_files",
            )

        return ToolResult(
            success=True,
            message="All required files are present",
            outputs={
                "root_path": str(root),
                "existing": existing,
            },
        )


class ValidateNonEmptyFilesTool(BaseTool):
    name = "validate_non_empty_files"
    description = "Validate that specified files exist and are not empty"

    def run(self, **kwargs) -> ToolResult:
        files = kwargs.get("files", [])

        if not files:
            return ToolResult(
                success=False,
                message="Missing 'files'",
                error="missing_files",
            )

        empty_files: List[str] = []
        missing_files: List[str] = []
        valid_files: List[str] = []

        for file_path in files:
            path = Path(file_path)

            if not path.exists():
                missing_files.append(str(path))
                continue

            if not path.is_file():
                missing_files.append(str(path))
                continue

            if path.stat().st_size == 0:
                empty_files.append(str(path))
            else:
                valid_files.append(str(path))

        if missing_files or empty_files:
            return ToolResult(
                success=False,
                message="Some files are missing or empty",
                outputs={
                    "valid_files": valid_files,
                    "missing_files": missing_files,
                    "empty_files": empty_files,
                },
                error="invalid_files",
            )

        return ToolResult(
            success=True,
            message="All files are present and non-empty",
            outputs={"valid_files": valid_files},
        )
class ValidateProjectScaffoldTool(BaseTool):
    name = "validate_project_scaffold"
    description = "Validate generated project scaffold against project spec"

    def run(self, **kwargs) -> ToolResult:
        root_path = kwargs.get("root_path")
        project_spec = kwargs.get("project_spec")

        if not root_path:
            return ToolResult(
                success=False,
                message="Missing 'root_path'",
                error="missing_root_path",
            )

        if project_spec is None:
            return ToolResult(
                success=False,
                message="Missing 'project_spec'",
                error="missing_project_spec",
            )

        if not isinstance(project_spec, ProjectSpec):
            return ToolResult(
                success=False,
                message="'project_spec' must be a ProjectSpec instance",
                error="invalid_project_spec",
            )

        root = Path(root_path)
        if not root.exists():
            return ToolResult(
                success=False,
                message=f"Project root does not exist: {root}",
                error="root_not_found",
            )

        required_files = [
            "README.md",
            ".env.example",
            "requirements.txt",
            "tests/test_smoke.py",
        ]

        stack = project_spec.tech_stack
        if stack is not None:
            if stack.backend:
                required_files.extend(
                    [
                        "backend/__init__.py",
                        "backend/config.py",
                        "backend/routes.py",
                        "backend/main.py",
                    ]
                )

            if stack.frontend:
                required_files.extend(
                    [
                        "frontend/index.html",
                        "frontend/app.js",
                    ]
                )

            if stack.database:
                required_files.extend(
                    [
                        "database/README.md",
                        "database/schema.sql",
                    ]
                )

            if stack.mobile:
                required_files.append("mobile/README.md")

        missing_files = []
        empty_files = []
        valid_files = []

        for rel_path in required_files:
            full_path = root / rel_path

            if not full_path.exists():
                missing_files.append(rel_path)
                continue

            if not full_path.is_file():
                missing_files.append(rel_path)
                continue

            if full_path.stat().st_size == 0 and rel_path != "backend/__init__.py":
                empty_files.append(rel_path)
                continue

            valid_files.append(rel_path)

        if missing_files or empty_files:
            return ToolResult(
                success=False,
                message="Project scaffold validation failed",
                outputs={
                    "root_path": str(root),
                    "valid_files": valid_files,
                    "missing_files": missing_files,
                    "empty_files": empty_files,
                },
                error="invalid_project_scaffold",
            )

        return ToolResult(
            success=True,
            message="Project scaffold validation passed",
            outputs={
                "root_path": str(root),
                "valid_files": valid_files,
            },
        )
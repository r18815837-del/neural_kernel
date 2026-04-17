from __future__ import annotations

from pathlib import Path

from .base import BaseTool, ToolResult


class CreateDirectoryTool(BaseTool):
    name = "create_directory"
    description = "Create a directory on disk"

    def run(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(
                success=False,
                message="Missing 'path'",
                error="missing_path",
            )

        try:
            directory = Path(path)
            directory.mkdir(parents=True, exist_ok=True)
            return ToolResult(
                success=True,
                message=f"Directory created: {directory}",
                outputs={"path": str(directory)},
                artifacts=[str(directory)],
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Failed to create directory: {path}",
                error=str(exc),
            )


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write text content to a file"

    def run(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        content = kwargs.get("content", "")

        if not path:
            return ToolResult(
                success=False,
                message="Missing 'path'",
                error="missing_path",
            )

        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(str(content), encoding="utf-8")

            return ToolResult(
                success=True,
                message=f"File written: {file_path}",
                outputs={"path": str(file_path)},
                artifacts=[str(file_path)],
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Failed to write file: {path}",
                error=str(exc),
            )


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read text content from a file"

    def run(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(
                success=False,
                message="Missing 'path'",
                error="missing_path",
            )

        try:
            file_path = Path(path)
            content = file_path.read_text(encoding="utf-8")

            return ToolResult(
                success=True,
                message=f"File read: {file_path}",
                outputs={
                    "path": str(file_path),
                    "content": content,
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Failed to read file: {path}",
                error=str(exc),
            )


class ListTreeTool(BaseTool):
    name = "list_tree"
    description = "List files and directories recursively"

    def run(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(
                success=False,
                message="Missing 'path'",
                error="missing_path",
            )

        try:
            root = Path(path)
            if not root.exists():
                return ToolResult(
                    success=False,
                    message=f"Path does not exist: {root}",
                    error="path_not_found",
                )

            items = []
            for item in root.rglob("*"):
                items.append(str(item))

            return ToolResult(
                success=True,
                message=f"Listed tree: {root}",
                outputs={
                    "path": str(root),
                    "items": items,
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Failed to list tree: {path}",
                error=str(exc),
            )
from __future__ import annotations

from pathlib import Path
import zipfile

from .base import BaseTool, ToolResult


class CreateZipTool(BaseTool):
    name = "create_zip"
    description = "Create a zip archive from a directory"

    def run(self, **kwargs) -> ToolResult:
        source_dir = kwargs.get("source_dir")
        output_path = kwargs.get("output_path")

        if not source_dir:
            return ToolResult(
                success=False,
                message="Missing 'source_dir'",
                error="missing_source_dir",
            )

        if not output_path:
            return ToolResult(
                success=False,
                message="Missing 'output_path'",
                error="missing_output_path",
            )

        try:
            source = Path(source_dir)
            if not source.exists():
                return ToolResult(
                    success=False,
                    message=f"Source directory does not exist: {source}",
                    error="source_not_found",
                )

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in source.rglob("*"):
                    if path.is_file():
                        zf.write(path, arcname=path.relative_to(source))

            return ToolResult(
                success=True,
                message=f"Zip archive created: {output}",
                outputs={"path": str(output)},
                artifacts=[str(output)],
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Failed to create zip: {output_path}",
                error=str(exc),
            )
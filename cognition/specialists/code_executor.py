"""Safe Python code executor with timeout and output capture.

Runs user code in a subprocess with strict limits:
  - Timeout (default 5 seconds)
  - Stdout/stderr capture
  - No network access hints
  - No file write access hints

Usage::

    executor = CodeExecutor(timeout=5)
    result = executor.run("print(2 + 2)")
    print(result.stdout)   # "4\\n"
    print(result.success)  # True
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of code execution."""
    stdout: str
    stderr: str
    success: bool
    exit_code: int
    timed_out: bool
    error_summary: str | None = None

    def to_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "success": self.success,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "error_summary": self.error_summary,
        }


# Disallowed imports / calls for basic safety.
_BLOCKED_PATTERNS = [
    "os.system",
    "subprocess",
    "shutil.rmtree",
    "open(",           # file writes
    "__import__",
    "exec(",
    "eval(",
    "compile(",
    "importlib",
]


class CodeExecutor:
    """Execute Python code safely in a subprocess."""

    def __init__(self, timeout: int = 5, max_output: int = 5000) -> None:
        self.timeout = timeout
        self.max_output = max_output

    def check_safety(self, code: str) -> str | None:
        """Return warning message if code looks unsafe, None if ok."""
        code_lower = code.lower()

        for pattern in _BLOCKED_PATTERNS:
            if pattern.lower() in code_lower:
                return f"Blocked: code contains '{pattern}' which is not allowed in sandbox."

        if "while True" in code and "break" not in code:
            return "Warning: infinite loop detected (while True without break)."

        return None

    def run(self, code: str) -> ExecutionResult:
        """Run Python code and return the result."""
        # Safety check.
        safety = self.check_safety(code)
        if safety:
            return ExecutionResult(
                stdout="",
                stderr=safety,
                success=False,
                exit_code=-1,
                timed_out=False,
                error_summary=safety,
            )

        # Wrap code to capture output.
        wrapped = textwrap.dedent(code)

        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                # Don't inherit parent env completely.
                env={"PATH": "", "PYTHONPATH": ""},
            )

            stdout = proc.stdout[:self.max_output]
            stderr = proc.stderr[:self.max_output]

            # Extract error summary from stderr.
            error_summary = None
            if proc.returncode != 0 and stderr:
                lines = stderr.strip().split('\n')
                error_summary = lines[-1] if lines else stderr[:200]

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                success=proc.returncode == 0,
                exit_code=proc.returncode,
                timed_out=False,
                error_summary=error_summary,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout} seconds.",
                success=False,
                exit_code=-1,
                timed_out=True,
                error_summary=f"Timeout after {self.timeout}s",
            )

        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                success=False,
                exit_code=-1,
                timed_out=False,
                error_summary=str(e),
            )

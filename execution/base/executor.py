from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

from .result import ExecutionResult
from .task import ExecutionTask


@dataclass
class TaskExecutor:
    handlers: Dict[str, Callable[[ExecutionTask], ExecutionResult]] = field(default_factory=dict)

    def register(self, task_type: str, handler: Callable[[ExecutionTask], ExecutionResult]) -> None:
        if not task_type.strip():
            raise ValueError("task_type cannot be empty")
        self.handlers[task_type] = handler

    def execute(self, task: ExecutionTask) -> ExecutionResult:
        task.validate()

        if task.task_type not in self.handlers:
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                message=f"No handler registered for task_type='{task.task_type}'",
                error="missing_handler",
            )

        handler = self.handlers[task.task_type]
        return handler(task)
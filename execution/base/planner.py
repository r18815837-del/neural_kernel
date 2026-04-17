from __future__ import annotations

from dataclasses import dataclass
from typing import List

from runtime.specs.project_spec import ProjectSpec
from .task import ExecutionTask


@dataclass
class TaskPlanner:
    def plan_project(
        self,
        project_spec: ProjectSpec,
        output_root: str = "build",
    ) -> List[ExecutionTask]:
        project_spec.validate()

        tasks: List[ExecutionTask] = []

        tasks.append(
            ExecutionTask(
                task_id="task_request_analysis",
                title="Analyze project request",
                task_type="request_analysis",
                description=f"Analyze request for project '{project_spec.project_name}'",
                inputs={
                    "project_name": project_spec.project_name,
                    "summary": project_spec.summary,
                    "project_type": project_spec.project_type,
                },
            )
        )

        tasks.append(
            ExecutionTask(
                task_id="task_product_scope",
                title="Define project scope",
                task_type="product_scope",
                description="Define scope, priorities and required features",
                inputs={
                    "project_name": project_spec.project_name,
                    "features": project_spec.features,
                },
                dependencies=["task_request_analysis"],
            )
        )

        tasks.append(
            ExecutionTask(
                task_id="task_prepare_structure",
                title="Prepare project structure",
                task_type="project_structure",
                description=f"Create base structure for project '{project_spec.project_name}'",
                inputs={
                    "project_name": project_spec.project_name,
                    "project_type": project_spec.project_type,
                    "output_format": project_spec.output_format,
                    "tech_stack": project_spec.tech_stack,
                },
                dependencies=["task_product_scope"],
            )
        )

        if project_spec.tech_stack is not None:
            tasks.append(
                ExecutionTask(
                    task_id="task_prepare_stack",
                    title="Prepare tech stack files",
                    task_type="tech_stack_setup",
                    description="Prepare stack-specific files and configuration",
                    inputs={
                        "tech_stack": project_spec.tech_stack,
                    },
                    dependencies=["task_prepare_structure"],
                )
            )

        for idx, feature in enumerate(project_spec.features, start=1):
            tasks.append(
                ExecutionTask(
                    task_id=f"task_feature_{idx}",
                    title=f"Implement feature: {feature.name}",
                    task_type="feature_implementation",
                    description=feature.description,
                    inputs={
                        "feature": feature,
                        "project_name": project_spec.project_name,
                    },
                    dependencies=["task_prepare_structure"],
                )
            )

        tasks.append(
            ExecutionTask(
                task_id="task_docs_generation",
                title="Generate project documentation",
                task_type="docs_generation",
                description="Generate README and supporting docs",
                inputs={
                    "project_name": project_spec.project_name,
                },
                dependencies=["task_prepare_structure"],
            )
        )

        tasks.append(
            ExecutionTask(
                task_id="task_quality_check",
                title="Validate generated project",
                task_type="quality_check",
                description="Run basic validation checks for generated project",
                inputs={
                    "project_name": project_spec.project_name,
                    "root_path": f"{output_root}/{project_spec.project_name}",
                    "project_spec": project_spec,
                },
                dependencies=[task.task_id for task in tasks],
            )
        )

        tasks.append(
            ExecutionTask(
                task_id="task_finalize_artifact",
                title="Finalize artifact",
                task_type="artifact_finalize",
                description="Finalize project files and prepare output artifact",
                inputs={
                    "project_name": project_spec.project_name,
                    "output_format": project_spec.output_format,
                },
                dependencies=["task_quality_check"],
            )
        )

        return tasks
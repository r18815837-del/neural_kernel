from __future__ import annotations

from dataclasses import dataclass

from company_agents.workflows.project_generation import ProjectGenerationWorkflow
from runtime.base.result import TaskResult
from runtime.specs.client_request import ClientRequest
from runtime.specs.requirement_parser import RequirementParser


@dataclass
class AssistantManager:
    requirement_parser: RequirementParser
    project_workflow: ProjectGenerationWorkflow

    def generate_project(
        self,
        client_request: ClientRequest,
        output_root: str,
    ) -> TaskResult:
        brief, project_spec, artifact_spec = self.requirement_parser.parse(client_request)

        result = self.project_workflow.run(
            project_spec=project_spec,
            artifact_spec=artifact_spec,
            output_root=output_root,
        )

        result.payload["client_brief"] = {
            "title": brief.title,
            "summary": brief.summary,
            "goals": brief.goals,
            "constraints": brief.constraints,
            "requested_features": brief.requested_features,
            "target_users": brief.target_users,
            "delivery_format": brief.delivery_format,
            "metadata": brief.metadata,
        }

        result.payload["project_spec"] = {
            "project_name": project_spec.project_name,
            "summary": project_spec.summary,
            "project_type": project_spec.project_type,
            "output_format": project_spec.output_format,
            "target_platforms": project_spec.target_platforms,
            "constraints": project_spec.constraints,
            "tech_stack": {
                "backend": project_spec.tech_stack.backend if project_spec.tech_stack else None,
                "frontend": project_spec.tech_stack.frontend if project_spec.tech_stack else None,
                "database": project_spec.tech_stack.database if project_spec.tech_stack else None,
                "mobile": project_spec.tech_stack.mobile if project_spec.tech_stack else None,
                "deployment": project_spec.tech_stack.deployment if project_spec.tech_stack else None,
                "testing": project_spec.tech_stack.testing if project_spec.tech_stack else [],
                "integrations": project_spec.tech_stack.integrations if project_spec.tech_stack else [],
            },
            "metadata": project_spec.metadata,
        }

        result.payload["artifact_spec"] = {
            "artifact_name": artifact_spec.artifact_name,
            "artifact_type": artifact_spec.artifact_type,
            "files": artifact_spec.files,
            "include_readme": artifact_spec.include_readme,
            "include_tests": artifact_spec.include_tests,
            "include_env_example": artifact_spec.include_env_example,
            "packaging": artifact_spec.packaging,
            "metadata": artifact_spec.metadata,
        }

        return result
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for absolute imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from company_agents.base.orchestrator import AgentOrchestrator
from company_agents.base.planner import AgentPlanner
from company_agents.company import (
    ArchitectAgent,
    BackendAgent,
    DocsAgent,
    IntakeAgent,
    ProductManagerAgent,
    QAAgent,
    ReleaseAgent,
)
from company_agents.company.llm_architect_agent import LLMArchitectAgent
from company_agents.company.llm_backend_agent import LLMBackendAgent
from company_agents.company.llm_docs_agent import LLMDocsAgent
from company_agents.company.llm_product_manager_agent import LLMProductManagerAgent
from company_agents.company.llm_qa_agent import LLMQAAgent
from company_agents.company.llm_release_agent import LLMReleaseAgent
from company_agents.workflows.project_generation import ProjectGenerationWorkflow
from nk_app.core.assistant_manager import AssistantManager
from artifacts import ArtifactBuilder, ArtifactValidator, ArtifactWriter, ZipPackager
from artifacts.generators import (
    DocsGenerator,
    EnvGenerator,
    FeatureCodeGenerator,
    MainAppGenerator,
    StructureGenerator,
    TestsGenerator,
    DockerGenerator,
    CICDGenerator,
    DependencyGenerator,
)
from execution.base.planner import TaskPlanner
from runtime.specs.client_request import ClientRequest
from runtime.specs.requirement_parser import RequirementParser
from runtime.specs.llm_requirement_parser import LLMRequirementParser
from llm.config import LLMConfig
from llm.factory import create_llm_client

logger = logging.getLogger(__name__)


def build_assistant_manager() -> AssistantManager:
    """Build standard AssistantManager with regex-based agents.

    Returns:
        AssistantManager instance with non-LLM agents.
    """
    orchestrator = AgentOrchestrator()
    orchestrator.register(IntakeAgent())
    orchestrator.register(ProductManagerAgent())
    orchestrator.register(ArchitectAgent())
    orchestrator.register(BackendAgent())
    orchestrator.register(DocsAgent())
    orchestrator.register(QAAgent())
    orchestrator.register(ReleaseAgent())

    artifact_builder = ArtifactBuilder(
        structure_generator=StructureGenerator(),
        docs_generator=DocsGenerator(),
        env_generator=EnvGenerator(),
        tests_generator=TestsGenerator(),
        writer=ArtifactWriter(),
        validator=ArtifactValidator(),
        zip_packager=ZipPackager(),
        feature_code_generator=FeatureCodeGenerator(),
        main_app_generator=MainAppGenerator(),
        docker_generator=DockerGenerator(),
        cicd_generator=CICDGenerator(),
        deps_generator=DependencyGenerator(),
    )

    workflow = ProjectGenerationWorkflow(
        task_planner=TaskPlanner(),
        agent_planner=AgentPlanner(),
        orchestrator=orchestrator,
        artifact_builder=artifact_builder,
    )

    return AssistantManager(
        requirement_parser=RequirementParser(),
        project_workflow=workflow,
    )


def build_llm_assistant_manager() -> AssistantManager | None:
    """Build AssistantManager with LLM-powered agents.

    Uses LLMConfig from environment to create LLM client.
    Falls back to non-LLM manager if LLM is not available.

    Returns:
        AssistantManager with LLM agents, or None if LLM unavailable.
    """
    try:
        # Try to create LLM config and client
        config = LLMConfig.from_env()
        llm_client = create_llm_client(config)

        if not llm_client:
            logger.info(
                "LLM client unavailable, falling back to regex-based agents"
            )
            return None

        logger.info(f"Creating LLM-powered manager with {config.provider}")

        # Create orchestrator with LLM agents
        orchestrator = AgentOrchestrator()
        orchestrator.register(IntakeAgent())
        orchestrator.register(LLMProductManagerAgent(llm_client=llm_client))
        orchestrator.register(LLMArchitectAgent(llm_client=llm_client))
        orchestrator.register(LLMBackendAgent(llm_client=llm_client))
        orchestrator.register(LLMDocsAgent(llm_client=llm_client))
        orchestrator.register(LLMQAAgent(llm_client=llm_client))
        orchestrator.register(LLMReleaseAgent(llm_client=llm_client))

        artifact_builder = ArtifactBuilder(
            structure_generator=StructureGenerator(),
            docs_generator=DocsGenerator(),
            env_generator=EnvGenerator(),
            tests_generator=TestsGenerator(),
            writer=ArtifactWriter(),
            validator=ArtifactValidator(),
            zip_packager=ZipPackager(),
            feature_code_generator=FeatureCodeGenerator(),
            main_app_generator=MainAppGenerator(),
            docker_generator=DockerGenerator(),
            cicd_generator=CICDGenerator(),
            deps_generator=DependencyGenerator(),
        )

        workflow = ProjectGenerationWorkflow(
            task_planner=TaskPlanner(),
            agent_planner=AgentPlanner(),
            orchestrator=orchestrator,
            artifact_builder=artifact_builder,
        )

        return AssistantManager(
            requirement_parser=LLMRequirementParser(
                llm_client=llm_client,
                use_llm=True,
            ),
            project_workflow=workflow,
        )

    except Exception as e:
        logger.warning(f"Failed to create LLM manager: {e}")
        return None


def main() -> None:
    """Run project generation with LLM if available, fallback to regex."""
    request_text = (
        "Сделай CRM для салона красоты с backend на FastAPI, "
        "frontend на React, базой PostgreSQL, админкой, ролями, "
        "клиентской базой и экспортом. Нужен zip."
    )

    # Try to use LLM manager first
    manager = build_llm_assistant_manager()

    # Fall back to standard manager if LLM unavailable
    if manager is None:
        logger.info("Using regex-based manager")
        manager = build_assistant_manager()
    else:
        logger.info("Using LLM-powered manager")

    request = ClientRequest(raw_text=request_text)

    result = manager.generate_project(
        client_request=request,
        output_root="build",
    )

    print("SUCCESS:", result.success)
    print("MESSAGE:", result.message)
    print("ARTIFACTS:", result.artifacts)

    # Use scaffold validation from the workflow itself (same parser/spec)
    scaffold_validation = result.payload.get("scaffold_validation", {})
    print("SCAFFOLD VALID:", result.success and "missing_files" not in str(scaffold_validation))
    print("SCAFFOLD DETAILS:", scaffold_validation)

    # Print key payload info
    project_info = result.payload.get("project_spec", {})
    print("PROJECT:", project_info.get("project_name", "unknown"))
    print("ARTIFACT PATH:", result.payload.get("artifact_path", "N/A"))

if __name__ == "__main__":
    main()
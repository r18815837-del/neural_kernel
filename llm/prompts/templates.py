"""Message templates for LLM agent roles with chain-of-agents context passing.

Each builder constructs a message list (system + user) that includes context
from previous pipeline stages, enabling agents to build on each other's work:
  Parser → PM → Architect → Backend → Docs → QA → Release
"""

from __future__ import annotations

import json
from typing import Any

from llm.prompts.system_prompts import (
    ARCHITECT_PROMPT,
    BACKEND_PROMPT,
    CODE_GENERATOR_PROMPT,
    DOCS_PROMPT,
    PRODUCT_MANAGER_PROMPT,
    QA_PROMPT,
    RELEASE_PROMPT,
    REQUIREMENT_PARSER_PROMPT,
)


def _safe_json(obj: Any, indent: int = 2) -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


def build_parse_requirements_prompt(raw_text: str) -> list[dict[str, str]]:
    """Build prompt for parsing user requirements into structured format."""
    return [
        {"role": "system", "content": REQUIREMENT_PARSER_PROMPT},
        {
            "role": "user",
            "content": f"""Analyze this project request and return structured requirements as JSON:

---
{raw_text}
---

Ensure your response is ONLY valid JSON with no markdown formatting or additional text.""",
        },
    ]


def build_product_manager_prompt(
    project_spec_dict: dict,
    *,
    parser_context: dict | None = None,
) -> list[dict[str, str]]:
    """Build prompt for product management analysis.

    Args:
        project_spec_dict: Project spec with features, type, etc.
        parser_context: Optional context from requirement parser (title, summary, goals).
    """
    features = ", ".join(
        [f.get("name", "unknown") for f in project_spec_dict.get("features", [])]
    )
    target_users = ", ".join(
        project_spec_dict.get("metadata", {}).get("target_users", ["end_user"])
    )

    context_section = ""
    if parser_context:
        context_section = f"""
## Context from Requirement Analysis
- Project Type: {parser_context.get('project_type', 'application')}
- Complexity: {parser_context.get('complexity', 'medium')}
- Goals: {', '.join(parser_context.get('goals', []))}
- Constraints: {', '.join(parser_context.get('constraints', []))}
"""

    user_content = f"""Analyze features for this project and provide strategic guidance:

## Project Info
- Project Name: {project_spec_dict.get('project_name', 'unnamed')}
- Project Type: {project_spec_dict.get('project_type', 'application')}
- Features requested: {features}
- Target Users: {target_users}
{context_section}
Provide comprehensive analysis including MVP scope, feature dependencies,
missing features, user stories, complexity estimates, and implementation phases.

Return ONLY valid JSON."""

    return [
        {"role": "system", "content": PRODUCT_MANAGER_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_architect_prompt(
    project_spec_dict: dict,
    *,
    pm_context: dict | None = None,
) -> list[dict[str, str]]:
    """Build prompt for architecture design.

    Args:
        project_spec_dict: Project spec dict.
        pm_context: Optional context from product manager (MVP, dependencies, phases).
    """
    features = ", ".join(
        [f.get("name", "unknown") for f in project_spec_dict.get("features", [])]
    )
    tech_stack = _safe_json(project_spec_dict.get("tech_stack", {}))

    context_section = ""
    if pm_context:
        mvp = ", ".join(pm_context.get("mvp_features", []))
        deps = _safe_json(pm_context.get("feature_dependencies", {}))
        phases = _safe_json(pm_context.get("implementation_phases", []))
        missing = _safe_json(pm_context.get("missing_features", []))
        context_section = f"""
## Product Manager Analysis (use this to guide your architecture)
- MVP Features: {mvp}
- Feature Dependencies: {deps}
- Implementation Phases: {phases}
- Suggested Missing Features: {missing}
"""

    user_content = f"""Design the complete system architecture for this project:

## Project Specification
- Project Name: {project_spec_dict.get('project_name', 'unnamed')}
- Project Type: {project_spec_dict.get('project_type', 'application')}
- Features: {features}
- Tech Stack: {tech_stack}
- Scale: {project_spec_dict.get('scale', 'medium')}
{context_section}
Design a production-ready architecture with directories, files, database schema,
API endpoints, design patterns, and security considerations.

Return ONLY valid JSON."""

    return [
        {"role": "system", "content": ARCHITECT_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_backend_code_prompt(
    feature_name: str,
    project_context: dict,
    *,
    architect_context: dict | None = None,
    pm_context: dict | None = None,
) -> list[dict[str, str]]:
    """Build prompt for generating backend code for a feature.

    Args:
        feature_name: Name of the feature to implement.
        project_context: Context dict with project_name, tech_stack, etc.
        architect_context: Optional architecture design (dirs, files, DB schema, endpoints).
        pm_context: Optional PM analysis (MVP scope, user stories).
    """
    backend = project_context.get("tech_stack", {}).get("backend", "Python/FastAPI")
    database = project_context.get("tech_stack", {}).get("database", "PostgreSQL")

    context_section = ""
    if architect_context:
        dirs = ", ".join(architect_context.get("recommended_directories", [])[:15])
        db_schema = _safe_json(architect_context.get("database_schema", {}))
        endpoints = _safe_json(architect_context.get("api_endpoints", []))
        patterns = ", ".join(architect_context.get("design_patterns", []))
        context_section += f"""
## Architecture Design (follow this structure)
- Directory Structure: {dirs}
- Design Patterns: {patterns}
- Database Schema: {db_schema}
- API Endpoints to Implement: {endpoints}
"""

    if pm_context:
        user_story = pm_context.get("user_stories", {}).get(feature_name, "")
        deps = pm_context.get("feature_dependencies", {}).get(feature_name, [])
        complexity = pm_context.get("feature_complexity", {}).get(feature_name, {})
        context_section += f"""
## Product Context
- User Story: {user_story}
- Dependencies: {', '.join(deps) if deps else 'none'}
- Complexity: {_safe_json(complexity) if isinstance(complexity, dict) else complexity}
"""

    user_content = f"""Generate production-ready backend code for this feature:

## Feature
- Name: {feature_name}
- Project: {project_context.get('project_name', 'unnamed')}
- Backend Framework: {backend}
- Database: {database}
- Description: {project_context.get('description', '')}
{context_section}
Requirements:
1. Complete, working code — no placeholders or TODOs
2. Proper error handling, validation, and logging
3. Type hints and docstrings on all functions
4. Follow {backend} best practices
5. Include models, schemas, services, repositories, and routes
6. Use proper HTTP methods and status codes
7. Handle edge cases (not found, duplicate, invalid input)

Return ONLY valid JSON with complete Python code as string values."""

    return [
        {"role": "system", "content": BACKEND_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_docs_prompt(
    project_spec_dict: dict,
    features: list[dict],
    *,
    architect_context: dict | None = None,
    backend_context: dict | None = None,
) -> list[dict[str, str]]:
    """Build prompt for generating project documentation.

    Args:
        project_spec_dict: Project info dict.
        features: List of feature dicts.
        architect_context: Optional architecture design for docs accuracy.
        backend_context: Optional backend implementation details.
    """
    feature_list = ", ".join([f.get("name", "unknown") for f in features])
    tech_stack = _safe_json(project_spec_dict.get("tech_stack", {}))
    target_users = ", ".join(
        project_spec_dict.get("target_users", ["developers", "end users"])
    )

    context_section = ""
    if architect_context:
        dirs = _safe_json(architect_context.get("recommended_directories", []))
        endpoints = _safe_json(architect_context.get("api_endpoints", []))
        context_section += f"""
## Architecture (use for accurate documentation)
- Directory Structure: {dirs}
- API Endpoints: {endpoints}
- Design Patterns: {', '.join(architect_context.get('design_patterns', []))}
"""

    if backend_context:
        deps = backend_context.get("all_dependencies", [])
        files = backend_context.get("all_generated_files", [])
        context_section += f"""
## Implementation Details
- Dependencies: {', '.join(deps) if deps else 'standard'}
- Generated Files: {', '.join(files[:20]) if files else 'standard scaffold'}
"""

    user_content = f"""Generate comprehensive documentation for this project:

## Project
- Name: {project_spec_dict.get('project_name', 'unnamed')}
- Summary: {project_spec_dict.get('summary', '')}
- Features: {feature_list}
- Tech Stack: {tech_stack}
- Target Users: {target_users}
{context_section}
Create complete documentation including README, API docs, contributing guide,
architecture overview, and deployment guide.

Return ONLY valid JSON with complete markdown content."""

    return [
        {"role": "system", "content": DOCS_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_qa_test_prompt(
    project_spec_dict: dict,
    features: list[dict],
    *,
    architect_context: dict | None = None,
    backend_context: dict | None = None,
) -> list[dict[str, str]]:
    """Build prompt for QA test generation.

    Args:
        project_spec_dict: Project info dict.
        features: List of feature dicts.
        architect_context: Optional architecture design.
        backend_context: Optional backend code details.
    """
    feature_list = ", ".join([f.get("name", "unknown") for f in features])
    tech_stack = _safe_json(project_spec_dict.get("tech_stack", {}))

    context_section = ""
    if architect_context:
        endpoints = _safe_json(architect_context.get("api_endpoints", []))
        db_schema = _safe_json(architect_context.get("database_schema", {}))
        context_section += f"""
## Architecture Context
- API Endpoints to Test: {endpoints}
- Database Schema: {db_schema}
"""

    if backend_context:
        test_hints = backend_context.get("all_test_hints", [])
        context_section += f"""
## Implementation Hints
- Test Scenarios from Backend: {_safe_json(test_hints)}
"""

    user_content = f"""Create comprehensive testing strategy and test code for this project:

## Project
- Name: {project_spec_dict.get('project_name', 'unnamed')}
- Features: {feature_list}
- Tech Stack: {tech_stack}
{context_section}
Generate complete test files with fixtures, unit tests, integration tests,
and quality gates.

Return ONLY valid JSON."""

    return [
        {"role": "system", "content": QA_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_release_prompt(
    project_spec_dict: dict,
    *,
    architect_context: dict | None = None,
    qa_context: dict | None = None,
    all_generated_files: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build prompt for release engineering.

    Args:
        project_spec_dict: Project info dict.
        architect_context: Optional architecture design.
        qa_context: Optional QA results and test coverage.
        all_generated_files: Optional list of all files in the project.
    """
    tech_stack = _safe_json(project_spec_dict.get("tech_stack", {}))

    context_section = ""
    if architect_context:
        context_section += f"""
## Architecture
- Design Patterns: {', '.join(architect_context.get('design_patterns', []))}
- Security Notes: {_safe_json(architect_context.get('security_notes', []))}
"""

    if qa_context:
        context_section += f"""
## QA Results
- Quality Gates: {_safe_json(qa_context.get('quality_gates', []))}
- Test Coverage: {qa_context.get('coverage_target', 'unknown')}%
"""

    if all_generated_files:
        context_section += f"""
## Project Files ({len(all_generated_files)} total)
- Key Files: {', '.join(all_generated_files[:20])}
"""

    user_content = f"""Prepare release package for this project:

## Project
- Name: {project_spec_dict.get('project_name', 'unnamed')}
- Tech Stack: {tech_stack}
- Output Format: {project_spec_dict.get('output_format', 'zip')}
{context_section}
Generate release notes, changelog, deployment config (Dockerfile, docker-compose,
CI/CD), and deployment checklist.

Return ONLY valid JSON."""

    return [
        {"role": "system", "content": RELEASE_PROMPT},
        {"role": "user", "content": user_content},
    ]

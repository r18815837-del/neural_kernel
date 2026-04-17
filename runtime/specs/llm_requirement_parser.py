"""LLM-powered requirement parser with fallback to regex-based parsing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from llm.base import BaseLLMClient
from llm.prompts.templates import build_parse_requirements_prompt
from runtime.specs.client_brief import ClientBrief
from runtime.specs.client_request import ClientRequest
from runtime.specs.feature_spec import FeatureSpec
from runtime.specs.project_spec import ProjectSpec
from runtime.specs.requirement_parser import RequirementParser
from runtime.specs.tech_stack_spec import TechStackSpec

logger = logging.getLogger(__name__)


@dataclass
class LLMRequirementParser(RequirementParser):
    """LLM-powered requirement parser that extends regex-based parsing.

    Falls back to regex parsing if LLM is unavailable or fails.
    """

    llm_client: BaseLLMClient | None = None
    use_llm: bool = True

    def parse_brief(self, request: ClientRequest) -> ClientBrief:
        """Parse client request into structured brief.

        Uses LLM if available, falls back to regex-based parsing.

        Args:
            request: ClientRequest with raw_text.

        Returns:
            ClientBrief with extracted information.
        """
        if self.llm_client and self.use_llm:
            try:
                return self._parse_brief_with_llm(request)
            except Exception as e:
                logger.warning(
                    f"LLM parsing failed, falling back to regex: {e}"
                )
                return super().parse_brief(request)

        return super().parse_brief(request)

    def brief_to_project_spec(self, brief: ClientBrief) -> ProjectSpec:
        """Convert brief to project spec.

        LLM can enhance feature descriptions and priorities if available.

        Args:
            brief: ClientBrief instance.

        Returns:
            ProjectSpec with enhanced feature information.
        """
        if self.llm_client and self.use_llm:
            try:
                return self._enhance_project_spec_with_llm(brief)
            except Exception as e:
                logger.warning(
                    f"LLM enhancement failed, using standard conversion: {e}"
                )

        return super().brief_to_project_spec(brief)

    def _parse_brief_with_llm(self, request: ClientRequest) -> ClientBrief:
        """Parse brief using LLM.

        Args:
            request: ClientRequest with raw_text.

        Returns:
            ClientBrief.

        Raises:
            ValueError: If LLM response is invalid.
            RuntimeError: If LLM call fails.
        """
        if not self.llm_client:
            raise RuntimeError("LLM client not available")

        raw_text = request.raw_text.strip()
        if not raw_text:
            raise ValueError("ClientRequest.raw_text cannot be empty")

        # Build and send prompt
        messages = build_parse_requirements_prompt(raw_text)

        try:
            response = self.llm_client.complete_json_sync(messages)
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e

        # Extract and validate response
        title = response.get("title", "Project")
        summary = response.get("summary", raw_text)
        goals = response.get("goals", [summary])
        constraints = response.get("constraints", [])
        target_users = response.get("target_users", ["end_user"])
        delivery_format = response.get("delivery_format", "zip")

        # Extract features with LLM-provided descriptions
        requested_features = []
        llm_features = response.get("features", [])

        for feature_dict in llm_features:
            feature_name = feature_dict.get("name", "")
            if feature_name:
                requested_features.append(feature_name)

        if not requested_features:
            requested_features.append("core_application")

        return ClientBrief(
            title=title,
            summary=summary,
            goals=goals if goals else [summary],
            constraints=constraints,
            requested_features=requested_features,
            target_users=target_users,
            delivery_format=delivery_format or self.default_output_format,
            metadata={
                "source": "llm_requirement_parser",
                "request_user_id": request.user_id,
                "request_session_id": request.session_id,
                "llm_features": llm_features,  # Store LLM data for later use
            },
        )

    def _enhance_project_spec_with_llm(
        self, brief: ClientBrief
    ) -> ProjectSpec:
        """Enhance project spec with LLM-provided descriptions and priorities.

        Args:
            brief: ClientBrief instance.

        Returns:
            Enhanced ProjectSpec.
        """
        # Get LLM features data if available
        llm_features = brief.metadata.get("llm_features", [])

        # Create features with LLM descriptions
        features = []
        for llm_feature in llm_features:
            name = llm_feature.get("name", "")
            if name:
                features.append(
                    FeatureSpec(
                        name=self._normalize_feature_name(name),
                        description=llm_feature.get(
                            "description",
                            self._describe_feature(name)
                        ),
                        priority=llm_feature.get("priority", "high"),
                        required=llm_feature.get("priority") in ["high", "critical"],
                    )
                )

        # Fallback if no LLM features
        if not features:
            for feature in brief.requested_features:
                features.append(
                    FeatureSpec(
                        name=self._normalize_feature_name(feature),
                        description=self._describe_feature(feature),
                        priority="high",
                        required=True,
                    )
                )

        tech_stack = self._infer_tech_stack(brief.summary)
        project_name = self._slugify(brief.title)

        return ProjectSpec(
            project_name=project_name,
            summary=brief.summary,
            project_type=self.default_project_type,
            features=features,
            tech_stack=tech_stack,
            output_format=brief.delivery_format or self.default_output_format,
            target_platforms=["web"],
            constraints=brief.constraints,
            metadata={
                "source": "llm_requirement_parser",
                "target_users": brief.target_users,
            },
        )
